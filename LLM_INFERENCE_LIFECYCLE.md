# How LLM Inference Works — End to End

A step-by-step trace of everything that happens from `ollama run gemma` (or a hosted model like
Claude Opus being loaded) through a user typing a prompt to tokens streaming back. Nothing
skipped. It ties together the pieces covered in the sibling docs:

- `PREFIX_CACHING_EXPLAINED.md` — token ids → embeddings → K/V → the KV cache, and the engine
  prefix caches.
- `PREFIX_PROMPT_CACHING_API.md` — the API prompt-cache layer.

Two deployment shapes are referenced throughout; the pipeline is the same, the scale differs:

- **Local** (`ollama run gemma`) — one user, llama.cpp/GGUF, your machine's CPU/Metal/GPU.
- **Hosted** (Claude Opus, an API endpoint) — many tenants, vLLM/TensorRT-LLM-class server,
  continuous batching, cross-request prefix caching, multi-GPU.

---

## The whole thing in one timeline

```
 ╔═══════════════ PHASE 0: LOAD (once per server/process start) ═══════════════╗
 ║ resolve model → read config → load weights to VRAM → carve KV pool →         ║
 ║ load tokenizer → warm up kernels → READY                                     ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
                                    │  (server now idle, waiting)
 ╔═══════════════ PER REQUEST ══════▼═══════════════════════════════════════════╗
 ║ 1 receive   → 2 chat template → 3 tokenize → 4 prefix-cache lookup →          ║
 ║ 5 schedule/batch →                                                           ║
 ║ ┌── PHASE A: PREFILL (process the whole prompt at once) ──────────────┐       ║
 ║ │ embed → L× (norm,QKV,attention,FFN) → write K/V to cache →           │       ║
 ║ │ logits for the LAST token only                                      │       ║
 ║ └──────────────────────────────────┬──────────────────────────────────┘       ║
 ║ ┌── PHASE B: DECODE (loop, one token per step) ◄──┐                  │        ║
 ║ │ sample next token → embed it → 1× forward over   │ repeat until     │        ║
 ║ │ ONE position → append K/V → logits → detokenize  │ EOS/stop/max     ◄┘       ║
 ║ │ → stream token out ──────────────────────────────┘                          ║
 ║ └────────────────────────────────────────────────────────────────────┐       ║
 ║ 6 finish → release/Release KV (prefix cache keeps shared blocks) → usage stats ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
```

Two performance regimes to keep in mind from the start:

- **Prefill** is **compute-bound** — one big parallel pass over all prompt tokens. Sets
  **TTFT** (time to first token).
- **Decode** is **memory-bandwidth-bound** — one token at a time, re-reading all weights + KV
  each step. Sets **TPOT/ITL** (time per output token / inter-token latency).

---

# PHASE 0 — Loading the model (`ollama run gemma` / server boot)

This happens **once**, before any prompt. It's the slow "cold start."

### Step 0.1 — Resolve and fetch the model

```
 model name ("gemma:7b", "claude-opus") ─► locate checkpoint files
   - weights:    *.safetensors (HF) or *.gguf (ollama/llama.cpp)
   - config:     config.json (HF) — architecture hyperparameters
                 (GGUF embeds this metadata inside the file)
   - tokenizer:  tokenizer.json / tokenizer.model / vocab (or embedded in GGUF)
```

`ollama run gemma` first **pulls** the GGUF blob from a registry if not cached locally, then
proceeds. A hosted service has the weights already on local NVMe.

### Step 0.2 — Read the config (the architecture)

The config fixes the **shapes** of every tensor — the "wiring diagram" from the weights
discussion:

```
 num_layers (L)        e.g. 28        n_heads / n_kv_heads   e.g. 16 / 8   (GQA)
 hidden_size (d_model) e.g. 3072      head_dim               e.g. 256
 vocab_size            e.g. 256k      max_context_len        e.g. 8192
 activation (SwiGLU), norm (RMSNorm), positional (RoPE), rope_theta, ...
```

This is read first because it determines how to interpret the raw weight bytes and how big the
KV cache will be.

### Step 0.3 — Load weights into GPU memory

```
 checkpoint on disk ──(mmap, dtype-convert/dequant)──► weight tensors in VRAM
```

- **dtype / quantization.** Full precision is fp16/bf16 (2 bytes/param). Local GGUF models are
  usually **quantized** (Q4_K_M ≈ 4 bits/param) to fit consumer GPUs — smaller and faster, with
  a small quality cost. The weights are dequantized on the fly during matmuls (or kept in low
  precision with quantized kernels).
- **Sharding (big models).** Opus-scale weights don't fit on one GPU, so they're split across
  many: **tensor parallelism** (each matmul split across GPUs, results all-reduced) and/or
  **pipeline parallelism** (different layers on different GPUs). `ollama` on one machine usually
  doesn't shard.
- Result: `model.embed_tokens.weight` (E), every layer's `W_Q/W_K/W_V/W_O`, FFN matrices, norms,
  and `lm_head` are now resident in VRAM. **Fixed for the whole server lifetime.**

### Step 0.4 — Carve the KV cache pool from leftover VRAM

```
 total VRAM
   − model weights            (fixed)
   − activation/workspace reserve (scratch for the forward pass)
   − framework/CUDA overhead
   = KV CACHE POOL  ──► divided into fixed-size blocks (e.g. 16 tokens each)
```

This is the pre-allocated block pool from the storage section. Its size decides how many tokens
(across all concurrent + cached sequences) can be held at once — often the real cap on
throughput and context length. (See the KV-memory math: KV can rival or exceed the weights.)

### Step 0.5 — Load the tokenizer

The vocabulary (id ↔ subword bytes) is loaded into host memory: BPE / SentencePiece / Tiktoken
merge rules + the vocab table. Used to encode input and decode output. Never touches the GPU.

### Step 0.6 — Warm up

A dummy forward pass is run to: trigger cuBLAS/cutlass **kernel autotuning** for the model's
shapes, capture **CUDA graphs** (so the decode loop has low per-step launch overhead), and fault
in memory. After this the server reports **READY** and idles, waiting for requests.

> Local nuance: `ollama` keeps the model resident for a few minutes after use, then unloads to
> free RAM/VRAM — the next call pays the load cost again.

---

# PER-REQUEST PRELUDE — from prompt to schedulable work

### Step 1 — Receive the request

The API gets the messages and **decoding parameters**:

```
 messages:    [ {role: system, ...}, {role: user, "..."} ]
 params:      temperature, top_p, top_k, max_tokens, stop sequences,
              repetition/frequency/presence penalties, seed, stream?
```

### Step 2 — Apply the chat template

Raw messages are rendered into one token stream with the model's **special tokens** and role
markers. Each model family has its own template — using the wrong one quietly degrades quality.
Gemma, for example:

```
 <bos><start_of_turn>user
 {user message}<end_of_turn>
 <start_of_turn>model
```

The trailing `<start_of_turn>model` is the cue that it's the model's turn to generate. (Claude,
Llama, Qwen, etc. each have different markers.)

### Step 3 — Tokenize

The templated text → **token ids** (integers), via the vocabulary — the encode step from the
pipeline doc. A leading `<bos>` / special tokens are inserted per the template.

```
 "...<start_of_turn>model\n"  ─tokenizer.encode─►  [2, 106, 1645, 108, ... ]
```

### Step 4 — Prefix-cache lookup

Before doing any compute, check whether this prompt's prefix is already cached (this is where
the sibling docs plug in):

- **API prompt cache** (`PromptCache`): hash the prefix up to the cache breakpoint; if present
  and within TTL → its KV blocks are reused. (Anthropic/OpenAI prompt caching.)
- **Engine prefix cache** (`BlockHashCache` / radix): chained-hash the prompt blocks; reuse the
  longest shared block-aligned prefix's KV.

Whatever prefix is found cached → its prefill is **skipped**. The shared system prompt across
requests is the big win here.

### Step 5 — Schedule and batch

```
 admission → allocate KV blocks for the NEW (uncached) tokens → join the running batch
```

- **Continuous batching** (hosted): the scheduler doesn't wait for a batch to fill. Every
  iteration it can admit new requests and interleave their prefill with other requests' decode
  (**chunked prefill**). This is what keeps GPUs busy and is why your numerics depend on batch
  composition (the nondeterminism point).
- If the KV pool is full, the request **queues** or an existing request is **preempted** (its KV
  evicted/recomputed later).
- Local single-user: batch size 1, no contention.

---

# PHASE A — PREFILL (process the entire prompt)

Goal: compute the KV for every prompt token (so decode can attend to it) and the logits for the
**last** position (to pick the first output token). All prompt tokens are processed **in
parallel** — one large matmul per layer — which is why prefill is compute-bound.

### Step A.1 — Embed

```
 token ids ──E[id]──► embedding vectors   (one d_model row per token)
```

### Step A.2 — Run the transformer stack (×L layers)

For each layer, on the full set of prompt positions at once:

```
 h ──RMSNorm──► ─┬─► Wq → q ─┐
                 ├─► Wk → k ──┤ apply RoPE to q,k  (positional info)
                 └─► Wv → v ──┘
                              │
        causal self-attention: for each position t,
            scores = q_t · kⱼ /√d   for j ≤ t      (causal mask: no looking ahead)
            out_t  = Σ softmax(scores)·vⱼ
                              │
        Wo → attention output ──► + residual ──► RMSNorm ──►
        FFN (SwiGLU: gate,up → act → down) ──► + residual ──► h(next layer)
```

Key things happening here:

- **K and V for every position are written into the request's KV cache blocks** — this is the
  artifact decode will reuse. (Q is used now and discarded; only K/V persist — see "what saves
  the compute".)
- **Cached prefix tokens are NOT recomputed** — their K/V are read from the blocks the prefix
  cache handed back; only the uncached suffix is computed.
- Each layer's input is the previous layer's output, so by upper layers each token's
  representation has mixed in all earlier tokens (the contextualization that forces prefix-level
  cache keys).

### Step A.3 — Logits for the last token

You ran all positions, but to start generating you only need the next-token distribution at the
**final** position:

```
 h_last ──lm_head (= Eᵀ or a separate matrix)──► logits  (one score per vocab entry, size = vocab_size)
```

Prefill ends. The time to get here is **TTFT**. The KV cache now holds the whole prompt.

---

# PHASE B — DECODE (autoregressive generation loop)

Now generate one token at a time. Each iteration is a **single-position** forward pass that
**reads the entire KV cache**. This is memory-bandwidth-bound: every step re-reads all model
weights and all KV from VRAM, so throughput is capped by bandwidth, not FLOPs — hence batching
many sequences together to amortize the weight reads.

### Step B.1 — Sample the next token

Turn the last logits into a token (the deliberate-nondeterminism step):

```
 logits
   │  ÷ temperature            (T=0 → greedy/argmax; higher T → flatter, more random)
   │  repetition / frequency / presence penalties, logit_bias
   │  top_k  (keep k highest)   then  top_p / nucleus (keep smallest set summing to p)
   ▼
 softmax → probability distribution → sample one token id (using the RNG seed)
```

(With `temperature = 0` this collapses to `argmax` — but recall it's still not bitwise
reproducible across runs unless the kernels are batch-invariant; see the nondeterminism doc.)

### Step B.2 — Feed the token back through the model

```
 new token id ──E[id]──► embedding ──► ONE forward pass over a SINGLE position:
     per layer: norm → Wq/Wk/Wv → RoPE → attention reads ALL cached K/V (0..t) →
                Wo → +residual → FFN → +residual
     APPEND this token's (k,v) to the KV cache  (cache grows by one position)
   ──lm_head──► logits for the next token
```

The only new KV computed is for this one token; everything before it is read from the cache.
That's the payoff of the KV cache: decode is O(context) memory reads per token instead of
recomputing the whole prefix every step.

### Step B.3 — Detokenize and stream

```
 token id ──tokenizer.decode──► text fragment ──► stream to client (SSE)
```

Subtleties handled here: a single character can span multiple tokens (and one token can be a
partial UTF-8 byte sequence), so detok is **incremental** — it buffers until it has a complete
character. **Stop sequences** are matched against the decoded text *across* token boundaries.

### Step B.4 — Loop or stop

Go back to B.1 with the new logits. Stop when any of:

```
 • the model emits an EOS / end-of-turn token (it decided it's done)
 • a user stop sequence is produced
 • max_tokens reached
 • (server) the request is cancelled / times out
```

---

# PER-REQUEST EPILOGUE

### Step 6 — Release resources and report

```
 • the request's KV blocks are released:
     - blocks unique to this request → freed back to the pool
     - blocks shared via the prefix cache → refcount--, kept resident for the TTL
       so the next request with the same prefix still hits  (PromptCache.removeEntry → engine.Release)
 • usage is reported: input tokens, output tokens, and (if prompt caching) cache-read vs
   cache-write tokens — the saved-prefill accounting
```

The server returns to idle (or keeps serving other batched requests). The shared prefix's KV
lingers, so the *next* user who sends the same system prompt skips its prefill entirely.

---

# Cross-cutting: continuous batching (why it all overlaps)

The phases above are drawn per-request, but a hosted server runs them **interleaved across many
requests every iteration**:

```
 iteration N:
   ┌─ request P: prefill chunk (tokens 200–263)        ← compute-heavy
   ├─ request Q: decode step (emit 1 token)            ┐
   ├─ request R: decode step (emit 1 token)            ├ all in ONE batched
   └─ request S: decode step (emit 1 token)            ┘ GPU pass
 iteration N+1: admit new request T, R finished → freed, ...
```

- **Iteration-level scheduling**: requests join/leave the batch every step, not at fixed batch
  boundaries — keeps the GPU saturated.
- **Chunked prefill**: a long prompt's prefill is split across iterations so it doesn't stall
  everyone else's decode.
- This batching is exactly what makes a request's logits depend on concurrent traffic (the
  source-2 nondeterminism).

---

# Cross-cutting: the two regimes and what each optimization targets

| | **Prefill** | **Decode** |
|---|---|---|
| Work | all prompt tokens at once | one token per step |
| Bottleneck | compute (FLOPs) | memory bandwidth (read weights + KV/token) |
| Latency metric | TTFT (time to first token) | TPOT / ITL (per-token latency) |
| Helped by | prefix caching (skip it), chunked prefill, more FLOPs | bigger batches, KV quantization, GQA/MQA, speculative decoding, FlashAttention |

A few optimizations worth naming (where they slot in):

- **FlashAttention** — fuses the attention softmax to avoid materializing the big scores matrix;
  speeds up both phases, saves memory. (Phase A.2 / B.2.)
- **PagedAttention** — the block-pool KV management from the storage doc; avoids reserving
  max-length per request. (Step 0.4 / 5.)
- **Speculative decoding** — a small "draft" model proposes several tokens, the big model
  verifies them in one batched pass; turns several bandwidth-bound decode steps into one. (Phase
  B.)
- **Prefix caching** — skip prefill for shared prompt prefixes. (Step 4.)
- **Quantization** — smaller weights (fit/bandwidth) and smaller KV (more concurrency). (Step
  0.3 / 0.4.)

---

# Recap — the minimal mental model

```
 LOAD ONCE:   weights → VRAM (fixed) ; KV pool carved from what's left ; tokenizer ready
 PER PROMPT:
   text → template → token ids → (reuse cached prefix KV) → schedule
   PREFILL:  embed all tokens → L layers → write K/V to cache → logits for last token   [compute-bound, TTFT]
   DECODE:   loop { sample → embed 1 token → 1-position pass reading all KV → append K/V → detok → stream }
             until EOS / stop / max_tokens                                               [bandwidth-bound, ITL]
   FINISH:   release KV (shared prefix kept for TTL) → usage stats
```

Everything else — paging, continuous batching, FlashAttention, speculative decoding, prefix
caching, quantization — is engineering to make those two loops cheaper, more parallel, or more
shareable. The core is unchanged: **embed, project to Q/K/V, attend, FFN, repeat L layers,
sample, feed back.**

---

## Where the sibling docs fit

| Concept in this trace | Detailed in |
|---|---|
| id → embedding → K/V → KV cache; chained hashing; VRAM block pool | `PREFIX_CACHING_EXPLAINED.md` |
| the prefix-cache lookup at Step 4 (engine: block-hash & radix) | `PREFIX_CACHING_EXPLAINED.md` |
| the API prompt cache (TTL, breakpoint, what saves the compute) | `PREFIX_PROMPT_CACHING_API.md` |
| why decode sampling / batching is non-deterministic | (the nondeterminism discussion) |
| code models of the three caches | `block_hash_cache.go`, `radix_cache.go`, `prompt_cache.go` |

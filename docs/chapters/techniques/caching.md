# Caching

You already met the **KV cache** in [Mechanics](../models/llm-inference-mechanics.md): during prefill
the engine computes a key and value for every token and stores them, then decode reuses them so
attention is linear-per-step instead of quadratic. **Every engine does this by default, within a single
request** — without it, inference would be unbearably slow, recomputing all prior K/V for every new
token.

This section is about the *next* level of reuse: **sharing the KV cache across requests**, where to
physically keep it, and how to route traffic so the cache actually gets hit.

## 5.3.1 Prefix caching and KV-cache reuse

If two requests start with the **same tokens**, they produce the **same K/V** for those tokens — so
the second request can skip prefill on the shared part and *read the first request's KV cache instead*.
That's **prefix caching**, and it cuts TTFT.

```
 request 1:  [ Weather  in  SF   ? ]   ← full prefill, KV cached
 request 2:  [ Weather  in  NYC  ? ]
               └── shared ──┘ ▲
                             first token that differs → prefix ends here
   reuse KV for "Weather in" (skip its prefill); prefill "NYC ?" fresh
```

Two-token savings won't move TTFT. But prefix caching can skip prefill on *thousands* of tokens in the
right domains:

- **Complex system prompts** — agents, chatbots, RAG scaffolds, tool definitions: long, *identical*
  preambles on every call.
- **Code completion** — the same thousands of lines of file context passed every keystroke.
- **Documents & retrieval** — repeated context ahead of each user question.
- **Multi-turn chat** — every turn re-sends the whole prior conversation via the chat template.

This is the same mechanism behind pay-per-token APIs charging less for **"cache hit"** input tokens —
reading cached K/V costs almost nothing. You can exploit it on your own deployments.

### The rule that falls out: put novel tokens last

A prefix ends at the **first token that differs**. So *where* your unique content sits decides your
savings:

```
 GOOD  [ <long shared system prompt> | <user's new question> ]   prefix = whole preamble ✓
 BAD   [ <user's new question> | <long shared system prompt> ]   prefix ends at token 1 ✗
```

In the book's second example — `[SF, weather, today, ?]` vs `[NYC, weather, today, ?]` — the *first*
token already differs, so **zero** is cached even though three of four tokens match. Same tokens,
opposite outcome, purely from ordering.

!!! key "Context engineering is cache engineering"
    Because prefixes end at the first unique token, **how you order your context determines your TTFT
    savings.** Keep everything shared and static at the front (system prompt, tools, retrieved docs);
    push the user's novel tokens to the very end. This is a free, lossless latency win available purely
    by laying out your prompt well.

### What is *actually* cached, and what it's keyed on

A subtlety that trips people up: prefix caching keys on **token ids**, never on text and never on
embeddings — and specifically on the **prefix of ids**, via a *chained* hash.[^prefix]

```
 "Weather in SF?"
    │ tokenizer encode (text → ids)
    ▼
 token ids  [ 15494, 304, 8765, 30 ]      ◄── the cache keys on THESE (ids), as a prefix
    │ per layer: project to K, V
    ▼
 K/V tensors per token, per layer         ◄── the cache VALUE (what's reused)
```

- **Keyed on the token-id prefix, chained.** Because each token's K/V depends on *every* token before
  it (causal attention), the cache key for position *n* must fold in ids `0..n` — a hash chained over
  the prefix. Two requests share a cache entry only where their id prefixes are byte-identical. A single
  different earlier token changes the model's internal representation of *everything* after it, so the
  prefix genuinely breaks there — even if the text looks the same to a human.
- **The value is K/V tensors, living in paged GPU memory.** vLLM's **PagedAttention** carves GPU memory
  into fixed-size **blocks** and stores K/V there; the lookup map holds only a **handle**
  (`hash → block_id`), not the tensors inline — OS-style paging that kills fragmentation. SGLang's
  **RadixAttention** indexes prefixes in a radix tree for the same end.
- **No text anywhere in the cache.** Text exists only at the tokenizer boundary (encode in, decode out).
  Everything inside is ids and tensors.

!!! info "Beyond prefixes: mid-sequence reuse"
    Prefix caching dominates *because* LLMs are autoregressive — a single novel token rewrites the
    representation downstream, so only true prefixes are safely reusable. Active research (CacheBlend,
    LMCache) reuses **non-prefix** chunks by correcting positional embeddings and selectively
    recomputing the entries that would otherwise be wrong — expanding reuse to arbitrary shared blocks
    (e.g. the same retrieved document appearing mid-prompt across requests).

## 5.3.2 Where to store the KV cache

The KV cache is valuable but **large**, and GPU VRAM is scarce. You configure how much VRAM the engine
hands to the cache — e.g. TensorRT-LLM's `kv_cache_config.free_gpu_memory_fraction = 0.8`. On a 180 GB
B200 using 100 GB for weights/buffers, that's ~64 GB for cache — which **fills fast**, and once full,
you evict, raising future miss rates.

To get more room, **offload** down the memory hierarchy — four tiers, descending bandwidth:

| Tier | Memory | Speed | Size |
|------|--------|-------|------|
| **G1** | Device VRAM | TB/s | 10s–100s of GB |
| **G2** | Host RAM | 10s–100s GB/s | 100s of GB – TBs |
| **G3** | Local SSD | 5–10 GB/s | TBs |
| **G4** | Networked SSD | GB/s | 10s of TBs |

The strategy: **keep hot blocks high (G1/G2), demote cold blocks to slower tiers** until needed.
GB200-class SKUs ship CPUs and interconnects with very fast **G2**, making them excellent for
offloading. **NVIDIA Dynamo's KVBM (KV Block Manager)** provides APIs to move blocks between tiers
automatically.

## 5.3.3 Cache-aware routing

Production runs **many replicas** behind a load balancer. The default routes by how *busy* each replica
is — fine, until you depend on prefix caching. A user deep in a conversation, or hammering questions
about one codebase, has their warm cache on **one specific replica**. Round-robin them elsewhere and
every request is a cold miss.

**Cache-aware routing** routes by *where the cache is*, not just load:

```
 default (load-only)        cache-aware
 user turn 5 ─► whoever      user turn 5 ─► the replica that served turns 1–4
                is free                       (warm KV → cache hit → fast, cheap)
```

A complementary option: use **G4 networked storage as a global KV cache** shared across replicas.
Routing still matters (a local G1 hit beats a remote G4 read), but a global cache means any replica can
*eventually* reach any precomputed sequence, and warm caches **survive nodes cycling or scaling down** —
important under autoscaling (Chapter 7).

## 5.3.4 Long context handling

**"Long context" is a circular definition: a sequence is long when its KV cache gets big enough to cause
problems.** Depending on model, hardware, and engine, that starts past common cutoffs — **32K, 64K,
128K** tokens. (Always load-test with very long inputs; long-context behavior won't show up on short
prompts.)

Labs stretch context windows with techniques like **RoPE** (Rotary Position Embedding) scaling, but a
bigger window means a bigger cache — and recall the KV cache makes attention scale **linearly with
sequence length**. At long context, **attention becomes the main consumer of VRAM** — the very resource
decode is already starved for.

Three general optimizations to the standard attention algorithm (model-specific tricks like sliding-
window, compressed, and sparse attention also exist):

- **FlashAttention** — fused kernels that compute attention with far fewer memory reads/writes (it never
  materializes the full score matrix). Lossless; especially helps compute-bound prefill.
- **PagedAttention** — the paging from §5.3.1: store KV in fixed-size blocks to kill fragmentation and
  duplication, so you fit more cache in the same VRAM.
- **Chunked prefill** — split a huge input into chunks and interleave them with decode, so one giant
  prefill doesn't monopolize the engine and stall everyone else's tokens.

But if, after all this, the KV cache *still* won't fit on one GPU, you're out of single-device options.
You parallelize across GPUs — the next section.

**Next:** [Parallelism →](parallelism.md)

[^prefix]:
    The token-id-prefix keying, chained hashing, paged-block storage, and the "handle, not tensors"
    detail are drawn from the author's own implementation notes (`PREFIX_CACHING_EXPLAINED.md`), which
    model vLLM-style Automatic Prefix Caching, SGLang RadixAttention, and API-style prompt caching.

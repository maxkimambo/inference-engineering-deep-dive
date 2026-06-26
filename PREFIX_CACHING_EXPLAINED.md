# Prefix Caching in LLM Inference — From Token ID to KV Tensor

A reference companion to the code in this package (`kvcache.go`, `block_hash_cache.go`,
`radix_cache.go`, `prompt_cache.go`). It explains *what* is being cached, *how* a token
id becomes the thing in the cache, *why* the cache key is what it is, and *where* the
cached data physically lives. Then it maps the three real-world strategies onto the three
implementations in this package.

---

## 0. TL;DR

- LLM "prefix caching" caches the **KV cache**: the attention **K**ey and **V**alue tensors
  computed for each token during *prefill*. Prefill is the expensive O(n²) step; reusing a
  shared prompt prefix's K/V skips it.
- The cache keys on **token ids** (integers from the tokenizer), never on text and never on
  embeddings — and specifically on the **prefix** of ids (via a *chained* hash), because a
  token's K/V depends on every token before it.
- The cache value (the K/V tensors) lives in a **pre-allocated pool of GPU memory** carved
  into fixed-size blocks. The lookup map only stores a **handle** (`hash → block_id`), not
  the tensors inline. This is OS-style paging — vLLM calls it *PagedAttention*.

---

## 1. Two different caches with the same name

| | Engine KV prefix cache | API prompt cache |
|---|---|---|
| Examples | vLLM Automatic Prefix Caching, SGLang RadixAttention | Anthropic / OpenAI prompt caching |
| Caches | K/V tensors in GPU memory | the prefix's KV server-side, exposed as a feature |
| Granularity | token blocks / radix nodes | prefix up to a caller-marked breakpoint |
| Eviction | refcount-pinned, then LRU under GPU memory pressure | sliding **TTL** (e.g. 5 min) + size |
| Has TTL? | no | yes |
| Has billing? | no | yes (cache-read vs cache-write vs input tokens) |

Both reuse a shared prompt prefix. They differ in how the prefix is indexed and what
triggers eviction. This package models all three (two engine-style, one API-style).

---

## 2. The pipeline: text → ids → embeddings → K/V → KV cache

```
 "hello world"
      │  (A) tokenizer encode: text → ids, via the VOCABULARY table
      ▼
 token ids:  [15339, 1917]            ◄── integers. THIS is a TokenID.
      │  (B) embedding lookup: id indexes a ROW of the embedding matrix
      ▼
 embedding vectors:  [ [0.02,-0.7,…], [0.5,0.1,…] ]   (d_model floats each)
      │  (C) per layer: project to Q,K,V with learned weight matrices
      ▼
 K/V tensors  (per token, per layer)  ◄── what we actually cache
```

Crucial ordering correction people trip on: **token ids come out of the tokenizer
*before* embedding.** Embedding is `embedding_matrix[id]` — a lookup *driven by* the id, not
something that produces the id.

---

## 3. Where each artifact lives

| Thing | Form | Stored where | In the cache? |
|---|---|---|---|
| token **text** (`"▁hello"`, bytes) | string/bytes | **tokenizer vocabulary** — fixed id↔text table, loaded once with the model | ❌ never |
| token **id** | `int` | the request's sequence metadata (`[]int`); in this model, `KVBlock.tokens` | ✅ ids only |
| **embedding** | `float[d_model]` | not stored per request — `embedding_matrix` is model *weights* `[vocab × d_model]`; rows looked up on the fly | ❌ |
| **K/V tensors** | `float[...]` per layer | the **KV cache** (GPU memory blocks) — the expensive thing prefix-caching reuses | ✅ (modeled as `KVBlock`) |

There is **no text anywhere** inside these caches, by design. Text exists only at the
tokenizer boundary (encode at input, decode at output). Everything inside is ids and
tensors.

---

## 4. Step A — id → embedding (a table lookup)

The embedding matrix `E` has shape `[vocab_size × d_model]`. Token id `x` simply selects a
row:

```
 e = E[x]          # one d_model-wide vector, e.g. 4096 floats
```

`E` is part of the model weights (the checkpoint tensor `model.embed_tokens.weight`). The
mapping id↔embedding is 1:1 and deterministic — the same id always yields the same row.
Note: this gives a single **vector** (a row), not a matrix.

---

## 5. Step C — embedding → K/V (the matmul)

This is the link that feels like magic but isn't. K and V are **linear projections** of the
hidden state by learned weight matrices:

```
 k_t = x_t · W_K          v_t = x_t · W_V
       └ two matrix multiplies against frozen model weights ┘
```

### Concrete arithmetic (toy dims: d_model = 4, d_head = 3)

```
 embedding vector for the token:
   x = [ 0.2,  -0.5,  0.1,  0.9 ]          shape 1×4

 W_K  (a learned weight matrix)            shape 4×3
   ┌  1.0   0.0  -1.0 ┐
   │  0.5   1.0   0.0 │
   │  0.0   0.5   1.0 │
   └ -1.0   0.0   0.5 ┘

 k = x · W_K   (each output = dot of x with a column):
   k[0] = 0.2·1.0 + (−0.5)·0.5 + 0.1·0.0 + 0.9·(−1.0) = −0.95
   k[1] = 0.2·0.0 + (−0.5)·1.0 + 0.1·0.5 + 0.9·0.0    = −0.45
   k[2] = 0.2·(−1.0)+(−0.5)·0.0 + 0.1·1.0 + 0.9·0.5   =  0.35
   k = [ −0.95, −0.45, 0.35 ]             shape 1×3   ← the K tensor for this token

 v = x · W_V   (same operation, different matrix)
```

That is the entire step: a vector goes in, gets linearly projected by two matrices, K and V
come out. No table, no lookup — a transform whose coefficients encode what the model learned.

### Two refinements so the picture is honest

1. **Multi-head:** `W_K` is really `[d_model × (n_heads · d_head)]`, so one matmul yields all
   heads' K at once; reshape the output into `n_heads` separate `d_head` vectors. Same
   operation, wider matrix.
2. **Per layer, and contextualized:** this happens at *every* layer with that layer's own
   `W_K^ℓ, W_V^ℓ`. At layer 0 the input `x` is the raw embedding; at layer ℓ>0 the input is
   the *previous layer's output*, which has already mixed in earlier tokens via attention.
   That is why the same id yields **different** K/V depending on its prefix — and why the
   cache key must be the *prefix*, not the token.

### Two flavors of "lookup" to keep straight

| Step | Operation | Nature |
|---|---|---|
| id → embedding | `E[id]` | **table lookup** — pick row `id` of `[vocab × d_model]` |
| embedding → K, V | `x·W_K`, `x·W_V` | **matrix multiply** — linear projection by learned weights |

---

## 6. Where do W_K / W_V come from? They *are* the model.

`W_K`, `W_V` (and `W_Q`, `W_O`, the FFN matrices, and `E`) are the model's **weights** —
learned parameters. Downloading "Llama-3-70B" downloads billions of numbers filling in these
matrices.

```
 A model = ARCHITECTURE (fixed code)  +  WEIGHTS (learned numbers)

 architecture decides the SHAPE of W_V:   [d_model × d_head·n_kv_heads]   (from model config)
 training decides the VALUES inside W_V:  the actual floats              (from gradient descent)
```

### How the values are produced — training (done once)

```
 1. INIT:     every entry of W_V = small random number
 2. FORWARD:  run the matmuls → model predicts next token
 3. LOSS:     compare prediction to the real next token → one number "how wrong"
 4. BACKPROP: compute ∂loss/∂(every weight), including every entry of W_V
 5. UPDATE:   optimizer nudges each weight a hair to reduce loss
 6. repeat 2–5 over trillions of tokens, weeks, thousands of GPUs
```

After training the weights are **frozen**. Nobody designs `W_V`'s values; they emerge from
minimizing prediction error.

### It is a named tensor in the checkpoint file

```
 model.layers.0.self_attn.q_proj.weight     ← W_Q for layer 0
 model.layers.0.self_attn.k_proj.weight     ← W_K for layer 0
 model.layers.0.self_attn.v_proj.weight     ← W_V for layer 0   ◄── here it is
 model.layers.0.self_attn.o_proj.weight     ← W_O
 model.layers.0.mlp.gate_proj.weight        ← FFN
 ...
 model.layers.79.self_attn.v_proj.weight    ← W_V for layer 79
 model.embed_tokens.weight                   ← E, the embedding table
```

Loading the model = reading these tensors off disk (`.safetensors` / GGUF) into GPU memory.
There are `L` copies of `W_V` (one per layer). These matrices are where essentially all the
"70B parameters" live.

### Lifecycle

```
 TRAINING TIME:   W_V's values are LEARNED (backprop)                 — done once
        │  saved to checkpoint as v_proj.weight
 LOAD TIME:       W_V read from disk into GPU memory                  — once per server boot
        │  now a frozen constant
 INFERENCE TIME:  k = x·W_K,  v = x·W_V   (plain matmul vs constant)  — every token
```

At inference `W_V` is just a constant you multiply by — same numbers for every request. The
only thing computed per request is the *product* `x·W_V`. There is no further table behind
`W_V`; it is the irreducible "the model" part.

---

## 7. Why cache K and V specifically — causal attention

Attention for token *t*:

```
 out_t = Σ_{j≤t}  softmax( q_t · k_j / √d )  ·  v_j
                  └──── attends over ALL previous tokens' K and V ────┘
```

Two facts:

1. To compute token *t* you need `k_j, v_j` for **every earlier token** `j`. Generating
   token 1000 re-reads the K/V of tokens 0–999.
2. Causal masking means *t* never attends to the future, so `k_j, v_j` depend **only on
   tokens 0…j** — they do not change when later tokens arrive.

⇒ Recomputing past K/V every step is pure waste. Compute once, store in the KV cache, reuse
every subsequent step. That is the entire reason the KV cache exists — and the reason a
*shared prefix* is reusable across different requests.

### What one token's KV actually is (not a single matrix)

Passing a token through the stack produces a **(K, V) pair at every layer**. All of them are
stored. The final output/logits are **not** cached — only the per-layer K/V, because that is
what future tokens' attention reads.

```
 token t through the stack:
   layer 0 → (k⁰, v⁰)
   layer 1 → (k¹, v¹)
      ⋮
   layer L-1 → (k^{L-1}, v^{L-1})

 stored per token  =  { (k,v) at all L layers }   ← 2·L tensors, not one matrix
 stored per block  =  that, for all block_size tokens

 per-token KV bytes ≈ 2 (K,V) × L layers × n_kv_heads × d_head × dtype_bytes
   e.g. 2 × 32 × 8 × 128 × 2B ≈ 128 KB / token
```

That size is why eviction matters.

---

## 8. Why the key is the prefix of ids — not the embedding, not a single id

The K/V of token *t* is a function of **tokens 0…t**. So the cache key must identify the
whole prefix. A single token's embedding (or id) cannot be the key.

### Counterexample that kills "key on the embedding/single id"

```
 "river bank"            "money bank"
        └─ id 3000 ┐          └─ id 3000 ┐   same id  → same embedding E[3000]
                   ▼                     ▼
   layer 0:  k,v identical (embedding only)   ← if you stopped here, an embedding key works
                   │                     │
   layer 5:  attention mixes in "river"   vs   attention mixes in "money"
                   ▼                     ▼
   k⁵, v⁵   DIFFERENT  ───────────────────►  must NOT share a cache entry
```

An embedding key would treat both "bank"s as one entry and serve `money bank`'s K/V to the
`river bank` request — wrong tokens, corrupted output / cross-request leak.

Also: the embedding is just `E[id]` (1:1 with the id), so it carries no more information than
the id while costing `d_model` floats and reintroducing float-equality fragility. It is
*strictly worse* than the id — and the id alone is still insufficient. Hence: key on the
prefix of ids.

| Key choice | Size | Equality | Verdict |
|---|---|---|---|
| token id | 1 int | exact | cheap, hashable — but doesn't capture the prefix |
| embedding | `d_model` floats | float ≈ float | strictly worse, same info |
| **chained hash of prefix ids** | 1 `uint64` | exact | **correct + compact** |

---

## 9. The chained hash — turning per-block hashing into prefix identity

(See `hashTokens` in `kvcache.go`.) Each block's hash folds in the hash of the block before
it, so a hash *is* a prefix identity.

```
 parent
 = 0
   │      ┌─────────── block 0 ───────────┐
   └────► hashTokens(0,  [1,2,3,4]) ──► h0 ─┐   h0 ≡ identity of prefix "1·2·3·4"
                                            │
          ┌─────────── block 1 ───────────┐ │
   ┌──────────────── parent = h0 ◄─────────┘
   └────► hashTokens(h0, [5,6,7,8]) ──► h1     h1 ≡ identity of prefix "1·2·3·4·5·6·7·8"
```

```
 h0 = F(  0 , 1,2,3,4 )
 h1 = F( h0 , 5,6,7,8 )       ← h0 baked in
 h2 = F( h1 , 9,10,11,12 )    ← h1 (and thus h0) baked in
```

### Inside one call — the byte stream fed to FNV-1a

`h1 = hashTokens(parent=h0, toks=[5,6,7,8])`. Everything is serialized to 8-byte
little-endian, **parent first**:

```
        parent(h0)        tok 5            tok 6            tok 7            tok 8
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
bytes │ ..h0 LE 8B.. │ │05 00 00 00 00│ │06 00 00 00 00│ │07 00 00 00 00│ │08 00 00 00 00│
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             ▼                ▼                ▼                ▼                ▼
   FNV-1a 64-bit, folded one byte at a time:  state = (state XOR byte) * FNV_prime
                                          │
                                          ▼
                                   Sum64() ─► h1   (uint64)
```

Parent going in first is the entire point — it makes `h1` depend on `h0`, hence on the whole
prefix.

### Why prefix identity = correct reuse

```
 Prompt A: [1 2 3 4 | 5 6 7 8]          Prompt B: [1 2 3 4 | 9 9 9 9]
            ▼block0  ▼block1                        ▼block0  ▼block1
 A:  h0 = F(0,1234)  h1 = F(h0,5678)     B:  h0' = F(0,1234)  h1' = F(h0,9999)

   block 0:  A.h0 == B.h0   →  one map entry  →  KV reused   (HIT)
   block 1:  A.h1 != B.h1'  →  two map entries →  KV recomputed (MISS)
```

Once they diverge they can never re-converge — every later `parent` is already different:

```
 h2  = F(h1 , …)        h2' = F(h1', …)        h1 ≠ h1'  ⇒  h2 ≠ h2'  ⇒ … forever
```

That single fact is the `sharing=false` short-circuit in `block_hash_cache.go`: after the
first miss, stop looking up, because no later block's hash can be in the map.

### The hash function used

| | Function | Properties | Why |
|---|---|---|---|
| **This model** | FNV-1a 64-bit (`hash/fnv`) | fast, deterministic, **not** collision-safe | stdlib, zero-dep, enough to show chaining |
| **vLLM default** | Python builtin `hash()` of `(parent, token_ids, extra_keys)` | fast 64-bit, low collision prob, weak | speed |
| **vLLM hardened** | SHA-256 (configurable, e.g. `--prefix-caching-hash-algo sha256`) | collision-resistant, slower, cross-process deterministic | correctness / security |

A block-hash **collision** is not just a perf bug: request B would be served request A's KV
under a colliding key — wrong output / cross-request leak. Hence the SHA-256 option, and hence
real engines keep the token ids next to each block to **verify on a hash hit** before reusing
(a collision guard) and to recompute the next block's chained hash. vLLM also mixes `extra_keys`
(LoRA id, multimodal hashes, cache salt / tenant) into the hash so identical ids under different
adapters/images/tenants don't collide.

### Implementation footnotes (from the code)

- `buf[:]` (not `buf`): `buf` is a `[8]byte` **array**; `Write`/`PutUint64` want a `[]byte`
  **slice**. `buf[:]` is the zero-cost reslice (a slice header over the stack array → no heap
  allocation per call).
- `_, _ = h.Write(...)`: `hash.Hash` embeds `io.Writer` so `Write` returns `(int, error)`,
  but the `hash.Hash` contract guarantees it never errors. The blank assignment just documents
  the deliberate ignore and silences strict (errcheck-style) linters.

---

## 10. Where the computed K/V is physically stored

**The map does not store the tensors inline.** It stores a **handle** — a block id / pointer
— into a separate, large slab of GPU memory where the floats live. Two structures, very
different sizes:

```
 ┌─────────────── INDEX (tiny, the "map") ───────────────┐
 │  map[ chained_hash ] ──► block_id 42                   │   just hash → address
 └───────────────────────────────────────────────────────┘
                                 │ points into
                                 ▼
 ┌─────────── STORAGE: KV-cache pool in VRAM (huge) ──────────────┐
 │ block 0 │ block 1 │ … │ block 42 │ … │ block N                 │
 │                         └─ the actual K/V floats for block_size │
 │                            tokens, across all L layers, live HERE│
 └────────────────────────────────────────────────────────────────┘
```

The pool is **allocated once at server startup**, sized to fill VRAM left over after the
weights, and carved into fixed-size blocks. Real per-layer shape (×2 for K and V, ×L layers):

```
 k_cache[layer] :  [ num_blocks , block_size , num_kv_heads , head_dim ]   fp16/bf16
                     └ block 42 is one slice along this axis ┘
```

- **Store** a new block: grab a free `block_id`, write the K/V floats into that slot (every
  layer), record `hash → block_id`.
- **Reuse** on a hit: read `hash → block_id`, point the request at that slot, `refcount++`.
- **Evict**: free a `block_id` back to the pool (only if `refcount == 0`).

### The analogy: OS paging (vLLM = "PagedAttention")

```
 virtual memory                    KV prefix cache
 ──────────────                    ───────────────
 page table:  vaddr → frame        index map:   prefix-hash → block_id
 physical RAM: frames of bytes     VRAM pool:   blocks of K/V floats
 page fault → load                 cache miss → run prefill, fill a block
 shared pages (COW)                shared prefix → two requests, same block (refcount)
```

The hash map is the page table (small, just translations). The VRAM block pool is physical
memory (where the data sits). Prefix sharing = two sequences' page tables pointing at the
same physical frame. One physical block can back many requests' prefixes at once — which is
exactly why eviction must be refcount-gated.

### In this Go model

```go
map[uint64]*KVBlock     // *KVBlock is the HANDLE (the block_id / pointer)
                        // the KVBlock struct stands in for the VRAM block
```

The toy keeps only the bookkeeping (`id`, `tokens`, `refcount`, `hash`) and **omits the
tensors**. In a real engine the struct is (or points to) the slot holding the
`2 · L · block_size · num_kv_heads · head_dim` floats in VRAM.

---

## 11. The block-hash cache process end to end (vLLM-style)

`BlockSize = 4`, prompt `[1,2,3,4,5,6,7,8,9]`.

```
 prompt:  [1 2 3 4 | 5 6 7 8 | 9]
           block 0   block 1   leftover (partial — never hashed, never cached)
```

### Request 1 — cold cache, all MISS

```
 blocks map: {}        sharing=true
 i=0  h0 ∉ map ─► MISS ─► sharing=false; makeRoom(); create A{rc:1,hash:h0}; map[h0]=A; Computed=4
 i=1  lookup SKIPPED (sharing==false); create B{rc:1,hash:h1}; map[h1]=B; Computed=8
 leftover "9":  Computed=9
 result: Blocks=[A,B]  Cached=0  Computed=9
```

### Free(r1) — blocks become evictable

```
 A.rc 1→0 ► PushFront(A);  B.rc 1→0 ► PushFront(B)
 evictable (MRU→LRU):  front [ B ][ A ] back        ← A is the next victim
 map still: { h0→A(rc0), h1→B(rc0) }                 (resident, just unpinned)
```

### Request 2 — `[1,2,3,4,5,6,7,8,99]` — shared prefix → HIT

```
 i=0  h0 ∈ map & sharing ─► HIT; Remove(A) from evictable; A.rc 0→1; Cached=4
 i=1  h1 ∈ map & sharing ─► HIT; Remove(B) from evictable; B.rc 0→1; Cached=8
 leftover "99": Computed=1
 result: Blocks=[A,B]  Cached=8  Computed=1     ← prefill skipped for 8 tokens; same pointers A,B
```

### Eviction — `makeRoom()` (BlockSize=2, capacity=2)

```
 after [1,2],Free:  map{A}        evictable:[A]
 after [3,4],Free:  map{A,B}      evictable:[B][A]      (len 2 == cap)
 serve [5,6]: MISS → makeRoom(): len(2)>=cap(2) & evictable non-empty
        victim = Back() = A (LRU); Remove(A); delete(map, A.hash)
        create C; map{B,C}
```

### The two-axis state of every block

```
                 refcount > 0                 refcount == 0
              ┌────────────────────┐      ┌─────────────────────────┐
  in map  →   │  PINNED (in use)   │      │  EVICTABLE (in LRU list) │
              │  cannot be evicted │ ◄──► │  victim when room needed │
              └────────────────────┘      └─────────────────────────┘
                   Allocate ▲                        │ Free
                            └────────────────────────┘
   not in map → evicted (KV discarded; next Allocate recomputes it)
```

The refcount gate — *not* the LRU — is the part that mirrors a real engine: the LRU picks the
victim, but a block a live request is decoding through is untouchable no matter how old.

---

## 11.5 Worked example: two chat requests sharing a system prompt

This is the canonical real use of prefix caching: many requests share the **same system
prompt**, so its K/V is computed once and reused by everyone. Let's trace two requests all the
way from text → tokens → hash → the *same physical memory block*.

### Setup (`BlockSize = 4`)

```
 SHARED system prompt:  "You are a helpful assistant."   → 8 tokens → exactly 2 blocks
   ids:  [101 102 103 104 | 105 106 107 108]
          └─── block S0 ───┘└─── block S1 ───┘

 Request A = system prompt + "What is 2+2?"     user tail → [201 202 203 204]   (1 block)
 Request B = system prompt + "Translate hi"     user tail → [301 302 303 304]   (1 block)
```

Lined up, the shared part is obvious — same ids until the user's question diverges:

```
 A:  [101 102 103 104 | 105 106 107 108 | 201 202 203 204]
 B:  [101 102 103 104 | 105 106 107 108 | 301 302 303 304]
      └──────── identical (system prompt) ───────┘ └─ differ ─┘
            block 0          block 1          block 2
```

### Request A arrives first — cold cache, everything is computed

Walk block by block, chaining the hash, and write each computed block to a free VRAM slot:

```
 parent = 0
 block 0  [101,102,103,104]   h0 = F(0,  101..104)   MISS → prefill → VRAM slot #0   map[h0]=#0
 parent = h0
 block 1  [105,106,107,108]   h1 = F(h0, 105..108)   MISS → prefill → VRAM slot #1   map[h1]=#1
 parent = h1
 block 2  [201,202,203,204]   h2 = F(h1, 201..204)   MISS → prefill → VRAM slot #2   map[h2]=#2

 A serves with blocks [#0, #1, #2], all pinned (refcount 1). Cached=0  Computed=12
```

State after A finishes and calls `Free` (refcounts drop to 0 → blocks become evictable but
stay resident):

```
 INDEX map:                         VRAM pool:
   h0 ──► slot #0   (rc 0)           #0 = K/V of [101..104]   ("You are a")
   h1 ──► slot #1   (rc 0)           #1 = K/V of [105..108]   ("helpful assistant.")
   h2 ──► slot #2   (rc 0)           #2 = K/V of [201..204]   ("What is 2+2?")
```

### Request B arrives — same system prompt → HIT → same blocks

B recomputes the *same chained hashes* for the shared blocks, because the ids — and therefore
`parent` at each step — are identical:

```
 parent = 0
 block 0  [101,102,103,104]   h0 = F(0,  101..104)   ← SAME h0 as A
                              h0 ∈ map → HIT → reuse VRAM slot #0   (rc 0→1)   Cached += 4
 parent = h0
 block 1  [105,106,107,108]   h1 = F(h0, 105..108)   ← SAME h1 as A
                              h1 ∈ map → HIT → reuse VRAM slot #1   (rc 0→1)   Cached += 4
 parent = h1
 block 2  [301,302,303,304]   h2'= F(h1, 301..304)   ← DIFFERENT (tail diverged)
                              h2'∉ map → MISS → prefill → VRAM slot #3   map[h2']=#3   Computed += 4

 B serves with blocks [#0, #1, #3]. Cached=8  Computed=4
```

The key moment: B's `block 0` produced **the exact same `h0`** as A's `block 0`, so the map
sent it straight to **physical slot #0** — the K/V already sitting in GPU memory. No embedding,
no matmul, no prefill for the entire system prompt. That is "arriving from a shared prefix to a
memory block."

### The picture — two requests, one shared set of physical blocks

This is the page-table view: each request has its own *block table* (logical → physical), but
the shared entries point at the **same physical slots**:

```
 Request A block table          VRAM pool (physical K/V)          Request B block table
 ─────────────────────          ────────────────────────          ─────────────────────
   logical 0 ─────────────────────►  #0  [101..104]  ◄───────────────── logical 0
   logical 1 ─────────────────────►  #1  [105..108]  ◄───────────────── logical 1
   logical 2 ───►  #2 [201..204]     #3  [301..304]  ◄───────────────── logical 2
                       ▲ A only            ▲ B only

   slot #0 refcount = 2   (A and B)     ← shared system-prompt block, pinned by BOTH
   slot #1 refcount = 2   (A and B)     ← shared
   slot #2 refcount = 1   (A only)
   slot #3 refcount = 1   (B only)
```

One physical block (`#0`, `#1`) backs *both* requests at once — exactly like two OS page tables
mapping the same physical frame. The refcount counts how many live requests lean on a block,
which is why a shared block can't be evicted while anyone is still using it.

### What B saved

```
 prompt length:          12 tokens
 served from cache (#0,#1): 8 tokens   ← system prompt, prefill SKIPPED
 computed (#3):             4 tokens   ← only the unique user question

 prefill work avoided: 8/12 = 67%
```

Scale that to a real 2,000-token system prompt shared across thousands of chat requests and the
first request pays for the system prompt's prefill once; every later request reads those blocks
for ~free. That is the entire economic point of prefix caching — and why the cache key is the
*prefix of token ids*: it's the only thing that makes "same system prompt" resolve to "same
physical block."

---

## 12. The three strategies, side by side

| | `BlockHashCache` (vLLM) | `RadixCache` (SGLang) | `PromptCache` (API layer) |
|---|---|---|---|
| Index | flat `map[chained_hash]→block` | radix tree over token ids | `map[prefix_hash]→entry` |
| Match granularity | fixed token blocks, block-aligned | any length (edge-split) | exact prefix to a breakpoint |
| Eviction | refcount-pinned, then LRU | LRU **leaf**, refcount-pinned | **TTL** + size LRU |
| TTL? | no | no | yes (sliding) |
| Result | blocks + cached/computed counts | cached/computed counts | hit + blocks + cached/computed counts |
| KV store | itself | itself | **layered on a `BlockHashCache`** |
| Closest to a hand-written trie | — | **yes** (it *is* a token radix tree) | — |

- **`block_hash_cache.go`** — prefix sharing falls out of *chained hashing* into a flat map;
  no tree. `sharing=false` after the first miss encodes that divergence is terminal.
- **`radix_cache.go`** — your prefix trie, productionized. `splitEdge` is the only thing a
  plain trie lacks: a long edge is stored compressed and split into two nodes only when a
  later request diverges partway through it. Eviction must pick a **leaf** (you can't drop an
  internal node whose descendants are still cached).
- **`prompt_cache.go`** — the API surface, **layered on a `BlockHashCache`**: a miss prefills
  the prefix through that engine and the entry holds the resulting KV blocks (pinned); a hit
  hands them back, skipping prefill, and `CachedTokens` reports the saved compute. Adds the two
  things the engine lacks: a **sliding TTL** (refreshed on hit; 5 min default at Anthropic) and a
  prefix-count cap, which on eviction `Release` the prefix's blocks back to the engine. (The real
  product also bills cache-reads vs cache-writes, omitted here as orthogonal to the mechanics.)

### Honest simplifications in the model (so you don't mistake the toy for the engine)

- KV **tensors are not modeled** — only the bookkeeping that drives reuse/eviction. In reality
  `KVBlock` is/points to a VRAM slot of `2·L·block_size·num_kv_heads·head_dim` floats.
- No **collision guard**: real engines compare stored token ids on a hash hit before reusing.
- No **extra_keys** (LoRA / multimodal / tenant salt) mixed into the hash.
- `RadixCache.lruLeaf` scans the tree **O(nodes)**; real SGLang keeps an explicit LRU list.
- Splitting an edge that another live request pins is glossed over (the new ancestor doesn't
  inherit the in-flight pin).
- FNV-1a, not SHA-256 — fine for a demo, not collision-safe for cross-tenant or persisted use.

None of these change the algorithm you're here to see.

---

## 13. File map

| File | Role |
|---|---|
| `kvcache.go` | shared types (`TokenID`, `KVBlock`), the chained `hashTokens`, `commonPrefixLen`, package doc |
| `block_hash_cache.go` | vLLM Automatic Prefix Caching: `map[hash]→block`, blocks, refcount + LRU |
| `radix_cache.go` | SGLang RadixAttention: token radix tree with edge-split, refcount + LRU-leaf |
| `prompt_cache.go` | API prompt cache layered on `BlockHashCache`: prefix entries hold KV-block handles, TTL + size eviction, `Serve` reports the saved prefill |
| `kvcache_test.go` | reuse + eviction proofs for all three (deterministic; injectable clock for TTL) |

Run: `go test ./kvcache/`

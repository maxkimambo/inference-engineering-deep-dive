# API-Layer Prompt Caching — Worked Example (hit / miss)

A companion to `PREFIX_CACHING_EXPLAINED.md`. That doc covers the **engine** KV prefix caches
(vLLM block-hash, SGLang radix). This one covers the **API product layer** — Anthropic /
OpenAI prompt caching, modeled by `prompt_cache.go` — and traces the same shared-system-prompt
scenario through it, focusing on **hit vs miss** (billing omitted on purpose).

---

## How this layer differs from the engine cache

Both reuse a shared prompt prefix, but the API layer adds two things and coarsens one:

| | Engine cache (block-hash / radix) | **API prompt cache (this doc)** |
|---|---|---|
| Key | chained hash of *each block* of ids | one hash of the whole prefix **up to a breakpoint** |
| Granularity | fine — block-aligned, partial reuse | coarse — all-or-nothing at the breakpoint |
| Eviction | refcount + LRU (GPU memory pressure) | **sliding TTL** (e.g. 5 min) + size LRU |
| Caller control | implicit (automatic) | explicit **cache breakpoint** (`cache_control`) |

The distinctive concept here is the **breakpoint**: the caller marks where the cacheable
prefix ends. Everything before it is the cacheable prefix; everything after is fresh input that
is *never* cached and always reprocessed.

```
 prompt tokens:  [ ........ cacheable prefix ........ | ... fresh input ... ]
                  └──────── hashed → one key ────────┘ └ after breakpoint ─┘
                                                       ▲
                                                  breakpoint
```

(Code: `PromptCache.Serve(tokens, breakpoint)` in `prompt_cache.go`. The key is
`hashTokens(0, tokens[:breakpoint])` — a single hash of the prefix, not per-block chaining.)

---

## Setup

Same scenario as the engine worked example: a system prompt shared across chat requests, with
the **breakpoint placed right after the system prompt**.

```
 SHARED system prompt:  "You are a helpful assistant."   → ids [101..108]   (8 tokens)
 breakpoint = 8         (cache everything up to here; the user's question stays fresh)

 Request A = system prompt + "What is 2+2?"     tail [201 202 203 204]
 Request B = system prompt + "Translate hi"     tail [301 302 303 304]

 A:  [101 102 103 104 105 106 107 108 | 201 202 203 204]
 B:  [101 102 103 104 105 106 107 108 | 301 302 303 304]
      └──────── cacheable prefix ──────┘ └─ fresh input ─┘
                  (breakpoint = 8)
```

TTL for this trace: **5 minutes**, sliding (refreshed on every hit). The model uses an
injectable clock, so the times below are exact, not wall-clock.

---

## The trace

### `10:00` — Request A — cold cache → MISS

```
 sweepExpired()            → nothing cached yet
 key  = hash(ids[:8])      = K_sys        (hash of the system prompt)
 K_sys ∈ entries?          → NO  →  MISS
 store entry{ key=K_sys, lastAccessed=10:00 }
 fresh input (tail [201..204]) → processed normally, never cached
```

```
 entries:  K_sys → { lastAccessed: 10:00 }        expires at 10:05 unless refreshed
```

The prefix is now cached. (On the real API this is the "cache write" — the server stashes the
system prompt's KV. Here we only care that K_sys is now present.)

### `10:02` — Request B — same system prompt → HIT

B's prefix `ids[:8]` is byte-identical to A's, so it hashes to the **same K_sys**:

```
 sweepExpired()            → K_sys last used 10:00, age 2 min < 5 min TTL → still alive
 key  = hash(ids[:8])      = K_sys        ← SAME key as A
 K_sys ∈ entries?          → YES →  HIT
 refresh lastAccessed = 10:02            ← sliding TTL: clock resets
 fresh input (tail [301..304]) → processed normally (B's question differs — never cached)
```

```
 entries:  K_sys → { lastAccessed: 10:02 }        expiry slides to 10:07
```

The system prompt was a **HIT** even though B's user question is completely different — because
only the prefix *before the breakpoint* is keyed. The differing tail lives after the breakpoint
and is irrelevant to the cache lookup.

### `10:06` — Request C — within the (slid) window → HIT

```
 sweepExpired()            → K_sys last used 10:02, age 4 min < 5 min → alive (thanks to B's refresh)
 K_sys ∈ entries?          → YES →  HIT
 refresh lastAccessed = 10:06            ← expiry slides to 10:11
```

Note: without B's `10:02` refresh, K_sys would have died at `10:05` and this would be a miss.
That's the **sliding** TTL — each hit extends the life; a hot prefix stays cached indefinitely.

### `10:13` — Request D — idle past TTL → MISS again

No requests touched K_sys since `10:06` (expiry `10:11`):

```
 sweepExpired()            → K_sys last used 10:06, age 7 min > 5 min → EXPIRED → removed
 K_sys ∈ entries?          → NO  →  MISS
 store entry{ key=K_sys, lastAccessed=10:13 }     ← re-written from scratch
```

Time, not memory pressure, evicted it. The very same system prompt is now a miss purely because
it went cold — the feature that engine caches do **not** have.

---

## The coarse-granularity gotcha: a *slightly* different prefix is a full MISS

Because the whole prefix hashes to **one** key, changing even a single token before the
breakpoint changes the key entirely — no partial reuse:

```
 Request E system prompt:  "You are a helpful assistant!"   ← one token differs (id 108 → 199)
   ids[:8] = [101 102 103 104 105 106 107 199]
   key = hash(ids[:8]) = K_sys'   ≠   K_sys      →  MISS  (re-cache the entire prefix)
```

Contrast with the engine block cache, which would still reuse the first block `[101..104]`
(same `h0`) and only recompute the diverging block. At the API layer it's **all-or-nothing at
the breakpoint**:

```
 engine block cache:   [101..104] HIT │ [105..108] differs → recompute from here
 API prompt cache:     entire prefix MISS — one byte changed the single key
```

Practical consequence: keep everything you want cached **stable and at the front** (system
prompt, tool definitions, few-shot examples), and put the volatile part (the user's turn)
**after** the breakpoint. A timestamp or per-user string slipped into the cached prefix
silently turns every request into a miss.

---

## State machine of one entry

```
        Serve, prefix not present                 Serve, prefix present & not expired
   ─────────────────────────────►  [ CACHED ]  ◄──────────────── HIT: refresh lastAccessed
   MISS: write entry, stamp time      │   ▲                      (sliding TTL slides forward)
                                       │   │
              idle > TTL (swept)  ─────┘   └─────  size cap exceeded: LRU-evicted
                     │                                   │
                     ▼                                   ▼
                  [ GONE ]  ── next Serve of this prefix is a MISS again
```

Two independent eviction pressures:
- **TTL** — entry idle longer than the window is swept (`sweepExpired`).
- **Size** — too many distinct prefixes → least-recently-used entry dropped (`evictBySize`).

Either can turn a future lookup of the same prefix into a miss.

---

## What the value side actually is — and what saves the compute

This is the link the hit/miss bool alone hides. The entry's **value is a handle to the
prefix's KV blocks**, and `PromptCache` is layered directly on top of the engine
`BlockHashCache` (the same one from the sibling doc) — that engine is where the KV physically
lives.

```go
type promptEntry struct {
    alloc        *AllocResult // handle to the prefix's KV blocks — the cached VALUE
    cachedTokens int          // full-block prefix tokens those blocks cover (saved on a hit)
    lastAccessed time.Time
    elem         *list.Element
}
```

- **MISS** → `engine.Allocate(prefix)` runs **prefill**: it computes the KV blocks for the
  prefix and stores them in GPU memory. The entry keeps the returned handle (`alloc`), pinned,
  so the blocks survive for the TTL.
- **HIT** → hand back `e.alloc.Blocks`. The prefill is **skipped entirely** — *that skipped
  prefill is the saved compute.*
- **EVICT** (TTL or size) → `engine.Release(e.alloc)` discards the prefix's blocks (refcount-
  aware, so a block still shared by another live prefix survives).

`Serve` returns a `PromptResult` that makes the saving measurable:

```go
type PromptResult struct {
    Hit            bool
    Blocks         []*KVBlock // the prefix's KV (same blocks reused on a hit)
    CachedTokens   int        // prefill SKIPPED by reuse  ← the saved compute
    ComputedTokens int        // prefill paid this request (uncached prefix tail + fresh input)
}
```

In `TestPromptCacheHitSkipsPrefill` (8-token prefix, 4 fresh tokens, block size 4):

```
 cold  MISS:  CachedTokens=0   ComputedTokens=12   (8 prefix + 4 fresh, all prefilled)
 warm  HIT:   CachedTokens=8   ComputedTokens=4    (8 prefix REUSED, only 4 fresh prefilled)
              └── r2.Blocks[0] == r1.Blocks[0] : the exact same KV blocks ──┘
```

That drop from 12 → 4 computed tokens is the prefill the cache saved — the whole economic
point. `CachedTokens` is the saving, expressed in tokens of prefill avoided.

## Map to the code (`prompt_cache.go`)

```go
key := hashTokens(0, tokens[:breakpoint])   // single hash of the whole prefix
if e, ok := c.entries[key]; ok {            // HIT
    e.lastAccessed = c.now()                //   slide the TTL
    c.lru.MoveToFront(e.elem)               //   bump recency
    res.Blocks = e.alloc.Blocks             //   reuse the precomputed KV (prefill skipped)
    res.CachedTokens = e.cachedTokens       //   ← the saved compute
} else {                                    // MISS
    alloc := c.engine.Allocate(prefix)      //   prefill: compute + store the KV blocks
    // ... store entry holding alloc, then evictBySize()
}
// sweepExpired() runs first; removeEntry() Releases the prefix's KV back to the engine
```

- `Serve(tokens, breakpoint)` — one request; `tokens[:breakpoint]` is the cacheable prefix.
- `engine.Allocate` / `engine.Release` — the underlying KV store; where prefill happens / where
  evicted blocks are discarded.
- `sweepExpired()` — TTL eviction (time). `evictBySize()` — LRU eviction (count).
- Injectable `now` clock — makes the `10:00 / 10:02 / …` trace deterministic in tests
  (`TestPromptCacheExpiresAfterTTL`, `TestPromptCacheEvictsBySize`).

---

## Honest simplifications vs. the real API

- **Single exact-prefix key.** Real Anthropic/OpenAI caching matches the longest cached prefix
  at ~block granularity and supports multiple breakpoints (Anthropic: up to 4). This model uses
  one exact hash to the breakpoint, so it's all-or-nothing — good for seeing hit/miss, coarser
  than production.
- **KV modeled, not real tensors.** The entry holds a handle (`AllocResult`) to blocks in the
  underlying `BlockHashCache`, which models GPU KV blocks as bookkeeping (no actual tensors) —
  see the sibling doc. The layering is real; the bytes are not.
- **Billing omitted** by request — in reality a hit bills the prefix at a cheap "cache-read"
  rate and a miss pays a small "cache-write" premium, but that's orthogonal to the hit/miss
  mechanics shown here.

---

## One-line summary

Same prefix **and** still within the TTL → **HIT** (and the TTL slides forward); a changed
prefix, or an idle one past the window, → **MISS**. The breakpoint decides what counts as "the
prefix"; time and size decide how long it survives.

# LLM Inference Mechanics

This is the core of the book. We trace a single request end to end: how text becomes numbers, how
those numbers flow through a transformer, how the next token is chosen, and why the whole thing
splits into two phases with completely different performance characteristics.

Read it top to bottom. Every term is defined where it first appears.

## The one-sentence model

> **An LLM is an autoregressive next-token predictor.**

- **Autoregressive** — it generates one token at a time, and each new token is fed back in as
  input to predict the next one. "Auto" (self) + "regressive" (feeding prior outputs back).

You give it `"The capital of France is"`; it predicts `" Paris"`; it appends that and predicts
again. Chat, code, reasoning — all of it is this loop run thousands of times. Everything else in
this section is the machinery that makes one turn of that loop happen, fast.

## Step 0 — Tokenization

Computers do math on numbers, not letters. So step zero is converting text into numbers. The unit
of conversion is the **token**.

- **Token** — a chunk of text: often a whole common word, a word-piece, or a punctuation mark.
  Modern LLMs use **subword tokenization** — frequent words are one token; rare words split into
  several. Rough rule: 1 token ≈ 4 characters of English ≈ ¾ of a word.
- **Tokenizer** — the component that chops text into tokens and maps each to an integer. It is
  **not** a neural network — just a fixed lookup table and a splitting algorithm (BPE or
  SentencePiece), decided before training and never changed.
- **Vocabulary** — the fixed list of every token the model knows, each with a unique integer id.
  Modern models have 100,000+ entries.
- **Token id** — the integer index into the vocabulary. This is the number the model actually
  consumes.

```
text        "Inference engineering makes AI apps fast."
              │ tokenizer (encode)
              ▼
tokens      [ In  ference  engineering  makes  AI  apps  fast  . ]
token ids   [ 644,  2251,      15009,    3727, 15592, 10721, 5043, 13 ]
              │ tokenizer (decode)  ◄── reverse lookup, same table
              ▼
text        "Inference engineering makes AI apps fast."
```

!!! key "Tokenizer efficiency is free latency"
    Fewer tokens for the same text = fewer forward passes = lower latency and cost. Newer models
    ship denser tokenizers for exactly this reason. The model never sees letters — by inference
    time, `"Paris"` is just the id `9847`.

### The three sequences and the context window

A request is made of up to three token sequences:

- **Input sequence** — the prompt, chat history, system prompt, tool definitions: everything you
  send in.
- **Reasoning sequence** *(optional)* — for reasoning models, an intermediate "thinking" output
  the model generates before its real answer.
- **Output sequence** — the response.

Together these must fit in the **context window** — the maximum number of tokens the model can
process and generate in one request. A `max_tokens` argument can further cap the output.

The raw input is a single string, but you rarely send a single string — you send roles (system /
user / assistant), past turns, maybe tool schemas. Flattening all of that into one token sequence
is the job of the **chat template**.

- **Chat template** — the model-specific rule for serializing structured input (roles, turns,
  tools) into one token sequence with special delimiter tokens. It differs subtly per model and
  **must be implemented exactly right** or quality silently degrades. Getting this wrong is one of
  the most common self-hosting bugs.

## Step 1 — Embeddings: from id to meaning

A token id like `9847` is just a label — id `9848` isn't "one more Paris." To compute with
*meaning*, each id is turned into a vector via a lookup.

- **Vector** — an ordered list of numbers, e.g. `[0.2, -0.5, 0.1, 0.9]`; think of it as a point in
  space.
- **Embedding** — the specific vector assigned to a token, encoding its meaning as a *location*.
  Similar meanings → nearby points. "king" and "queen" sit close; "king" and "banana" far apart.
- **Embedding matrix `E`** — a table with one row per vocabulary entry, shape
  `[vocab_size × d_model]`. Turning an id into an embedding is just "go to that row."

```
E   (vocab_size rows × d_model columns)
  row 0      [ … d_model numbers … ]
  …
  row 9847   [ 0.2, -0.5, 0.1, 0.9, … ]   ◄── embedding for "Paris"
  …

lookup:   embedding = E[token_id]
```

The embedding is the form a token travels in for the rest of the network. From here on, "the
token" means its evolving hidden-state vector, length `d_model`.

## Step 2 — The transformer stack

The body of an LLM is a tall stack of identical **transformer blocks** — dozens to hundreds of
them — wrapped by an embedding layer at the bottom and an output head at the top.

```
   token ids
      │
 ┌────▼─────────────┐
 │ Embedding layer  │  id → vector  (the lookup above)
 └────┬─────────────┘
      │  hidden state  (d_model)
 ┌────▼─────────────┐
 │ Transformer blk  │ ┐
 ├──────────────────┤ │
 │ Transformer blk  │ │  × N  (e.g. 32, 80, 94 …)
 ├──────────────────┤ │
 │       …          │ ┘
 └────┬─────────────┘
      │  hidden state
 ┌────▼─────────────┐
 │ Output head      │  vector → logits  (one per vocab token)
 │ (LM head)        │
 └────┬─────────────┘
      │
   logits  →  next token
```

- **Embedding layer** — the input layer; ids → embeddings.
- **Transformer blocks** — the hidden layers; each refines the hidden state.
- **Output layer / LM head** — converts the final hidden state into **logits**: one raw score per
  vocabulary token.

### Inside a transformer block

Each block has three kinds of sublayer:

1. **Attention** — lets each token look at other tokens and pull in relevant context.
2. **Feed-forward network (FFN / MLP)** — a small multi-layer perceptron (two linear layers + an
   activation) applied to each token independently. This is where most of the *weights* live.
3. **Normalization** — cheap element-wise rescaling that keeps the numbers stable between
   sublayers (LayerNorm / RMSNorm). A rounding error in cost.

The flow within a block is roughly:

```
 hidden ──► norm ──► ATTENTION ──►(+)──► norm ──► FFN ──►(+)──► hidden'
   │                               ▲                       ▲
   └──────── residual ─────────────┘     └─── residual ────┘
```

The `(+)` are **residual connections** — the block adds its work *back onto* its input rather than
replacing it. This is what lets you stack 80 blocks without the signal degrading.

!!! key "Where the weights are"
    The **FFN linear layers are the majority of an LLM's parameters**; attention is the
    second-largest. Norms and activations are negligible. So when someone says "70B parameters,"
    most of those numbers are FFN weight matrices that must be *read from memory every forward
    pass*. Remember this for [Bottlenecks](bottlenecks.md).

### Reading the architecture from `config.json`

Every model on Hugging Face ships a `config.json` — a few dozen lines describing the architecture.
An **architecture** is the set of training-time decisions about each component's nature and shape.
A name like `Qwen3MoeForCausalLM` parses as:

| Piece | Meaning |
|-------|---------|
| `Qwen` | model family / brand |
| `3` | major version of the architecture |
| `Moe` | it's a Mixture-of-Experts model (see below) |
| `ForCausalLM` | a **causal** language model — predicts the next token from *previous* tokens only |

> **Causal** vs **masked**: a causal LM only sees leftward context (the past). A masked LM (BERT)
> fills a blank using both sides. All generative LLMs today are causal — that one-directional
> constraint is enforced by the *causal mask* in attention, below.

The same config also gives you `hidden_size` (`d_model`), `num_hidden_layers` (N), the number of
attention heads, and the vocab size — everything you need to estimate memory footprint.

## Step 3 — Attention, properly

Attention is the one genuinely novel operation, and the one that drives inference cost. Take it
slowly.

**The problem it solves:** in *"I decided to write a book because I thought it would be easy, but
it was actually hard,"* what does "it" refer to? A human knows "it" = "writing a book." Attention
is how a transformer lets the token "it" *look back* at earlier tokens and decide which ones it
depends on.

### Q, K, V

For each token, the block computes three vectors by multiplying the hidden state through three
learned weight matrices (\(W_Q, W_K, W_V\)):

- **Query (Q)** — "what am I looking for?" The vector for the token doing the looking.
- **Key (K)** — "what do I offer?" A vector for each token that can be looked *at*.
- **Value (V)** — "what I'll hand over if you attend to me." The content actually pulled in.

The analogy: a **query** is a search box, each prior token advertises a **key** (like a search-result
title), and the **value** is the page content you retrieve from the matches.

### Scaled dot-product attention

The standard form:

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
\]

Walk through it mechanically:

1. **\(QK^{\top}\)** — dot every query against every key. The dot product of two vectors is large
   when they point the same way, so this scores *how relevant each prior token is to the current
   one*. Result: a grid of scores, one per (query, key) pair.
2. **\(\div \sqrt{d_k}\)** — divide by the square root of the key dimension. Without this, large
   `d_k` makes the dot products huge, which pushes softmax into a near-one-hot spike with
   vanishing gradients and brittle focus. The \(\sqrt{d_k}\) scaling keeps the score variance
   stable so attention stays smooth. *(This is the step most explainers skip — it's not cosmetic.)*
3. **\(\text{softmax}(\cdot)\)** — turn each row of scores into a probability distribution that
   sums to 1. Now each prior token has a *weight* — how much this token attends to it.
4. **\(\cdot V\)** — take a weighted sum of the value vectors using those weights. The output is a
   blend of the prior tokens' content, weighted by relevance.

> **softmax**, concretely: it exponentiates each score (making big ones dominate) then normalizes
> so they sum to 1. `[2.0, 1.0, 0.1] → [0.66, 0.24, 0.10]`. It's how "raw scores" become "how much
> to attend."

### Multi-head attention

Attention sublayers are **multi-head**: the operation runs several times in parallel, each a
**head** with its own \(W_Q, W_K, W_V\). Different heads specialize — one tracks subject-verb
agreement, another tracks coreference ("it" → "book"), another tracks position. Their outputs are
concatenated and mixed by one more linear layer.

- **Self-attention** — Q, K, V all come from the *same* sequence. LLMs use this.
- **Cross-attention** — Q comes from one sequence, K and V from another. Used in image/multimodal
  models to condition generation on a text prompt. (More in
  [Image & Video Generation](image-video-generation.md).)

### The causal mask

A generator must not peek at the future — when predicting token 5 it can't see tokens 6+. The
**causal mask** enforces this by setting the attention scores for all *future* positions to
\(-\infty\) before softmax, so their weight becomes zero.

```
        attends to →
          t1   t2   t3   t4
   t1  [  ✓    ✗    ✗    ✗  ]      ✓ = allowed (past or self)
   t2  [  ✓    ✓    ✗    ✗  ]      ✗ = masked to −∞ (future)
   t3  [  ✓    ✓    ✓    ✗  ]
   t4  [  ✓    ✓    ✓    ✓  ]
```

This lower-triangular shape is *the* reason an LLM is causal, and — as we'll see — the reason the
KV cache works at all.

### Why attention is quadratic… and how the KV cache makes it linear

Attention relates every token to every prior token. For a sequence of length \(n\), that's on the
order of \(n^2\) score computations — **quadratic in sequence length**. Double the context,
quadruple the attention work. This is why long context is expensive.

But notice something about the causal mask: when you generate token \(n+1\), the keys and values
for tokens \(1 \ldots n\) are *identical* to what they were on the previous step. The mask
guarantees the past never depends on the future, so past K and V never change. Recomputing them
every step would be enormous waste.

So we don't. We **cache** them.

- **KV cache** — the stored key and value vectors for every token processed so far. Built during
  prefill, then on each decode step we compute K and V for *only the one new token*, append them,
  and reuse the rest.

```
decode step for a new token:
   new token ─► compute its Q, K, V
                      │        │
                      │        └─► append to KV cache  ───┐
                      │                                    │
                      └─► attend Q against  [ all cached K ]  and  [ all cached V ]
                                                                          │
                                                            weighted sum ─┘─► output
```

With the cache, each decode step does work proportional to \(n\) (attend against \(n\) cached
entries), not \(n^2\). **The KV cache turns quadratic attention into linear-per-step attention.**
It lives in GPU memory, is the single biggest consumer of memory after the weights, and is the
subject of much of Chapter 5.

!!! key "The KV cache is the hinge of inference engineering"
    Building it (prefill) and reading it (decode) are the two operations that dominate runtime.
    Almost every technique you'll learn later — paged attention, prefix caching, cache-aware
    routing, quantizing the cache, disaggregation — is about managing this one data structure.

#### Sizing the cache (and why GQA exists)

The cache stores K and V for every layer, every head, every token. The size is roughly:

\[
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}}
\times \text{seq\_len} \times \text{bytes\_per\_value}
\]

The leading `2` is for K *and* V. Plug in a 70B-class model at 8k context and you get *gigabytes*
— per request. Multiply by your batch size and the cache, not the weights, becomes your memory
ceiling.

This cost is exactly why modern models reduce `n_kv_heads`:

- **MHA (Multi-Head Attention)** — every query head has its own K/V head. Biggest cache.
- **MQA (Multi-Query Attention)** — *all* query heads share a *single* K/V head. Tiny cache, some
  quality loss.
- **GQA (Grouped-Query Attention)** — the middle ground used by most current models: groups of
  query heads share a K/V head. The figure's "64 Q-heads, 8 KV-heads" is GQA — an 8× smaller cache
  than MHA for nearly the same quality.

When you read `num_attention_heads: 64, num_key_value_heads: 8` in a config, *that's the model
telling you it traded a little quality for an 8× smaller KV cache*. Now you know why.

## Step 4 — From hidden state to the next token

After the final block, the LM head projects the last token's hidden state to **logits** — one raw
score per vocabulary token (so the logit vector is `vocab_size` long, often 100k+).

```
 hidden state ──► LM head (matmul) ──► logits        ──► softmax ──► probabilities
                                       [ vocab-long ]                [ sums to 1 ]
   A     →   .0001                         Abs  →  0.5967  →  80%
   Ab    →   .0004                         Act  →  0.0983  →  12%
   Abs   →   .5967     ◄── highest          …
   …                                       next token:  "Abs"
```

Logits aren't probabilities yet — softmax makes them so. Then a token is **sampled** (a weighted
random draw). You steer that draw with inference arguments:

| Argument | Acts on | Effect |
|----------|---------|--------|
| **Temperature** | logits, *before* softmax | scales them; <1 sharpens (safer), >1 flattens (more random); 0 = always pick the top token (deterministic) |
| **Top-k** | after softmax | keep only the `k` most likely tokens, renormalize, sample among them |
| **Top-p** (nucleus) | after softmax | keep the smallest set whose probabilities sum to `p`, sample among them |

> Lower temperature / smaller k / smaller p → more predictable output. Temperature 0 or top-k 1 →
> fully deterministic (always the argmax).

Two more mechanisms ride on this step:

- **Logit biasing / structured output** — to force valid JSON or a schema, the engine masks out
  logits for tokens that would break the grammar *before* sampling, every step. Correct
  implementation here is what makes reliable tool-calling possible.
- **Stop token** — a special vocab entry meaning "end of output." Generation loops until the model
  samples it (or hits the context window / `max_tokens`).

## Putting it together: prefill vs decode

Now the payoff. Inference has **two phases** with opposite performance profiles. This split is the
most important operational fact in the whole book.

=== "Prefill"

    **Process the entire input sequence at once** to build the KV cache.

    - All input tokens go through every layer **in parallel** — one big batched matmul per layer.
    - Produces the KV cache for every input token, plus the first output token.
    - Lots of math, weights reused across many tokens in the batch → **compute-bound**.
    - Determines **TTFT** (Time To First Token) — how long until the user sees anything.

    ```
    prompt: [t1 t2 t3 t4 t5]  ──►  one forward pass over all 5  ──►  KV cache + token #6
    ```

=== "Decode"

    **Generate output tokens one at a time**, autoregressively.

    - Each step runs a **full forward pass for a single new token**, reusing the KV cache.
    - To produce one token you must read *all the model's weights* from memory — but you only do a
      sliver of math with them (one token's worth). Tons of data movement, little compute →
      **memory-bound**.
    - Determines **TPS** (Tokens Per Second / inter-token latency) — how fast text streams out.

    ```
    token #6 ─► fwd pass ─► token #7 ─► fwd pass ─► token #8 ─► … until stop
                (reuse KV)              (reuse KV)
    ```

!!! key "Why this asymmetry is everything"
    **Prefill is compute-bound; decode is memory-bound.** They stress different GPU resources, so
    they want different optimizations — and ideally different scheduling, batching, even different
    hardware. This single sentence is the seed of continuous batching, chunked prefill,
    speculative decoding, and prefill/decode *disaggregation* (Chapter 5). We prove the
    compute-vs-memory claim with arithmetic in [Bottlenecks](bottlenecks.md).

## Mixture of Experts (a sparsity trick)

One more architecture you must recognize. The **density** of a network is how many connections it
has. Dense networks hold more knowledge but cost more to run. **Mixture of Experts (MoE)** adds
*sparsity* to the FFN sublayers: instead of one giant FFN, the block holds many smaller FFNs (the
**experts**) plus a tiny **router** that sends each token to only a few of them — "activating" those
experts.

- **Total parameters** — every expert's weights; sets the memory footprint (you must *store* them
  all).
- **Active parameters** — only the experts a given token actually uses; sets the *compute* per
  token.

Example: **Qwen3-235B-A22B** has 235B total but activates **22B** per token. The `A22B` literally
means "22B active." With 128 experts and the router picking 8 per layer across 94 layers, each
token touches a small, *different* subset.

```
            ┌── router (tiny) ── picks 8 of 128 ──┐
 token ────►│                                      │──► only those 8 experts run
            └──────────────────────────────────────┘
   store 128 experts (memory)   ·   run 8 (compute)
```

!!! warning "MoE's catch in production"
    For a *single* local request, MoE is gloriously efficient — few active params, low compute.
    But in **batched** serving, different requests hit *different* experts, so across a full batch
    you end up activating almost all of them anyway. You pay the full memory cost regardless, and
    only recover the win at scale through **Expert Parallelism** (Chapter 5.4). MoE shines for
    large models (100B+); models under ~32B usually stay dense, where the whole model is
    effectively one expert.

---

You can now trace a token from string to next-token, name every matmul, and say which phase is
compute- vs memory-bound. Next we make the compute-vs-memory claim quantitative.

**Next:** [Calculating Bottlenecks →](bottlenecks.md)

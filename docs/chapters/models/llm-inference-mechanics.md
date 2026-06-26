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

#### What Q, K, and V actually look like

Those symbols have stayed shapeless. Concretely, **each of Q, K, and V is just a vector of
`d_head` numbers** — the per-head slice from multi-head attention below (real models use
`d_head = 128`; we'll use 4 so it fits on the page).

Take the text so far as *"The book was good, it…"* and let the current token `"it"` attend over
three earlier tokens. Every token already has its Q/K/V — produced by multiplying its hidden state
through \(W_Q, W_K, W_V\):

```
              Q, K, V are each a 4-number vector here (d_head = 4)

current token  "it"    Q = [ 1.0,  0.5, -0.5,  2.0 ]

prior token    "The"   K = [ 0.2,  0.1,  0.0,  0.1 ]   V = [ 0.1, 0.0, 0.2, 0.1 ]
prior token    "book"  K = [ 0.9,  0.4, -0.3,  1.8 ]   V = [ 0.7, 0.9, 0.2, 0.8 ]
prior token    "was"   K = [ 0.1, -0.2,  0.3,  0.2 ]   V = [ 0.0, 0.3, 0.1, 0.0 ]
```

**Steps 1–2** — score `"it"`'s query against each prior key (dot product, then ÷ √4 = ÷ 2):

```
score(it, The)  = (1.0*0.2 + 0.5*0.1  + -0.5*0.0  + 2.0*0.1) / 2 = 0.225
score(it, book) = (1.0*0.9 + 0.5*0.4  + -0.5*-0.3 + 2.0*1.8) / 2 = 2.425   ◄ it & book align
score(it, was)  = (1.0*0.1 + 0.5*-0.2 + -0.5*0.3  + 2.0*0.2) / 2 = 0.125
```

**Step 3** — softmax the three scores into attention weights:

```
softmax([0.225, 2.425, 0.125]) = [ 0.091, 0.826, 0.083 ]
                                    The    book   was
```

`"it"` places **83% of its attention on "book"** — it has resolved the reference.

**Step 4** — the output is the weighted sum of the *Value* vectors:

```
output = 0.091*V(The) + 0.826*V(book) + 0.083*V(was)
       = [ 0.587, 0.768, 0.192, 0.670 ]      ← again a d_head-length vector
```

That output — dominated by book's value — is what this head contributes for `"it"`. It's the same
length as the inputs (`d_head`), so it flows straight back into the pipeline; across all heads
these outputs concatenate back up to `d_model`.

!!! note "Scale check"
    Here `d_head = 4` and 3 prior tokens. A real decode step has `d_head ≈ 128` and *thousands* of
    prior tokens — each contributing a K and a V vector read from the **KV cache**. Identical four
    steps, just bigger. That "thousands of K/V vectors" is exactly why the KV cache exists and why
    its size dominates memory.

### Multi-head attention

So far we've described *one* attention operation over the token's full `d_model`-wide vector. Real
models don't do that — they **split the vector into several smaller slices and run attention on
each slice independently**. Each independent attention is a **head**.

- **Attention head** — one self-contained attention unit. It has its *own* learned
  \(W_Q, W_K, W_V\) and runs the full scaled-dot-product attention on a *slice* of the hidden
  state. A transformer block contains many heads running in parallel.
- **`d_head`** — the width of one head's slice. With hidden size `d_model` and `h` heads,
  `d_head = d_model / h`. Example: a `d_model = 4096` vector split across `32` heads gives
  `d_head = 128` — each head attends inside its own 128-dimensional subspace.

Mechanically, per token: split the `d_model` vector into `h` chunks → each head computes its own
Q/K/V and does attention on its chunk → concatenate the `h` outputs back to `d_model` → one final
linear layer mixes them.

```
 token hidden state  (d_model = 4096)
        │  split into 32 slices of 128
   ┌────┼─────┬─────┬─ … ─┐
 head1  head2 head3  …  head32     ← each runs its own attention (own Wq,Wk,Wv)
   └────┼─────┴─────┴─ … ─┘
        │  concatenate back to 4096
        ▼
   output projection (one matmul)  →  d_model
```

**Why split at all — the purpose.** A single attention produces exactly *one* weighted blend of
prior tokens per step: the token gets *one* "focus." Language needs more than one at a time. In
*"the keys that he left **are** on the table,"* the verb "are" depends on "keys" (subject-verb
agreement) **and** on nearby words (position) **and** possibly a pronoun elsewhere (coreference). A
single focus can't track all three. **Multiple heads give each token several focuses at once** —
picture independent spotlights, each free to attend to a different kind of relationship. No one
tells a head what to specialize in; training discovers it. Inspect a trained model afterward and
you'll often find recognizable roles — a coreference head, a previous-token head, a syntax head.

!!! info "Where the head count lives: `num_attention_heads`"
    The number of heads is a fixed architectural decision, listed in the model's `config.json` as
    `num_attention_heads` (the count of **query** heads). The related `num_key_value_heads`
    controls how many *key/value* heads exist — usually fewer, which is the GQA trick in the next
    subsection. The identity to remember: `num_attention_heads * d_head = d_model`.

!!! warning "\"Head\" is overloaded — two unrelated things share the name"
    - **Attention head** (this section): one of many parallel attention units *inside every
      transformer block*. A model has, e.g., 32 of them **per layer**, so hundreds in total.
    - **Output head / LM head** ([the stack diagram](#step-2-the-transformer-stack) and Step 4):
      the **single** final projection that turns the last hidden state into vocabulary logits.
      Exactly one exists, at the very top of the stack.

    Same word, different jobs. "How many heads does the model have?" almost always means *attention
    heads per layer* — the `num_attention_heads` value.

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

After the final block, the LM head projects the last token's hidden state to **logits**.

- **Logit** — a single *raw, unnormalized* score the model assigns to one vocabulary token,
  answering "how strongly do I favor this as the next token?" There's one logit per token, so the
  output is a vector `vocab_size` long (often 100k+). A logit is just a real number — it can be
  negative, zero, or large; **bigger means more favored**. Crucially, logits are **not**
  probabilities: they don't sit between 0 and 1 and they don't sum to 1. Converting them into
  probabilities is a separate step — **softmax** (the same function from inside attention).

**A concrete example.** Say the model has just read *"The capital of France is"* and must choose
the next token. The LM head emits one logit for *every* token in the vocabulary — 100k+ of them —
but almost all are tiny. Here are the handful that score highest, before and after softmax:

```
prompt:  "The capital of France is ___"

 LM head ──► logits ──────────── softmax ──► probability
   candidate token    logit                   prob
   " Paris"            6.0        ───────►     86.1%   ◄── highest
   " Lyon"             3.5        ───────►      7.1%
   " London"           3.0        ───────►      4.3%
   " a"                2.0        ───────►      1.6%
   " the"              1.5        ───────►      1.0%
   …(99,995 other tokens, each ≈ 0%)…
        ▲                              ▲
   raw scores, not                probabilities,
   probabilities                  summing to 1
```

The raw logits mean nothing on their own — softmax exponentiates each and divides by the total, so
larger logits dominate and the whole 100k-long vector becomes probabilities that sum to 1. The
model then **samples** from this distribution: a weighted random draw. With `" Paris"` holding 86%
of the probability mass, that's almost always the token that comes out. Append `" Paris"` to the
text, feed the whole sequence back in, and the loop predicts the token after that.

You steer that final draw with inference arguments:

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

# Speculative Decoding

Speculative decoding is the cleverest free lunch in inference. It raises decode throughput **without
changing a single output token** — it's lossless — by exploiting a gap you already proved exists in
[Bottlenecks](../models/bottlenecks.md).

## The insight: decode leaves compute idle

Decode is **memory-bound**. Each step reads the whole model from memory to generate *one* token, so
the FLOP units sit mostly idle waiting on bandwidth — at low-to-moderate batch sizes the GPU's
compute is barely touched.

So: what if we used that idle compute to *guess ahead*? Generate several candidate tokens in one pass,
then check them. If the guesses are usually right, we get multiple tokens per trip through memory
instead of one.

That's the whole idea. Every algorithm below is a different way to produce the guesses.

!!! key "Speculative decoding improves TPS, never TTFT"
    It accelerates **decode** (token generation), so it raises tokens-per-second / lowers inter-token
    latency. It does nothing for **prefill**, so **TTFT is unchanged**. If your problem is "first token
    is slow," this is the wrong tool — look at caching and disaggregation.

## The common mechanism

Every speculative scheme shares three steps:

1. A **speculator** proposes one or more **draft tokens** — cheap guesses at what comes next.
2. The **target model** (the real model you're accelerating) runs a forward pass that **verifies** all
   the draft tokens *at once* — checking whether each matches what it would have produced.
3. The target accepts every valid draft token *up to the first wrong one*, and generates one more
   itself, completing the pass.

The payoff: one forward pass now yields **N+1 tokens**, where N = accepted drafts.

!!! key "Why verifying is cheap but generating is expensive — the sudoku analogy"
    Generating a token means a full memory-bound forward pass. But **verifying** N draft tokens is one
    forward pass that scores them in parallel — nearly the cost of generating a single token, because
    it's the same weight read. Solving a sudoku is hard; *checking* a finished sudoku is easy. The
    target model checks the draft's sudoku.

```
 draft:    "Look at the"  + [ tel  le  vis  ion ]   ← speculator guesses 4 tokens
 verify:   target scores all 4 in ONE pass
                          [ tel✓ le✓ vis✗ ion ]     ← first 2 match, 3rd wrong
 accept:   "Look at the"  + [ tel  le ] + [ scope ] ← keep 2 valid, target adds 1
                          → 3 tokens from 1 pass
```

Note the **cascade rule**: once a draft token is rejected, *every* token after it is discarded too —
they were predicated on a guess that turned out wrong.

## What governs the speedup

Three factors, and you tune against all three:

1. **Draft token cost** — time to produce one guess. Not free; it takes compute and memory.
2. **Draft sequence length** — guesses per pass. More is better *only if* they're accepted.
3. **Token acceptance rate** — the fraction of drafts the target accepts. The dominant factor.

Acceptance is **high early in a draft sequence and decays with depth** — the further you guess, the
likelier you've diverged. Combined with the cascade rule, this means: **aim for short, high-confidence
draft sequences.** A long draft that's wrong at position 3 wasted positions 3–N.

!!! warning "Two things that kill speculative decoding"
    - **High temperature.** Higher temperature flattens the token distribution, making the next token
      genuinely less predictable → lower acceptance → less benefit. (Even subject matter matters: a
      draft model stronger in code than history will have higher acceptance on code.)
    - **High batch size.** Speculation spends spare compute. At high batch sizes that compute is already
      *saturated* serving the batch, so verification has no room — engines **dynamically disable
      speculation** as batch size climbs. This is the antagonism flagged in the [chapter
      intro](index.md): speculation and big-batch throughput compete.

## The algorithms

### Draft-target speculative decoding

The original method: a **separate, smaller draft model** proposes; the **target model** verifies.

The key decision is *which* draft model. A good one has a **high acceptance rate** while costing little
to run — usually a smaller member of the *same family* (shared tokenizer and behavior). Rule of thumb:
**the draft should be ≥10× smaller** than the target. Fine-tuning or distilling the draft toward the
target raises acceptance.

- **Pro:** works out of the box, no training.
- **Con:** the most overhead of any method — the engine must store the draft's weights, activations,
  and KV cache, spend compute on draft prefill, and *coordinate two models* so they don't fight for
  resources (engines like TensorRT-LLM handle the orchestration).

### Medusa

Instead of a second model, **Medusa fine-tunes the target itself to predict further ahead.** A normal
LLM has one decoder head (Step 4 of [Mechanics](../models/llm-inference-mechanics.md)); Medusa grafts
on **2–4 extra heads** that each generate a subsequent draft token in the same pass. Drafts are
verified on the next forward pass, as usual.

Medusa removed the two-model overhead but is still limited on draft count and acceptance rate. It's not
widely used in production — but it pointed the way to EAGLE.

### EAGLE

The problem with an off-the-shelf draft model: a standalone 0.5B LLM was designed to be a *good little
model on cheap hardware*, not to speculate on a B200. It's inefficient and its acceptance is mediocre.

**EAGLE** is a purpose-built draft model, trained from scratch to generate up to **~8 draft tokens**
(2× Medusa) at **high acceptance**. Its trick: during inference the target accumulates rich context in
its **hidden states** between layers — information ordinary draft models never see. EAGLE is trained to
**take hidden states as input** (specifically an early, a middle, and a late layer's) and emit draft
tokens. It's often **under 1B parameters** and scales well with more training.

```
  target's hidden states          draft tokens
  [ early ]                        [ 644, 2251, 15009, … ]
  [ middle ] ──► EAGLE  ─────────►  (verified next pass)
  [ late ]
```

Crucially, EAGLE **attaches to the same module as the target**, so one forward pass runs *both* — no
CPU round-trips to coordinate two models (draft-target's other big cost). **EAGLE is the go-to
speculation method today** for engineers able to train EAGLE heads, and is well supported across
engines. Like all speculation, it wants low batch sizes.

### N-gram speculation and lookahead decoding

A different mechanism entirely — **no draft model at all.**

Alongside building the KV cache, the engine builds an **n-gram dictionary** mapping a starting token to
sequences of tokens *seen in the input*. During decode, the just-generated token indexes the
dictionary, and any matching suffix becomes the draft, verified normally.

```
 input contains:  "from transformers import AutoModelForSequenceClassification"
 n-gram dict:      "import" → ["AutoModel…", "AutoTokenizer", "TrainingArguments"]
 decode generates "import" → draft the suffix from the dictionary → verify
```

- **Pro:** drafts can be **very long** (10+ tokens vs EAGLE's ~8), since they're copied from real text.
- **Con:** acceptance is only high when **output closely mirrors input** — so n-gram shines in **code
  completion and code revision** (repetitive, predictable syntax), where it *beats* EAGLE, and is weak
  elsewhere.

**Lookahead decoding** is a cousin that *generates* n-grams during inference to fill the dictionary,
rather than only harvesting them from the input. More general (doesn't need repetitive context) but
costs extra compute to manufacture the n-grams.

## Choosing

| Method | Setup cost | Draft len | Best for |
|--------|-----------|-----------|----------|
| **Draft-target** | none (off-the-shelf) | short | quick wins, no training budget |
| **Medusa** | fine-tune target | short | historical; superseded |
| **EAGLE** | train EAGLE head | ~8 | **general-purpose default** |
| **N-gram** | none | 10+ | **code** / input-echoing tasks |
| **Lookahead** | none | variable | excess-compute systems, general |

Every method has the same north star: **fewer forward passes per output sequence.** All trade some
extra work-per-pass for far fewer passes, and all want the spare compute of low batch sizes.

**Next:** [Caching →](caching.md)

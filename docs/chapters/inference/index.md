# Chapter 0 · Inference

Every other chapter in this book is about *how* — how a transformer computes, how a GPU moves bytes,
how to quantize or cache or scale. This one is about *what* and *why*: what the problem actually is,
and why it's worth a whole book. Skip it and the techniques later have no target to aim at; read it
and every optimization that follows has a number it's trying to move.

## What inference is

A model has two lives. In the first — **training** — it's shown mountains of data and its billions of
parameters are slowly adjusted until it's good at a task. That happens once (or a handful of times),
on a cluster, over days or weeks, and then it's done. In the second life — **inference** — those
frozen parameters are *used*: you hand the model an input, it runs the math once, and out comes an
answer. Training is teaching; inference is the model doing its job.

That second life is the one that matters in production, because it never ends. Training is a project
with a finish line. Inference is a service that runs every time anyone uses the product — a million
times a day, every day, for as long as the model is deployed. The trained weights are an asset;
inference is the act of spending that asset to produce value, one request at a time.

!!! key "Inference is the recurring cost, and recurring costs are where engineering pays off"
    You train a model once and serve it forever. Over a deployed model's life, the total compute
    spent *running* it dwarfs the compute spent *creating* it — which means a 2× improvement in
    inference efficiency isn't a one-time win, it's a permanent discount on every future token. This
    is the whole economic reason inference engineering exists as a discipline separate from training:
    the leverage is enormous because the cost is perpetual.

## Why it's a hard, distinct problem

It's tempting to think inference is just "training without the backward pass" — easier, since you're
only running the model forward. The opposite is true in the ways that matter for engineering, because
inference runs under constraints training never faces:

- **A clock.** Training is judged by the final model's quality; nobody cares if an epoch takes six
  hours. Inference is judged by a user waiting *right now* — every millisecond of latency is a person
  staring at a blinking cursor. Inference has a real-time deadline; training doesn't.
- **Unpredictable, bursty demand.** A training run is one known, steady workload you schedule. A
  serving system faces traffic that spikes at lunchtime, varies by request, and must be met with
  hardware you provisioned in advance — too little and requests fail, too much and you burn money on
  idle GPUs.
- **A per-request economy.** Every single inference call costs a measurable slice of a GPU's time, so
  the unit economics — *cost per token* — directly set your margins. Training is a capital expense;
  inference is cost-of-goods-sold.
- **A peculiar compute shape.** As Chapter 2 will prove, generating text happens one token at a time,
  each token needing the *entire* model re-read from memory — a pattern that leaves the GPU's math
  units mostly idle and makes the obvious "just add more compute" instinct useless. Inference is
  bottlenecked in a place training usually isn't.

Put together, these make serving a model its own engineering discipline with its own tools, metrics,
and failure modes — the subject of this book.

## The tension at the center of everything

Almost every decision in inference engineering is a trade between three things that pull against each
other:

!!! key "Latency, throughput, and cost — pick your balance, you can't max all three"
    - **Latency** — how fast a *single* request is answered (the user's experience).
    - **Throughput** — how many requests the system handles *at once* (the system's efficiency).
    - **Cost** — the dollars per token you spend getting there.

    These fight. The classic move to raise throughput — batching many requests onto one GPU so its
    expensive hardware stays busy — makes each individual request *wait longer*, hurting latency. The
    move to crush latency — give each request a whole GPU to itself — wrecks throughput and cost.
    There is no setting that wins all three; there is only the balance that fits *your* workload. The
    entire rest of this book is techniques for buying back one corner of this triangle without paying
    too much of another.

Get this triangle in your head now. When you read about quantization, batching, speculative decoding,
or disaggregation later, the right question is always: *which corner does this buy, and at whose
expense?*

## The map

This chapter frames the problem in three moves:

| § | Section | What it gives you |
|---|---------|-------------------|
| [0.1](training-vs-inference.md) | **Training vs inference** | Why the two are different workloads, and why that difference is the whole game |
| [0.2](latency-throughput-and-metrics.md) | **Latency, throughput & the metrics** | The numbers you actually serve to — TTFT, inter-token latency, throughput, percentiles |
| [0.3](the-economics-of-a-token.md) | **The economics of a token** | Cost per token, goodput, and what "optimize" really means in dollars |

## Learning objectives

By the end of this chapter you can:

- [x] Explain how inference differs from training in compute pattern, timing, and economics
- [x] Name the three competing goals of a serving system and give a concrete trade between them
- [x] Define TTFT, inter-token latency, and throughput, and say which one a given workload lives or dies by
- [x] Read a latency requirement as a percentile SLO, not an average
- [x] Estimate the cost per million tokens of a deployment from its hardware and throughput

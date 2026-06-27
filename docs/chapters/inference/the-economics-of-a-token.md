# The Economics of a Token - Tokenomics

Every latency and throughput decision you just learned to make is, underneath, a decision about money.
A token is a unit of work with a price, and a serving system is a machine for producing tokens as
cheaply as your latency promise allows. This section makes the cost concrete — how to compute it, the
metric that actually matters (it's not raw throughput), and why a few percent of efficiency is worth
chasing so hard.

## What a token costs

The cost of a token comes from one ratio: the price of the hardware over how many tokens that hardware
produces in the same time.

\[
\frac{\$}{\text{million tokens}} = \frac{\text{GPU \$/hour}}{\text{tokens/hour}}
\]

Two illustrative deployments — the same arithmetic, different hardware:

| GPU | Price | Throughput | Cost per 1M tokens |
|-----|------:|-----------:|-------------------:|
| 1× L4 | \$0.70/hr | 800 tok/s | **\$0.24** |
| 1× H100 | \$3.00/hr | 2,500 tok/s | **\$0.33** |

!!! key "Cost per token is set by utilization, not by the size of the GPU"
    Notice the cheaper-per-token option here is the *smaller* GPU — because cost per token is
    `price ÷ throughput`, and a GPU only earns its price when it's kept busy. A powerful GPU running
    at low batch is *expensive* per token (you're paying for hardware you're not using); the same GPU
    packed to high batch can be the cheapest. This is why "buy the biggest GPU" is the wrong instinct
    (Chapter 3) and why batching is an economic lever, not just a performance one. The denominator —
    throughput at high utilization — is where the savings live. (These throughputs are illustrative;
    the real crossover depends on model, batch, and precision — Chapters 2–3.)

## Goodput: the only throughput that counts

Raw throughput has a trap. A system can post huge tokens-per-second numbers by cramming an enormous
batch onto the GPU — while every request blows past its latency SLO and every user has rage-quit. Those
tokens were *produced* but not *useful*. The honest metric corrects for this:

!!! key "Goodput = throughput that actually met the latency target"
    **Goodput** counts only the tokens delivered *within* your latency SLO; tokens that arrived too
    late don't count, because a too-slow answer is a failed answer. It's the difference between "the
    GPU emitted tokens" and "the GPU served users." Optimizing raw throughput at the expense of
    latency can *raise* throughput while *lowering* goodput — you did more work and helped fewer
    people. Every capacity and batching decision in this book is really a goodput decision: **the most
    useful tokens per dollar, not the most tokens per dollar.** When a vendor benchmark quotes a giant
    throughput number with no latency bound, it's quoting the metric that doesn't matter.

## Why a few percent compounds

Recall the chapter's opening claim: inference is the *recurring* cost. That's what makes efficiency so
valuable. A training run is a fixed bill you pay once. Inference is a meter that never stops, so any
improvement to cost-per-token applies to **every future token, forever**.

Put numbers on it. A product serving a billion tokens a day at \$0.40 per million:

\[
\frac{1{,}000{,}000{,}000}{1{,}000{,}000} \times \$0.40 \times 365 \approx \$146{,}000 \text{ / year}
\]

At ten or a hundred billion tokens a day — ordinary for a popular product — that's millions a year,
and it recurs annually. Now a 2× efficiency gain (from quantization, better batching, a smarter
engine) isn't a one-time saving — it *halves that perpetual meter*. The same logic runs in reverse for
quality: a technique that cuts cost 30% but quietly degrades outputs is rarely worth it, because the
revenue from good answers also compounds.

!!! key "Optimization, defined in one line"
    Every technique in this book does exactly one thing: **raise goodput per dollar** — more
    useful tokens within the latency SLO, per GPU-hour you pay for. Quantization shrinks the bytes per
    token; batching and speculative decoding raise tokens per GPU; caching skips tokens you'd have
    recomputed; the right hardware and rung (Chapters 3, 8, 9) lower the dollar in the denominator.
    When you evaluate any optimization, reduce it to this fraction and ask which part it moves and what
    it costs the others.
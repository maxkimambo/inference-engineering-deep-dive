# Choosing a GPU

Everything so far was build-up to this: you're in a cloud console with a model, a latency target, and
a budget, and you have to pick. This section is the decision procedure — grounded in the three
ceilings (capacity, bandwidth, compute) and the wire between them — plus the current generations and
three worked choices end to end.

## The decision procedure

Run these gates **in order**. The first one that fails decides the next move; you don't get to the
compute question until the memory questions pass.

!!! key "Five gates, in order"
    1. **Capacity** — does `weights + peak KV cache` fit in HBM? If no → quantize, or go multi-GPU
       (and you're now in Chapter 5's parallelism, on an NVLink box from § 3.4).
    2. **Decode latency** — is `HBM bandwidth ÷ weight bytes` ≥ your required tokens/s per stream? If
       no → more bandwidth (H200/B200), or quantize to move fewer bytes.
    3. **Prefill latency (TTFT)** — is `(2 * P * prompt) / dense_FLOPS` within your TTFT budget? If no →
       more FLOPS, FP8, or prefix caching.
    4. **Throughput / cost** — at your batch size, what's the `$ per million tokens`? This picks
       *between* options that all pass 1–3.
    5. **Interconnect** — if multi-GPU, are the GPUs on NVLink (for TP) or only PCIe (pipeline/replicas
       only)? § 3.4.

Notice capacity and bandwidth — the *memory* questions — come first. The instinct to lead with "how
many FLOPS" is a training-shaped reflex; for serving it's the fourth question, not the first.

## The current field

Dense numbers (halve-the-headline already applied), as of the 2025–26 generation:

| GPU | Arch | HBM | Bandwidth | FP16 dense | FP8 dense | TDP | Interconnect | Best at |
|-----|------|----:|----------:|-----------:|----------:|----:|--------------|---------|
| **A100 80GB** | Ampere | 80 GB | 2.0 TB/s | 312 | — (no FP8) | 400 W | NVLink3 600 GB/s | cheap workhorse; ≤13B, or 70B across 2–4 |
| **L4** | Ada | 24 GB | 0.30 TB/s | ~121¹ | ~242¹ | 72 W | none | cheapest small-model / edge serving |
| **L40S** | Ada | 48 GB | 0.86 TB/s | 362 | 733 | 350 W | none (PCIe) | mid models, no-NVLink budget boxes |
| **H100 SXM** | Hopper | 80 GB | 3.35 TB/s | 989 | 1,979 | 700 W | NVLink4 900 GB/s | the default serving GPU |
| **H200 SXM** | Hopper | 141 GB | 4.8 TB/s | 989 | 1,979 | 700 W | NVLink4 900 GB/s | bigger models + faster decode, same compute |
| **B200** | Blackwell | 192 GB | 8.0 TB/s | 2,250 | 4,500² | ~1000 W | NVLink5 1,800 GB/s | frontier models, max throughput |
| **MI300X** | CDNA3 (AMD) | 192 GB | 5.3 TB/s | 1,305 | 2,610 | 750 W | Infinity Fabric | huge HBM per card; ROCm software |
| **TPU v5e** | Google | 16 GB | 0.82 TB/s | 197 (bf16) | — (INT8 394) | — | ICI | cost-efficient GCP serving |
| **TPU v6e** | Google | 32 GB | 1.6 TB/s | 918 (bf16) | — | — | ICI | Trillium; GCP at scale |

<small>¹ L4's headline 242 TFLOPS is the with-sparsity figure; dense FP16 ≈121, dense FP8 ≈242. ² B200
also adds **FP4** at ≈9,000 dense TFLOPS — its signature inference feature.</small>

!!! key "The H200 is the cleanest lesson on the whole table"
    Same compute die as the H100, so *identical* FLOPS — yet it's the better **inference** GPU,
    because it adds 76% more HBM capacity (141 vs 80 GB → a 70B model finally fits with KV room) and
    43% more bandwidth (4.8 vs 3.35 TB/s → 43% faster decode). If you ranked GPUs by the compute
    headline you'd call them equal and pick wrong. For decode-heavy serving, **memory is the product.**

## Worked choice 1 — serve Llama-3 8B, cheaply

**Gates.** 8B BF16 weights = 16 GB; KV at 8K context is small (~1–2 GB for a modest batch). Fits a
**24 GB L4** with room to spare. Decode: `300 GB/s ÷ 16 GB ≈ 19 tok/s` per stream in BF16 — and
quantizing to INT8 (8 GB) doubles that to ~38 tok/s while freeing more KV room. Prefill is light at
8B.

**Decision.** The **L4** at 72 W is the cost-optimal choice for a small model — you don't pay for
HBM or FLOPS you can't use. Step up to **L40S** (48 GB, 0.86 TB/s) only if you need higher decode
speed or bigger batches than the L4's thin 300 GB/s allows. There is no reason to put an 8B model on
an H100 unless you're packing many models per GPU or need the bandwidth for very high per-stream TPS.

!!! note "The most common overspend"
    Putting a small model on a big GPU because it's the default in a tutorial. An 8B model on an H100
    uses ~20% of its HBM and a sliver of its bandwidth — you're renting a truck to carry a backpack.
    Right-size *down*; use MIG (§ 3.1) or smaller cards.

## Worked choice 2 — serve Llama-3 70B with good latency

**Gates.** 70B BF16 = 140 GB → **fails the capacity gate on any single 80 GB GPU** (§ 3.2). Three
real paths:

| Path | Fits? | Decode TPS | Trade |
|------|-------|-----------:|-------|
| 1× H200, INT8 (70 GB) | yes, ~70 GB room for KV | ~69 tok/s | simplest; one GPU, no interconnect |
| 1× H100, INT4 (35 GB) | yes, lots of KV room | ~96 tok/s | aggressive quantization; watch quality |
| 2× H100, BF16 (TP=2) | yes, split across NVLink | high, full precision | needs NVLink box; best quality |

**Decision.** If you can get **H200s**, one card at INT8 is the lowest-complexity production answer —
it clears all gates with a single GPU and no interconnect to reason about. If you're on **H100s**,
either accept INT4 on one card (cheapest, if quality holds) or run **TP=2 across NVLink** for full
BF16 quality (§ 3.4 — and never attempt this split over PCIe). This is the canonical "70B is the
first model that forces a real hardware decision" case.

## Worked choice 3 — serve a frontier MoE (e.g. 400B–700B)

**Gates.** Hundreds of billions of parameters (even a sparse MoE keeps *all* experts resident in HBM)
blow past any single card — you're in **multi-GPU, single-node** territory, and the node must be an
**NVLink/NVSwitch** box (8× H100/H200) or an **NVL72** rack so tensor- and expert-parallel collectives
stay on the fast wire (§ 3.4). HBM capacity per card now dominates: this is where **B200 (192 GB)** and
**MI300X (192 GB)** earn their place — fewer cards to hold the model means fewer collective hops.

**Decision.** An **8× H200** or **8× B200** NVLink node, model sharded with **TP within the node + EP
for the experts** (Chapter 5). Reach for **NVL72** when even an 8-GPU node can't hold the model or
you need many of them to act as one memory pool. The hardware decision and the parallelism decision
are now the same decision — which is why this chapter and Chapter 5 are read together.

## Cost: the gate that picks between survivors

Gates 1–3 leave you with options that all *work*; **cost-per-token** picks among them. The method:
\[
\frac{\$}{\text{million tokens}} = \frac{\text{GPU \$/hour}}{\text{tokens/hour at your batch size}}
\]
The lever is the denominator: **batching** raises tokens/hour per GPU dramatically (Chapter 2's climb
up the roofline), so the cheapest-per-token option is usually "the smallest/cheapest GPU that still
fits the model and sustains a *large* batch within your latency budget," not the fastest GPU. A model
that fits on an L40S served at batch 64 can beat an H100 served at batch 4 on `$/token` even though
the H100 wins every single-stream benchmark.

!!! key "Two different questions, two different GPUs"
    **"Lowest latency per request?"** → most bandwidth, smallest batch → H200/B200.
    **"Lowest cost per token?"** → cheapest GPU that fits the model and sustains a big batch → often
    L40S/A100/H100 at high occupancy.
    A latency SLA and a cost target pull toward *different* hardware. Know which one you're optimizing
    before you pick — Chapter 7 turns this into capacity planning and autoscaling.

## AMD, TPU, and local — when the non-default wins

- **AMD MI300X** — 192 GB of HBM at 5.3 TB/s is a genuinely strong inference card; its gate is
  **software maturity** (ROCm + the engine's AMD support). Choose it when its memory advantage or
  price/availability beats NVIDIA *and* your inference engine supports it well. Validate the stack,
  not just the spec sheet.
- **Google TPU (v5e / v6e)** — compelling `$/token` inside GCP for models with first-class JAX/XLA
  support, wired by Google's ICI fabric instead of NVLink. The lock-in is the software ecosystem, not
  the silicon. Choose it when you're already on GCP and your model/serving stack targets TPUs.
- **Local / edge** — consumer cards (e.g. an RTX-class GPU with 24 GB GDDR) and Apple-silicon unified
  memory invert the constraints: capacity is tiny and bandwidth lower, so you lean hard on
  quantization (INT4 and below) and small models. The *reasoning* is identical — capacity gate first,
  then bandwidth — only the numbers shrink.

---

That closes the hardware chapter. You can read any accelerator's spec sheet and, in order, answer
will-it-fit, will-it-decode-fast-enough, will-it-prefill-in-budget, what-does-it-cost, and how-do-I-
wire-more-of-them. Chapter 4 puts software on top of this silicon — the CUDA stack and the inference
engines that turn these ceilings into served tokens — and Chapter 5's techniques are, every one of
them, a move against one of the three ceilings you just learned to measure.

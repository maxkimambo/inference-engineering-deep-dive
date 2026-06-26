# Compute and Precision

Memory tells you whether the model fits and how fast it decodes. **Compute** — the FLOPS number —
tells you how fast it *prefills*, how big a batch you can drive, and how long an image takes to
generate. It's the second ceiling of the roofline, and it comes wrapped in the single most misread
figure on any GPU datasheet. This section makes you fluent in what a TFLOP actually buys, why
shrinking the number format buys more of them, and how to read the headline without being fooled.

## What a FLOP is, and what "peak" means

A **FLOP** is one floating-point operation — a single multiply *or* a single add. The fundamental
operation of a matmul is the **multiply-accumulate** (MAC): `a*b + c`, which counts as **2 FLOPs**.
Peak throughput is just counting silicon:
\[
\text{peak FLOPS} = (\text{number of math lanes}) \times (\text{FLOPs per lane per clock}) \times (\text{clock rate})
\]
The Tensor Cores (§ 3.1) dominate the lane count for matrix math, which is why the "Tensor" rows on a
datasheet are 10–30× the plain FP32 row. **Peak is a ceiling you never fully reach.** The fraction
you do reach is **MFU — Model FLOPs Utilization** — and 40–60% is healthy for real inference. Always
mentally discount a peak number by ~2× to estimate reality.

## The asterisk: dense vs sparse

Here is the trap. NVIDIA's H100 datasheet leads with **"1,979 TFLOPS FP16."** That number is *with
structured sparsity* — and almost no production LLM uses it. The honest, dense number is **989
TFLOPS**, exactly half.

**Structured (2:4) sparsity** is a hardware feature where, in every group of 4 weights, 2 are forced
to zero. The Tensor Core then skips the zeros and does only the nonzero math, roughly doubling
throughput. The catch: your model must actually be *pruned* into that 2-of-4 pattern, and recover its
accuracy afterward — a training-time commitment most deployed models haven't made. So the 2× is real
silicon, but it's locked behind a constraint your weights probably don't satisfy.

!!! key "Halve every headline FLOPS number unless you've pruned for sparsity"
    Datasheets quote the **sparse** figure as the hero number (it's bigger). For ordinary
    dense-weight inference, use the **dense** figure — the smaller one, often printed in a footnote.
    H100: 989 (dense) not 1,979 (sparse). When comparing GPUs or predicting prefill, compare *dense*
    to *dense*, or you'll overestimate by exactly 2×. This is the most common way vendors' and
    blogs' numbers drift from what you measure.

We use **dense** numbers everywhere in this book for exactly this reason.

## Why lower precision means more FLOPS

The Tensor Core's throughput depends on the **bit-width** of the numbers it multiplies. A datapath of
fixed physical width holds twice as many 8-bit values as 16-bit values, so it does roughly twice the
operations per clock. Each halving of precision roughly *doubles* peak throughput — and, on decode,
*halves* the bytes you move (§ 3.2). Precision is the one knob that helps on **both** sides of the
roofline at once, which is why Chapter 5 spends a whole section on quantization.

H100 dense Tensor throughput, by format:

| Format | Dense TFLOPS (H100) | vs FP16 |
|--------|--------------------:|:-------:|
| TF32 | 495 | 0.5× |
| FP16 / BF16 | 989 | 1× |
| FP8 (E4M3/E5M2) | 1,979 | 2× |
| INT8 | 1,979 (TOPS) | 2× |

Blackwell (B200) extends the ladder downward to **FP4**, doubling again (≈9,000 dense FP4 TFLOPS) —
the headline reason Blackwell is a generational leap for inference specifically.

## The number formats, decoded

A floating-point format splits its bits into **sign + exponent + mantissa**. The exponent sets
**dynamic range** (how big and small a number it can represent); the mantissa sets **precision** (how
finely it resolves values near each other). Inference engineering is the art of spending bits where
the model needs them.

| Format | Layout (S·E·M) | Bits | Range vs FP32 | Why it's used |
|--------|----------------|------|---------------|---------------|
| **FP32** | 1·8·23 | 32 | full | the reference; too slow/big for serving |
| **TF32** | 1·8·10 | 19* | full | FP32's range, fewer mantissa bits; transparent speedup on Ampere+ |
| **FP16** | 1·5·10 | 16 | **narrow** | high precision but *overflows* easily — needs loss scaling |
| **BF16** | 1·8·7 | 16 | **full** | FP32's range, less precision → the de-facto default; rarely overflows |
| **FP8 E4M3** | 1·4·3 | 8 | small | weights & activations (more precision, less range) |
| **FP8 E5M2** | 1·5·2 | 8 | larger | gradients / wide-range tensors (more range, less precision) |
| **FP4 E2M1** | 1·2·1 | 4 | tiny | Blackwell inference; needs careful per-block scaling |
| **INT8** | integer | 8 | fixed-point | quantized inference with a scale factor (Chapter 5) |

!!! key "BF16 beats FP16 for one structural reason: range"
    FP16 spends more bits on the mantissa (10 vs 7) but fewer on the exponent (5 vs 8), so it's
    *more precise* but has a *much narrower* dynamic range — large activations overflow to infinity,
    which is why FP16 training needs fiddly loss-scaling. BF16 keeps FP32's full exponent (same
    range) and sacrifices precision the model doesn't miss. For inference you almost always want
    BF16. (This is also why "qwen 64-bit" or "FP16 by default" are red flags — no serving model is
    FP32/FP64, and FP16-vs-BF16 is a deliberate range choice, not interchangeable.)

The numeric *mechanics* of mapping real weights into INT8/INT4 — scale, zero-point, per-channel vs
per-tensor — are Chapter 5's quantization section. Here the point is narrower and about hardware:
**a number format is a physical throughput setting on the Tensor Core, and smaller is faster on both
roofline axes.**

## Compute applied: predicting prefill (TTFT)

Decode is memory-bound (§ 3.2); **prefill is compute-bound**, so this is where FLOPS earns its keep.
A forward pass costs about **2 FLOPs per parameter per token**, so prefilling an `N`-token prompt
through a `P`-parameter model is `~2 * P * N` FLOPs. For Llama-3 70B:

| Prompt | Work | H100 FP16 (dense) | H100 FP8 (dense) | A100 FP16 |
|--------|------|------------------:|-----------------:|----------:|
| 1,000 tokens | 140 TFLOP | ~280 ms | ~140 ms | ~900 ms |
| 4,000 tokens | 560 TFLOP | ~1.1 s | ~570 ms | ~3.6 s |

(at ~50% MFU; ideal would be 2× faster). Three things to take from it:

1. **Prefill scales linearly with prompt length** — doubling the prompt doubles TTFT. Long-context
   prompts are a compute problem, which is exactly why prefix caching (Chapter 5) — *skipping*
   re-prefill of shared prompt prefixes — is so valuable.
2. **FP8 halves prefill latency** on Hopper, because prefill lives on the compute axis where the 2×
   FLOPS is real.
3. **The A100 is ~3× slower at prefill** than the H100 (312 vs 989 dense TFLOPS) — a gap you'd miss
   if you only looked at memory bandwidth.

## When decode becomes compute-bound too

Decode is memory-bound *per stream*, but batching changes the picture. When you process a batch of
sequences together, each weight read from HBM is reused across all of them — arithmetic intensity
rises roughly with batch size (Chapter 2's "climb the diagonal"). Push the batch large enough and you
cross the **ridge point** (~295 ops/byte on H100): decode flips from memory-bound to compute-bound,
and now the FLOPS number caps your *throughput*.

!!! key "Both ceilings matter — at different operating points"
    - **Latency-critical, low batch** → memory-bound → buy **bandwidth** (H200 over H100).
    - **Throughput-critical, high batch** → compute-bound → buy **FLOPS** (and lower precision).
    - **Prefill / image / video** → compute-bound → buy **FLOPS**.

    A real server does prefill *and* batched decode, so it touches both ceilings. This is the whole
    argument for **disaggregation** (Chapter 5): split the two phases onto hardware tuned for their
    different bottlenecks instead of compromising on one GPU.

---

You can now read every line of a GPU datasheet and say what it does for your workload: capacity gates
fit, bandwidth gates decode, FLOPS gates prefill and large-batch throughput — and the sparse headline
is half of what you'll see. One ceiling remains, the one that appears the moment a model outgrows a
single GPU: the wire to the next one.

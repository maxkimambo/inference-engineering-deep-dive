# Quantization

**Quantization** is storing and computing a model's numbers in a *lower-precision* format than it
was trained in — FP16 weights becoming FP8 or INT4, say. It's the highest-leverage technique in
this chapter because it attacks *both* phases at once, and the only lossy one, which is why most of
the work is about *not* losing quality.

Recall the [roofline](../models/bottlenecks.md): cutting precision in half helps on both walls.

- **Prefill** (compute-bound) → lower-precision Tensor Cores do roughly **2× the FLOPS**.
- **Decode** (memory-bound) → each value is **half the bytes**, so you move twice as much per second
  of bandwidth — effectively doubling the resource decode is starved for.

In practice you don't get a clean 2× — there's overhead converting and handling low-precision data —
but **dropping one precision level typically buys 30–50% better performance** for LLMs.

!!! key "Why quantization is risky: errors compound"
    A forward pass is thousands of dependent operations; in decode, each token's KV feeds every later
    token. Small rounding errors don't stay small. Watch precision compound through a trivial example —
    squaring and cubing π at three precisions:

    | π precision | π² | π³ |
    |-------------|-----|-----|
    | 3.14159 | 9.869588 | 31.006198 |
    | 3.14 | 9.8596 | 30.959144 |
    | 3 | 9 | 27 |

    The error in the *input* precision is amplified by every operation downstream. Most of
    quantization research is (a) preventing these errors and (b) minimizing their impact on the final
    output.

## 5.1.1 Number formats

To reason about quantization you must read number formats fluently. Every format has a **precision**
(bit count), a **type** (integer or floating-point), and — once quantized — a **scale factor** that
maps the low-precision values back toward the original range.

Two derived properties decide how well a format represents real model values:

- **Dynamic range** — the spread between the smallest and largest representable value. This is what
  lets a format capture *outliers* without clipping them.
- **Granularity** — how many values share a single scale factor. Finer granularity = less chance of
  one outlier distorting its neighbors, at the cost of storing more scale factors.

### Integer vs floating-point: it's about dynamic range

An *n*-bit format has 2ⁿ distinct codes either way — INT8 and FP8 both have 256 — but they *place*
those codes differently.

- An **integer** format spaces its codes **evenly**. 256 equal steps across a range.
- A **floating-point** format spaces them **logarithmically** — dense near zero, sparse far out —
  using three fields:

```
  FP8 in E4M3 layout (8 bits)
  ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │S│ E E E E │ M M M │
  └─┴─────────┴───────┘
   sign  exponent  mantissa
   (1)     (4)       (3)

  value ≈ (−1)^S × 1.MMM × 2^(EEEE − bias)
```

- **Sign (S)** — one bit, positive or negative.
- **Exponent (E)** — sets the *scale* (which power-of-two bucket). More exponent bits = more dynamic
  range.
- **Mantissa (M)** — the precision *within* a bucket. More mantissa bits = finer resolution.

`E4M3` means 4 exponent + 3 mantissa bits; `E5M2` trades a mantissa bit for an exponent bit — *less*
precision, *more* range. Same 8 bits, different balance.

!!! key "Why floating-point wins for inference: outliers"
    Model weights and activations are mostly small, with rare large **outliers** that carry
    disproportionate meaning. An integer format spreads its 256 codes evenly, so it either clips the
    outliers or wastes resolution on the dense middle. Floating-point's logarithmic spacing keeps fine
    resolution near zero *and* reaches the outliers. That extra dynamic range is why **production,
    quality-sensitive inference sticks to floating-point formats** — integer formats lack the range and
    are reserved for size-critical local/edge inference.

### The formats you'll meet

| Name | Abbr | First arch | Notes |
|------|------|-----------|-------|
| 64-bit float | FP64 | Fermi (2010) | scientific computing only, never inference |
| 32-bit float | FP32 | Kepler (2012) | sometimes training, almost never inference |
| 16-bit float | FP16 | Pascal (2016) | common native training/inference precision |
| Brain float 16 | BF16 | Ampere (2020) | FP16's range, less mantissa — the usual native format |
| 8-bit float | FP8 | Hopper (2022) | the inference **sweet spot** |
| Mixed-precision FP8 | MXFP8 | Blackwell (2024) | microscaling FP8 |
| 8-bit integer | INT8 | Pascal (2016) | size-critical, lacks range |
| 6-bit float | FP6 | Blackwell (exp.) | AMD adopting quickly |
| 4-bit float | FP4 | Blackwell (2024) | aggressive; quality-risky |
| Mixed-precision FP4 | MXFP4 | Blackwell (2024) | microscaling FP4 |
| NVIDIA FP4 | NVFP4 | Blackwell (proprietary) | finest-grain 4-bit |
| 4-bit integer | INT4 | Turing (2018) | local/edge only |

The practical landscape: **16, 8, and 4-bit are the inference precisions.** FP8 (and microscaling
MXFP8) is the current sweet spot — big speedups, little quality loss, flexible enough even for the KV
cache. FP4/NVFP4 is promising but quality-risky.

### Granularity: tensor, channel, block — and microscaling

A single scale factor stretched across too many values gets dragged around by outliers. Granularity
controls how finely you slice:

- **Per-tensor** — one scale for the entire QKV tensor. Cheapest, coarsest.
- **Per-channel** — a scale per feature vector (row/column). Middle ground.
- **Per-block (group)** — split each vector into blocks of *N* values, one scale per block. Finest,
  most metadata.

```
 per-tensor   [ one scale for everything ............................ ]   coarse
 per-channel  [ scale | scale | scale | scale | scale | scale ....... ]
 per-block    [s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s]   fine
```

**Microscaling** formats bake fine granularity into the format itself. **MXFP8/MXFP4** compute a
blockwise scale every **32** values, clawing back the dynamic range a raw 8/4-bit format loses.
**NVFP4** goes finer still — block size **16** plus a secondary FP32 *global* scale — specifically to
fight 4-bit quality loss.

!!! info "The microscaling trade"
    Finer scaling means more scale factors to *store* and *apply*, eating into the speedup — and you
    now apply both block and tensor scales. Blackwell offsets this by applying scale factors directly
    in the Tensor Cores. Net: microscaling formats give you 4-bit storage with closer-to-8-bit
    quality, at a small compute cost the hardware mostly hides.

## How quantization actually maps numbers: the affine transform

The formats above are the *destination*. The *mechanism* that moves a high-precision value into a
low-precision grid is the **affine mapping** — worth seeing once, because scale and zero-point are
exactly the knobs the formats above are configuring.[^kalyan]

To quantize a real value \(x\) to an integer code \(q\):

\[
q = \text{round}\!\left(\frac{x}{S}\right) + Z
\qquad\text{and to recover it}\qquad
x \approx S\,(q - Z)
\]

- **Scale `S`** — the size of one quantization step: the real-world distance between adjacent integer
  codes. \(S = \dfrac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}\). Smaller `S` = finer resolution but
  narrower range. *This is the "scale factor" the formats and granularity sections keep referencing —
  per-tensor gives every value the same `S`; per-block gives each block its own.*
- **Zero-point `Z`** — the integer code that represents real **0**. It shifts the grid so that zero
  lands *exactly* on a code, with no rounding error.

```
  real values     x_min ───────────── 0 ──────────────────── x_max
                    │                  │                        │
  quantize (÷S, +Z) ▼                  ▼                        ▼
  integer codes    q_min ───────────── Z ──────────────────── q_max
                  (e.g. -128)      (exact zero)            (e.g. 127)
```

!!! key "Why exact zero matters (the 'tare' intuition)"
    Zero is special — padding, masked attention positions, and post-ReLU activations are *all
    literally zero*, and they're everywhere. If your mapping represented 0 as "approximately 0.04,"
    that tiny error would be injected into millions of values and compound. The zero-point is the
    **tare** on a kitchen scale: you zero it before weighing so the container's mass never pollutes
    the measurement. Force real 0 onto an exact code and a whole class of error vanishes.

    When \(Z = 0\) the mapping is **symmetric** (range centered on zero, slightly faster); when
    \(Z \neq 0\) it's **asymmetric/affine** (better for lopsided ranges like ReLU outputs, which are
    never negative). That choice is one of the real knobs in a quantization recipe.

## 5.1.2 Quantization approaches

After picking a precision, two questions remain: **when** do you quantize, and **what** do you
quantize?

### When: during training vs after

- **Quantization-aware training (QAT)** — train the weights *and* the scale factors together, so the
  converged model is already accurate at the target precision. Best quality; only the model's creator
  can do it. Some labs ship QAT models (GPT-OSS in MXFP4, Kimi K2 Thinking in INT4).
- **Post-training quantization (PTQ)** — convert *finished* weights to a new precision, computing scale
  factors and preserving accuracy via **calibration** (running sample data through to measure real
  value ranges). This is what inference engineers do, since you work with finished open weights.

A leading PTQ tool is **NVIDIA TensorRT Model Optimizer (ModelOpt)** — also does pruning, distillation,
sparsity; outputs run on vLLM, SGLang, and TensorRT-LLM.

### What: the sensitivity ladder

Not all components tolerate quantization equally. **Quantize from least to most sensitive, and stop
before you hurt quality:**

```
   QUANTIZATION RISK         safe ▲
   ┌───────────────────────┐      │
   │ 1. Weights (linear)   │  least sensitive — biggest, most redundant
   │ 2. Activations        │  somewhat sensitive
   │ 3. KV cache           │  moderately sensitive
   │ 4. Attention (softmax)│  highly sensitive — quantize last, if ever
   └───────────────────────┘      │
                            risky ▼
```

1. **Weights** (especially linear layers) — least sensitive; they're the bulk of the model and
   individually redundant.
2. **Activations** — the intermediate outputs; somewhat sensitive. (The activation *functions*
   themselves are rarely quantized — too tiny to matter.)
3. **KV cache** — moderately sensitive, but quantizing it is a *force multiplier*: more cache fits in
   memory and reads faster, boosting prefix caching and disaggregation directly.
4. **Attention** (the softmax path) — highly sensitive. All but the most aggressive schemes run
   **softmax in full precision.**

!!! key "Why attention is the riskiest to quantize"
    Two compounding reasons. First, softmax is exponential — it's exquisitely sensitive to dynamic
    range, and low precision distorts the distribution. Second, and worse: **each token's attention
    depends on every prior token's KV.** A precision error in attention doesn't stay local — it feeds
    forward into thousands of downstream tokens and snowballs. The π-cubing table at the top of this
    page, but a thousand steps deep. That's why the sensitivity ladder ends here.

Even within "safe" components you can be selective: early and late layers (input/output) are more
sensitive than the middle, so they're often left in original precision. **A solid moderate recipe:**
FP8 (ideally microscaling MXFP8 for its dynamic range) on select linear layers, activations, and often
the KV cache — leaving attention's internals alone.

!!! info "Integer formats and the local-inference exception"
    The data-center rule is "stick to floating-point." Local/edge inference flips it: when squeezing
    DeepSeek onto a MacBook, size beats range. **GGUF** is the popular format for distributing heavily
    quantized models on Hugging Face, and *dynamic* quantizations (Unsloth's famous 1.58-bit) keep
    sensitive layers high-precision while crushing the rest — averaging out to sub-2-bit. Brilliant for
    local, but production quality-sensitive work should stay in floating-point.

## 5.1.3 Measuring quality impact

The bar for production quantization is **zero perceptible quality loss**. You can't eyeball that — you
measure it, three ways, always apples-to-apples against the original weights:

1. **Perplexity** — the cheapest check. Give the model known text and measure how well it predicts the
   actual next tokens. **Perplexity is how "surprised" the model is** by correct text — lower is
   better. After quantizing, you want a *minimal increase*. Fast, but coarse.
2. **Intelligence benchmarks** — run a public suite (MMLU, SWE-bench) and compare scores. You want a
   *minimal reduction*.
3. **Custom evals** — a product-specific evaluation matching your real usage. The most meaningful, and
   the one that should gate a deploy.

!!! warning "Quantization is a dial, not a switch"
    Because LLMs are non-deterministic, scores vary run to run — you're looking for a difference
    *indistinguishable from noise*, not zero. And you have continuous control: FP8 instead of FP4, or
    weights-only instead of weights+KV, trade a little speed for a lot less risk. Run all three checks,
    pick the most aggressive setting that still passes your custom eval, and no further.

---

Quantization makes every *other* technique cheaper — fewer bytes to cache, transfer, and parallelize.
Next: spending the compute that decode leaves idle.

**Next:** [Speculative Decoding →](speculative-decoding.md)

[^kalyan]:
    The affine-mapping treatment (scale `S`, zero-point `Z`, the tare analogy, per-tensor/channel/block
    granularity) follows Vivek Kalyanarangan, *Quantization and Fast Inference: A Practitioner's Guide
    to Efficient AI* (Manning, 2026), ch. 2 — recommended for a from-first-principles build-up of the
    fixed-point and floating-point machinery underneath these formats.

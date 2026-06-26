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

## Worked example: taking Qwen from BF16 to INT4

Let's run the whole pipeline on a real target: **Qwen2.5-7B**, shipped in **BF16** (16-bit — no model
is 64-bit), quantized down to **INT4 weights**. This is the most common "shrink it to 4-bit" job, and
its proper name is **W4A16** — *4-bit weights, 16-bit activations*. The activations stay at 16-bit on
purpose: the [sensitivity ladder](#what-the-sensitivity-ladder) says weights tolerate quantization best,
so we crush them and leave everything else alone.

### Step 1 — the math, on eight real weights

Quantization happens per **group** of weights sharing one scale (here a group of 8; production uses
~128). Take one group from a weight row and quantize it **symmetrically** (zero-point `Z = 0`, standard
for weights) into signed INT4, whose codes run `[-7, 7]` (qmax = 7):

```
weights (BF16)  w = [ 0.12, -0.41,  0.93, -0.05,  0.55, -0.88,  0.27,  0.02 ]

1. absmax       = max(|w|)           = 0.93
2. scale S      = absmax / qmax      = 0.93 / 7   = 0.13286
3. quantize     q = round(w / S), clamp to [-7, 7]
                  = [  1,   -3,    7,    0,    4,   -7,    2,    0  ]   ◄ stored as INT4
4. dequantize   ŵ = q * S
                  = [0.133, -0.399, 0.93, 0.0, 0.531, -0.93, 0.266, 0.0]

   error  ŵ − w  = [+0.013, +0.011, 0.0, +0.05, −0.019, −0.05, −0.004, −0.02]
   max abs error = 0.05
```

What to notice:

- **Each weight is now 4 bits** — an integer in `[-7, 7]` — plus **one shared `S` per group**. That `S`
  is the per-block scale factor from the [granularity](#granularity-tensor-channel-block-and-microscaling)
  section, made concrete.
- **`0.93` is exact** (it set the scale), but `−0.05` rounded all the way to `0` — small values near a
  big outlier lose the most. That's *precisely* why **group size matters**: a smaller group means a
  local outlier inflates the scale for fewer neighbors. Halve the group, and `−0.05` might land in a
  group whose absmax is `0.41`, getting a finer scale and surviving.
- The errors look tiny, but recall the π-cubing table at the top of this page — across billions of
  weights and thousands of dependent steps, the recipe's job is to keep them from compounding.

### Step 2 — the payoff

Do that for all of Qwen2.5-7B's ~7.6B weights:

| | Size | Note |
|---|------|------|
| BF16 weights | **15.2 GB** | 2 bytes each |
| INT4 weights | 3.8 GB | 0.5 byte each |
| + group scales (group=128) | +0.12 GB | one BF16 scale per 128 weights |
| **Effective INT4** | **≈ 3.9 GB** | **≈ 3.9× smaller** |

A model that needed an 80 GB A100 now fits on a 12 GB consumer GPU — and decode, being memory-bound,
gets faster because it moves ¼ the weight bytes per token.

### Step 3 — but don't actually use round-to-nearest

The Step 1 math is **round-to-nearest (RTN)** — the simplest scheme, and at INT4 it loses too much
quality to ship. Real W4A16 uses a smarter **PTQ algorithm** that spends a little calibration compute to
place the codes better:

- **GPTQ** — quantizes weights one column at a time, using second-order (Hessian) information to
  **compensate the not-yet-quantized weights** for each rounding error. Minimizes the layer's *output*
  error, not each weight's error.
- **AWQ** (Activation-aware Weight Quantization) — notices that a few weight **channels** matter far more
  (judged by activation magnitude) and **scales those salient channels up** before quantizing, so
  rounding hurts them less. Often the best quality/speed for INT4.

Both still produce a W4A16 checkpoint; they just choose `q` more cleverly than `round(w/S)`.

### Step 4 — the production recipe

Putting the whole chapter's machinery into an actual workflow:

```
1. DECIDE THE RECIPE
   format      INT4 weights, BF16 activations   (W4A16)
   granularity per-group, group_size = 128       (finer = better quality, more scales)
   scope       linear-layer weights only;
               keep embeddings, LM head, and attention in BF16   (sensitivity ladder)

2. CALIBRATE
   run a few hundred representative samples through the model so the
   algorithm sees real activation/weight ranges  (GPTQ/AWQ need this)

3. QUANTIZE
   run AWQ or GPTQ via a tool — llm-compressor, AutoAWQ, or NVIDIA
   ModelOpt — producing a quantized safetensors (or GGUF for local/llama.cpp)

4. EVALUATE  (§5.1.3, against the original BF16 weights)
   perplexity delta ≈ noise?   MMLU/your-eval drop ≈ noise?
   if a custom eval regresses → back off: try INT4→ a microscaling FP4,
   or quantize fewer layers, or W4A16 → W8A16

5. DEPLOY
   load the checkpoint on vLLM / SGLang / TensorRT-LLM — they read the
   quant format and run INT4 Tensor-Core kernels
```

!!! key "The one-paragraph version"
    To take Qwen from BF16 to INT4: choose **W4A16, per-group (128), weights-only**; **calibrate** on
    representative data; run **AWQ or GPTQ** (never plain round-to-nearest at 4-bit); **evaluate** the
    perplexity/benchmark/custom-eval deltas against the BF16 original; and if quality regresses, **dial
    back** — a higher-granularity FP4, fewer quantized layers, or 8-bit. You get a ~4× smaller, faster
    model, and the entire job is managing the quality you trade for it.

## Hands-on: quantizing Qwen with `llm-compressor`

Here is the recipe above as runnable code. We use **`llm-compressor`** — the vLLM-native quantization
library — because its output loads straight into vLLM/SGLang and its layer-targeting is exactly the
control you want.[^llmc]

!!! warning "This runs on a GPU box, not your laptop"
    Quantizing 7B needs a CUDA GPU (~24 GB) and downloads the BF16 weights (~15 GB) first. The code is
    correct against the current `llm-compressor` API; run it where there's a GPU.

### Install

```bash
pip install llmcompressor   # pulls in transformers, datasets, compressed-tensors
```

### The minimal full script

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")   # (1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = GPTQModifier(            # (2)
    targets="Linear",             # which modules to quantize
    scheme="W4A16",               # 4-bit weights, 16-bit activations, group_size 128
    ignore=["lm_head"],           # which modules to SKIP
)

oneshot(
    model=model,
    dataset="HuggingFaceH4/ultrachat_200k",   # (3) calibration data
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,              # 256–512 is plenty; more = slower
)

SAVE_DIR = "Qwen2.5-7B-Instruct-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)   # (4)
tokenizer.save_pretrained(SAVE_DIR)
```

1. `dtype="auto"` loads the model in its native BF16.
2. `GPTQModifier` is the smart PTQ from Step 3 — not round-to-nearest. Swap one import to use AWQ
   instead (below).
3. Calibration: a few hundred samples so GPTQ measures real value ranges. **Use domain-matched data**
   — for a code model, calibrate on code, not chat.
4. `save_compressed=True` writes the packed 4-bit checkpoint; vLLM reads it directly.

That's the whole thing. The interesting part — and what you asked for — is the `targets`/`ignore`
control.

### Targeting different layers

`targets` says *what to quantize*; `ignore` says *what to leave in BF16*. Both accept module **class
names**, exact **module paths**, or **regex** with a `re:` prefix. This is how you walk the [sensitivity
ladder](#what-the-sensitivity-ladder) in practice — quantize the robust layers hard, protect the
sensitive ones.

First, know Qwen's module paths (printable with `print(model)`):

```
model.embed_tokens                      ← input embedding   (sensitive)
model.layers.{0..27}.self_attn.q_proj   ← attention projections
                     .self_attn.k_proj
                     .self_attn.v_proj
                     .self_attn.o_proj
model.layers.{0..27}.mlp.gate_proj      ← MLP / FFN  (biggest, most robust)
                     .mlp.up_proj
                     .mlp.down_proj
lm_head                                 ← output head       (sensitive)
```

Now, recipes from least to most conservative:

=== "Default (skip the head)"

    ```python
    GPTQModifier(targets="Linear", scheme="W4A16",
                 ignore=["lm_head"])
    ```
    Quantize every Linear, keep only the output head in BF16. The standard starting point.

=== "Protect the edge layers"

    ```python
    GPTQModifier(targets="Linear", scheme="W4A16",
                 ignore=["lm_head",
                         "re:model\\.layers\\.(0|1|26|27)\\..*"])  # first 2 + last 2
    ```
    Early and late layers are more sensitive (page §5.1.2). Leave the first two and last two
    transformer blocks in BF16, quantize the middle 24. Costs a little memory, recovers quality.

=== "MLP-only (skip attention)"

    ```python
    GPTQModifier(targets="Linear", scheme="W4A16",
                 ignore=["lm_head", "re:.*self_attn.*"])  # only MLP Linears quantized
    ```
    Attention is the most sensitive component. Quantize just the big, robust MLP projections and leave
    all attention projections in BF16 — most of the size win, least of the risk.

=== "Spare a known-sensitive projection"

    ```python
    GPTQModifier(targets="Linear", scheme="W4A16",
                 ignore=["lm_head", "re:.*down_proj"])  # down_proj often sensitive
    ```
    The MLP `down_proj` frequently carries outliers and degrades worst under 4-bit. Skip just that one
    projection across all layers.

!!! key "The targeting workflow"
    Start with **default**, run your evals (§5.1.3). If quality regresses, don't abandon 4-bit — **add
    the regression's likely culprits to `ignore`** and re-run: edge layers, then attention, then
    `down_proj`. You're searching for the smallest set of BF16 exceptions that recovers quality, which
    is mixed-precision quantization done by hand. Each module you move to `ignore` costs a little memory
    and buys a little quality.

!!! info "Memory tip for big models"
    Add `sequential_targets=["Qwen2DecoderLayer"]` to the modifier to quantize **one decoder layer at a
    time**, keeping only that layer's activations in memory. Essential when the model barely fits —
    it's how the same recipe scales from 7B to 70B+.

### Use AWQ instead (one import)

AWQ often edges out GPTQ at 4-bit (it protects salient channels — Step 3). Same call, different
modifier:

```python
from llmcompressor.modifiers.awq import AWQModifier

recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
# ...identical oneshot(...) and save_pretrained(...)
```

### Run it on vLLM

The compressed checkpoint needs no special flags — vLLM detects the quantization from the saved config:

```bash
vllm serve ./Qwen2.5-7B-Instruct-W4A16-G128
```

Then evaluate (§5.1.3): compare perplexity and your custom eval against the original `Qwen2.5-7B-Instruct`.
If it passes, you've got a ~4× smaller model serving on a quarter of the VRAM.

---

Quantization makes every *other* technique cheaper — fewer bytes to cache, transfer, and parallelize.
Next: spending the compute that decode leaves idle.

**Next:** [Speculative Decoding →](speculative-decoding.md)

[^kalyan]:
    The affine-mapping treatment (scale `S`, zero-point `Z`, the tare analogy, per-tensor/channel/block
    granularity) follows Vivek Kalyanarangan, *Quantization and Fast Inference: A Practitioner's Guide
    to Efficient AI* (Manning, 2026), ch. 2 — recommended for a from-first-principles build-up of the
    fixed-point and floating-point machinery underneath these formats.

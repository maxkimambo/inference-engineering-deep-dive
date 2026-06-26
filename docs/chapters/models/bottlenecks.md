# Calculating Inference Bottlenecks

The previous section *claimed* prefill is compute-bound and decode is memory-bound. This section
**proves it** with arithmetic you can redo on a napkin. Once you can do this, "is this workload
compute- or memory-bound?" stops being a vibe and becomes a number you compute.

## Two resources, one race

In a perfectly optimized system every resource is busy all the time. On a GPU there are two that
matter for inference:

- **Compute** — floating-point operations per second (**FLOPS**) the GPU can do.
- **Memory bandwidth** — bytes per second the GPU can move between its memory (HBM) and its
  compute units.

In the ideal world, compute never idles waiting on memory and bandwidth never idles waiting on
compute. In the real world there's always a **bottleneck**: one resource is saturated while the
other sits partly idle. Finding which one is the entire game — *if an operation is memory-bound,
no amount of compute optimization makes it faster, and vice versa.*

The headline results we're about to derive:

| Operation | Bottleneck | Governs |
|-----------|-----------|---------|
| LLM **prefill** (build KV cache) | **compute** | TTFT |
| LLM **decode** (generate tokens) | **memory** | TPS |
| Image / video generation | **compute** | generation time |

## The ops:byte ratio of a GPU

Every GPU has a compute speed and a memory bandwidth. Their ratio tells you how much math the GPU
*wants* to do per byte it reads, to stay balanced.

Take an **NVIDIA H100** in FP16:

- Compute: **989 TFLOPS** (dense FP16)
- Bandwidth: **3.35 TB/s**

\[
\text{ops:byte} = \frac{989 \times 10^{12} \ \text{ops/s}}{3.35 \times 10^{12} \ \text{bytes/s}}
\approx 295 \ \text{ops/byte}
\]

!!! key "What 295 means"
    For an H100 to be **perfectly balanced**, your workload must do **~295 floating-point
    operations for every byte it reads from memory**. Do *more* math per byte → you're
    compute-bound (compute is the wall). Do *less* → you're memory-bound (bandwidth is the wall,
    compute starves). 295 is the break-even line.

The ops:byte ratio is a property of the *hardware*, measured per second. To compare a *workload*
against it, we need the equivalent ratio for the algorithm — its arithmetic intensity.

## Arithmetic intensity of a workload

- **Arithmetic intensity** (a.k.a. operational intensity) — the ratio of compute work to memory
  traffic for one run of an algorithm.

\[
\text{intensity} = \frac{\text{work (FLOPs)}}{\text{memory traffic (bytes)}}
\]

Where ops:byte is measured per second on the *hardware*, arithmetic intensity is measured across a
single execution of a *function*. Compare the two:

- intensity **>** ops:byte → **compute-bound** (you've saturated the math units)
- intensity **<** ops:byte → **memory-bound** (you've saturated the memory bus)

## The roofline model

The standard way to visualize this is the **roofline** chart: performance (y) against arithmetic
intensity (x).

```
 performance
    │                 ┌─────────────────────  ◄ compute ceiling (peak FLOPS)
    │               ╱   COMPUTE BOUND
    │             ╱      (intensity > ops:byte → hits the flat roof)
    │           ╱
    │         ╱
    │       ╱  MEMORY BOUND
    │     ╱    (intensity < ops:byte → hits the slanted roof)
    │   ╱
    │ ╱  ◄ bandwidth ceiling (slope = peak bytes/s)
    └────────────┼────────────────────────► arithmetic intensity
              ops:byte
            (the ridge point, ~295 on H100)
```

- The **diagonal** is the bandwidth ceiling — at low intensity you're memory-limited and
  performance rises with intensity.
- The **horizontal** is the compute ceiling — past the ridge point, more intensity buys nothing
  because the math units are maxed.
- The **ridge point** sits exactly at the hardware's ops:byte ratio (~295 for the H100).

To find a system's bottleneck, compute the arithmetic intensity of its most expensive operation
and see which side of the ridge it lands on. For LLM inference, that operation is **attention**.

## Worked example: why decode is memory-bound

Let's compute the arithmetic intensity of a **single decode step's attention** and watch it land
far below 295. Setup (matching the standard, unoptimized attention algorithm):

- Sequence length **N = 4096**
- Attention-head dimension **d = 128**
- FP16 → **2 bytes per value**
- Matrices: `Q, K, V` are `N×d`; the score matrix `S = QKᵀ` and probability matrix `P` are `N×N`;
  output `O` is `N×d`

The algorithm has three steps, each a *read → compute → write*:

| Step | Read (bytes) | Compute (FLOPs) | Write (bytes) |
|------|--------------|-----------------|---------------|
| `S = QKᵀ` | `2·N·d + 2·N·d` | `(2d)·(N·N)` | `2·N·N` |
| `P = softmax(S)` | `2·N·N` | `3·(N·N)` | `2·N·N` |
| `O = P·V` | `2·N·N + 2·N·d` | `(2N)·(N·d)` | `2·N·d` |

**Total memory traffic** (sum of reads + writes):

\[
(2\cdot2Nd + 2Nd) + (2NN + 2NN) + (2NN + 2Nd + 2Nd) = 8N^2 + 8Nd \ \text{bytes}
\]

**Total compute** (sum of the middle column):

\[
2dN^2 + 3N^2 + 2N^2 d = 4N^2 d + 3N^2 \ \text{FLOPs}
\]

**Arithmetic intensity:**

\[
\frac{4N^2 d + 3N^2}{8N^2 + 8Nd} \approx 62 \ \text{ops/byte}
\]

!!! key "62 ≪ 295 → memory-bound, proven"
    Decode attention does only **~62 operations per byte** it moves. The H100 wants **295**. So on
    decode the GPU is reading data far faster than it can find math to do with it — **memory
    bandwidth is the wall**, and the expensive FLOP units sit ~80% idle. *This is why decode is
    memory-bound, not a heuristic — a ratio.*

The deeper cause: on decode you generate **one token at a time**, so each forward pass loads the
*entire* model's weights from memory to do a single token's worth of math. Enormous reads, tiny
compute → low intensity. On prefill you push the *whole input sequence* through those same weights
in one parallel pass — the weights are read once and reused across thousands of tokens, so compute
dwarfs the memory traffic → high intensity → compute-bound.

!!! note "This is an exercise, not a daily chore"
    You won't hand-derive arithmetic intensity in production — engines and profilers tell you where
    you're bound. But doing it once builds the intuition that every Chapter 5 technique relies on.

## Turning the knowledge into moves

Knowing the bottleneck tells you which optimizations can possibly help:

- **Decode is memory-bound** → *raise* arithmetic intensity to climb the diagonal roofline. The
  canonical move is **batching**: process many requests' tokens together so each weight read is
  reused across more math. Continuous batching, speculative decoding, and quantization (fewer
  bytes per weight) all attack the same axis. This is why a decode-heavy server's throughput jumps
  when you batch but its single-request latency doesn't.
- **Prefill is compute-bound** → you're already on the flat roof; throwing batching at it won't
  help latency. The wins come from faster kernels (**FlashAttention**), better hardware, or doing
  *less* work (prefix caching to skip re-prefilling shared prompts).
- **Image/video is compute-bound** → like prefill: attention over a whole latent canvas every
  denoising step is heavy math. Optimize kernels, quantize, and cut the number of steps.

!!! key "The mental model to keep"
    **A workload is a point on the roofline. An optimization is a move — either up the diagonal
    (more intensity, beats a memory wall) or a higher roof (more/faster hardware, beats a compute
    wall).** When you read Chapter 5, ask of each technique: *which wall does this beat, and which
    way does it move the point?*

---

That completes the conceptual core. You can now trace a token, name every matmul, identify the
phase, and prove its bottleneck. The rest of the book is about acting on this knowledge —
hardware, software, and the techniques that move points around the roofline.

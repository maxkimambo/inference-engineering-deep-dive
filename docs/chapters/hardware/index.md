# Chapter 3 · Hardware

Chapter 2 ended with a mental model: **a workload is a point on the roofline, and an optimization
is a move** — up the diagonal (beat a memory wall) or to a higher roof (beat a compute wall). But
the roofline itself is drawn by the *hardware*. Its ceiling is the GPU's peak FLOPS; its slope is
the GPU's memory bandwidth; its ridge point is their ratio. Change the GPU and you redraw the whole
chart.

This chapter is about reading that machine. Not trivia — the specific, load-bearing facts that let
you answer the three questions an inference engineer is paid to answer:

1. **Will it fit?** (capacity — does the model plus its KV cache live in memory at all)
2. **Will it be fast enough?** (bandwidth and compute — predicted before you rent anything)
3. **What should I buy?** (the choice — and the cost of getting it wrong)

!!! key "A GPU is two machines bolted together: a calculator and a hose"
    Every performance question reduces to which one is the bottleneck. The **calculator** is the
    array of math units (peak FLOPS). The **hose** is the pipe to memory (peak bytes/s). LLM decode
    starves the calculator because the hose can't feed it fast enough; prefill and image generation
    saturate the calculator. You cannot reason about a GPU as one number ("how fast is it") — you
    reason about it as *two* numbers and the ratio between them. Everything in this chapter is a
    consequence of that split.

We build it from the bottom up — what's physically inside the chip, then how memory and compute and
interconnect each impose their own ceiling — and end where you started reading: standing in a cloud
console, choosing.

## The map

| § | Section | The question it answers |
|---|---------|-------------------------|
| [3.1](gpu-anatomy.md) | **Anatomy of a GPU** | Why a GPU and not a CPU? What is an SM, a CUDA core, a Tensor Core? |
| [3.2](memory-hierarchy.md) | **The memory hierarchy** | Where do weights and the KV cache live, and why does it dominate inference? |
| [3.3](compute-and-precision.md) | **Compute & precision** | What is a TFLOP really, why does lower precision buy more of them, and what's the "sparsity" asterisk on every datasheet? |
| [3.4](interconnect.md) | **Interconnect** | When one GPU isn't enough, what wires the next one in — and how much does crossing that wire cost? |
| [3.5](choosing-a-gpu.md) | **Choosing a GPU** | Given a model and a latency budget, which accelerator — and how to not overpay? |

## The one ordering that matters

Most spec sheets lead with the compute number (the big "PFLOPS" headline). For **inference
serving** that is usually the *least* important number. The order that predicts whether a deployment
works is almost the reverse:

!!! key "Read a spec sheet for inference in this order"
    1. **Memory capacity** — if the model + KV cache don't fit, nothing else matters. This is a
       hard yes/no gate.
    2. **Memory bandwidth** — decode is memory-bound, so this sets your tokens-per-second per stream.
    3. **Interconnect** — if you need multiple GPUs (you often do), the wire between them becomes
       the new bottleneck.
    4. **Compute (FLOPS)** — matters for prefill, large batches, and image/video. Important, but it's
       the *last* gate for a decode-heavy LLM server, not the first.

    Training flips 1 and 4. Inference engineers who carry over the training instinct ("buy the most
    FLOPS") consistently overpay and under-provision memory.

## Learning objectives

By the end of this chapter you can:

- [x] Explain why GPUs beat CPUs for inference in terms of throughput vs latency and SIMT
- [x] Name what lives in registers, SRAM, L2, and HBM — and the bandwidth/latency cliff between them
- [x] Compute whether a given model + KV cache fits on a given GPU, and predict its decode TPS
- [x] Read "1,979 TFLOPS" on a datasheet and say what it *actually* delivers for your workload
- [x] Rank NVLink, PCIe, and InfiniBand by bandwidth and know which parallelism each can afford
- [x] Pick a GPU (and a count) from a model size and latency budget, and defend the cost trade

---
hide:
  - navigation
  - toc
---

# Inference Engineering — Deep

> A from-the-ground-up guide to how generative-model inference *actually* works — and how to make it fast.

Most inference guides tell you **what** the knobs do. This one explains **why** they exist, so
that when you hit a latency wall at 2 a.m. you can reason about it from first principles instead
of pattern-matching on a blog post.

It assumes you can read code and know what an array, a number, and computer memory are. Every
ML-specific term is **defined the first time it appears**, with an analogy and usually a tiny
worked example. It is long on purpose.

## How to read this

<div class="grid cards" markdown>

-   :material-numeric-0-box: __Foundations first__

    Chapters 0–2 build the mental furniture: what inference is, the prerequisites, and the
    mechanics of a forward pass. Read these top-to-bottom.

-   :material-chip: __Then the machine__

    Chapters 3–4 cover the hardware (GPUs, memory hierarchy) and the software stack (CUDA,
    inference engines) that run the math.

-   :material-tune: __Then the craft__

    Chapters 5–6 are the techniques (quantization, speculative decoding, caching, parallelism)
    and the modalities (vision, audio, embeddings) where they're applied.

-   :material-rocket-launch: __Then production__

    Chapter 7 is autoscaling, cold starts, multi-cloud capacity, and observability.

</div>

## The map

| # | Chapter | What you'll be able to do |
|---|---------|---------------------------|
| 0 | [Inference](chapters/inference/index.md) | Frame the problem: training vs inference, latency vs throughput |
| 1 | [Prerequisites](chapters/prerequisites/index.md) | Pick a model, define your latency budget, measure it honestly |
| 2 | [Models](chapters/models/index.md) | Trace a token through a transformer; find the bottleneck |
| 3 | [Hardware](chapters/hardware/index.md) | Read a GPU spec sheet and predict performance |
| 4 | [Software](chapters/software/index.md) | Choose and reason about an inference engine |
| 5 | [Techniques](chapters/techniques/index.md) | Quantize, cache, speculate, and parallelize on purpose |
| 6 | [Modalities](chapters/modalities/index.md) | Apply the techniques to vision, audio, and embeddings |
| 7 | [Production](chapters/production/index.md) | Scale, deploy, and observe a real serving system |

!!! note "Status"
    Chapter 2 (Models) is written to full depth as the reference for tone and rigor. The
    remaining chapters are scaffolded and filled in iteratively.

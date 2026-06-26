---
hide:
  - navigation
  - toc
---

# Inference Engineering — Deep Dive

> A from-the-ground-up guide to how generative-model inference *actually* works — and how to make it fast.

Most inference guides tell you **what** the knobs do. This one explains **why** they exist, so
that when you hit a latency wall at 2 a.m. you can reason about it from first principles instead
of pattern-matching on a blog post.

It assumes you can read code and know what an array, a number, and computer memory are. Every
ML-specific term is **defined the first time it appears**, with an analogy and usually a tiny
worked example. It is long on purpose.

!!! quote "Why this site exists"
    I wanted to understand the *entire* LLM-inference black box — not just operate the knobs, but
    know **why** every piece works the way it does, end to end. This site is me working that out in
    the open, going deeper on the points other guides skip, and sharing the learnings so they're
    useful to anyone else chasing the same understanding. — Max

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
    This is still work in progress chapters are being added.

## Sources & acknowledgements

This site is **heavily based on two excellent books**, which provided the structure and much of the
source material. I reorganized, went deeper on the points I wanted to understand fully, and added my
own worked examples, diagrams, and hands-on guides — but the foundations are theirs, and I'm grateful
for both:

- ***Inference Engineering*** — Philip Kiely (Baseten Books, 2026). The eight-chapter arc and the
  framing of this site follow the book; its companion site is
  [inferenceengineering.tech](https://inferenceengineering.tech).
- ***Quantization and Fast Inference: A Practitioner's Guide to Efficient AI*** — Vivek
  Kalyanarangan (Manning, 2026). The basis for the deeper quantization material (number formats, the
  affine mapping, scale and zero-point).

This is an independent personal learning project — not affiliated with or endorsed by either author
or publisher. If you want the canonical, authoritative treatments, **read the books.**


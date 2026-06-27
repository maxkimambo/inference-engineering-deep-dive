---
hide:
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

## The map

Read in order — it's bottom-up, and each chapter builds on the last. Every ML-specific term is
defined where it first appears. Each row is what the chapter covers and the concrete skill you walk
away with.

| # | Chapter | What it covers — and what you'll be able to do |
|---|---------|------------------------------------------------|
| 0 | [Inference](chapters/inference/index.md) | What inference *is*, training vs inference, latency vs throughput. **Gain:** frame a workload and state its goal in the right terms. |
| 1 | [Prerequisites](chapters/prerequisites/index.md) | Choosing a model and defining a latency budget (TTFT, tokens/sec). **Gain:** pick a model and set + measure an honest budget. |
| 2 | [Models](chapters/models/index.md) | A token's journey through a transformer — attention, the KV cache, prefill vs decode, the roofline. **Gain:** trace a token end-to-end and *prove* where a workload is bottlenecked. |
| 3 | [Hardware](chapters/hardware/index.md) | GPU anatomy (SMs, Tensor Cores), the memory hierarchy, compute & number formats, interconnect. **Gain:** read a spec sheet and predict fit, decode speed, and cost — then choose a GPU. |
| 4 | [Software](chapters/software/index.md) | The CUDA stack and the inference engines (vLLM, TensorRT-LLM) that drive the hardware. **Gain:** choose and reason about an inference engine. |
| 5 | [Techniques](chapters/techniques/index.md) | Quantization, speculative decoding, caching, parallelism, disaggregation. **Gain:** move a workload along the roofline on purpose and pick TP/EP/PP with intent. |
| 6 | [Modalities](chapters/modalities/index.md) | Applying the inference toolkit to vision, audio, and embeddings. **Gain:** carry the reasoning across modalities. |
| 7 | [Production](chapters/production/index.md) | Containerization, autoscaling, cold starts, multi-cloud capacity, deployment, observability. **Gain:** scale, deploy, and observe a real serving system. |
| 8 | [Infrastructure](chapters/infrastructure/index.md) | Kubernetes for ML, GPU scheduling, infrastructure as code, orchestration, multi-cloud — hands-on. **Gain:** build the GPU platform — schedule GPUs, declare clusters as code, serve with scale-to-zero and failover. |
| 9 | [Google Cloud](chapters/google-cloud/index.md) | Deploying on the managed rungs — Vertex AI endpoints, Model Garden & MaaS, Cloud Run GPU, custom containers. **Gain:** pick the right serving surface (call it → run it) and deploy on it. |

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


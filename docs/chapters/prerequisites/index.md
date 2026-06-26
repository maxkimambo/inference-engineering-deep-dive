# Chapter 1 · Prerequisites

!!! note "Scaffolded — not yet written to depth"
    Outlined below.

Before optimizing inference you need to know **what you're serving, to whom, and what "good"
means**. This chapter is about framing the problem so the later chapters have a target.

## Planned sections

- **Scale and specialization** — when a small specialized model beats a large general one
- **About your app** — AI-native vs feature; online vs offline; consumer vs B2B, and how each
  shapes the latency budget
- **Model selection** — evaluation, fine-tuning for domain quality, distillation
- **Measuring latency and throughput** — percentiles (p50/p95/p99), TTFT, TPS, and end-to-end
  metrics that actually reflect user experience

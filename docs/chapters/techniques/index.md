# Chapter 5 · Techniques

!!! note "Scaffolded — not yet written to depth"
    Outlined below. The quantization section will draw on Kalyanarangan's *Quantization and Fast
    Inference* for number-format depth.

## Planned sections

- **Quantization** — number formats (FP32 → FP16/BF16 → FP8 → INT8/INT4), approaches
  (PTQ vs QAT, weight-only vs activation), measuring quality impact
- **Speculative decoding** — draft-target, Medusa, EAGLE, n-gram / lookahead
- **Caching** — prefix/KV-cache reuse, where to store the cache, cache-aware routing, long context
- **Parallelism** — tensor (latency), expert (throughput), multi-node
- **Disaggregation** — separating prefill and decode; dynamic disaggregation with Dynamo

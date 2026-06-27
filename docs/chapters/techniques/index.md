# Chapter 5 · Techniques

Chapters 2–4 built the picture: what a forward pass does, where it's bottlenecked, the hardware
that runs it, and the software that drives it. This chapter is the craft — the applied research you
actually deploy to move a workload's point on the roofline.

One organizing principle runs through everything here:

!!! key "The more constraints you can introduce, the more performance you can extract"
    A general system that handles every case optimally handles none. Every technique in this
    chapter *narrows* the problem — fix the precision, assume a draft model is usually right, assume
    prompts share a prefix, pin prefill and decode to separate hardware — and trades that lost
    generality for speed. Inference engineering is largely the art of choosing which constraints
    your traffic can afford.

A second principle stacks on top: **the more traffic you have, the more techniques pay off.** Higher
parallelism, KV-aware routing, and disaggregation only earn their complexity at volume — many GPUs,
often many nodes, serving one model. At low traffic they're overhead.

## The foundation, and the five categories

Everything below assumes one technique that isn't really optional — **batching** — so the chapter
starts there, then layers five optimizations on top:

| § | Technique | What it attacks | Lossy? |
|---|-----------|-----------------|--------|
| [5.0](batching.md) | **Batching** *(foundation)* | idle GPU on memory-bound decode → throughput | no |
| [5.1](quantization.md) | **Quantization** | bytes per weight/value → both phases faster | yes (managed) |
| [5.2](speculative-decoding.md) | **Speculative decoding** | decode's idle compute → higher TPS | no |
| [5.3](caching.md) | **Caching** | redundant prefill → lower TTFT | no |
| [5.4](parallelism.md) | **Parallelism** | model/KV too big for one GPU → fit + speed | no |
| [5.5](disaggregation.md) | **Disaggregation** | prefill/decode fighting for one GPU → specialize | no |

Quantization is the only lossy one — every other technique is exact. If you work in a quality-critical
domain and can't risk *any* output change, you still have four of five tools available.

!!! warning "Techniques interact — sometimes they fight"
    Optimizations are not independent. Some are **symbiotic**: quantizing the KV cache makes
    disaggregation cheaper (less to transfer) and caching denser (more fits in memory). Some are
    **antagonistic**: raising batch size to feed quantization's throughput *starves* speculative
    decoding of the spare compute it needs. The goal is a *balanced* set that delivers more than the
    sum of its parts — which is why these are knobs to tune, not boxes to check.

## Learning objectives

By the end of this chapter you can:

- [x] Explain how continuous batching works, why it never tangles requests' KV caches, and how to size it to an SLO
- [x] Read a number format (`E4M3`, `MXFP8`, `INT4`) and explain its dynamic-range/precision trade
- [x] Order model components by quantization sensitivity and justify the order
- [x] Explain why speculative decoding raises TPS but never TTFT, and what caps its benefit
- [x] Compute KV-cache reuse from a prompt's structure and lay out context to maximize cache hits
- [x] Size the minimum GPU count for a model and pick TP vs EP vs PP for the situation
- [x] Decide whether a workload justifies disaggregation, and read `xPyD` deployment notation

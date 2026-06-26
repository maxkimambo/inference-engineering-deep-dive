# Chapter 0 · Inference

!!! note "Scaffolded — not yet written to depth"
    This chapter is outlined below. Chapter 2 (Models) is the depth reference; we fill the rest
    in iteratively.

**Inference** is using a trained model to produce an output — as opposed to **training**, which
is the (vastly more expensive, one-time) process of *creating* the model's weights. You train
once; you run inference billions of times. That asymmetry is why inference engineering exists as
a discipline.

## Planned sections

- What inference is, and how it differs from training (compute profile, who pays the cost)
- The two metrics that govern everything: **latency** (time to a result) and **throughput**
  (results per second), and why they trade off
- Why a model that's cheap to *call* can be expensive to *serve*
- The shape of the rest of the book

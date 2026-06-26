# Chapter 7 · Production

Everything so far made a *single instance* fast. This chapter is about what happens when real
traffic arrives — and the uncomfortable truth that **no matter how fast one instance is, enough
traffic will overwhelm it.** That's not a PyTorch problem or a CUDA problem. It's an infrastructure
problem, and it needs a different mindset and different tools.

Three shifts define production inference:

- **From one box to a fleet.** You now run many replicas across many GPUs, often many clusters, and
  must decide how many, where, and when.
- **From per-token to per-GPU economics.** You stop paying a public API per million tokens and start
  paying directly for hardware. You gain control of your unit economics and lose the simplicity of a
  linear price.
- **From server time to end-to-end latency.** On-server prefill+decode is only *part* of what the
  user feels. The network, the queue, the client's session setup, and the protocol all add up — and
  you have to optimize the whole path.

!!! key "The production principle"
    Reliability, autoscaling, and global capacity only earn their complexity **at volume**. A
    low-traffic app is better off on a pay-per-token API. This chapter is what you reach for once
    your product's growth — and its viral spikes — make dedicated infrastructure cheaper *and* more
    reliable than renting tokens.

## The sections

<div class="grid cards" markdown>

-   :material-package-variant-closed: __[Containerization](containerization.md)__

    Packaging inference into a portable, reproducible image — layers, the fragile dependency chain,
    version pinning, and NVIDIA's pre-built NIMs.

-   :material-arrow-expand-all: __[Autoscaling](autoscaling.md)__

    Matching replicas to demand — Kubernetes, utilization vs traffic signals, batching and
    concurrency, cold starts, routing/queueing, scale-to-zero, and per-component scaling.

-   :material-earth: __[Multi-Cloud Capacity](multi-cloud-capacity.md)__

    Going global — a control plane across providers, GPU procurement, geo-aware balancing,
    reliability postures, and security/compliance.

-   :material-rocket-launch-outline: __[Testing & Deployment](testing-and-deployment.md)__

    Shipping safely — load and shadow testing, canary deploys, cost estimation, and observability.

-   :material-account-network: __[Client Code](client-code.md)__

    The half of latency you forget — session reuse, async inference, and streaming protocols
    (HTTP/WebSockets/gRPC).

</div>

## Worked example & hands-on

- [**Putting It Together**](worked-example.md) — a single B2B chat deployment (INT4 Qwen on L4)
  decided end to end: replica sizing, autoscaler knobs, cold-start budget, routing, dedicated-vs-API
  cost, reliability, and alerts — with consistent numbers. Read this once you've skimmed the
  sections; it ties them together.
- [**A Quantization Pipeline on GKE**](quantization-pipeline-gke.md) — productionize the Chapter 5
  quantization job as a scale-to-zero GPU batch job on Google Kubernetes Engine: containerize, run with
  Workload Identity, write to Cloud Storage, and roll it out to a vLLM serving Deployment. A concrete
  application of §7.1–§7.4.

## Learning objectives

By the end of this chapter you can:

- [x] Build a lean, version-pinned inference container and explain why pinning is non-negotiable
- [x] Configure a traffic-based autoscaler (the five factors) and pick a batching strategy
- [x] Break a cold start into its four stages and shorten each
- [x] Decide when scale-to-zero helps and when it signals you're not ready for dedicated infra
- [x] Estimate dedicated-deployment cost and compare it honestly to a per-token API
- [x] Name the metrics an inference service must emit, and why server time ≠ user-felt latency

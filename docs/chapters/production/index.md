# Chapter 7 · Production

!!! note "Scaffolded — not yet written to depth"
    Outlined below.

## Planned sections

- **Containerization** — dependency management, NIMs
- **Autoscaling** — concurrency and batch sizing, cold starts, routing/load balancing/queueing,
  scale-to-zero, independent component scaling
- **Multi-cloud capacity** — GPU procurement, geo-aware load balancing, reliability, security
- **Testing and deployment** — zero-downtime deploys, cost estimation, observability
- **Client code** — how the caller's choices affect serving cost

## Hands-on guides

- [**A Quantization Pipeline on GKE**](quantization-pipeline-gke.md) — productionize the Chapter 5
  quantization job as a scale-to-zero GPU batch job on Google Kubernetes Engine: containerize, run with
  Workload Identity, write to Cloud Storage, and roll it out to a vLLM serving Deployment.

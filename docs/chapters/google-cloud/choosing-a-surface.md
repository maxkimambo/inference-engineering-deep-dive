# Choosing a Surface

You've seen all four rungs. Now the decision the chapter opened with: given a real workload, which one?
This is a procedure, not a preference — run the gates in order and the surface falls out, the same way
the GPU choice fell out of capacity-then-bandwidth in Chapter 3.

## The decision procedure

!!! key "Climb to the highest rung that meets your requirements — stop at the first that does"
    1. **Can a managed model serve it?** If Gemini (frontier) or an open model via **MaaS** meets your
       quality, latency, and data-residency needs → **use the API**. No infrastructure. Stop here;
       most prototypes and many products never need to leave this rung.
    2. **Need open weights, but the model fits one GPU and traffic is bursty?** → **Cloud Run GPU**
       (serverless, scale-to-zero) — or a **Vertex Endpoint at `min_replica=0`**. You self-host without
       a cluster, paying near-zero at rest.
    3. **Steady traffic, and you want the ML platform** (Model Registry, traffic-split canaries,
       monitoring, batch)? → **Vertex AI Endpoint**, warm (`min_replica ≥ 1`), single- or modest
       multi-GPU machine.
    4. **Multi-GPU topology, custom scheduling/routing, or cost-critical scale?** → **GKE** (Chapter 8):
       tensor parallelism on NVLink, gang scheduling, KV-cache-aware routing, every knob — at the price
       of operating it.

The order matters: each gate you pass *up* hands Google more operations. Engineers who skip to gate 4
by reflex ("we'll run it on Kubernetes") often rebuild, badly, what gate 1–3 gives for free.

## The surfaces side by side

| | Gemini / MaaS | Cloud Run GPU | Vertex Endpoint | GKE (Ch. 8) |
|---|---|---|---|---|
| **You manage** | nothing | a container | a model + machine spec | the platform |
| **Billing** | per token | per second | per GPU-hour | per node-hour |
| **Scale-to-zero** | n/a (always on) | **yes**, native | yes (`min_replica=0`) | yes (Karpenter/CA) |
| **Cold start** | none | ~tens of s | ~tens of s–min | ~minutes |
| **Multi-GPU / TP** | n/a | **no** (1 GPU/instance) | yes (multi-GPU machine) | **yes**, full control |
| **Custom engine/topology** | no | yes (any container) | yes (custom container) | **yes**, anything |
| **ML lifecycle** (registry, canary, monitoring) | partial | minimal | **yes** | you build it |
| **Ops burden** | none | low | low–medium | **high** |

## Cost: the crossover that decides between rungs

The surfaces don't just differ in features — they bill on **different models**, and the cheapest one
flips with utilization:

- **Per token** (MaaS/Gemini) — zero idle cost, scales linearly with usage. Cheapest at **low or
  spiky** volume; gets expensive as steady volume climbs (you're paying a margin on every token).[^maas]
- **Per second** (Cloud Run) — pay only while an instance runs. Cheapest for **bursty** traffic that's
  idle much of the day; scale-to-zero is the whole win.[^crgpu]
- **Per GPU-hour / node-hour** (Vertex / GKE) — you rent the hardware whole. Cheapest **per token only
  at high utilization**, where continuous batching (Ch. 2) packs the GPU and divides its hourly cost
  across many concurrent requests.

!!! key "Match the billing model to your traffic shape, not your instinct"
    A request that runs 30 minutes a day belongs on **per-token or per-second** billing — renting a
    GPU-hour to sit idle 95% of the time is the classic overspend. A GPU pinned at 70% utilization
    around the clock belongs on **per-hour** self-hosting — paying a per-token margin on that volume is
    the opposite overspend. The crossover is utilization; compute `$/million tokens` for each (Ch. 3's
    method) at *your* traffic and let the number, not the architecture diagram, decide.

## Worked picks

| Workload | Surface | Why |
|----------|---------|-----|
| Prototype needing frontier quality | **Gemini API** | best managed model, zero infra, ship today |
| Internal tool, open 7B, idle most of the day | **Cloud Run GPU** (scale-to-zero) | pay only when used; one `gcloud run deploy` |
| Low-volume open-model feature | **MaaS** | per-token, no GPU quota, instant |
| Production chat, steady, needs canary + monitoring | **Vertex Endpoint** (warm) | ML platform: registry, traffic split, GPU autoscale |
| 70B+, high steady volume, cost-critical, KV-aware routing | **GKE** (Ch. 8) | multi-GPU TP on NVLink, custom routing, best $/token at scale |

Notice the same model (an open 7B) lands on *different* rungs purely by **traffic shape and
requirements** — bursty internal → Cloud Run; low-volume → MaaS; steady production → Vertex. The model
doesn't pick the surface; the workload does.

## How this chapter sits against the rest

This was the **managed** view of everything the book built bottom-up: the roofline and bottlenecks
(Ch. 2), the GPU you'd choose (Ch. 3), the engine like vLLM (Ch. 4), the techniques (Ch. 5), the
production concerns (Ch. 7), and the platform (Ch. 8) — all still true, now with Google operating the
machinery on the upper rungs. The fundamentals didn't change; your **job** did, from building the
system to choosing how much of it to build.

!!! key "The one sentence to carry out"
    Google Cloud gives you a ladder from *call it* to *run it*; the engineering is picking the highest
    rung that meets your control, latency, and cost requirements — because every rung you climb is
    operations you no longer own, and every rung you skip downward is operations you just took on.

---

That closes the Google Cloud deployment chapter, and with it the arc from a single token's arithmetic
to a model served, at the rung that fits, on the world's GPUs.

[^crgpu]: Google Cloud docs — *Configure GPU for Cloud Run services* (instance-based per-second billing, scale-to-zero, no per-request fee): <https://docs.cloud.google.com/run/docs/configuring/services/gpu>
[^maas]: Google Cloud docs — *Use open models with Model as a Service (MaaS)* (serverless, per-token pricing): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-maas>

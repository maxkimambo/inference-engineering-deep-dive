# Chapter 9 · Deploying on Google Cloud

Chapter 8 built a GPU platform from first principles on GKE — maximum control, maximum operational
surface. But "run it yourself on Kubernetes" is the *bottom* of a ladder, not the only rung. Google
Cloud offers a spectrum of serving surfaces, and most of the time the right engineering decision is to
climb *up* the ladder — let Google operate more of the stack — until you hit the first rung that gives
you the control you actually need. This chapter is that ladder: what each surface does, when it wins,
and how to deploy on it hands-on.

!!! key "Don't deploy what you can call; don't operate what you can rent"
    The cheapest inference system is the one you don't run. Before standing up a GPU, ask in order:
    *Can I just call a managed API (Gemini, or an open model via MaaS)? If not, can a serverless box
    (Cloud Run GPU) scale-to-zero do it? If not, can managed model serving (Vertex AI Endpoints) do
    it? Only if none fit do I run it myself (GKE, Chapter 8).* Every rung you climb hands Google more
    of the operational burden — patching, autoscaling, cold-start machinery — in exchange for less
    control. Engineering maturity is choosing the **highest** rung that meets your requirements, not
    the lowest.

## The ladder of serving surfaces

From most-managed (call it) to least-managed (run it), trading operational burden for control:

| Surface | You provide | Google operates | Control | § |
|---------|-------------|-----------------|---------|---|
| **Gemini / MaaS API** | a request | *everything* — model, hardware, scaling | lowest | [9.2](model-garden.md) |
| **Cloud Run GPU** | a container | the GPU host, scaling, scale-to-zero | low–medium | [9.3](cloud-run-and-custom.md) |
| **Vertex AI Endpoints** | a model + machine spec | the serving runtime, autoscaling, rollout | medium | [9.1](vertex-ai-endpoints.md) |
| **GKE** (Chapter 8) | the whole platform | the control plane only | highest | Ch. 8 |

Two surfaces deserve their framing up front:

- **Vertex AI** is Google Cloud's managed ML/GenAI platform. For inference it gives you **Model
  Registry → Endpoint**: you register a model, attach hardware, and Vertex runs the serving container,
  autoscaling, health-checking, and version rollout for you. It also fronts **Model Garden** (one-click
  open models) and **MaaS** (serverless per-token APIs).[^ladder]
- **Cloud Run GPU** is *serverless containers with an L4 (or Blackwell) attached*: you hand it a
  container, it scales from zero to many and back, billed by the second. It splits the difference
  between "call an API" and "run Kubernetes."

## How to read this chapter

It complements Chapter 8 rather than repeating it. Chapter 8 is the **self-managed** rung (you run the
GPUs); this chapter is **everything above it**. The same fundamentals you've built carry over
unchanged — capacity gates (Ch. 3), the bottleneck roofline (Ch. 2), cold starts and autoscaling
(Ch. 7) — only now Google operates the machinery, and your job shifts from *building* it to *choosing
and configuring* it.

| § | Section | What you'll do |
|---|---------|----------------|
| [9.1](vertex-ai-endpoints.md) | **Vertex AI Endpoints** | Deploy a model to a managed endpoint; autoscale it; canary with traffic split |
| [9.2](model-garden.md) | **Model Garden & MaaS** | Deploy an open model (Gemma) one-click; call a serverless per-token API |
| [9.3](cloud-run-and-custom.md) | **Cloud Run GPU & custom containers** | Serve vLLM serverless, scale-to-zero; meet the Vertex custom-container contract |
| [9.4](choosing-a-surface.md) | **Choosing a surface** | Pick Gemini vs Cloud Run vs Vertex vs GKE from requirements and cost |

## Learning objectives

By the end of this chapter you can:

- [x] Place a workload on the right Google Cloud serving rung from its control and ops requirements
- [x] Deploy a model to a Vertex AI Endpoint with GPU autoscaling, scale-to-zero, and a canary split
- [x] Deploy an open model from Model Garden, and decide self-deploy vs Model-as-a-Service
- [x] Serve vLLM on Cloud Run GPU with scale-to-zero, and reason about its cold-start cost
- [x] Meet Vertex's custom-container serving contract (`AIP_*` routes) with your own image
- [x] Defend a surface choice on cost model (per-token vs per-second vs per-node) and latency

[^ladder]: Google Cloud docs — *Choose an open model serving option* (the managed-to-self-managed spectrum: MaaS vs self-deploy, and the prebuilt serving engines): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/choose-serving-option>

# Autoscaling

**Autoscaling** dynamically adjusts how many replicas of a model run, with one goal: *always have
enough capacity to meet your latency SLAs, without paying for idle GPUs.* Get it wrong in one
direction and you miss SLAs during spikes; wrong in the other and you burn money on idle hardware
during lulls.

```
 WITHOUT autoscaling          WITH autoscaling
 traffic ╱╲    ← missed SLAs   traffic ╱╲
 ───────╱──╲──  (under-provisioned)   ╱──╲    GPU count tracks the curve
 GPUs ──────── flat            GPUs ──╱────╲── steps up and down with demand
   ↑ wasted spend in the lulls          ↑ matched: no waste, no misses
```

- **SLA** (Service-Level Agreement) — the latency promise you're holding to, e.g. "p95 end-to-end
  under 300 ms."

## Kubernetes, briefly

Autoscalers run on **Kubernetes** — an open-source container orchestrator that pools hardware into a
**cluster** and provisions/deprovisions compute. A cluster has two planes:

- **Control plane** — makes routing and scaling decisions.
- **Worker plane** — runs the actual containerized replicas, each on an instance with the GPUs and
  hardware the container needs.

A cluster runs many replicas of many models. The question autoscaling answers: *how many replicas of
each?*

## Two signals: utilization vs traffic

- **Utilization-based** — scale on GPU signals (memory, compute usage). A *lagging* indicator: by the
  time utilization spikes, you're already behind.
- **Traffic-based** — scale on request volume. Can be *proactive* — react to incoming load before it
  saturates the GPUs.

They don't always agree: in prefill, a few requests with hundreds of thousands of *uncached* input
tokens can drive far higher utilization than many small, high-cache-hit requests. **Use both** —
traffic to act early, utilization to confirm — to keep resources matched to demand.

### The five knobs of a traffic-based autoscaler

| Knob | Question it answers |
|------|---------------------|
| **Min replicas** | how many always stay running, regardless of traffic? |
| **Max replicas** | the ceiling when traffic is high? |
| **Autoscaling window** | over what sliding timeframe is traffic measured? |
| **Scale-down delay** | how long to wait after a scale-down is suggested, in case of another spike? |
| **Concurrency target** | how many requests can each replica handle at once? |

Tuning is a balance: a longer **scale-down delay** prevents premature scale-downs on spiky traffic,
but costs money if traffic has truly cooled. There's no universal setting — it follows your traffic
shape.

## 7.2.1 Concurrency and batch sizing

The **concurrency target** above is meaningless unless you know how many requests one replica can
actually handle — which comes down to **batching**.

- **Static batching** — wait until the batch is *full*, then run. Early requests wait a long time.
- **Dynamic batching** — run when the batch is full *or* a time cutoff passes, whichever comes first.
- **Continuous batching** — run continuously, swapping new requests into freed slots at the **token
  level**. This is what production engines (vLLM, SGLang, TensorRT-LLM — which calls it *in-flight
  batching*) implement, and it minimizes latency versus static.

```
 static:      [wait for full batch]──────► run all together   (early reqs idle)
 dynamic:     [wait: full OR timeout]─────► run                (bounded wait)
 continuous:  ─run─run─run─ swap each finished slot for a waiting request ─►
```

!!! key "Batch size is the latency↔throughput dial, and it must match the autoscaler"
    Bigger batches → more **throughput**, but worse **per-user latency**. Test across batch sizes for
    your model/instance/SLA/budget. Crucially, the replica's batch size and the autoscaler's
    **concurrency target must match**: when every active replica hits its max concurrency, the
    autoscaler spins up more; when replicas are firing *half-full* batches, it scales down. A
    mismatch makes the autoscaler blind to real load.

## 7.2.2 Cold starts

A **cold start** is the time to bring a *new* replica online. It governs how aggressively you can
scale down: if cold starts are slow, you can't confidently shed capacity, so you over-provision "just
in case" — paying the idle-GPU tax autoscaling was meant to avoid.

Four stages, each optimized separately:

```
 │ GPU procurement │ image loading │ model weight loading │ engine startup │
 └─ §7.3.1, mostly ─┘└── make smaller / more bandwidth ──┘└─ caching helps ─┘
    the provider's job
```

1. **GPU procurement** — how fast you can add GPUs to the cluster and assign them. Mostly a function
   of your cloud provider (and a *negotiable* term in a GPU contract — see [§7.3.1](multi-cloud-capacity.md#731-gpu-procurement)).
2. **Image loading** — pulling the (multi-GB) container onto the new instance. *Make images smaller.*
3. **Model weight loading** — writing the weights (often *hundreds* of GB) onto the instance. *Make
   them smaller or get more bandwidth.*
4. **Engine startup** — starting the inference engine, including any compilation.

What you control:

- **Smaller images and weights load faster.** This is a second payoff of
  [quantization](../techniques/quantization.md) — INT4 weights aren't just faster to *run*, they're
  ~¼ the bytes to *load* at cold start.
- **Load weights over fast, nearby storage.** Pulling from a third party (Hugging Face) is capped by
  their egress; an S3 bucket adds latency and transfer cost. For multi-hundred-GB models you need
  *gigabytes/second* — best achieved loading over the network from a source cached in the same
  datacenter as the GPU.
- **Don't bake big weights into the image.** For small models, baking weights in simplified caching.
  Now that models are tens-to-hundreds of billions of params, the weights dwarf the image and load
  better *separately*.
- **Cache compiled engines.** vLLM/SGLang start fast, but TensorRT-LLM and PyTorch-compiled models
  have a multi-minute compilation step. Both support **engine caching** — but a cached engine only
  loads into an instance with the *exact* GPU type, CUDA version, and dependencies it was built for.

## 7.2.3 Routing, load balancing, and queueing

Once multiple replicas are online, something must decide which request goes where. Two roles:

- **Router** — works at the *request* level: "where *should* this request go?"
- **Load balancer** — works at the *system* level: "where *could* this request go?" — evening load
  across options.

In real systems these aren't singular; routing happens throughout the stack with load balancers
injected at key points.

Naive even-splitting ("3 replicas, 12 requests → 4 each") fails because **requests aren't equal** —
one with 10,000 input tokens isn't one with 100. Smart routing uses inference-engine and orchestrator
(NVIDIA Dynamo) signals:

- **KV-cache-aware routing** — send a request to a replica that already holds a matching prefix in
  its KV cache (the [cache-aware routing](../techniques/caching.md#533-cache-aware-routing) from Ch 5,
  now a fleet-level concern).
- **LoRA-aware routing** — send it to a replica that already has the needed LoRA adapter in memory.

For the concrete tools that implement this (sticky sessions, GKE Inference Gateway's
prefix-cache-aware Endpoint Picker, engine routers) and how to wire them into a cluster, see the
[worked example, step 5](worked-example.md#5-route-and-queue-how-kv-cache-aware-routing-actually-works-723-ch-5).

### Queueing

Routing and balancing aren't enough: when traffic exceeds capacity, requests need somewhere to
**wait** while the system scales up. A **queue** is that primitive — FIFO by default, or priority
(e.g. paid users ahead of free) for richer policies.

!!! warning "Feed new replicas from the queue immediately"
    When a fresh replica comes online, the queue must *see it at once* and assign it up to its
    concurrency limit from the backlog. Otherwise new requests keep piling onto the old replicas
    while the new one sits idle — you scaled up but didn't relieve the pressure.

## 7.2.4 Scale to zero

Advanced autoscalers can **scale to zero** — drop to *no* active replicas when there's no traffic,
then spin up on the next request. It needs two things:

- **Fast cold starts** — a user is waiting live on that first request.
- **Robust queueing** — to hold the request until a replica is live.

It's a great fit for **bursty, latency-tolerant** workloads: development/testing, periodic agents
(business-hours-only in one region), offline daily batch jobs (exactly the [GKE quantization
job](quantization-pipeline-gke.md), which scales its GPU pool 0→1→0 per run).

!!! warning "Scale-to-zero as a crutch is a smell"
    If you're leaning on scale-to-zero to keep costs down for a **latency-sensitive app with light,
    unscheduled traffic**, that's usually a sign the app isn't ready for dedicated infrastructure
    yet. Use a pay-per-token API until you reach the scale where dedicated GPUs are genuinely cheaper.

## 7.2.5 Independent component scaling

Modern AI apps are **multi-model, multi-stage pipelines** — and the stages have *different* hardware
needs. A voice-activity-detector model that chunks audio needs a tiny slice of a GPU; the LLM
transcribing that audio might need a full multi-GPU node. Their scaling parameters differ too.

```
 dictation pipeline
   audio ─► VAD (fractional/MIG GPU) ─► Whisper (H100 MIG) ─► LLM (B200 node) ─► text
            └─ each stage scales independently, right-sized to its own load ─┘
```

So **decompose autoscaling per stage** — right-size and scale each step on its own load, avoiding
both bottlenecks (an under-scaled stage) and overprovisioning (an over-scaled one).

!!! key "But keep the whole pipeline in one cluster"
    Independent scaling, *co-located* hardware. If intra-cluster messaging is ~10 ms and cross-cluster
    is ~50 ms, that 40 ms gap across a 5-stage pipeline is **200 ms — two-thirds of a 300 ms SLA**,
    spent entirely on network hops. Scale the stages separately; keep them physically together.

**Next:** [Multi-Cloud Capacity →](multi-cloud-capacity.md)

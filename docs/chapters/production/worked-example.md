# Putting It Together: A Production Chat Deployment

The five sections each explained one piece. This page wires them into **one concrete deployment**,
deciding every production knob from first principles with real numbers. Nothing new is introduced —
it's the whole chapter (and a callback to Chapters 2–6) applied to a single scenario.

!!! abstract "The scenario"
    A **B2B SaaS** is adding an in-app **AI chat assistant**, serving **Qwen2.5-7B quantized to INT4**
    (the [model from Chapter 5](../techniques/quantization.md#worked-example-taking-qwen-from-bf16-to-int4)).

    - **Traffic:** business-hours-heavy, **~20 req/s peak**, **~1 req/s** overnight, clear daily/weekly
      cycle. Users in the **US and EU**.
    - **SLA:** p95 **TTFT < 500 ms**, stream at **≥ 30 TPS**.
    - **Request shape:** ~1,500 input tokens (system prompt + history + question), ~320 output tokens.

We'll make every decision the chapter set up, in order.

## 1 — Pick the hardware (Ch 3 + Ch 5)

INT4 Qwen2.5-7B is **~3.9 GB** of weights. That comfortably fits an **NVIDIA L4 (24 GB)** with ~20 GB
left for KV cache and activations — no need for an A100/H100 or any [parallelism](../techniques/parallelism.md).
One L4 per replica, on a `g2-standard-8` instance.

> This is the first payoff of quantization compounding: the [Chapter 5 INT4
> work](../techniques/quantization.md) didn't just speed up decode — it shrank the model onto a
> cheaper GPU *and* cut cold-start load time (step 4).

## 2 — Find one replica's concurrency, then size the fleet (§7.2.1)

Benchmark one L4 replica under **continuous batching**: it sustains the SLA up to **~32 concurrent
requests**. So set the **concurrency target = 32**, and make the engine's batch size match.

Now size the fleet with **Little's Law** — concurrent requests = arrival rate × time-in-system. A
streaming request holds its slot for its whole generation: 320 tokens ÷ 40 TPS ≈ **8 s**.

```
 peak:    20 req/s × 8 s = 160 concurrent  →  ⌈160 / 32⌉ = 5 replicas
 trough:   1 req/s × 8 s =   8 concurrent  →  ⌈  8 / 32⌉ = 1 replica
```

## 3 — Configure the autoscaler (§7.2)

Plug the sizing into the [five knobs](autoscaling.md#the-five-knobs-of-a-traffic-based-autoscaler):

| Knob | Value | Why |
|------|-------|-----|
| **Min replicas** | **1** | never scale to zero — live users wait on the first token (step 4) |
| **Max replicas** | **8** | peak is 5; headroom for spikes above forecast |
| **Autoscaling window** | **30 s** | react quickly to the steep morning ramp |
| **Scale-down delay** | **5 min** | traffic is spiky; don't shed capacity that you'll re-acquire in minutes |
| **Concurrency target** | **32** | matches the measured per-replica batch size |

Scale on **traffic** (proactive, catches the ramp) confirmed by **utilization** (catches the
occasional 200k-token paste that one request alone can saturate).

## 4 — Budget the cold start (§7.2.2)

Why min replicas is **1**, not 0: a new replica takes ~50 s to come online, far longer than a chat
user will wait.

```
 GPU procurement │ image load │ INT4 weight load │ engine start │
   ~10 s (warm     ~20 s (4 GB   ~6 s (3.9 GB from   ~14 s (vLLM)  = ~50 s
   pool)           lean image)   a regional cache)
```

Each stage was shortened deliberately: a **warm node pool** (negotiated GPU availability), a
**lean pinned image** ([§7.1](containerization.md)), **INT4 weights** loaded from a **same-region
cache** (not Hugging Face egress), and **vLLM** (fast start, no compile step). 50 s is survivable
behind the queue during a ramp — 0 active replicas would not be.

!!! key "Scale-to-zero would be a mistake here"
    This is the §7.2.4 "smell" made concrete: a *latency-sensitive* app with *steady-enough business
    traffic* should keep a warm floor. Scale-to-zero is for the [overnight batch
    job](quantization-pipeline-gke.md), not the live chat path.

## 5 — Route and queue: how KV-cache-aware routing actually works (§7.2.3 + Ch 5)

A multi-turn conversation re-sends its whole history every turn, so if turn 5 lands on the replica
that served turns 1–4, it skips prefill on those turns via the
[prefix cache](../techniques/caching.md#531-prefix-caching-and-kv-cache-reuse) → far lower TTFT. The
catch is the part the theory glossed: **a plain Kubernetes `Service` load-balances round-robin and has
no idea which replica holds which prefix.** Getting the routing right is what makes this real. You have
three tiers of options.

### First, the two mechanisms

There are two ways to land a conversation's turns on a warm cache:

- **Session affinity (stickiness)** — don't track caches at all; just guarantee every turn of a session
  hits the **same replica**, by consistent-hashing a session/user id. That replica's *automatic prefix
  caching* — **vLLM Automatic Prefix Caching** (`--enable-prefix-caching`) or **SGLang RadixAttention** —
  then reuses the KV for free. Simple and robust; no cache introspection. Downsides: a hot session can
  imbalance load, and a replica restart loses that session's cache.
- **True cache-aware routing** — a smart router *tracks* which prefixes live on which replica (KV-cache
  indexes, utilization, queue depth) and scores replicas per request to maximize prefix overlap *while*
  balancing load. Higher hit rate and better balance; needs a router that introspects the engines.

### The tools you actually have

| Option | What it is | Effort to hook up |
|--------|-----------|-------------------|
| **Sticky sessions (DIY)** | consistent-hash routing at the gateway you already run (Envoy/Istio/NGINX) on a session header | a load-balancer config — minutes |
| **GKE Inference Gateway** *(recommended here)* | managed, **prefix-cache-aware**; `InferencePool` + **Endpoint Picker**, powered by llm-d, built on the Gateway API Inference Extension[^gie] | install CRDs, wrap your vLLM Deployment in an `InferencePool` |
| **Engine routers (portable)** | vLLM **production-stack** / **llm-d**, **SGLang router**, or **NVIDIA Dynamo's** KV-aware router | run the router as a Deployment in front of the engine pods |

You almost never write a cache-tracking router from scratch anymore — the bar is "config, not code."

### Wiring it on GKE (this deployment)

Since we're already on GKE, use **GKE Inference Gateway**. The Endpoint Picker (EPP) tracks each
replica's prefix-cache indexes, KV utilization, and queue depth, and routes each request to the
best-scoring replica — prefix-aware routing with **no router code from you** (Google reports up to ~96%
TTFT improvement on prefix-heavy workloads):

1. Run the vLLM replicas as a normal Deployment, with prefix caching on:
   `args: ["--model", "/models/Qwen2.5-7B-Instruct-W4A16-G128", "--enable-prefix-caching"]`.
2. Group those pods in an **`InferencePool`** and attach the Endpoint Picker.
3. Bind your hostname to the pool with an **`HTTPRoute`**. Done — the EPP does the cache-aware scoring.

```yaml
# group the vLLM pods + attach the cache-aware Endpoint Picker
apiVersion: inference.networking.x-k8s.io/v1alpha2   # check the GA CRD version in the quickstart
kind: InferencePool
metadata: { name: qwen-pool }
spec:
  targetPortNumber: 8000
  selector: { app: qwen-vllm }      # ← the serving Deployment's pods
  extensionRef: { name: gke-epp }   # ← the Endpoint Picker that scores prefix-cache match
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata: { name: qwen-route }
spec:
  parentRefs: [{ name: inference-gateway }]
  rules:
    - backendRefs:
        - group: inference.networking.x-k8s.io
          kind: InferencePool
          name: qwen-pool
```

(Exact CRD `apiVersion`s move fast — follow the GKE Inference Gateway quickstart[^gw] for the current
manifests.)

!!! tip "When sticky sessions are enough — and when to build your own"
    If you're **not** on a gateway that supports this, the 80/20 is **DIY sticky sessions**: consistent-hash
    on a `session-id` header (Istio `DestinationRule` `consistentHash`, or an Envoy hash policy) so a
    conversation pins to one replica, and let vLLM's automatic prefix caching do the rest. It's minutes
    of config and captures most of the benefit. **Build your own cache-tracking router only** if you're
    off Kubernetes or have an exotic policy the EPP can't express — rarely worth it now that llm-d /
    GKE Inference Gateway exist.

### Queueing

Routing decides *where*; the **queue** decides what happens when every replica is full during the 50 s
scale-up. Two layers cooperate: each vLLM replica has its own request queue (the EPP reads its depth to
avoid piling onto a busy replica), and the gateway holds overflow. Configure a **priority queue** so
**paid tenants jump ahead** of free-trial traffic under load.

## 6 — Decide dedicated vs API on cost (§7.4.2)

The whole reason to run your own infra is unit economics. Estimate a month both ways (illustrative
rates):

```python
# Dedicated — autoscaled L4 GPU-hours over the month
gpu_hours = 22*(10*4 + 14*1) + 8*(24*1)   # weekdays (10h busy@~4 + 14h@1) + weekends@1
          = 1380
dedicated = 1380 * 0.85                    # ≈ $1,173 / month  (L4 instance-hour)

# Per-token API — same 10M requests
api = 15_000*0.18  +  3_200*0.18           # 15,000M input + 3,200M output tokens
    = $3,276 / month
```

**Dedicated is ~2.8× cheaper** at this volume — *and* gives you latency control and EU data residency
the API can't. Below ~3–4M requests/month the API would win; this product is past the crossover.

!!! key "Add engineering time — the real TCO"
    The GPU bill ($1,173) isn't the whole cost. The engineers building and operating this stack are a
    real line item. Count them: dedicated still wins here, but **total cost of ownership** — not the
    hardware invoice — is the honest comparison, and it's why low-volume apps should stay on the API.

## 7 — Make it reliable and global (§7.3)

- **Two regions** — `us-central1` and `europe-west4` — for **geo-proximity** (no ~80 ms transatlantic
  hop on every request) and **EU data residency**. A global load balancer sends each user to their
  region.
- **Reliability** — **active-passive** per region (a hot standby absorbs a zonal failure). At ~5–8 L4s
  running continuously you're well under the [one-failure-per-50k-GPU-hours](multi-cloud-capacity.md#733-building-for-reliability)
  rate, but cordon/cycle automation is still wired in for when a node does fail.

## 8 — Watch the right things (§7.4.3)

Alert on the metrics that map to the SLA and the failure modes above:

- **p95 TTFT** crossing 500 ms → the SLA breach itself.
- **Queue depth** rising while **replica count** is flat → autoscaler not keeping up (raise max or
  shorten the window).
- **5XX rate** → a sick replica; cordon the node.
- **TTFT up but input-size up too** → not a regression, just longer prompts (the §7.4.3 "metrics only
  make sense together" point).

All piped into the existing **Grafana/PagerDuty**, not a siloed dashboard.

## The result

```
        US users ──► global LB ──► us-central1  [vLLM × 1–8 L4, KV-aware routing]  ┐
        EU users ──► global LB ──► europe-west4 [vLLM × 1–8 L4, KV-aware routing]  │ active-passive
                                          ▲              ▲                         ┘
                                  autoscaler (1↔8)   priority queue
                                  warm floor=1, 50s cold start, $1.2k/mo/region
```

## What each chapter contributed

| Decision | Came from |
|----------|-----------|
| INT4 model on a cheap L4, fast to load | Ch 5 Quantization · Ch 3 Hardware |
| Concurrency target = batch size; Little's-Law fleet sizing | §7.2.1 |
| Five autoscaler knobs with real values | §7.2 |
| Warm floor (min=1) justified by the 50 s cold-start budget | §7.2.2 / §7.2.4 |
| KV-cache-aware routing for multi-turn chat | Ch 5 Caching · §7.2.3 |
| Dedicated beats API 2.8×, plus TCO caveat | §7.4.2 |
| Two regions for latency + EU residency, active-passive | §7.3 |
| SLA-mapped alerts in existing tooling | §7.4.3 |

That's inference engineering end to end: every fast-single-replica trick from Chapters 2–6, wrapped in
the production machinery of Chapter 7, tuned to one product's traffic and SLA.

To see the *automation* behind a piece of this — building and shipping the quantized model itself —
work through the [Quantization Pipeline on GKE](quantization-pipeline-gke.md).

[^gie]:
    GKE Inference Gateway tracks each model server's prefix-cache indexes, KV-cache utilization, and
    queue depth, scoring servers by longest prefix match while balancing load — built on the
    [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension)
    and llm-d's Endpoint Picker.
    [About GKE Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway).

[^gw]:
    [GKE Inference Gateway prefix caching](https://cloud.google.com/blog/products/containers-kubernetes/gke-inference-gateway-prefix-caching-accelerates-ai-inference)
    (Google Cloud blog) reports up to ~96% TTFT improvement on prefix-heavy workloads; see the product
    docs for the current `InferencePool` / `HTTPRoute` quickstart manifests.

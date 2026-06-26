# Multi-Cloud Capacity Management

Autoscaling within one cluster works up to a point. A high-volume product serving a *global* user
base needs **thousands of GPUs distributed around the world** — more than any single cluster, often
more than any single provider can give you in the regions you need.

The naive version — siloed pools of compute in different clouds — leaves you unable to move workloads
between them fluidly; shifting a workload across clouds becomes a tedious, error-prone manual job.

**True multi-cloud inference** treats distinct compute pools as *fungible*: a multi-region,
multi-provider "bin packing" layer that, like Kubernetes within one cluster, takes a **global view**
to enable self-healing and global scheduling.

```
   CLOUD A           CLOUD B            CLOUD C
  [workload]        [workload]         [workload]
  [workload] ──┐    [workload]    ┌── [workload]
               └──► GLOBAL CONTROL PLANE ◄──┘
                  (deploy + global scaling decisions, event streams)
```

- **Control plane** (global) — handles model deployment and global scaling decisions; consumes
  real-time event streams.
- **Workload planes** (per cluster/region) — serve traffic and make in-cluster scaling decisions;
  report utilization and demand.

Separation matters for blast radius: if the control plane or one workload plane fails, the *other*
workloads keep serving.

What it unlocks: **capacity** (pool across providers), **redundancy** (survive a provider outage),
**latency** (run near users), **compliance** (honor data-sovereignty rules).

## 7.3.1 GPU procurement

Three kinds of GPU supplier, in rough order of price and polish:

- **Hyperscalers** — AWS, GCP, Azure. Most reliable, most expensive.
- **Neoclouds** — GPU-focused clouds (CoreWeave, Nebius). Often better availability/price for GPUs.
- **Resellers** — secondary markets (SF Compute). Spot capacity, opportunistic pricing.

The first challenge is simply *getting* capacity — the latest hardware is scarce, big contiguous
clusters (hundreds of nodes) are scarcer, and providers reserve their best supply for their largest
customers on long-term commitments. You'll often span multiple providers to get the right GPUs in the
right regions.

Three procurement mechanisms, which you blend:

| Mechanism | What | Use for |
|-----------|------|---------|
| **Reserved** | hundreds/thousands of GPUs for months/years at a discount | a low-cost **baseline** of steady capacity |
| **On-demand** | individual instances up to a quota, high per-hour cost | handling **peaks** above the baseline |
| **Spot** | discounted, pre-emptible on short notice (minutes) | **interruptible** peak/batch work |

!!! key "The standard blend"
    Large-scale inference runs a **reserved baseline** for steady load, plus a **mix of on-demand and
    spot** for peaks — distributed across clusters worldwide for proximity to users. Reserved keeps
    the floor cheap; on-demand/spot absorb the spikes without paying peak prices year-round.

## 7.3.2 Geo-aware load balancing

Just as a cluster's load balancer keeps every GPU evenly fed, a multi-cluster system needs a
**global** load balancer — one that's *geography-aware*.

Rule of thumb: **~5 ms of latency per time zone crossed.** New York → San Francisco is ~15 ms *one
way*. Against tight latency budgets, that's enormous.

So: don't let a request sit queued in one region when capacity is free nearby, but also **don't
habitually send a Singapore user's request to San Francisco.** Run workloads as close to end users as
possible — the global balancer trades a little cross-region spillover for keeping the common case
local.

## 7.3.3 Building for reliability

**GPUs fail. Plan for it.** In the Llama 3 paper, Meta ran 16,000 GPUs for 54 days and hit **419
unexpected interruptions**, mostly hardware — roughly **one failure per 50,000 GPU-hours**.[^llama3]

That sounds rare until you do the arithmetic:

```
 one node = 8 GPUs, running inference for a year
   8 GPUs × 24 h × 365 d ≈ 70,000 GPU-hours  >  50,000
   → expect at least one hardware failure per node-year. Plan for it, don't be surprised by it.
```

GPU health is a **node-level** concern: when one GPU fails, the others on its node often fail next or
must be pulled for maintenance. Proactively noting failures, **cordoning** nodes, and **cycling**
pods keeps clusters healthy. And GPUs aren't the only risk — providers have scheduled maintenance and
their own outages, so *every* layer must be reinforced.

Multi-cloud enables two high-availability postures:

- **Active-active** — multiple regions/clusters serve live traffic *simultaneously*. If one plane
  fails, traffic continues on the others seamlessly. Higher cost, best resilience.
- **Active-passive** — a "hot standby" cluster sits ready but idle; on failure, traffic cuts over to
  it. Cheaper, with a brief failover.

## 7.3.4 Security and compliance

For AI to power *mission-critical* apps, inference must be secure and compliant. Conversations center
on three assets:

- **User data** — inputs and outputs must be protected.
- **Model weights** — for fine-tuned/proprietary models, the weights are an invaluable trade secret.
- **Infrastructure** — the GPUs and access to intelligence are themselves abuse targets.

!!! key "The cheapest security win: don't store what you don't need"
    Not retaining user inputs/outputs shrinks your attack surface for free. It isn't always possible
    (logging requirements, training-data agreements), but if you *don't* need to retain user data,
    don't — you can't leak what you never kept.

Otherwise, securing inference is like securing any containerized workload: **data encryption,
container security, network and access controls, workload isolation**, all validated by third-party
penetration testing.

Multi-cloud helps with **compliance** in two ways:

- **Certification inheritance** — to be SOC 2 Type II or HIPAA compliant, your *providers* generally
  must be too; being able to move workloads to compliant providers is valuable.
- **Data residency** — some industries/countries require user data to be processed *in-country*. One
  cluster near Toronto and another near New York lets you keep Canadian data in Canada and US data in
  the US, with minimal added latency for users across the region.

**Next:** [Testing & Deployment →](testing-and-deployment.md)

[^llama3]:
    Grattafiori et al., *The Llama 3 Herd of Models* (2024) — reports 419 unexpected interruptions over
    a 54-day run on 16,384 GPUs, the majority hardware-related.

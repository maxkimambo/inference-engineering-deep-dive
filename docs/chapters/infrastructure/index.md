# Chapter 8 · Infrastructure Deep Dive

Chapter 3 taught you to choose a GPU. Chapter 7 told you to containerize the server, autoscale the
replicas, and chase GPU capacity across clouds — and along the way it said *"Kubernetes, briefly."*
This chapter is the un-abbreviated version: the **platform layer** that turns one GPU's worth of
reasoning into a fleet you can declare, schedule, place, and move.

It matters because the economics are brutal and specific. A GPU node costs **10–40× a CPU node**, is
**chronically scarce**, and runs workloads that violate almost every assumption Kubernetes was born
with. Kubernetes grew up scheduling small, stateless, interchangeable web pods onto cheap, abundant
CPUs. A model server is the opposite: it wants a *whole* expensive accelerator (or a carefully shared
slice of one), pulls a 20 GB image, spends minutes loading weights before it can serve, and is
worthless if the scheduler strands it one GPU short of a multi-GPU group. Naively running it like a
web app wastes the most expensive hardware in your account.

!!! key "The platform's one job: put GPU work on the right silicon, reproducibly, and keep it there"
    Every topic in this chapter is a facet of that sentence. **Scheduling** decides *which* GPU a pod
    lands on (and whether it shares one). **Infrastructure as Code** makes the cluster and its GPU
    pools *reproducible* instead of hand-clicked. **Orchestration patterns** keep the server *healthy
    and rolling* despite slow starts and stateful loads. **Multi-cloud** lets you *move* the workload
    when one provider runs dry. Get the platform right and the per-GPU performance from Chapter 3
    actually reaches production; get it wrong and you pay for idle accelerators.

This is a *platform* chapter, not a serving chapter — it sits beneath everything in Chapter 7. Where
the two touch (autoscaling, capacity), Chapter 7 owns the *serving* view (scale replicas to meet
traffic) and this chapter owns the *platform* view (scale *nodes*, schedule *GPUs*, declare the
cluster).

## The map

| § | Section | The question it answers |
|---|---------|-------------------------|
| [8.1](kubernetes-for-ml.md) | **Kubernetes for ML** | What are the moving parts, and what's different about running models on them vs web apps? |
| [8.2](gpu-scheduling.md) | **GPU scheduling & resources** | How does a pod actually get a GPU — whole, shared, or a multi-GPU group — and who decides? |
| [8.3](infrastructure-as-code.md) | **Infrastructure as Code** | How do you make a GPU cluster reproducible — Terraform or Pulumi, and GitOps on top? |
| [8.4](orchestration-patterns.md) | **Orchestration patterns** | What pod/deployment shapes keep a slow-starting, weight-loading GPU server healthy? |
| [8.5](multi-cloud.md) | **Multi-cloud strategies** | How do you deploy the *same* workload across clouds without rewriting it per provider? |

## The mindset shift from web infra

If you've run web services on Kubernetes, the instincts that served you there will mislead you here.
Hold these reversals in mind as you read:

| Web-app instinct | GPU-infra reality |
|------------------|-------------------|
| Nodes are cheap and fungible | Nodes are scarce, costly, and a specific SKU you fought to get |
| Pods are small fractions of a node | A pod often wants the *whole* node (or a fenced GPU slice) |
| Start in milliseconds | Start in *minutes* — image pull + driver + weight load |
| Scale up instantly on demand | GPU capacity may not *exist* to scale into; you queue or fail over |
| Any node will do | Topology matters — NVLink neighbors, same zone, MIG profile (Ch. 3) |
| Lose a pod, reschedule, no harm | Losing a node can strand a multi-GPU group and kill a job |

!!! key "Why this is its own chapter, not a section"
    The cost asymmetry changes the goal of the whole platform. On CPU infra you optimize for
    *developer velocity and availability*; the hardware is an afterthought. On GPU infra you optimize
    for **utilization of a scarce, expensive resource** — every percent of idle GPU is money on fire.
    That single inversion is why GPU scheduling, gang scheduling, scale-to-zero, and bin-packing get
    the attention they do here. You are no longer managing servers; you are managing a GPU *budget*.

## Learning objectives

By the end of this chapter you can:

- [x] Name the Kubernetes objects a model server is built from and say why each is (or isn't) the right fit
- [x] Get a pod a whole GPU, a shared slice (time-slicing/MPS/MIG), or a gang of GPUs — and pick between them
- [x] Explain the device-plugin model and where Dynamic Resource Allocation (DRA, GA in 1.34) replaces it
- [x] Provision a GKE GPU node pool in Terraform, and say when Pulumi or GitOps fits better
- [x] Configure probes, init containers, and rollouts that tolerate minutes-long model loads
- [x] Design a multi-cloud deployment that survives one provider running out of GPUs

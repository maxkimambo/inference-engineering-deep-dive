# GPU Scheduling and Resource Management

This is the heart of GPU infrastructure: how a pod actually acquires an accelerator. Getting it right
is the difference between 80% fleet utilization and paying for half-idle GPUs. The section moves from
the simplest case (one pod, one whole GPU) outward to the hard cases (sharing one GPU safely, and
scheduling a *gang* of GPUs all-or-nothing), and ends at the node autoscaler that conjures GPU nodes
into existence.

## Step 1: GPUs are a different kind of resource

CPU and memory are **compressible** and divisible — Kubernetes hands out `250m` of a CPU, overcommits
nodes, and throttles when contended. **GPUs are neither.** By default a GPU is an indivisible,
non-overcommittable unit: a pod either holds a whole one or none. So a GPU request looks like this,
and `requests` must equal `limits`:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1        # whole GPUs only; you cannot ask for 0.5
```

`nvidia.com/gpu` is not a built-in resource like `cpu` — it's an **extended resource** advertised to
the scheduler by a **device plugin** running on each GPU node. That plugin is the bridge between "the
node has hardware" and "the scheduler can allocate it."

### The GPU Operator installs the plumbing

You don't wire that bridge by hand. The **NVIDIA GPU Operator** is a Kubernetes operator that, on
every GPU node, installs and manages the whole stack: the **driver**, the **container toolkit** (so
containers see the GPU), the **device plugin** (advertises `nvidia.com/gpu`), **Node Feature
Discovery** (labels nodes with GPU model, memory, MIG capability), and **DCGM** (the exporter that
feeds GPU metrics to your monitoring — Chapter 7's observability). On managed clusters (GKE/EKS/AKS)
the cloud often installs a managed equivalent. Either way: **something has to advertise the GPU, and
that something is operator-managed, not a manual `apt install`.**

## Step 2: placing the pod on the *right* GPU

Advertising GPUs isn't enough — you must steer pods to the correct node and keep the wrong pods off
expensive ones. Three mechanisms, each answering a different question:

- **`nodeSelector` / node affinity** — *"this pod needs an H100 in `us-central1`."* You match against
  the labels NFD and the cloud put on nodes (`nvidia.com/gpu.product=H100-SXM`, `topology.kubernetes.io/zone`).
- **Taints & tolerations** — *"keep everyone else off my GPU nodes."* You **taint** GPU nodes
  (`nvidia.com/gpu=present:NoSchedule`) so ordinary pods can't land there and waste them; only pods
  that explicitly **tolerate** the taint may. This one is non-optional in practice — without it, a
  logging sidecar can occupy a $30/hr node.
- **Topology / affinity for multi-GPU** — *"put these 8 pods on NVLink-connected GPUs in one node."*
  Pod affinity and the scheduler's topology awareness keep a tensor-parallel group on the fast wire
  (Chapter 3, § 3.4) instead of scattering it across nodes where collectives crawl.

```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
nodeSelector:
  nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
```

!!! key "Taint your GPU nodes on day one"
    The most common money leak in a young GPU cluster is non-GPU pods scheduled onto GPU nodes
    because nobody tainted them. Taint every GPU node and tolerate the taint only in GPU workloads.
    It costs one line and saves whole accelerators.

## Step 3: from count-based to attribute-based — DRA

The device-plugin model has a blind spot: it counts GPUs but can't *describe* them. You ask for "1
GPU," not "1 GPU with ≥40 GB free and a 3g.40gb MIG profile." For heterogeneous fleets that's a real
limitation. **Dynamic Resource Allocation (DRA)** fixes it, and as of **Kubernetes 1.34 (GA, 2026)**
it's the modern path — NVIDIA donated the DRA driver to the CNCF.

DRA replaces "give me N of this resource" with **structured requests against device attributes**. You
describe what you need; a DRA driver matches it to real hardware:

```yaml
# A ResourceClaim says WHAT you need, by attribute — not just a count
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata: { name: big-gpu }
spec:
  devices:
    requests:
      - name: gpu
        deviceClassName: gpu.nvidia.com
        selectors:
          - cel:
              expression: "device.attributes['memory'].quantity >= '40Gi'"
```

| | Device plugin (classic) | DRA (1.34 GA) |
|---|---|---|
| Request model | count: `nvidia.com/gpu: 1` | attributes: memory, MIG profile, compute capability |
| Sharing | clunky (global ConfigMap) | first-class, per-claim |
| Heterogeneous fleets | poor (all GPUs look alike) | strong (match by spec) |
| Maturity | ubiquitous, battle-tested | GA in 1.34; adopt as clusters upgrade |

!!! key "Which to use: device plugin today, DRA as you upgrade"
    If you're on a managed cluster today, the **device plugin** (`nvidia.com/gpu: N`) is what you'll
    use and it's completely fine for homogeneous pools. **DRA** is the direction of travel and the
    right choice when your fleet is mixed (different GPU models, MIG profiles) or you need fine-grained
    sharing — it lets the scheduler reason about *which* GPU, not just *how many*. Don't run both the
    device plugin and the DRA driver for the same GPUs; they conflict.

## Step 4: sharing one GPU when a whole one is overkill

A whole H100 for an 8B model is the "truck for a backpack" waste from Chapter 3. Three ways to put
multiple workloads on one GPU, trading isolation for simplicity:

| Mechanism | How it shares | Isolation | Works on | Use when |
|-----------|---------------|-----------|----------|----------|
| **Time-slicing** | round-robins GPU time between pods | **none** (shared memory, no limits) | any CUDA GPU | dev/test, bursty low-risk sharing; cheapest to enable (a ConfigMap) |
| **MPS** (Multi-Process Service) | runs kernels *concurrently*, spatially | partial (some memory/compute limits) | most GPUs | many small models with predictable footprints |
| **MIG** (Multi-Instance GPU) | **hardware** partition into ≤7 instances | **strong** (dedicated SMs, L2, memory, fault domain) | A100/H100/H200/B-class | multi-tenant, hard QoS, noisy-neighbor isolation |

Time-slicing is just oversubscription — two pods take turns, neither protected from the other running
it out of memory. MIG (introduced in Chapter 3, § 3.1) is the opposite: physically fenced slices that
can't see or starve each other, at the cost of only working on data-center GPUs and partitioning
ahead of time.

!!! key "Isolation is the axis you're trading on"
    Pick by *blast radius*. Internal dev sharing → **time-slicing** (free, no isolation needed).
    Many small production models on one card → **MPS** (concurrency without the MIG hardware
    requirement). Multiple tenants / strict QoS where one workload must never touch another's memory →
    **MIG**. The wrong default is treating time-slicing as a production multi-tenant solution — it has
    *no* memory isolation, so one OOM takes down every co-tenant.

## Step 5: scheduling a *gang* — all GPUs or none

Here's the problem the default scheduler can't solve. A distributed training job or a multi-GPU
serving group needs **all** its GPUs simultaneously. The default scheduler places pods *one at a
time* and greedily — so it may place 6 of your 8 pods, find no room for the last 2, and leave the 6
**holding expensive GPUs idle, waiting forever** for siblings that never come. Two such jobs can even
deadlock, each holding what the other needs.

**Gang scheduling** fixes this: schedule all *N* pods together or none of them. The ecosystem splits
the job in two:

- **Kueue** — *admission control and quota*. It holds a job **suspended** until the cluster (or the
  node autoscaler) can satisfy the *whole* request, then admits it. Prevents partial placement by
  never letting the job start early. The lightweight, Kubernetes-native default.
- **Volcano** — a *batch scheduler* replacing the default for these workloads, adding true gang
  scheduling, fair-share across teams, and priority queues. Heavier; reach for it for serious
  multi-tenant training/batch estates.
- **KAI Scheduler** (NVIDIA, CNCF-donated) — an AI-workload-aware scheduler that's become a common
  reference point for GPU-first clusters.

!!! key "Gang scheduling is non-negotiable for multi-GPU jobs"
    The instant a single logical workload spans more than one pod-with-a-GPU, the default scheduler's
    one-pod-at-a-time greed will eventually strand GPUs. Use **Kueue** for quota + all-or-nothing
    admission as the default; add **Volcano** when you need strict gang semantics and fair-share
    across competing teams. This is the platform-side complement to Chapter 3's "tensor parallelism
    must stay on NVLink": gang scheduling makes sure the group lands *together* in the first place.

## Step 6: where do the GPU nodes come from? Node autoscaling

Pods need nodes, and GPU nodes are too expensive to leave running idle. The **node autoscaler**
creates and destroys them to match pending pods:

- **Cluster Autoscaler** — scales predefined node *groups* up and down. Reliable, works everywhere
  (good for multi-cloud/hybrid), but node-group-bound and slower (~3–4 min to add a node).
- **Karpenter** — provisions nodes by calling cloud APIs directly from pending-pod requirements:
  faster (~45–60 s), bin-packs better, natively does **scale-to-zero** (no idle GPU floor) and
  **Spot** strategies. The cost-optimal choice for bursty/intermittent GPU work — you pay for the
  accelerator only while a job runs.

!!! key "Scale-to-zero is the GPU cost lever"
    A GPU node left running overnight at zero traffic is pure waste. **Karpenter** terminating idle
    GPU nodes (and **Kueue** queueing jobs while it spins capacity up) turns a fixed GPU bill into a
    usage-based one — frequently a 30–50% cut for non-24/7 workloads. The trade is **cold starts**
    (Chapter 7, § 7.2.2): scaling from zero means a node provision + image pull + weight load before
    the first token. Scale-to-zero and cold-start mitigation are two ends of the same decision.

## Try it: prove the taint, then tolerate it (on the lab cluster)

This is the experiment that makes taints/tolerations click. On the GKE lab cluster from § 8.6 (or any
cluster with a tainted GPU node), first *read* how the scheduler sees the hardware, then watch a taint
block a pod and a toleration unblock it:

```bash
# 1. What GPUs does the scheduler know about, and what's the taint?
kubectl get nodes -o custom-columns=\
NODE:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
kubectl describe node -l cloud.google.com/gke-accelerator | grep -i taint
#   → nvidia.com/gpu=present:NoSchedule

# 2. Request a GPU but DON'T tolerate the taint → it can never schedule
kubectl run blocked --image=nvidia/cuda:12.4.1-base-ubuntu22.04 \
  --overrides='{"spec":{"containers":[{"name":"c","image":"nvidia/cuda:12.4.1-base-ubuntu22.04",
  "command":["nvidia-smi"],"resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}'
kubectl get pod blocked          # stays Pending — the taint repels it
kubectl describe pod blocked | grep -i "untolerated\|taint"

# 3. Add the toleration (see gpu-smoke.yaml above) → it schedules and runs
kubectl delete pod blocked
kubectl apply -f gpu-smoke.yaml  # has the toleration; lands on the GPU node
```

The skill you just built: when a GPU pod is stuck `Pending`, your first move is `kubectl describe pod`
→ look for an untolerated taint or an unsatisfiable resource request. That diagnosis covers most
real-world "why won't my GPU pod schedule?" tickets.

---

You can now get a pod onto a GPU in every mode that matters: whole, placed, attribute-matched, shared,
ganged, and on a node that autoscaled into being. The next problem is making all of this
*reproducible* — declared in code, not clicked into a console.

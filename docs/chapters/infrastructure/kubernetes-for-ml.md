# Kubernetes for ML

You don't *need* Kubernetes to serve a model — a single container on a single GPU VM works fine, and
for one model on one GPU that's the right answer. Kubernetes earns its complexity at *fleet* scale:
many GPUs, many models, many requests, where you need something to pack scarce accelerators
efficiently, restart what crashes, roll out new versions without downtime, and do it reproducibly.
This section builds the platform from its one core idea, then shows precisely what changes when the
workload is a model instead of a web app.

## The one idea: a reconciliation loop

Strip away the hundreds of objects and Kubernetes is a single pattern repeated everywhere: **you
declare desired state; controllers continuously drive actual state toward it.** You don't tell
Kubernetes "start this container" (imperative); you tell it "I want 3 replicas of this server to
exist" (declarative), write that to the cluster's database, and a controller notices reality has 2,
not 3, and starts one. Crash a pod and the same loop replaces it. This **control loop** — observe,
diff, act, repeat — is the whole philosophy, and it's exactly why Kubernetes is good at keeping
expensive, failure-prone GPU workloads running without a human babysitting them.

!!! key "Declarative is the feature, not the syntax"
    The payoff of "declare desired state" is that your entire system is a *document* — versionable,
    reviewable, reproducible (§ 8.3's whole premise). The cost is a mindset shift: you never act on
    the cluster directly, you change the desired state and let it converge. Fighting this — `kubectl`
    hand-edits, manual pod restarts — is the single most common way teams turn Kubernetes from an
    asset into a liability.

## The moving parts

A cluster is a **control plane** (the brain) plus **worker nodes** (the muscle, where your GPUs live).

**Control plane:**

- **API server** — the front door. Everything (you, controllers, the scheduler) reads and writes
  desired state through it. The only component that talks to the database.
- **etcd** — the database. A consistent key-value store holding the entire desired + observed state
  of the cluster. Lose etcd, lose the cluster's memory.
- **Scheduler** — decides *which node* each new pod runs on, given its resource requests and
  constraints. For GPU work this component's decisions are the whole ballgame (§ 8.2).
- **Controller manager** — runs the reconciliation loops (the "make actual match desired" logic for
  Deployments, Jobs, and the rest).

**Each worker node:**

- **kubelet** — the node's agent; takes pod specs from the API server and makes the container runtime
  run them, then reports health back.
- **Container runtime** (containerd) — actually pulls images and runs containers.
- **kube-proxy** — wires up pod networking so Services resolve.

For GPU nodes, add two more (installed by the GPU Operator, § 8.2): the **driver** and the **device
plugin / DRA driver** that makes the GPU visible to the scheduler as an allocatable resource.

## The objects you'll actually use

Kubernetes has dozens of object types; a model platform leans on a handful. The skill is matching the
*workload's shape* to the right object.

| Object | What it is | Use it for |
|--------|-----------|------------|
| **Pod** | one or more co-located containers, the atomic unit | never directly — it's what the others manage |
| **Deployment** | a controller keeping *N* identical, stateless pods alive + rolling updates | **the model server** — replicas behind a Service |
| **ReplicaSet** | the thing a Deployment uses to hold *N* pods | you don't touch it; Deployment owns it |
| **Service** | a stable virtual IP/DNS load-balancing across a Deployment's pods | exposing the server to clients |
| **Job** / **CronJob** | run-to-completion (or scheduled) work | **batch**: quantization (Ch. 5), evals, offline inference |
| **StatefulSet** | pods with stable identities + per-pod storage | rarely — only for sharded/leader-based serving |
| **ConfigMap** / **Secret** | non-secret / secret config injected into pods | model names, endpoints, API keys, HF tokens |
| **PersistentVolumeClaim** | a request for durable storage | caching model weights across restarts (§ 8.4) |

!!! key "Serving is a Deployment; the pipeline jobs are Jobs"
    A model **server** is a stateless replica set that should always be running → **Deployment**
    behind a **Service**. The quantization run from Chapter 5, an eval sweep, or a batch-inference
    pass are **run-to-completion** → **Jobs** (often gang-scheduled, § 8.2). Reaching for a
    StatefulSet for ordinary serving is a common over-engineering smell — you want fungible replicas,
    not stable identities, unless you're sharding a single model's KV/state across pods.

## What's actually different about ML workloads

Everything above is generic Kubernetes. Here is where models break the web-app assumptions and force
the rest of this chapter:

**1. A pod wants a whole accelerator, not a CPU slice.** Web pods request fractional CPU
("`250m` = a quarter core"); the scheduler bin-packs dozens per node. A model pod requests
`nvidia.com/gpu: 1` — an indivisible, scarce unit — or a carefully configured *share* of one. This is
a different scheduling problem entirely (§ 8.2).

**2. Images are enormous.** A CUDA + PyTorch + engine image is **5–20 GB**. First pull onto a fresh
node can take minutes and dominates cold-start time. Mitigations — slim images (Ch. 7), pre-pulled
base layers, image streaming, secondary boot disks — are platform concerns, not afterthoughts.

**3. The container isn't ready when it starts.** A web server accepts traffic in milliseconds. A
model server must **load tens of GB of weights from storage into HBM** first — seconds to minutes
during which it's running but useless. Get the **readiness probe** wrong and Kubernetes routes
traffic to a pod still loading, and clients get errors (§ 8.4 fixes this precisely).

**4. Nodes are a specific SKU, and topology matters.** You don't want "a node," you want "an A100
node in `us-central1`, on the NVLink island next to its 7 siblings." That's expressed with **node
labels, taints/tolerations, and affinity** (§ 8.2), tying directly back to Chapter 3's interconnect
and capacity reasoning.

**5. Some jobs need *all their GPUs at once or none*.** A multi-GPU serving group or a distributed
job is worthless with 7 of 8 GPUs — the 8th pod waiting strands the other 7 (expensive idle silicon,
and possibly deadlock). This is the **gang scheduling** problem, unique to this world (§ 8.2).

!!! key "When NOT to use Kubernetes"
    Mirror of Chapter 5's "more traffic, more techniques pay off." One model on one or two GPUs with
    steady traffic → a plain container on a GPU VM (or a managed endpoint) is simpler and cheaper to
    operate; Kubernetes is pure overhead. Reach for it when you have **many GPUs to pack, many
    models/versions to roll, bursty or batch workloads to schedule, or a reproducibility/compliance
    mandate.** The platform is a fixed cost you amortize over fleet scale — below that scale it loses.

## Try it: feel the reconciliation loop (free, no GPU)

You can experience the one core idea — declare desired state, controllers converge to it — on a free
local cluster in two minutes. Needs Docker + [`kind`](https://kind.sigs.k8s.io/):

```bash
kind create cluster                                   # a throwaway cluster on your laptop
kubectl create deployment web --image=nginx --replicas=3
kubectl get pods -w &                                 # watch in the background

# Now fight the controller: delete a pod and watch desired(3) beat actual(2)
kubectl delete pod "$(kubectl get pod -l app=web -o name | head -1 | cut -d/ -f2)"
#   → the Deployment immediately recreates it. You declared 3; it keeps 3.

kubectl scale deployment web --replicas=5             # change desired state → it converges
kubectl create job hello --image=busybox -- echo "batch done"   # run-to-completion = Job
kubectl get job hello                                 # COMPLETIONS 1/1

kind delete cluster                                   # clean up
```

You just watched the loop that keeps GPU servers alive without a babysitter — and saw *why* a server
is a **Deployment** (kept running) while a quant pass is a **Job** (run to completion). The skill:
never `kubectl run` a pod directly again — change desired state and let it converge.

---

You now know the platform's structure and the five ways models stress it. The first and sharpest of
those stresses — getting a pod onto a GPU at all, whole or shared or in a gang — is the next section.

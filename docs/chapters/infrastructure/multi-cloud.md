# Multi-Cloud Deployment Strategies

Chapter 7 (§ 7.3) made the *case* for spanning clouds: GPUs are scarce, and no single provider
reliably has the capacity, the price, or the regions you need. This section is the **how** — deploying
the *same* workload across providers without rewriting it per cloud, and the patterns for routing and
failover between them. The unifying tool is the one this whole chapter is built on: Kubernetes as a
portability layer.

!!! key "Multi-cloud is a tax you pay for a reason, not a default"
    Spanning clouds adds real cost: per-provider substrate, a lowest-common-denominator constraint,
    cross-cloud networking and data-egress bills, and doubled operational surface. Don't do it for
    fashion. Do it when one of three forces requires it: **GPU capacity resilience** (one cloud runs
    dry, you fail over), **cost arbitrage** (GPUs are materially cheaper elsewhere), or **compliance /
    data residency**. Start single-cloud, *design for portability*, and add the second cloud when a
    concrete force demands it.

## What's portable, and what isn't

Kubernetes is the reason this is feasible at all: a `Deployment`, `Service`, and probe config (§ 8.4)
run essentially unchanged on GKE, EKS, or AKS. Your *workload* is portable. The **substrate underneath
it is not** — and that boundary is the whole game:

| Layer | Portable across clouds? |
|-------|-------------------------|
| Model server manifests (Deployment, Service, probes) | **Yes** — same YAML |
| Scheduling config (Kueue, taints, GPU requests) | **Mostly** — minor per-cloud label differences |
| Cluster + GPU node pools (machine types, drivers) | **No** — per-cloud Terraform (§ 8.3) |
| Networking, IAM, storage classes, load balancers | **No** — provider-specific |
| GPU SKUs and their names | **No** — `a3-highgpu-8g` vs `p5.48xlarge` |

So the realistic architecture is **portable workload, per-cloud substrate**: one set of Kubernetes
manifests, deployed by one CD plane, onto clusters that were each stood up by cloud-specific IaC.

## Three levels of multi-cloud machinery

You adopt only as much as the need justifies:

1. **Portable workload, per-cloud IaC (the default).** Terraform/OpenTofu modules per provider build
   each cluster; one **GitOps** controller (ArgoCD/Flux, § 8.3) syncs the *same* app manifests to all
   of them. Simple, explicit, and enough for most.
2. **Fleet / unified control plane.** Tools that manage many clusters as one (GKE fleets/Anthos,
   Cluster API, Rancher) — central policy, config, and rollout across clouds. Worth it when you run
   *many* clusters and need consistent governance.
3. **Capacity-arbitrage abstraction.** Tools like **SkyPilot** treat multiple clouds as one GPU pool
   and place a job wherever capacity is cheapest/available right now. Powerful for *batch* and
   training (Chapter 5 quantization runs, evals); less so for always-on serving with sticky state.

## The hard parts that don't abstract away

Portability stops at the workload. These are the realities that bite, and they're all consequences of
physics and economics, not tooling gaps:

- **Data gravity.** Weights and datasets are large and *heavy to move*. Cross-cloud egress is slow and
  expensive, so each cloud needs its *own* copy of the weights near its GPUs — the mounted-cache
  decision from § 8.4, replicated per provider. You move *traffic* between clouds cheaply; you move
  *data* between clouds reluctantly.
- **Cross-cloud networking is the slow wire.** Chapter 3's interconnect hierarchy has a final, slowest
  rung: between clouds. Never split a latency-sensitive, tightly-coupled workload (tensor-parallel
  serving) *across* clouds — keep each model instance whole within one cloud, and load-balance whole
  requests between clouds.
- **Identity, secrets, observability** each fork per provider and must be unified deliberately
  (federated identity, a secrets layer, one metrics backend) — or you get N disconnected silos.

## Routing and failover patterns

How requests actually find a healthy cluster with capacity. Pick by how much you'll pay for standby:

| Pattern | How it works | Cost | Use when |
|---------|--------------|------|----------|
| **Active–active** | a global LB / DNS / multi-cluster Gateway splits live traffic across clouds, geo-routed | both clouds always paid | steady high traffic; latency + resilience both matter |
| **Active–passive (failover)** | primary serves; secondary stands by, takes over on outage/capacity-out | standby is cheap (scaled low/zero) | resilience without doubling spend |
| **Burst / overflow** | primary cloud serves baseline; spill to a second cloud only when capacity runs out | pay overflow only when used | a primary that's usually-but-not-always enough |

The mechanism is a **capacity- and health-aware global front door** — a multi-cluster Gateway or a
global load balancer with health checks — that routes each request to a cluster that is up *and* has
GPU headroom. Combined with § 8.2's scale-to-zero on the passive side, active–passive failover gives
you GPU-outage insurance for close to the cost of the primary alone.

!!! key "Keep instances whole within a cloud; balance requests between clouds"
    The one rule that prevents the worst multi-cloud mistakes: a single model instance and its
    tightly-coupled parallelism live *entirely inside one cloud* (on one NVLink island, Chapter 3);
    the multi-cloud layer only ever routes *whole requests* between independent, self-contained
    instances. Split a model across clouds and the cross-cloud wire — orders of magnitude slower than
    NVLink — will dominate. Multi-cloud is a *replication and routing* strategy, never a *parallelism*
    strategy.

## Try it: one manifest, two "clouds" (free, no cloud)

The claim that "your workload is portable, the substrate isn't" is something you can *prove* locally.
Stand up two `kind` clusters as stand-ins for two clouds and deploy the **identical** manifest to
both — then switch context the way a global load balancer would shift traffic:

```bash
kind create cluster --name cloud-a
kind create cluster --name cloud-b

cat > app.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata: { name: model }
spec:
  replicas: 2
  selector: { matchLabels: { app: model } }
  template:
    metadata: { labels: { app: model } }
    spec: { containers: [{ name: c, image: nginx }] }   # stands in for the model server
EOF

# SAME yaml, both 'clouds' — this is the portability multi-cloud relies on
for ctx in kind-cloud-a kind-cloud-b; do
  kubectl --context "$ctx" apply -f app.yaml
done
kubectl --context kind-cloud-a get deploy model   # serving here…
kubectl --context kind-cloud-b get deploy model   # …and here, unchanged

# 'Fail over': drain cloud-a, traffic would shift to cloud-b
kubectl --context kind-cloud-a scale deploy model --replicas=0

kind delete cluster --name cloud-a; kind delete cluster --name cloud-b
```

The same `model` Deployment ran on two independent clusters with zero per-"cloud" changes — and a
front door would simply route to whichever is healthy. That's the whole portability thesis of § 8.5,
made concrete: **portable workload, separate substrate, capacity-aware routing.**

---

That completes the platform's concepts: schedule GPUs, declare the cluster, orchestrate the server,
span clouds. Concepts stick when you run them — so the final section is a single continuous lab that
builds this whole platform on Google Kubernetes Engine, from `terraform apply` to a model serving,
gang-scheduled, scaling to zero, with a failover path.

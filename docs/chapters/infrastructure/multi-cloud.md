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

## Try it: split real traffic across GCP and Scaleway

Time to make this concrete on two *actual* clouds — GKE on Google Cloud and **Kapsule** on
**Scaleway** (a European provider, handy for EU data residency) — with **Cloudflare Load Balancing**
as the cloud-neutral front door splitting live traffic between them. This is the real architecture,
and it runs on the stack you already have (`kimambo.de` is on Cloudflare). It builds on the GKE lab
cluster from § 8.6.

!!! warning "Real clouds, real money"
    This spins up a GPU node on Scaleway too, and Cloudflare Load Balancing is a paid add-on
    (~\$5/mo + usage). Keep both GPU pools at `min=0`, and run the teardown at the end. A full run is a
    few euros.

**1. Stand up the second cloud — same workload, different substrate.** Kapsule auto-installs the
NVIDIA GPU Operator on any GPU pool (the same device-plugin + driver stack GKE gave you in § 8.2), so
the *workload* is unchanged — only the Terraform/CLI that builds the cluster differs:

```bash
# Scaleway: cluster + an L4 GPU pool that scales to zero (mirrors the GKE pool)
scw k8s cluster create name=infra-lab-scw version=$(scw k8s version list -o json | jq -r '.[0].name') \
  cni=cilium region=fr-par
CID=$(scw k8s cluster list name=infra-lab-scw -o json | jq -r '.[0].id')
scw k8s pool create cluster-id=$CID name=l4 node-type=L4-1-24G \
  size=1 min-size=0 max-size=2 autoscaling=true   # L4-1-24G = 1× NVIDIA L4
scw k8s kubeconfig install $CID                    # adds the Scaleway context

# Deploy the IDENTICAL vllm.yaml from § 8.6 Step 3 — not one line changes
kubectl --context infra-lab-scw apply -f vllm.yaml
# Expose it: a LoadBalancer Service → Scaleway CCM provisions an external IP
kubectl --context infra-lab-scw expose deployment qwen --type=LoadBalancer \
  --port 80 --target-port 8000 --name qwen-lb
SCW_IP=$(kubectl --context infra-lab-scw get svc qwen-lb \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
```

Each cloud pulls its *own* copy of the weights from Hugging Face — the **data-gravity** rule in
action: you replicate weights per cloud, you don't stream them across the gap.

**2. Expose the GKE side the same way** and grab its IP:

```bash
kubectl --context gke_${PROJECT_ID}_${REGION}_infra-lab expose deployment qwen \
  --type=LoadBalancer --port 80 --target-port 8000 --name qwen-lb
GCP_IP=$(kubectl --context gke_${PROJECT_ID}_${REGION}_infra-lab get svc qwen-lb \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
```

**3. Put Cloudflare in front and split the traffic.** Create one **health monitor**, two **origin
pools** (one per cloud), and a **load balancer** on a hostname, with weighted steering. In Terraform
(on-theme with § 8.3 — or click the same shapes in *Cloudflare → Traffic → Load Balancing*):

```hcl
resource "cloudflare_load_balancer_monitor" "health" {
  account_id = var.cf_account_id
  type = "http"; method = "GET"; path = "/health"; expected_codes = "200"
  interval = 15; retries = 2; timeout = 5      # marks a cloud down fast
}
resource "cloudflare_load_balancer_pool" "gcp" {
  account_id = var.cf_account_id; name = "gcp-gke"
  monitor = cloudflare_load_balancer_monitor.health.id
  origins { name = "gke"; address = var.gcp_ip; enabled = true }
}
resource "cloudflare_load_balancer_pool" "scaleway" {
  account_id = var.cf_account_id; name = "scaleway-kapsule"
  monitor = cloudflare_load_balancer_monitor.health.id
  origins { name = "kapsule"; address = var.scw_ip; enabled = true }
}
resource "cloudflare_load_balancer" "infer" {
  zone_id = var.cf_zone_id; name = "infer.kimambo.de"
  default_pool_ids = [cloudflare_load_balancer_pool.gcp.id,
                      cloudflare_load_balancer_pool.scaleway.id]
  fallback_pool_id = cloudflare_load_balancer_pool.gcp.id
  steering_policy  = "random"                   # weighted split across pools
  random_steering { pool_weights = {
    (cloudflare_load_balancer_pool.gcp.id)      = 0.7   # 70% GCP
    (cloudflare_load_balancer_pool.scaleway.id) = 0.3   # 30% Scaleway
  } }
}
```

**4. Watch the split — and the failover.** Hit the hostname in a loop; responses now come from
*both* clouds in roughly 70/30 proportion. Then break one cloud and watch Cloudflare's health monitor
drain it within ~30 s, shifting **all** traffic to the survivor — active-active turning into automatic
failover, no human in the loop:

```bash
for i in $(seq 20); do curl -s https://infer.kimambo.de/v1/models | jq -r '.data[0].id'; done
# now kill GCP's serving side; health checks fail; traffic moves 100% to Scaleway
kubectl --context gke_${PROJECT_ID}_${REGION}_infra-lab scale deployment qwen --replicas=0
for i in $(seq 20); do curl -s -o /dev/null -w '%{http_code}\n' https://infer.kimambo.de/v1/chat/completions \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"hi"}]}'; done
# still 200s — Scaleway absorbed it
```

You just ran one model across two clouds behind a single hostname, with weighted distribution and
health-based failover — the § 8.5 thesis end to end: **portable workload, per-cloud substrate,
capacity-aware global routing.** Swap the weights to do cost arbitrage (more traffic to whichever
cloud is cheaper that month), or set Scaleway's weight to 0 for cheap active-passive insurance.

### Alternative front door: Google Cloud global external ALB

You don't have to use Cloudflare. Google's **global external Application Load Balancer** can front
Scaleway too — you reach the external cloud with an **internet NEG** (a backend that points at a
public IP/FQDN outside GCP), then weight-split across the two backend services in the URL map:

```bash
# Scaleway as an external backend: an internet NEG pointing at its public LB IP
gcloud compute network-endpoint-groups create scw-neg \
  --global --network-endpoint-type=INTERNET_IP_PORT
gcloud compute network-endpoint-groups update scw-neg --global \
  --add-endpoint="ip=$SCW_IP,port=80"

# One backend service per cloud (GKE via its own standalone NEG; Scaleway via scw-neg)
gcloud compute backend-services create be-scw --global \
  --load-balancing-scheme=EXTERNAL_MANAGED --protocol=HTTP
gcloud compute backend-services add-backend be-scw --global \
  --network-endpoint-group=scw-neg --global-network-endpoint-group
# ...be-gke wired to the GKE Service's NEG the same way...

# Weighted split lives in the URL map's routeAction (weights 0–1000 → 70/30):
#   defaultRouteAction.weightedBackendServices: [{be-gke, weight:700},{be-scw, weight:300}]
gcloud compute url-maps import infer-map --global --source=urlmap.yaml
```

!!! key "Cloudflare vs Google global ALB: who's in the critical path?"
    Both split weighted traffic with health-based failover. The architectural difference is **where
    the front door lives**:

    - **Cloudflare LB** — provider-neutral anycast/DNS at the edge. Reaches any public IP, no GCP
      dependency, and it survives *either* cloud (including GCP) having a bad day. Best when the front
      door is your **insurance against a whole-provider outage**. (And you already run it.)
    - **Google global ALB** — GCP-native, composes with **Cloud Armor** (WAF/DDoS) and **Cloud CDN**,
      single pane if you're GCP-centric. But the front door *is in GCP*, so GCP sits in the path even
      for Scaleway-bound requests — great for **cost/capacity splitting**, weaker as insurance against
      GCP itself.

    Rule of thumb: optimizing **cost/capacity** across clouds → either works, pick your ecosystem.
    Optimizing **resilience against a provider outage** → put the front door at a *neutral* third
    party (Cloudflare/DNS), so no single cloud you're hedging against is also the thing routing around
    it.

**Tear down:**

```bash
kubectl --context infra-lab-scw delete svc qwen-lb
kubectl --context gke_${PROJECT_ID}_${REGION}_infra-lab delete svc qwen-lb
scw k8s cluster delete cluster-id=$CID with-additional-resources=true
terraform destroy   # removes the Cloudflare LB, pools, and monitor
# if you built the Google ALB path instead, delete its url-map, backend-services, and scw-neg too
```

## Keeping a conversation on its cluster (cache affinity)

The weighted split has a hidden cost the moment you care about **prefix caching** (Chapter 5, § 5.3):
a stateless 70/30 split routes *each request independently*, so turn 2 of a conversation can land on
the *other* cloud — where none of turn 1's KV cache exists. The new cluster must **cold re-prefill the
entire history**, the exact work prefix caching exists to skip. Load-spreading and cache-locality pull
against each other, and a naive front door silently picks spreading.

The fix is **two-tier routing**, the same shape whether your clusters sit in two regions or two clouds:

!!! key "Pin the conversation to a cluster, then route to the replica inside it"
    - **Tier 1 — global front door: session affinity.** Pin each *conversation* (not each request) to
      one cloud, keyed on a **session/conversation id**, not raw client IP (coarse, and broken behind
      mobile NAT). New conversations still obey the weighted split; existing ones stick.
    - **Tier 2 — in-cluster: KV-cache-aware routing.** Affinity gets the request to the right
      *cluster*; a cache-aware router inside it (GKE Inference Gateway's InferencePool + Endpoint
      Picker — Chapter 7's worked example) gets it to the right *replica* that holds the prefix.

    Coarse, cheap stickiness globally; fine, prefix-level routing locally, where the KV cache lives.

Pin by a session header you control. With **Cloudflare**, affinity spans the LB's pools (your two
clouds) directly:

```hcl
resource "cloudflare_load_balancer" "infer" {
  # ...pools + weighted steering as before...
  session_affinity     = "header"
  session_affinity_ttl = 1800                                  # 30 min
  session_affinity_attributes { headers = ["X-Session-Id"] }   # by conversation, not IP
}
```

With the **Google ALB**, make the same-session→same-cloud mapping deterministic with **consistent
hashing** on the header (`--session-affinity=HEADER_FIELD`, locality policy `RING_HASH`/`MAGLEV`,
`consistentHash.httpHeaderName: X-Session-Id`) — so a session always hashes to the same backend, and
rebalancing disturbs only a *fraction* of sessions instead of all of them.

!!! key "KV cache is a latency optimization, not correctness state"
    If affinity ever breaks — a cloud fails over, the cookie expires, you drain a hot cluster — the
    worst case is **one slow turn**: the new cluster cold-prefills the history, then runs warm again.
    You never get a *wrong* answer, only a briefly slower one. So you don't need perfect stickiness;
    you tune affinity *strength* (TTL, hash) against load-balancing *freedom*, knowing a miss is
    bounded and self-healing. This is what makes the whole split safe to run.

It's also why you **don't replicate the KV cache across the cross-cloud wire** (§ 8.5 data gravity;
Chapter 3, § 3.4): shipping gigabytes of KV between clouds to save a cache hit costs far more than the
re-prefill it avoids. Affinity buys reuse *within* a cloud; a failover pays one cold prefill and
re-warms locally. In multi-cloud, **sticky-by-session globally + cache-aware-within-cluster** is the
whole game.

!!! warning "The honest tension: weighted splitting vs session affinity"
    An astute reader will catch the contradiction. Earlier this section sold weighted splitting as
    *precise* traffic distribution (70/30); then the affinity rule pins every conversation to one
    cluster. Both can't be fully true at once. The reconciliation: **the weight governs session
    *admission*, not live requests.** Once affinity holds, realized GPU load per cloud is
    `sessions * length * intensity` — it equals 70/30 only if sessions are statistically uniform, and
    drifts otherwise. Concretely:

    - You **can't chase a spot-price drop mid-conversation** without eating a re-prefill — cost
      arbitrage steers *new* traffic, not the backlog.
    - A cluster can get **stuck hot** from accumulated long-lived sessions, even at a "low" weight.
    - Split precision and cache hit-rate trade off through **affinity TTL**: short TTL → closer to true
      70/30, more cache misses; long TTL → better hits, more load drift.

    There's no clean resolution because there isn't one to have — it's the load-balancing-vs-locality
    tension, surfaced. Treat the weight as a *steering input*, monitor **realized** per-cloud GPU
    utilization (Chapter 7's observability), and adjust. The saving grace is the same bound as before:
    being wrong costs a re-prefill, never a wrong answer.

---

That completes the platform's concepts: schedule GPUs, declare the cluster, orchestrate the server,
span clouds. Concepts stick when you run them — so the final section is a single continuous lab that
builds this whole platform on Google Kubernetes Engine, from `terraform apply` to a model serving,
gang-scheduled, scaling to zero, with a failover path.

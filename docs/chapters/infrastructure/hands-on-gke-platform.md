# Hands-on: A GPU Inference Platform on GKE

Everything in this chapter, built for real. You'll start from an empty Google Cloud project and end
with a model serving requests on an autoscaling, scale-to-zero GPU platform you provisioned as code,
scheduled deliberately, and made resilient. Each step maps to a section you just read, and several
include a **break-it / observe / fix-it** exercise — because the failures teach more than the happy
path.

We use a single **NVIDIA L4** (24 GB, ~\$0.70/hr on Spot, and quota you can actually get) serving a
~7B model. The *skills* are identical at H100/8-GPU scale; only the SKU and the bill change.

!!! warning "This costs money — and you must tear it down"
    GPU nodes bill by the second. The whole lab run is a few dollars if you **destroy everything at
    the end** (final step). Set a budget alert, and don't leave a GPU node pool running overnight.
    Scale-to-zero (Step 4) protects you while working; `terraform destroy` protects you after.

## Step 0 — Prerequisites

```bash
# tools: gcloud, kubectl, terraform (or opentofu)
gcloud version && kubectl version --client && terraform version

# a project with billing enabled, and the APIs on
export PROJECT_ID="your-project"
export REGION="us-central1"
gcloud config set project "$PROJECT_ID"
gcloud services enable container.googleapis.com compute.googleapis.com
```

You need **L4 GPU quota** in your region (`gcloud compute regions describe $REGION` → look for
`NVIDIA_L4_GPUS`). If it's 0, request an increase in the console (*IAM & Admin → Quotas*) before
continuing — this is itself the Chapter 7 lesson that GPU capacity is something you *request*, not
assume.

## Step 1 — Provision the cluster as code (§ 8.3)

A `main.tf` that creates a GKE Standard cluster and an **L4 node pool that scales to zero** on Spot —
every § 8.2 decision (autoscaling floor of 0, Spot, the managed driver) encoded as reviewable code:

```hcl
provider "google" { project = var.project_id; region = var.region }
variable "project_id" {}
variable "region" { default = "us-central1" }

resource "google_container_cluster" "main" {
  name                     = "infra-lab"
  location                 = var.region
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
}

# Small CPU pool for system pods + the control-plane add-ons
resource "google_container_node_pool" "cpu" {
  name       = "cpu-pool"
  cluster    = google_container_cluster.main.id
  node_count = 1
  node_config { machine_type = "e2-standard-4" }
}

# The GPU pool: L4, Spot, scale-to-zero
resource "google_container_node_pool" "gpu" {
  name     = "l4-pool"
  cluster  = google_container_cluster.main.id
  location = var.region

  autoscaling { min_node_count = 0; max_node_count = 2 }   # ← scale to zero

  node_config {
    machine_type = "g2-standard-8"   # 1× L4
    spot         = true              # cheap, interruptible

    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config { gpu_driver_version = "LATEST" }
    }
    # GKE auto-taints GPU nodes nvidia.com/gpu=present:NoSchedule — we rely on it (§ 8.2)
  }
}
```

```bash
terraform init
terraform plan      # read the diff — this is the review step IaC buys you
terraform apply -var "project_id=$PROJECT_ID"

# wire kubectl to the new cluster
gcloud container clusters get-credentials infra-lab --region "$REGION"
kubectl get nodes
```

You'll see the CPU node and **zero GPU nodes** — the L4 pool is at its floor of 0. That's the platform
holding no GPU bill until something needs one.

## Step 2 — Get a pod onto a GPU (§ 8.2)

First, confirm GKE installed the device plugin and that GPU nodes will advertise the resource. The
pool is empty, so requesting a GPU should *trigger a node to appear*. Apply a throwaway pod that runs
`nvidia-smi`:

```yaml
# gpu-smoke.yaml
apiVersion: v1
kind: Pod
metadata: { name: gpu-smoke }
spec:
  restartPolicy: Never
  tolerations:                       # tolerate GKE's GPU node taint (§ 8.2)
    - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
  containers:
    - name: smi
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources: { limits: { nvidia.com/gpu: 1 } }   # whole GPU, indivisible
```

```bash
kubectl apply -f gpu-smoke.yaml
kubectl get pods -w          # Pending → (node autoscales up ~2-4 min) → Running → Completed
kubectl logs gpu-smoke       # the nvidia-smi table: your L4, from inside a pod
```

What you just watched: a pod requested `nvidia.com/gpu: 1`, the scheduler had nowhere to put it, the
**cluster autoscaler created a GPU node**, the pod landed and saw the L4. That Pending→Running gap is
the cold-start cost of scale-to-zero (§ 8.2). Clean up: `kubectl delete pod gpu-smoke`.

!!! note "GPU sharing on this pool"
    L4 supports **time-slicing** (add `--gpu-sharing-strategy=time-sharing --max-shared-clients-per-gpu=2`
    when creating the pool) so two pods share one L4 — useful for packing small models, with *no*
    memory isolation (§ 8.2). **MIG** needs an A100/H100/H200, so it's not available on L4 — exactly
    the hardware-tier distinction from Chapter 3.

## Step 3 — Serve a model, and learn the probe lesson the hard way (§ 8.4)

Now the real workload: vLLM serving an open 7B model. We'll deploy it **wrong first** — no
`startupProbe` — to see the signature failure, then fix it.

```yaml
# vllm-broken.yaml  — deliberately missing a startupProbe
apiVersion: apps/v1
kind: Deployment
metadata: { name: qwen }
spec:
  replicas: 1
  selector: { matchLabels: { app: qwen } }
  template:
    metadata: { labels: { app: qwen } }
    spec:
      tolerations: [{ key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }]
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model", "Qwen/Qwen2.5-7B-Instruct", "--max-model-len", "8192",
                 "--gpu-memory-utilization", "0.92"]
          ports: [{ containerPort: 8000 }]
          resources: { limits: { nvidia.com/gpu: 1 } }
          livenessProbe:                 # ← fires during the long model load
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 10
            failureThreshold: 3          # kills the pod ~30s in — before weights load
```

```bash
kubectl apply -f vllm-broken.yaml
kubectl get pods -w
```

**Observe the failure.** The pod starts, downloads + loads ~15 GB of weights (minutes), the liveness
probe fails at ~30 s because the server isn't up yet, and Kubernetes **kills and restarts it — forever**:
`CrashLoopBackOff`, never serving. This is the § 8.4 crash loop, live. Confirm the cause:

```bash
kubectl describe pod -l app=qwen | grep -A3 "Liveness\|Killing\|Back-off"
```

Now **fix it** — add a generous `startupProbe` that guards the boot window, and a `readinessProbe`
that only admits traffic once the model can serve:

```yaml
# vllm.yaml  — corrected probes (replace the container's probes block)
          startupProbe:                  # up to 5 min to finish loading (§ 8.4)
            httpGet: { path: /health, port: 8000 }
            failureThreshold: 30
            periodSeconds: 10
          readinessProbe:                # gates the Service — only when it can generate
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 5
          livenessProbe:                 # now only matters AFTER startup passes
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 10
```

```bash
kubectl apply -f vllm.yaml
kubectl get pods -w           # Running, then READY 1/1 once weights are in HBM

# expose and call it — a real served completion
kubectl expose deployment qwen --port 8000 --target-port 8000
kubectl port-forward svc/qwen 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role":"user","content":"In one sentence, why is decode memory-bound?"}]
}' | python3 -m json.tool
```

You now have a model answering — and you've felt *why* the startup probe is the most important ten
lines in § 8.4, instead of being told.

## Step 4 — Gang scheduling and scale-to-zero (§ 8.2)

**Quota-gated admission with Kueue.** Install Kueue, then define a GPU quota and watch it *suspend* a
job that would exceed it — the all-or-nothing admission that stops partial placement:

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/latest/download/manifests.yaml
kubectl -n kueue-system rollout status deploy/kueue-controller-manager
```

```yaml
# queue.yaml — one flavor, a ClusterQueue capped at 1 GPU, a LocalQueue
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata: { name: l4 }
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata: { name: gpu-cq }
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["nvidia.com/gpu"]
      flavors:
        - name: l4
          resources: [{ name: "nvidia.com/gpu", nominalQuota: 1 }]   # only 1 GPU of quota
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata: { name: gpu-lq, namespace: default }
spec: { clusterQueue: gpu-cq }
```

```bash
kubectl apply -f queue.yaml
# submit TWO 1-GPU jobs into a 1-GPU quota; the second must wait, not partially place
for i in 1 2; do
  kubectl create job gpujob-$i --image=nvidia/cuda:12.4.1-base-ubuntu22.04 -- \
    bash -c 'nvidia-smi; sleep 120'
  kubectl label job gpujob-$i kueue.x-k8s.io/queue-name=gpu-lq --overwrite
done
kubectl get workloads     # one Admitted, one with admission pending — quota enforced
```

Job 2 sits **suspended** until job 1 releases its GPU — Kueue refusing to over-commit, the § 8.2
behavior that prevents stranded GPUs at multi-GPU scale (where it's *gang* admission, not just quota).

**Scale-to-zero.** Delete everything GPU-bound and watch the platform retire the node and the bill:

```bash
kubectl delete deploy qwen; kubectl delete job gpujob-1 gpujob-2
kubectl get nodes -w      # after the idle window, the L4 node is removed → 0 GPU nodes
```

The node pool returns to its floor of 0. On GKE this is the **cluster autoscaler**; on EKS the
equivalent is **Karpenter** (§ 8.2). Either way you've closed the loop: GPUs exist only while work
needs them.

## Step 5 — A failover path (§ 8.5)

The capstone, scoped honestly: the cheapest resilience is a **second region** first (same cloud, same
manifests), then a second *cloud* once that works. The portability payoff is that **Step 3's
`vllm.yaml` is unchanged** — only the substrate (Step 1's Terraform) is restamped per region/cloud.

The GKE-native runnable path uses a **fleet + Multi-Cluster Gateway**:

```bash
# register both clusters to a fleet, enable multi-cluster networking
gcloud container fleet memberships register infra-lab-uc1 --gke-cluster=us-central1/infra-lab
gcloud container fleet memberships register infra-lab-ue1 --gke-cluster=us-east1/infra-lab-east
gcloud container fleet ingress enable --config-membership=infra-lab-uc1
```

You then deploy a `MultiClusterService` (exports the `qwen` Service across the fleet) and a
`MultiClusterGateway` (one global anycast IP, health-checked). Requests land on a healthy cluster
*with GPU headroom*; when one region is out of L4 capacity or down, traffic shifts to the other —
the **active–passive failover** of § 8.5, with the passive side scaled to zero so insurance is nearly
free. Extending to a *second cloud* swaps the global front door for a cloud-neutral one (a global LB
or DNS failover) and keeps each model instance whole within its cloud — never split across the slow
cross-cloud wire.

!!! note "Why Step 5 is lighter than the rest"
    Cross-cloud failover is genuinely more setup than one lab should run end-to-end, and most teams
    earn resilience with a second *region* long before a second *cloud*. The skill that transfers is
    the architecture: **portable workload, per-substrate IaC, capacity-aware global routing** — the
    same three ideas, whether the second cluster is a region away or a cloud away.

## Tear down (do this!)

```bash
terraform destroy -var "project_id=$PROJECT_ID"   # deletes cluster, node pools, the GPU bill
# if you made a 2nd cluster/region, destroy that too
```

Confirm `gcloud container clusters list` is empty. The GPU meter is now off.

---

You've built the whole chapter: a cluster declared in Terraform, a GPU obtained by a deliberately
scheduled pod, a model server that taught you the probe lesson by failing first, quota-gated admission
and scale-to-zero that turn GPUs into a usage-based cost, and a failover architecture. That's the
platform layer — the foundation the serving system in Chapter 7 runs on, and the practical complement
to the silicon you learned to choose in Chapter 3.

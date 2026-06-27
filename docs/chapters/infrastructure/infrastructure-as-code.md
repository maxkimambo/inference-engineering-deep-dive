# Infrastructure as Code

The cluster from the last two sections — VPCs, the control plane, GPU node pools with the right
machine types, taints, autoscaling bounds, driver versions — is *dozens* of interdependent resources.
You can click them into a cloud console once. You cannot click them into existence *identically* in a
second region, recreate them after an outage, code-review a change to them, or know six months later
why the node pool has that exact taint. **Infrastructure as Code (IaC)** makes the infrastructure a
versioned document instead of a pile of console state — the same "declare desired state, let it
converge" idea from § 8.1, now applied to the cloud substrate beneath Kubernetes.

!!! key "The thing IaC actually buys you: reproducibility and review"
    An IaC definition is the difference between "our cluster" being a fragile, undocumented artifact
    in one engineer's account and being a **reviewable, versionable, re-runnable artifact** in git.
    You get identical staging/prod environments, disaster recovery by re-apply, a diff for every
    change, and an audit trail. For GPU infra — where a misconfigured node pool wastes thousands a
    month and a region failover may be your capacity strategy (§ 8.5) — this isn't hygiene, it's the
    foundation.

## Terraform: the declarative default

**Terraform** (and its open-source fork **OpenTofu**) is the industry default. You write **HCL**, a
declarative configuration language: you describe the resources you *want*, Terraform computes the diff
against a **state file** (its record of what currently exists), and `terraform apply` makes the cloud
match. The workflow is `plan` (show me the diff) → `apply` (make it so).

A GKE GPU node pool — the substrate for everything in § 8.2 — is a few lines:

```hcl
resource "google_container_node_pool" "h100" {
  name    = "h100-pool"
  cluster = google_container_cluster.main.id

  autoscaling {
    min_node_count = 0          # scale to zero (§ 8.2) — no idle GPU floor
    max_node_count = 8
  }

  node_config {
    machine_type = "a3-highgpu-8g"   # 8× H100, NVLink island (Ch. 3)
    spot         = true               # interruptible, cheap GPU capacity

    guest_accelerator {
      type  = "nvidia-h100-80gb"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"  # GKE installs the driver for you
      }
    }

    taint {                            # keep non-GPU pods off (§ 8.2)
      key = "nvidia.com/gpu"; value = "present"; effect = "NO_SCHEDULE"
    }
  }
}
```

Read what's encoded there: every Chapter 3 and § 8.2 decision — the NVLink machine type, scale-to-zero,
Spot for cost, the driver install, the taint — is now a reviewable line of code, not console
tribal knowledge. **Modules** let you parameterize this (region, GPU type, size) and stamp out
identical clusters across environments and clouds.

## Pulumi: IaC in a real language

**Pulumi** does the same job — declarative desired state, a diff engine, providers — but you express
it in a **general-purpose language** (TypeScript, Python, Go, C#) instead of HCL. The trade is direct:

| | Terraform / OpenTofu | Pulumi |
|---|---|---|
| Language | HCL (purpose-built, declarative) | TS / Python / Go / C# |
| Logic (loops, conditionals, types) | limited, awkward | native — real `for`, types, tests |
| Ecosystem / maturity | largest; the default | smaller but solid |
| Team fit | infra/ops, multi-cloud | software teams who want code, not config |

!!! key "Choose by who maintains it and how dynamic it is"
    **Terraform/OpenTofu** is the right default — the biggest provider ecosystem, the lingua franca of
    ops, and declarative-by-design keeps infra boring (which you want). Pick **Pulumi** when your infra
    has *genuine programmatic logic* (generate N node pools from a config, strong typing, unit tests)
    and the people who own it are software engineers fluent in its language. Don't adopt Pulumi just
    to "write infra in Python" — the dynamism is a double-edged sword; imperative power makes it easier
    to build something clever and unreproducible.

## State and drift: the one discipline

Both tools keep a **state file** — their model of reality. The cardinal sin is **drift**: someone
hand-edits the cluster in the console, the live infra diverges from the state, and your IaC is now
lying. The next `apply` may revert their fix or fail on an unexpected diff. The rule is the same as
§ 8.1's: **never touch the infrastructure outside the code.** Detect drift with `terraform plan`
(it shows reality-vs-desired), store state remotely and locked (so two engineers can't corrupt it),
and treat the console as read-only.

## IaC vs GitOps: two layers, not two choices

A frequent confusion: where does **GitOps** (ArgoCD, Flux) fit, and is it an alternative to Terraform?
No — they own **different layers**:

- **IaC (Terraform/Pulumi)** provisions the **substrate**: cloud accounts, networks, the cluster
  itself, GPU node pools. Things *outside* Kubernetes that Kubernetes runs on.
- **GitOps (ArgoCD/Flux)** continuously reconciles the **contents** of the cluster: your Deployments,
  Services, scheduling configs, the GPU Operator — the in-cluster YAML from § 8.1–8.2. A controller
  watches a git repo and makes the cluster match it, the reconciliation loop applied to your manifests.

```
        git repo (desired state)
        ├── infra/    →  Terraform   →  builds the cluster + GPU node pools
        └── apps/     →  ArgoCD/Flux →  fills the cluster with workloads
```

!!! key "Terraform builds the cluster; GitOps fills it"
    Provision **infrastructure** with IaC; deploy **workloads** with GitOps. They compose: Terraform
    stands up the GKE cluster and H100 pool, then ArgoCD syncs your model-server Deployments, Kueue
    config, and scheduling policy into it from git. Anti-patterns to avoid: deploying *applications*
    through Terraform (slow, brittle, couples app rollouts to infra state) or provisioning *cloud
    infra* through raw `kubectl` (no plan, no review, instant drift). Keep the layers clean and each
    tool does what it's good at.

## Try it: plan → apply → drift → reconcile (free, no cloud)

You can feel the entire IaC discipline — including drift detection — with zero cloud cost, using
Terraform's `local_file` resource as a stand-in for "infrastructure":

```bash
mkdir iac-demo && cd iac-demo
cat > main.tf <<'EOF'
resource "local_file" "cluster" {
  filename = "cluster.txt"
  content  = "node_pool = l4, min = 0, max = 2"   # desired state
}
EOF

terraform init
terraform plan                 # "+ 1 to add" — the diff IaC buys you
terraform apply -auto-approve  # creates cluster.txt; records it in terraform.tfstate

# Now commit the cardinal sin: hand-edit the 'infrastructure' out of band
echo "someone clicked in the console" > cluster.txt

terraform plan                 # ← Terraform DETECTS the drift: reality ≠ desired
terraform apply -auto-approve  # reconciles reality back to the declared state
cat cluster.txt                # your desired content is restored

cd .. && rm -rf iac-demo
```

Swap `local_file` for `google_container_node_pool` and this is *exactly* how you'll manage the GPU
pool from § 8.6 — same `plan`/`apply`/state/drift loop, real money on the line. The skill: treat the
console as read-only and let `terraform plan` be your truth about what's actually deployed.

---

The cluster is now reproducible code and its workloads sync from git. What remains is making the
*workload itself* behave on this platform — the pod shapes and rollout patterns that survive a server
which takes minutes to become useful. That's the next section.

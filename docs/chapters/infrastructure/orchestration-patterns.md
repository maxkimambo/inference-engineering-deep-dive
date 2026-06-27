# Container Orchestration Patterns

Kubernetes' reconciliation loop (§ 8.1) assumes a container that starts fast and is either up or down.
A model server breaks that assumption: it runs for *minutes* — pulling a 20 GB image, then loading
tens of GB of weights into HBM — before it can serve a single token, and "up" (process running) is not
"ready" (weights loaded). This section is the set of pod and deployment patterns that make a
slow-starting, stateful-feeling GPU workload behave correctly under a platform built for fast,
stateless ones. Get these wrong and you get the signature GPU-serving failures: crash loops during
load, traffic to half-loaded pods, and rollouts that strand capacity.

## The model-server pod, anatomy

A production model-server pod is usually three roles, not one container:

- **Init container** — runs to completion *before* the main container: downloads and verifies model
  weights into a shared volume (or warms a cache). Separating this means the serving container starts
  with weights already on local disk, and a download failure fails fast and visibly instead of
  mid-serve.
- **Main container** — the inference engine (Chapter 4) holding the GPU.
- **Sidecars** — the metrics exporter (DCGM/engine metrics → Chapter 7's observability), maybe a
  proxy or request-logging container, sharing the pod's network.

## Probes: the highest-leverage 10 lines you'll write

Kubernetes decides *is this pod alive? ready for traffic?* via three probes, and for a slow-loading
model server the defaults are actively dangerous. The three are distinct and you need all three:

- **`startupProbe`** — "is it done booting *yet*?" Holds off the other two probes until it passes.
  **This is the one that saves you.** Without it, the liveness probe starts checking immediately,
  fails while the model is still loading, and Kubernetes **kills the pod mid-load → permanent crash
  loop.** Give it a long budget (loads can take minutes).
- **`readinessProbe`** — "should it get traffic *now*?" Must only pass once **weights are in HBM and
  the engine can actually generate**. If it passes too early, the Service routes requests to a pod
  that returns errors. Readiness is what gates the load balancer.
- **`livenessProbe`** — "is it wedged?" Restarts a hung pod (deadlocked CUDA context, stuck engine).
  Only meaningful *after* startup completes.

```yaml
startupProbe:                 # tolerate a long load: 30 * 10s = 5 min before giving up
  httpGet: { path: /health, port: 8000 }
  failureThreshold: 30
  periodSeconds: 10
readinessProbe:               # only true when the model can serve
  httpGet: { path: /ready, port: 8000 }
  periodSeconds: 5
livenessProbe:                # restart if wedged — but startupProbe guards the boot window
  httpGet: { path: /health, port: 8000 }
  periodSeconds: 10
```

!!! key "No startupProbe = crash-loop on every cold start"
    The number-one self-inflicted GPU-serving outage is a liveness probe with no startup probe: the
    pod is healthily loading a 70B model, liveness fails at 30 s, Kubernetes restarts it, and it
    never finishes loading — forever. Always pair a generous **startupProbe** with a **readinessProbe**
    that reflects *true* serving readiness (weights in HBM), not just "process is up." These ten lines
    prevent more incidents than any other config in this chapter.

## Where do the weights come from? The cold-start trade

Loading weights is the dominant cold-start cost (Chapter 7, § 7.2.2). Four strategies, trading image
size against start latency against operational complexity:

| Strategy | Start latency | Image size | Trade |
|----------|---------------|-----------|-------|
| **Bake weights into the image** | fast (already local) | huge (20–100 GB+) | slow to pull/push; rebuild per weight change |
| **Download at init** (from object storage) | slow (network-bound) | small | needs storage + bandwidth; PCIe/network floor (Ch. 3, § 3.4) |
| **Mounted cache** (GCS FUSE / read-only PVC) | medium, shared | small | weights live once, many pods mount; cache warms over time |
| **Image streaming** (lazy-pull layers) | fast first bytes | n/a | platform feature (GKE image streaming); start before full pull |

!!! key "Right answer depends on how often weights change and how often you cold-start"
    Stable weights + frequent cold starts (scale-to-zero) → **bake them in** or use a **warm mounted
    cache** so each new pod doesn't re-download. Frequently changing weights → **download at init**
    or **stream**, keeping images small. There's no universal winner; it's the cold-start half of the
    scale-to-zero decision from § 8.2 — you're choosing where the minutes go.

## Rollouts when GPUs are scarce

A normal Deployment does a **rolling update**: spin up new pods, then retire old ones, controlled by
`maxSurge` (extra pods allowed during the roll) and `maxUnavailable`. This quietly assumes spare
capacity exists to surge into — and for GPUs **it often doesn't**. If every GPU is occupied,
`maxSurge: 1` asks for a GPU that isn't there, and the rollout stalls.

- **Surge needs a spare GPU.** Either keep headroom, let the node autoscaler (§ 8.2) add a node for
  the surge pod, or set `maxSurge: 0` / `maxUnavailable: 1` (retire-then-replace — accepts a brief
  capacity dip to avoid needing an extra GPU).
- **Canary / blue-green** for quality-sensitive rollouts: shift a slice of traffic to the new version
  and watch output quality before full cutover. This is Chapter 7's testing-and-deployment territory;
  the platform mechanism is a second Deployment + weighted Service/Gateway.

## Surviving disruption: PDBs, Spot, and spread

Three patterns keep a fleet resilient against the platform moving underneath it:

- **PodDisruptionBudget (PDB)** — caps how many replicas can be *voluntarily* evicted at once (node
  upgrades, drains). `minAvailable: 2` means a node upgrade can't take your serving capacity below 2
  pods. Essential when one node hosts several GPU replicas.
- **Spot/preemption handling** — Spot GPUs (§ 8.2/§ 8.3) are cheap but reclaimable on short notice.
  Handle the termination signal: drain in-flight requests, and for *training* jobs checkpoint
  regularly so a reclaim costs minutes, not the whole run.
- **Topology spread constraints** — spread replicas across zones/nodes so one node or zone failure
  doesn't drop all of them. The counter-pull to § 8.2's affinity (which packs a *single* multi-GPU
  group together): spread *independent replicas*, pack *one tightly-coupled group*.

## The operator pattern: when to stop hand-rolling

Everything above — init containers, three probes, weight caching, rollout policy, PDBs, autoscaling
hooks — is a lot of YAML to get right per model. A **serving operator** (KServe, NVIDIA NIM Operator,
or a custom one) encapsulates it behind a high-level object: you declare `InferenceService: llama-70b`
and the operator generates the correct Deployment, probes, scaling, and routing.

!!! key "Adopt an operator when you're running many models, not your first"
    For one or two model servers, write the YAML directly — you'll understand exactly what runs, and
    an operator is overhead. Once you're standing up *many* models or letting teams self-serve, an
    operator like **KServe** pays off: it bakes these patterns in so every team gets correct probes,
    scale-to-zero, and canary rollouts without re-deriving them (and re-making the crash-loop mistake).
    Same trade as § 8.1's "when to use Kubernetes," one level up: platform abstractions earn their
    keep at fleet scale.

## Try it: reproduce the crash-loop, then fix it (free, no GPU)

The startup-probe lesson costs nothing to learn — you don't need a GPU or a slow model, just a
container that *pretends* to take 60 s to load. On a free `kind` cluster:

```yaml
# slowboot.yaml — simulates a model that takes 60s to become ready
apiVersion: v1
kind: Pod
metadata: { name: slowboot }
spec:
  containers:
    - name: app
      image: busybox
      command: ["sh","-c","sleep 60; touch /tmp/ready; sleep 3600"]  # 'loading' for 60s
      livenessProbe:
        exec: { command: ["cat","/tmp/ready"] }
        periodSeconds: 5
        failureThreshold: 2          # declares it dead ~10s in — before the 60s 'load' finishes
```

```bash
kind create cluster
kubectl apply -f slowboot.yaml
kubectl get pod slowboot -w          # RESTARTS climbs → CrashLoopBackOff. It never finishes loading.
```

Now add the guard. Put this `startupProbe` in the container (alongside the others) and re-apply:

```yaml
      startupProbe:
        exec: { command: ["cat","/tmp/ready"] }
        failureThreshold: 30         # up to 30*5 = 150s to finish 'loading' before liveness starts
        periodSeconds: 5
```

```bash
kubectl apply -f slowboot.yaml
kubectl get pod slowboot -w          # survives the 60s load, then Running and stable
kind delete cluster
```

You just reproduced the single most common GPU-serving outage with a `busybox` container and fixed it
in three lines — the muscle memory that turns § 8.6 Step 3 from a surprise into a checklist.

---

You can now run a single model server correctly on this platform — placed, scheduled, reproducible,
and resilient. The last question is what happens when *one cloud can't give you the GPUs at all*, and
the workload has to span providers.

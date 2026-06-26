# Hands-on: a Quantization Pipeline on GKE

In [Chapter 5](../techniques/quantization.md#hands-on-quantizing-qwen-on-google-cloud-with-llm-compressor)
we quantized Qwen by hand on a throwaway GPU VM. That's fine once. In production you want it
**repeatable, automated, and cheap when idle**: a new model version lands, a job spins up a GPU,
quantizes, writes the checkpoint to a bucket, and the GPU disappears — then serving picks up the new
weights. This page builds exactly that on **Google Kubernetes Engine (GKE)**.

## The shape

```
  trigger (manual / CronJob / event)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │ GKE batch Job  (Kubernetes Job)          │
  │   • lands on the GPU node pool…          │   GPU node pool
  │   • …which scales 0→1 just for this job  │◄─ (L4, Spot, autoscale min=0)
  │   • pull model from Hugging Face          │
  │   • llm-compressor → INT4 checkpoint      │
  │   • write to GCS  (via Workload Identity) │
  └───────────────┬─────────────────────────┘
                  │ checkpoint in gs://…/models/
                  ▼
  vLLM Deployment  ── rolling update ──► serves the quantized model
        (node pool scales 1→0 after the job; no idle GPU bill)
```

The whole design goal: **pay for the GPU only while the job runs.** Everything below serves that.

!!! info "Prerequisites"
    A GKE cluster with **Workload Identity** enabled and the **GCS FUSE CSI driver** addon; an
    **Artifact Registry** repo for the job image; a **Cloud Storage** bucket for checkpoints; and
    **GPU quota** for L4 in your region. Enable the addons on an existing cluster with:
    ```bash
    gcloud container clusters update CLUSTER --location REGION \
      --workload-pool=PROJECT_ID.svc.id.goog --update-addons=GcsFuseCsiDriver=ENABLED
    ```

## 1 — A GPU node pool that scales to zero

The cost trick: a dedicated node pool with `--min-nodes=0`. It holds **no GPU nodes** (and bills
nothing) until a Pod requests a GPU, then scales up for the job and back to zero after.

```bash
gcloud container node-pools create gpu-quant \
  --cluster=CLUSTER --location=REGION \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1,gpu-driver-version=default \  # GKE installs the driver
  --enable-autoscaling --num-nodes=0 --min-nodes=0 --max-nodes=3 \   # ← scale to zero
  --spot \                                                            # ~60–70% cheaper
  --node-locations=REGION-a
```

GKE automatically **taints** GPU nodes (`nvidia.com/gpu=present:NoSchedule`) so only GPU workloads land
there — your Job will carry a matching toleration.

!!! key "Why a separate scale-to-zero pool, not one big VM"
    A standing GPU VM (Chapter 5's approach) bills 24/7 even when idle. A `min-nodes=0` pool bills
    **only for the minutes a job actually runs**, and the cluster autoscaler tears the node down
    afterward. For a job that runs occasionally, that's the difference between a few dollars a month and
    a few hundred.

## 2 — Containerize the job

The job is a tiny image: the quantization library plus a script driven entirely by environment
variables, so one image quantizes *any* model with *any* recipe.

`quantize.py`:

```python
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID   = os.environ["MODEL_ID"]                       # e.g. Qwen/Qwen2.5-7B-Instruct
OUTPUT_DIR = os.environ["OUTPUT_DIR"]                     # a path on the mounted bucket
SCHEME     = os.environ.get("SCHEME", "W4A16")
IGNORE     = os.environ.get("IGNORE", "lm_head").split(",")
SAMPLES    = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "512"))

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

oneshot(
    model=model,
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=GPTQModifier(targets="Linear", scheme=SCHEME, ignore=IGNORE),
    max_seq_length=2048,
    num_calibration_samples=SAMPLES,
)

model.save_pretrained(OUTPUT_DIR, save_compressed=True)   # OUTPUT_DIR is the GCS mount
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"wrote quantized checkpoint to {OUTPUT_DIR}")
```

`Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN pip install --no-cache-dir llmcompressor
COPY quantize.py /app/quantize.py
ENTRYPOINT ["python", "/app/quantize.py"]
```

Build and push to Artifact Registry (Cloud Build keeps it off your laptop):

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/quantize:latest
```

## 3 — Give the job bucket access with Workload Identity

**No service-account keys.** Workload Identity lets the Job's Kubernetes ServiceAccount *impersonate* a
Google service account that has bucket permissions — credentials are short-lived and never leave Google.

```bash
# 1. a Google service account for the job, with write access to the bucket
gcloud iam service-accounts create quant-job
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \
  --member="serviceAccount:quant-job@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# 2. a Kubernetes service account, bound to the Google one
kubectl create serviceaccount quant-ksa
gcloud iam service-accounts add-iam-policy-binding \
  quant-job@PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:PROJECT_ID.svc.id.goog[default/quant-ksa]"

# 3. link them with an annotation
kubectl annotate serviceaccount quant-ksa \
  iam.gke.io/gcp-service-account=quant-job@PROJECT_ID.iam.gserviceaccount.com
```

## 4 — The Kubernetes Job

This is the heart of it. The Job requests one GPU, tolerates the GPU taint, mounts the bucket via GCS
FUSE (so `save_pretrained` writes straight to Cloud Storage), and cleans itself up when done.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: quantize-qwen25-7b
spec:
  backoffLimit: 2                 # retry twice (Spot nodes can be preempted)
  ttlSecondsAfterFinished: 3600   # auto-delete the Job object an hour after it finishes
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"          # enable the FUSE sidecar
    spec:
      serviceAccountName: quant-ksa          # ← Workload Identity
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: quantize
          image: REGION-docker.pkg.dev/PROJECT_ID/REPO/quantize:latest
          env:
            - { name: MODEL_ID,   value: "Qwen/Qwen2.5-7B-Instruct" }
            - { name: OUTPUT_DIR, value: "/data/Qwen2.5-7B-Instruct-W4A16-G128" }
            - { name: SCHEME,     value: "W4A16" }
            - { name: IGNORE,     value: "lm_head" }       # comma-separated to target more layers
          resources:
            limits:
              nvidia.com/gpu: "1"
          volumeMounts:
            - { name: ckpt, mountPath: /data }
      volumes:
        - name: ckpt
          csi:
            driver: gcsfuse.csi.storage.gke.io
            volumeAttributes:
              bucketName: YOUR_BUCKET
              mountOptions: "implicit-dirs"
```

Run it and watch the node pool wake up:

```bash
kubectl apply -f quantize-job.yaml
kubectl get pods -w                       # Pending → (node scales 0→1) → Running → Completed
kubectl logs -f job/quantize-qwen25-7b
```

When the Pod completes, the checkpoint is in `gs://YOUR_BUCKET/Qwen2.5-7B-Instruct-W4A16-G128/`, and the
autoscaler removes the GPU node within minutes — back to zero GPU spend.

!!! warning "Spot preemption is normal — design for it"
    `--spot` nodes can be reclaimed mid-job. `backoffLimit: 2` lets the Job retry on a fresh node. Since
    quantization is a deterministic batch job with no external side effects until the final write, a
    restart is harmless — it just re-runs. Don't use Spot for *latency-sensitive serving*; do use it for
    *interruptible batch* like this.

## 5 — Make it repeatable

The image is already parameterized, so productionizing is about *triggering*:

- **On a schedule** — wrap the same pod spec in a **`CronJob`** to re-quantize when base models or
  calibration data refresh.
- **Event-driven** — have a new model landing in a bucket or registry fire **Eventarc → a Job** (or a
  Cloud Build trigger that `kubectl apply`s it). This is the "new model version → auto-quantize" loop.
- **At fleet scale** — if you quantize many models and contend for limited GPU quota, put a batch
  queue like **Kueue** in front so Jobs queue for GPU capacity instead of failing to schedule.

!!! tip "Parameterize per model"
    Template the Job (Helm/Kustomize, or just `envsubst`) on `MODEL_ID`, `OUTPUT_DIR`, `SCHEME`, and
    `IGNORE`. One pipeline then quantizes your whole model catalog — and the [layer-targeting
    recipes](../techniques/quantization.md#targeting-different-layers) from Chapter 5 become per-model
    config (`IGNORE="lm_head,re:.*down_proj"`), not code changes.

## 6 — Hand off to serving

A serving Deployment consumes the checkpoint from the same bucket — mount it read-only with GCS FUSE so
no weights bake into the serving image:

```yaml
# vLLM serving Deployment (sketch)
spec:
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      serviceAccountName: quant-ksa
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args: ["--model", "/models/Qwen2.5-7B-Instruct-W4A16-G128"]
          resources:
            limits: { nvidia.com/gpu: "1" }
          volumeMounts:
            - { name: models, mountPath: /models, readOnly: true }
      volumes:
        - name: models
          csi:
            driver: gcsfuse.csi.storage.gke.io
            readOnly: true
            volumeAttributes: { bucketName: YOUR_BUCKET }
```

Publishing a new quantization is then a **rolling update**: point the Deployment's `--model` arg at the
new checkpoint directory and `kubectl apply` — GKE drains old Pods only as new ones become ready, so
serving never drops (the zero-downtime deploy pattern from §7.4). Because INT4 needs ~¼ the VRAM, the
serving pool can run smaller, denser GPU nodes than a BF16 deployment would.

## What you built

A closed loop: **trigger → scale-from-zero GPU job → quantize → bucket → rolling serve**, with
short-lived credentials, Spot-priced compute, and no idle GPU bill. Swap `MODEL_ID`/`IGNORE` to handle
any model and any layer-targeting recipe; swap `CronJob`/Eventarc to change *when* it runs.

The Chapter 5 techniques tell you *what* to do to a model; this is the production plumbing that does it
**reliably and on every model**, which is what Chapter 7 is about.

---

Sources for the GKE specifics:
[GPUs in GKE Standard node pools](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpus),
[GKE automatic GPU driver install](https://cloud.google.com/blog/products/containers-kubernetes/gke-can-now-automatically-install-nvidia-gpu-drivers).

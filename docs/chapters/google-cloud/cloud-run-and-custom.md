# Cloud Run GPU & Custom Containers

Two rungs remain between "call an API" and "run Kubernetes." **Cloud Run GPU** is serverless containers
with a GPU bolted on — you hand Google a container, it scales from zero, bills by the second. And when
a prebuilt serving image won't do, **Vertex custom containers** let you bring your own image to a
managed endpoint by meeting a small contract. Together they cover "I have a container, not a cluster."

## Cloud Run GPU: serverless inference

Cloud Run runs your container and scales instances on request load — now with an **NVIDIA L4 (24 GB)**
or **RTX PRO 6000 Blackwell (96 GB)** attachable per instance.[^crgpu] The defining feature is **true
scale-to-zero**: no requests → zero instances → **$0**, then it spins an instance back up on the next
request. One GPU per instance, billed by the second.

```bash
# Serve vLLM serverless, scale-to-zero, weights streamed from GCS via Cloud Storage FUSE
gcloud run deploy vllm-qwen \
  --image=vllm/vllm-openai:latest \
  --gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu=8 --memory=32Gi \                 # L4 wants ≥4 CPU/16Gi; 8/32 recommended
  --min-instances=0 --max-instances=3 \   # ← scale-to-zero
  --concurrency=40 \                      # vLLM batches many users per instance (Ch. 2)
  --no-cpu-throttling --port=8000 --region=us-central1 \
  --args=--model=Qwen/Qwen2.5-7B-Instruct,--max-model-len=8192
```

!!! key "Cloud Run GPU is the serverless sweet spot — for single-GPU models"
    It nails **bursty and scale-to-zero** workloads: an internal tool, a demo, an endpoint that's idle
    most of the day. You pay only while serving, and it's a `gcloud run deploy`, not a cluster. Its
    hard limit is **one GPU per instance** — there's no tensor parallelism across instances (Ch. 3,
    § 3.4 needs NVLink within a box), so the model must fit and run on a *single* L4 or Blackwell.
    Need multi-GPU serving, gang scheduling, or custom topology? That's GKE (Ch. 8). Need ML-platform
    features (Model Registry, traffic split, batch)? That's Vertex (§ 9.1).

**Cold starts are the tax.** Scaling from zero means instance start + container pull + **weight load**
before the first token — on the order of ~20 s for a small model (Ch. 7, § 7.2.2). Two levers, both
familiar:

- **`--min-instances=1`** keeps one instance warm — you trade scale-to-zero's \$0 idle for predictable
  latency (the § 9.1 / Chapter 8 dial again).
- **Cloud Storage FUSE** mounts weights from a GCS bucket instead of baking a 20 GB image, so the
  container pulls fast and reads weights on demand — the managed cousin of Chapter 8's mounted-cache
  weight-loading strategy.

The cost model is **per-second instance billing** (an L4 instance is roughly \$0.67/hr while running,
no per-request fee), and **min-instances are billed at full rate even when idle** — so scale-to-zero
is exactly what makes it cheap. Push `--concurrency` as high as quality allows: vLLM's continuous
batching (Ch. 2) means one warm GPU serves many users, dividing the hourly cost across all of them.

## Vertex custom containers: bring your own image

Model Garden's prebuilt containers (§ 9.2) cover the common engines. When you need something else —
custom pre/post-processing, an engine Google doesn't package, special dependencies — you bring your own
image to a Vertex Endpoint by satisfying a small **serving contract**:[^vtxcustom]

| Requirement | Detail |
|-------------|--------|
| **HTTP server** | listen on `AIP_HTTP_PORT` (default `8080`) |
| **Health route** | `GET` `AIP_HEALTH_ROUTE` → **200** when ready to serve |
| **Predict route** | `POST` `AIP_PREDICT_ROUTE`, request `{"instances": [...]}` → response `{"predictions": ...}` |
| **Weights** | download from `AIP_STORAGE_URI` (a GCS path Vertex sets) at startup |

```python
# Register a custom image against the contract, then deploy as in § 9.1
model = aiplatform.Model.upload(
    display_name="my-custom-server",
    serving_container_image_uri="us-central1-docker.pkg.dev/PROJECT/repo/my-server:latest",
    serving_container_ports=[8080],
    serving_container_health_route="/health",     # → AIP_HEALTH_ROUTE
    serving_container_predict_route="/predict",   # → AIP_PREDICT_ROUTE
)
```

!!! key "The contract is the whole point — meet it and you inherit the platform"
    Vertex doesn't care what's *inside* your container as long as it speaks `AIP_HTTP_PORT` + health +
    predict. Meet those four lines and your arbitrary image gets the full § 9.1 platform for free —
    autoscaling, scale-to-zero, traffic-split canaries, monitoring — with no cluster. This is the
    managed-platform version of Chapter 8's probe contract: implement the interface, inherit the
    orchestration. (vLLM's OpenAI-style routes differ from the `instances`/`predictions` shape, which
    is why the prebuilt Model Garden vLLM container exists — it does that mapping for you.)

## Try it: serverless vLLM that costs nothing at rest

```bash
gcloud run deploy vllm-qwen --image=vllm/vllm-openai:latest \
  --gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu=8 --memory=32Gi --min-instances=0 --max-instances=2 \
  --concurrency=40 --no-cpu-throttling --port=8000 --region=us-central1 \
  --args=--model=Qwen/Qwen2.5-7B-Instruct,--max-model-len=8192 --allow-unauthenticated

URL=$(gcloud run services describe vllm-qwen --region=us-central1 --format='value(status.url)')
# first call pays the cold start (~tens of s); subsequent calls are warm
curl -s "$URL/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"hi"}]}'
# leave it idle → it scales to 0 → $0. Tear down:
gcloud run services delete vllm-qwen --region=us-central1
```

You served a 7B model with one command, paid only while it ran, and watched it cost nothing at rest —
the serverless rung, end to end.

---

You've now seen every rung: a per-token API, serverless containers, managed endpoints, and (Chapter 8)
your own cluster. The last question is the one that started the chapter — given a real workload, which
rung? — answered as a decision procedure.

[^crgpu]: Google Cloud docs — *Configure GPU for Cloud Run services* (NVIDIA L4 24 GB / RTX PRO 6000 Blackwell 96 GB, one GPU per instance, scale-to-zero, L4 min 4 CPU/16 GiB, instance-based billing with no per-request fee, `--gpu`/`--gpu-type`/`--no-gpu-zonal-redundancy`): <https://docs.cloud.google.com/run/docs/configuring/services/gpu>
[^vtxcustom]: Google Cloud docs — *Custom container requirements for inference* (`AIP_HTTP_PORT` default 8080, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE`, `AIP_STORAGE_URI`; `{"instances": …}` → `{"predictions": …}`): <https://docs.cloud.google.com/vertex-ai/docs/predictions/use-custom-container>

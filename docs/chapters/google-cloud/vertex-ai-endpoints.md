# Vertex AI Endpoints

Vertex AI Endpoints are the managed version of the entire platform you hand-built in Chapter 8. Every
moving part you wired together there — the Deployment, the Service, the autoscaler, the health probes,
the rollout policy, the node pool — has a Vertex equivalent that Google operates for you. The skill
here isn't *building* those pieces; it's *recognizing* them under new names and configuring them well.

!!! key "A Vertex Endpoint is a Deployment you don't operate"
    | Chapter 8 (you run it) | Vertex AI (Google runs it) |
    |------------------------|----------------------------|
    | GPU node pool (machine type, accelerator) | **`DedicatedResources.machineSpec`**[^vtxcompute] |
    | Deployment + replicas | a **DeployedModel** on an Endpoint |
    | HorizontalPodAutoscaler | **`AutoscalingMetricSpec`** (min/max replica) |
    | readiness/startup probes | health route on the serving container |
    | rolling update / canary | **traffic split** across DeployedModels |
    | Service / load balancer | the Endpoint's managed URL |

    Same concepts (Ch. 8), zero cluster to maintain. What you lose is fine-grained control — no custom
    scheduler, no Kueue, no arbitrary sidecars; what you gain is never patching a node again.

## The model → endpoint flow

Vertex serving is two objects. A **Model** is a registered artifact in the **Model Registry**: weights
(or a pointer to them) plus a **serving container** — the image that runs inference, either a Google
prebuilt one (vLLM, Hex-LLM, TGI — § 9.2) or your own (§ 9.3). An **Endpoint** is the managed,
load-balanced URL clients call; you **deploy** one or more Models onto it with attached hardware.[^vtxdeploy]

```python
from google.cloud import aiplatform
aiplatform.init(project=PROJECT, location="us-central1")

# 1. Register a model: a prebuilt vLLM serving container + where the weights live
model = aiplatform.Model.upload(
    display_name="qwen-7b",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest",
    serving_container_args=["--model=Qwen/Qwen2.5-7B-Instruct", "--max-model-len=8192"],
    serving_container_ports=[8080],
    serving_container_health_route="/health",
    serving_container_predict_route="/generate",
)

# 2. Create the endpoint (the managed URL) and deploy the model with GPU + autoscaling
endpoint = aiplatform.Endpoint.create(display_name="qwen-ep")
endpoint.deploy(
    model=model,
    machine_type="g2-standard-12",          # 1× L4 (Ch. 3) — the machineSpec
    accelerator_type="NVIDIA_L4",
    accelerator_count=1,
    min_replica_count=1,                     # set 0 for scale-to-zero (below)
    max_replica_count=3,
)
```

That's the whole Chapter-8 lab — node pool, Deployment, Service, probes, HPA — in two SDK calls.
Google provisions the GPU, runs the container, health-checks it, fronts it with a URL, and will
autoscale it.

## Autoscaling: the one dial you actually turn

Vertex scales replicas between `min_replica_count` and `max_replica_count` to hold a **target
utilization** — by default **60%** of CPU *or* GPU, whichever is higher.[^vtxautoscale] On a GPU
deployment it tracks GPU duty cycle. The only knobs are `min_replica_count`, `max_replica_count`, `required_replica_count`,
and the `AutoscalingMetricSpec` (which metric + target). That deliberate sparseness is the point:
Vertex hides the five-knob traffic autoscaler from Chapter 7 behind one target number.

!!! key "min_replica_count is the cost dial, and 0 changes the economics"
    - `min_replica_count = 1+` → always-warm; you pay for at least one GPU 24/7, but latency is
      consistent (no cold start).
    - `min_replica_count = 0` → **scale-to-zero**: Vertex tears down the last replica when idle and you
      pay nothing between requests — at the cost of a **cold start** (Ch. 7, § 7.2.2: node + container
      + weight load, tens of seconds to minutes) on the next request.

    This is the same scale-to-zero trade as Chapter 8's Karpenter/cluster-autoscaler, now a single
    field. Use `0` for spiky/dev/internal traffic; use `1+` behind a latency SLA. It's the highest-
    leverage decision on the whole endpoint.

GPU-based autoscaling has a structural caveat worth knowing: decode is memory-bound (Ch. 2), so a
decode-heavy replica can be *fully* loaded on memory bandwidth while GPU *duty cycle* still reads
below 60% — and Vertex won't scale up. Watch tail latency, not just the utilization metric, and lower
the target if requests queue.

## Dedicated endpoints, and why streaming needs them

By default an endpoint shares Google-managed front-end infrastructure. A **dedicated endpoint**[^vtxdeploy]
gives the DeployedModel its own dedicated DNS and network path — required for **streaming responses**
(server-sent token streaming, the norm for chat UIs) and gRPC, and it isolates your traffic from
noisy neighbors. For any interactive LLM endpoint you'll almost always want a dedicated endpoint so
you can stream tokens as they generate rather than waiting for the full completion.

## Canary with traffic split

Rolling out a new model version safely is Chapter 7's canary, expressed as a **traffic split** across
DeployedModels on one endpoint. Deploy the new version with a small slice, watch quality and latency,
then shift:

```python
# v2 lands on the SAME endpoint, taking 10% while v1 keeps 90%
endpoint.deploy(model=model_v2, machine_type="g2-standard-12",
                accelerator_type="NVIDIA_L4", accelerator_count=1,
                min_replica_count=1, max_replica_count=3,
                traffic_split={"0": 90, "0-new": 10})   # ids: existing vs new DeployedModel

# happy with v2? shift all traffic, then undeploy v1
endpoint.update(traffic_split={"<v2_id>": 100})
```

Same idea as a Kubernetes weighted rollout (Ch. 8, § 8.4), but Vertex holds both versions behind one
URL and does the splitting — no second Deployment, Gateway, or mesh to manage.

## Try it: deploy, scale-to-zero, canary

```bash
# the SDK flow above also exists as gcloud — register, create, deploy:
gcloud ai models upload --region=us-central1 --display-name=qwen-7b \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest \
  --container-args="--model=Qwen/Qwen2.5-7B-Instruct" \
  --container-ports=8080 --container-health-route=/health --container-predict-route=/generate
gcloud ai endpoints create --region=us-central1 --display-name=qwen-ep
gcloud ai endpoints deploy-model ENDPOINT_ID --region=us-central1 \
  --model=MODEL_ID --machine-type=g2-standard-12 \
  --accelerator=type=nvidia-l4,count=1 --min-replica-count=0 --max-replica-count=2   # scale-to-zero

# call it (first request after idle pays the cold start):
gcloud ai endpoints predict ENDPOINT_ID --region=us-central1 --json-request=request.json

# observe scale-to-zero: leave it idle, watch replicas drain to 0 in the console → $0 between requests
# clean up so you stop paying:
gcloud ai endpoints undeploy-model ENDPOINT_ID --region=us-central1 --deployed-model-id=DEPLOYED_ID
gcloud ai endpoints delete ENDPOINT_ID --region=us-central1
```

You just ran the Chapter-8 platform — provisioned GPU, autoscaled serving, scale-to-zero, canary — with
no cluster, no node pool, no probes to tune. That's the rung trade made concrete.

---

The serving container in those calls was a *prebuilt* one from Model Garden. That's the next rung up:
deploy a whole open model in one click, or skip deployment entirely with a per-token API.

[^vtxdeploy]: Google Cloud docs — *Deploy a model to an endpoint* (endpoints, dedicated public endpoints, deploying multiple models with a traffic split): <https://docs.cloud.google.com/vertex-ai/docs/general/deployment>
[^vtxcompute]: Google Cloud docs — *Configure compute resources for inference* (`DedicatedResources.machineSpec`: `machine_type`, `accelerator_type`, `accelerator_count`, `gpu_partition_size`): <https://docs.cloud.google.com/vertex-ai/docs/predictions/configure-compute>
[^vtxautoscale]: Google Cloud docs — *Scale inference nodes by using autoscaling* (default 60% CPU/GPU target, `min_replica_count`/`max_replica_count`/`required_replica_count`/`AutoscalingMetricSpec`, scale-to-zero at `min_replica_count=0`): <https://docs.cloud.google.com/vertex-ai/docs/predictions/autoscaling>

# Model Garden & Model-as-a-Service

Vertex § 9.1 showed *how* to put a model on an endpoint. **Model Garden** is the catalog you pull the
model *from* — Google's own **Gemma**, partner open weights (Llama, Mistral, Qwen, DeepSeek), and
Hugging Face models — each pre-packaged so deployment is a click or one SDK call instead of a
Dockerfile. And for many of those models you can skip deployment entirely with a **per-token API**.
This section is about choosing *how* to consume an open model, which is often a bigger cost and latency
decision than which model you pick.

## Three ways to consume a model

| Way | What you run | Billing | Control | Use when |
|-----|--------------|---------|---------|----------|
| **MaaS** (Model-as-a-Service) | nothing — a serverless API | **per token** | low | bursty/low volume, no infra wanted, fast start |
| **Self-deploy** (Model Garden → Endpoint) | a dedicated endpoint + GPUs | **per GPU-hour** | high | steady volume, fixed latency, VPC isolation |
| **Gemini API** | nothing — Google's frontier model | per token | n/a | you want the best managed model, not an open one |

The first decision in any Google Cloud LLM project is which of these three you're on — and the default
should be the *most* managed one that meets your needs (§ 9.0's ladder).

## Self-deploy: one model, prebuilt serving

When you self-deploy an open model, Model Garden registers it to the Model Registry with a **prebuilt
serving container** and deploys it to a dedicated endpoint — the § 9.1 flow, with the container and
args filled in for you. You choose the serving engine:

- **vLLM** — the GPU default (Ch. 4): continuous batching, paged KV cache, OpenAI-compatible. What most
  open-model GPU deployments use.
- **Hex-LLM** — Google's own high-throughput LLM engine, **optimized for TPU** (Ch. 3's TPU v5e/v6e).[^mgserve]
  If you're serving on TPUs for `$/token` efficiency, this is the Google-native path vLLM-on-GPU's TPU
  counterpart.
- **TGI** (Text Generation Inference) — Hugging Face's server, also offered.

```python
# Deploy Gemma from Model Garden in one SDK call — container + args are chosen for you
from vertexai import model_garden
model = model_garden.OpenModel("google/gemma-3-9b-it")
endpoint = model.deploy(
    machine_type="g2-standard-12", accelerator_type="NVIDIA_L4", accelerator_count=1,
    min_replica_count=0, max_replica_count=2,    # scale-to-zero, same as § 9.1
)
```

The console equivalent is literally a **Deploy** button on the model's Garden card. Either way you land
exactly where § 9.1 left you — a managed endpoint with autoscaling and traffic split — having written
no serving code.

## MaaS: skip deployment entirely

For many open models (Llama, DeepSeek, Qwen, and others), Vertex offers **Model-as-a-Service**: a
serverless, OpenAI-compatible endpoint Google operates, billed **per token** like any hosted API.[^maas] No
GPU, no endpoint, no scaling — you call it. It's the open-model equivalent of calling Gemini.

```python
# MaaS / Gemini both speak the OpenAI API via Vertex — no infra at all
import openai, google.auth, google.auth.transport.requests
creds, project = google.auth.default()
creds.refresh(google.auth.transport.requests.Request())
client = openai.OpenAI(
    base_url=f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project}/locations/us-central1/endpoints/openapi",
    api_key=creds.token)
resp = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct-maas",
    messages=[{"role": "user", "content": "Why is decode memory-bound?"}])
print(resp.choices[0].message.content)
```

!!! key "MaaS vs self-deploy is a per-token vs per-hour decision"
    The crossover is **utilization**. MaaS bills per token with zero idle cost — unbeatable at low or
    bursty volume, and it starts instantly (no cold start, no quota for GPUs). Self-deploy bills per
    GPU-hour — cheaper *per token* once a GPU is busy enough, and it gives you **fixed latency, VPC
    isolation, version pinning, and no shared rate limits**. Rule of thumb: prototype and low traffic →
    **MaaS**; steady high traffic or a latency/compliance requirement → **self-deploy**. It's the same
    "rent vs run" trade as the chapter's ladder, one rung lower.

!!! note "Gemma vs Gemini, in one line"
    **Gemma** = Google's *open-weight* models you can self-deploy or hit via MaaS (and run anywhere,
    including the GKE lab in Ch. 8). **Gemini** = Google's *proprietary frontier* model, API-only via
    Vertex. Reach for Gemma when you need the weights (customization, on-prem, cost control); reach for
    Gemini when you want the strongest managed model and don't care where it runs.

## Try it: deploy Gemma, then call a model with zero infra

```bash
# A) self-deploy Gemma from Model Garden (CLI) — lands on a managed endpoint
gcloud ai model-garden models list --model-filter=gemma
gcloud ai model-garden models deploy \
  --model=google/gemma-3-9b-it --region=us-central1 \
  --machine-type=g2-standard-12 --accelerator-type=NVIDIA_L4 \
  --min-replica-count=0 --max-replica-count=2     # one command → a serving endpoint

# B) or skip all of that — call an open model via MaaS, per token, no GPU
curl -s -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/us-central1/endpoints/openapi/chat/completions" \
  -d '{"model":"meta/llama-3.1-8b-instruct-maas",
       "messages":[{"role":"user","content":"hello"}]}'
```

Run both and feel the rung difference directly: (A) gives you a dedicated GPU endpoint you control and
pay hourly for; (B) returns a token in a second with nothing provisioned and a per-token bill. Same
model family, opposite ends of the operate-vs-rent trade.

---

Between "call an API" and "run a dedicated endpoint" sits one more rung worth knowing: serverless
*containers* with a GPU attached — Cloud Run — plus what it takes to bring your own serving image to
Vertex.

[^mgserve]: Google Cloud docs — *Choose an open model serving option* (prebuilt serving containers: vLLM for GPU/TPU, Hex-LLM for TPU, TGI; self-deploy vs MaaS): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/choose-serving-option>
[^maas]: Google Cloud docs — *Use open models with Model as a Service (MaaS)* (serverless, per-token, OpenAI-compatible; Llama/DeepSeek/Qwen/Mistral/Gemma): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-maas>

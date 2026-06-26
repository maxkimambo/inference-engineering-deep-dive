# Client Code

Inference engineering draws on CUDA, Kubernetes, and everything between — but one area is routinely
overlooked when optimizing latency: the **client**. The code making the request is half of every
call, and it sits on the user's side of the network, where the latency they actually feel is decided.

Two sides to any call:

- **Client** — the browser, agent, or app making the request.
- **Server** — the inference service that handles it and returns results.

!!! key "On-server time is only a fraction of end-to-end latency"
    ```
     │  network  │ queue │       model server (prefill+decode)       │  network  │
     └────────── what the USER feels (end-to-end) ──────────────────────────────┘
    ```
    Everything in Chapters 2–6 optimized the *model server* box. But the user experiences the whole
    bar. The network hops, the queue wait, and the client's own overhead can rival the inference time
    — and no amount of kernel tuning touches them.

The industry-standard client is the **OpenAI SDK**, which works against many compatible providers,
not just OpenAI. Frameworks like LangChain, Vercel AI SDK, LiteLLM, and LlamaIndex also act as
clients. Whether you use a library or your own code, the client can introduce **latency overhead** or
**throughput bottlenecks** — and for real-time apps you may need a protocol beyond plain HTTP.

## 7.5.1 Client latency overhead

Establishing a session between client and server — depending on the connection and protocol — takes a
few dozen milliseconds.

!!! key "Session setup can eat 10% of your latency budget"
    In a system with a **300 ms p95** end-to-end SLA, a **TLS handshake** alone can cost ~30 ms —
    **10% of the budget spent before inference even starts.** The fix: **reuse sessions** across
    requests from the same client. The OpenAI SDK does this silently under the hood; when you build
    your *own* client for a non-standard modality, you must follow the same best practice yourself.

## 7.5.2 Asynchronous inference

Some systems are built for **throughput, not latency** — bulk document processing, corpus embedding.
These aren't latency-sensitive, so switch to **asynchronous jobs**: a "fire and forget" model.

The problem with synchronous requests: they have a timeout (usually a few minutes), after which they
*fail*. An async job sidesteps that — it **immediately acknowledges** the request and later delivers
the result to a **webhook** supplied in the original call.

```
 sync:   client ──request──► server ...(must finish before timeout)... ──response──► client
 async:  client ──request──► server ──"accepted, id=…"──► client
                                  └─(minutes/hours later)─► POST result to client's webhook
```

Async time limits are measured in **hours, not minutes**. Paired with robust server-side
[queueing](autoscaling.md#queueing), async requests make high-throughput, latency-insensitive systems
far more robust and efficient.

## 7.5.3 Streaming and protocol support

Streaming makes apps feel instant. For LLMs, **streaming text over HTTP is enough** — tokens arrive
as they're generated. But other modalities, especially **live voice and video**, need input *and*
output streams carrying far more data, and a one-shot HTTP request/response doesn't fit.

```
 HTTP:       client ──request──► server ──response──► client ──╳ connection closed
             good for text chat; one round trip, then done

 WebSocket:  client ⇄ handshake ⇄ server ──── continuous bi-directional stream ────⇄
             good for live audio/video; stays open
```

The two common bi-directional streaming protocols:

| Protocol | Data | Trade-off |
|----------|------|-----------|
| **WebSockets** | unstructured, real-time (audio streams) | flexible; *no* schema enforcement — the server parses/validates downstream |
| **gRPC** | structured, well-defined service-to-service | schema-enforced (no parsing burden); the validation layer makes it slightly slower than WebSockets |

!!! warning "WebSocket concurrency is a hard, per-replica limit"
    A server supports up to a fixed, developer-configured number of concurrent WebSocket connections.
    Once that's reached, **new connections can't be established** until a slot frees or another replica
    scales up. For live-streaming modalities, connection count — not just token throughput — becomes a
    scaling dimension your [autoscaler](autoscaling.md) must account for.

---

That completes the production picture. The Chapters 2–6 work makes a single replica fast; Chapter 7 is
everything that turns one fast replica into a fleet that stays fast, available, and affordable under
real, global, spiky traffic — and remembers that the user feels the *whole* path, not just the GPU.

For a concrete, end-to-end application of this chapter on Google Cloud, see the hands-on:
[**A Quantization Pipeline on GKE**](quantization-pipeline-gke.md).

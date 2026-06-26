# Disaggregation

Disaggregation is the technique that takes the single most important fact in this book —
**prefill and decode have opposite performance profiles** — and acts on it structurally, by running
them on *separate hardware*.

It combines three ideas you've now met:

1. **Prefill is compute-bound (sets TTFT); decode is memory-bound (sets TPS).** Different bottlenecks.
2. **Specialization improves performance** — from kernel choice to engine tuning.
3. **You can parallelize serving across many GPUs/nodes** if you avoid interconnect bottlenecks.

## The problem it solves: prefill and decode interfering

Run prefill and decode on the *same* node under heavy traffic and they fight. Ideally prefill (hungry
for compute) and decode (hungry for memory bandwidth) would coexist, each using the resource the other
doesn't. But with larger batches and more compute-intensive optimizations, **they start competing** — a
big prefill stalls everyone's decode tokens, and vice versa.

```
 co-located:   [ prefill ⚔ decode ]  on one GPU  → they contend, TTFT and TPS both suffer
 disaggregated:[ prefill ] ──KV──► [ decode ]    → each engine specialized, no contention
```

## 5.5.1 How it works

**Disaggregation** (disaggregated serving) splits prefill and decode into **separate engines on
separate GPUs or nodes.** Inference becomes three steps:

1. The **prefill engine** takes the input, builds the KV cache, and computes the **first token**.
2. It **transfers the KV cache** over the hardware interconnect to a **decode engine**.
3. The **decode engine** generates all remaining tokens.

```
   PREFILL workers          DECODE workers
   ┌───────┐  first token ──────────────► [ remaining tokens → user ]
   │ ████  │ ───── KV cache ─────► ┌──────┐
   └───────┘   (over interconnect) │ ██   │
   compute-heavy, low TP           memory-heavy, high TP
```

The payoff beyond ending contention: **you tune each engine independently.** The compute-bound prefill
engine can run a *lower* TP (it's not latency-bound the same way); the memory-bound decode engine runs a
*higher* TP for fast token generation. Two specialists beat one generalist.

### Conditional disaggregation

Naively transferring every request's KV is wasteful for short prompts. **Conditional disaggregation**
sends the request to the **decode engine first**, which checks:

- Is the input already cached, or short enough to prefill locally? → **handle it locally**, skip the
  transfer.
- Otherwise → **hand off to the prefill engine** for disaggregated serving.

This adapts to real, mixed traffic instead of paying the KV-transfer tax on every request.

## 5.5.2 When to use it

Disaggregation is powerful but costs **multiple GPUs and real engineering**. Reach for it **only when
all three hold**:

1. **High volume** — roughly **100M–1B+ tokens/day** (depending on model size).
2. **Large model** — **≥100B parameters**.
3. **Prefill-heavy traffic** — long input sequences.

!!! warning "If the conditions don't hold, you're burning money"
    - Miss (1) or (2) → you're paying for extra hardware for minimal gain.
    - Miss (3) — traffic is short-sequence or prefix-cache-heavy → you're better off spending those GPUs
      on **horizontal replicas**, since decode engines are more efficient for short sequences and cache
      hits anyway.

    **The textbook fit:** a frontier (trillion-param) LLM in a **code editor**, where many developers
    paste large, varied chunks of code as context — tons of tokens, mostly prefill, on a huge model.

## 5.5.3 Dynamic disaggregation with NVIDIA Dynamo

Real traffic is heterogeneous and shifts over the day, so a fixed prefill/decode split is suboptimal.
**NVIDIA Dynamo** provides production-ready, *dynamic* disaggregation:

- A **prefill queue** to hold requests when all prefill engines are saturated.
- **Conditional disaggregation** with prefill routing based on configurable thresholds (input-sequence
  length after prefix-cache, queue size).
- Efficient **NIXL-based KV transfer** from prefill to decode, including a kernel to transpose KV blocks
  when the two engines run **different TP configurations** (which they will, since you tuned them
  separately).

Together these let the **number of prefill and decode engines flex at runtime** to track changing
traffic.

!!! info "Reading `xPyD` notation"
    Disaggregation is **not** one-to-one. Real systems run multiple of each, written **`xPyD`** — *x*
    prefill engines, *y* decode engines. **`5P3D`** = five prefill + three decode engines serving one
    model. A prefill-heavy code-assistant might run more P's; a chat workload more D's. Dynamic
    disaggregation means that ratio is a runtime dial, retuned as usage changes.

---

That completes the techniques. Each one narrows the problem to buy speed: quantization fixes precision,
speculation bets on predictability, caching assumes shared prefixes, parallelism splits the model,
disaggregation specializes the hardware. The art — per the [chapter intro](index.md) — is composing a
*balanced* set for your specific traffic, where the symbiotic gains outweigh the antagonistic ones.

**Next:** [Chapter 6 · Modalities →](../modalities/index.md)

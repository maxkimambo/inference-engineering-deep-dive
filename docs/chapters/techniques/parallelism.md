# Parallelism

Every frontier LLM is **too big for one GPU**. GPUs grew, but models grew faster, and that won't
reverse. When the weights plus KV cache exceed a single device, you split the model across many — and
the whole game becomes doing that *without* drowning in inter-GPU communication.

## First, size the problem

A clean rule of thumb: **in FP8, one billion parameters of weights ≈ 1 GB of VRAM** (FP8 = 1 byte each).

Take **DeepSeek-V3.1**, 671B parameters. The weights alone are ~671 GB — a single 180 GB B200 OOMs
instantly. But weights are only half the story; you must also fit the **KV cache**, which often eats
**80%+ of the *remaining* VRAM**. So the real sizing multiplies weights by a KV allowance and rounds up
to the next instance:

```python
# Minimum GPUs for DeepSeek in FP8
bits_precision    = 8        # FP8
params            = 671      # billions
kv_cache_alloc    = 1.8      # weights + ~80% headroom for KV cache

vram_required = (bits_precision / 8) * params * kv_cache_alloc
              # = 1 * 671 * 1.8 ≈ 1200 GB

b200_sizes = [180, 360, 720, 1440]        # 1, 2, 4, 8 GPUs
round_up(vram_required, b200_sizes)        # → 1440 GB  =  8× B200 (a full node)
```

Four B200s (720 GB) could *technically* hold the 671 GB of weights — but with no room for KV cache,
they couldn't serve any real sequence length or batch size. **A full node of eight B200s** is the
minimum to serve DeepSeek on production traffic. And for midsize models (GPT-OSS) you often want *more*
than the minimum anyway, to enable bigger KV caches and better per-user latency.

## The constraint that shapes everything: communication

Splitting a model means the GPUs must constantly exchange intermediate results. From Chapter 3, the
interconnects are **NVLink/NVSwitch within a node** and **InfiniBand between nodes** — both fast, but a
*fraction* of VRAM bandwidth.

!!! key "Why parallelism is a topology problem"
    Decode is memory-bound and latency-sensitive; every cross-GPU exchange adds latency at
    interconnect speed, not VRAM speed. So multi-GPU inference must be **designed around the
    interconnect** — keep chatty communication on fast NVLink, push only what must cross the slow
    InfiniBand between nodes. This discipline is called **topology-aware parallelism**, and it's why the
    *right* split depends on your hardware layout, not just the model.

## The three forms

| Method | Mechanism | Drawback | Best for |
|--------|-----------|----------|----------|
| **Pipeline (PP)** | split *layers* across GPUs | pipeline bubbles → poor latency/utilization | multi-node only |
| **Tensor (TP)** | split *tensors within each layer* across GPUs | heavy sync (all-reduce) every layer → needs fast interconnect | **low-latency, single node** |
| **Expert (EP)** | shard whole *MoE experts* across GPUs | needs token routing between GPUs | **throughput, MoE models** |

### 5.4.1 Tensor parallelism — for latency

**TP is your default for multi-GPU inference.** It splits *every layer* apart and distributes the
fragments, so the cost of reading weights and doing the matmul for each layer is *shared* across GPUs.
Works for dense (Llama 405B) and MoE alike.

```
 Tensor Parallelism — each layer's weights sliced across 8 GPUs
   layer L:  [ B200 | B200 | B200 | B200 | B200 | B200 | B200 | B200 ]
              each holds a slice of L's weight matrix, computes its part,
              then ──► all-reduce ──► combine into one output ──► next layer
```

The catch: after each layer, the partial results must be combined in an **all-reduce** before the next
layer can start — a synchronization across *all* TP GPUs, *every layer*. On fast intra-node
NVLink/NVSwitch this overhead is minimal; across slow InfiniBand it's crippling.

!!! key "More TP = lower per-user latency (within a node)"
    Increasing tensor parallelism raises TPS per user — *assuming* the model is large enough and
    sequences long enough that the faster forward pass outweighs the all-reduce cost (true for most
    frontier models). But because the all-reduce is so chatty, **TP wants to stay inside one node.**
    That single fact drives the multi-node strategy below.

### 5.4.2 Expert parallelism — for throughput

For MoE models, **EP places whole experts on different GPUs.** 128 experts in "EP8" across 8 GPUs →
each GPU hosts 16 full experts.

```
 Expert Parallelism — experts distributed; router replicated
   B200: E0  E1        B200: E2  E3
   B200: E4  E5        B200: E6  E7    …  each token routed to its experts
   (the tiny router is copied to every GPU, not split)
```

Each token still takes just as long, but the *system* handles **more simultaneous tokens** — pure
throughput. And critically, **EP needs less communication than TP**: the router is small and replicated
per-GPU, so the only cross-GPU traffic is passing tokens to their experts — there's no per-layer
all-reduce to collect results. That lighter footprint lets **EP scale to multi-node** and to systems
with limited interconnect.

!!! info "Real deployments mix TP and EP"
    The common frontier-MoE pattern: **TP for the attention layers** (latency-sensitive, dense) and
    **EP for the sparse MoE layers** (throughput-sensitive). One deployment, both benefits — TP where
    you need low latency, EP where you need scale. (Context Parallelism, a third data-parallel axis, is
    rare in LLM inference but essential for video — see Chapter 6.6.)

### 5.4.3 Multi-node inference

When even eight GPUs aren't enough — a huge model at high precision, multi-million-token inputs, or just
chasing maximum speed — you cross nodes over InfiniBand. Two challenges: **infrastructure** (reliably
provisioning interconnected nodes across clouds — Chapter 7) and **parallelism** (communicating
efficiently over InfiniBand, which is much slower than NVLink).

Because TP's per-layer all-reduce is too chatty for InfiniBand, you combine strategies:

- **Dense models → `TP8PP2`**: tensor parallelism *within* each node (fast NVLink), pipeline parallelism
  *between* nodes (only layer-boundary data crosses InfiniBand). Generally the lowest per-user latency.
- **MoE models → `EP16`**: expert parallelism across all 16 GPUs, since EP's lighter communication
  tolerates InfiniBand. Generally the highest system throughput.

```
   FIRST NODE  [ 8× B200, NVLink/NVSwitch internally ]
                          │  InfiniBand (slow — minimize traffic across it)
   SECOND NODE [ 8× B200, NVLink/NVSwitch internally ]
```

!!! warning "Don't reach for multi-node too early"
    Unless the model *and* its KV cache genuinely exceed one node, multi-node is usually a poor use of
    hardware — you pay InfiniBand latency for little gain. You're often better off using extra nodes for
    **horizontal scaling** (more replicas) or **disaggregated serving** (next section) than for making a
    single replica span nodes.

**Next:** [Disaggregation →](disaggregation.md)

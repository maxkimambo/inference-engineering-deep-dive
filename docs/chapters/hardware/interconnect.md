# Interconnect: Scaling Beyond One GPU

The moment a model or its KV cache outgrows a single GPU — which, as § 3.2 showed, happens at 70B in
BF16 — you need a second GPU, and the wire between them becomes a new ceiling on the roofline. This
section is the *physical layer* underneath Chapter 5's parallelism. Chapter 5 decides **what** to
split (tensors, experts, layers); this section is about **the wire that split data has to cross**,
and why choosing the wrong wire makes multi-GPU inference slower than single-GPU.

## The interconnect hierarchy

Just like memory, interconnect is a hierarchy — fast-and-local to slow-and-distant — and the cliffs
between tiers are enormous. Per GPU, roughly:

| Link | Connects | Bandwidth (per GPU) | Latency | Tier |
|------|----------|--------------------:|---------|------|
| **NVLink 5** (Blackwell) | GPU ↔ GPU, same node | **1,800 GB/s** | ~sub-µs | scale-**up** |
| **NVLink 4** (Hopper) | GPU ↔ GPU, same node | **900 GB/s** | ~sub-µs | scale-**up** |
| **NVLink 3** (Ampere) | GPU ↔ GPU, same node | 600 GB/s | ~sub-µs | scale-**up** |
| **PCIe 5.0 ×16** | GPU ↔ CPU/host, or GPU↔GPU w/o NVLink | ~64 GB/s | ~µs | host link |
| **InfiniBand NDR** (per port) | node ↔ node, over a switch fabric | ~50 GB/s | ~1–2 µs | scale-**out** |
| **RoCE** (RDMA / Ethernet) | node ↔ node | ~25–50 GB/s | ~µs | scale-**out** |

!!! key "Crossing a node boundary is ~14× slower than staying inside one"
    NVLink (intra-node) runs at **900–1,800 GB/s**; PCIe and a single InfiniBand port run at **~50–64
    GB/s**. That ~14× cliff between "same box" and "next box" is the most important fact about
    multi-GPU inference. It dictates *which* parallelism strategy can cross *which* boundary — put a
    chatty, latency-sensitive split across a slow wire and your expensive GPUs sit idle waiting on the
    network.

A few terms you'll meet in a cloud console:

- **NVSwitch** — a crossbar chip that gives *every* GPU in a node a full-bandwidth path to every
  other, so an 8-GPU box behaves like a fully-connected mesh rather than a ring. This is what makes
  intra-node all-to-all collectives fast.
- **NVL72** (Blackwell) — 72 GPUs wired into a *single NVLink domain* across a whole rack, so they
  address each other's memory at NVLink speed. It collapses the node boundary: a rack acts like one
  giant 13 TB-of-HBM GPU. This is NVIDIA's answer to models too big for 8-GPU nodes.
- **Scale-up vs scale-out** — *scale-up* adds GPUs inside one fast NVLink domain (limited count, huge
  bandwidth); *scale-out* adds more nodes over InfiniBand/Ethernet (near-unlimited count, far less
  bandwidth per hop). You almost always scale **up first, out second.**

## Why the wire is a roofline ceiling

Parallelism isn't free: split a model across GPUs and they must *exchange activations* at each step.
The cost of that exchange is a **collective communication** operation, and the dominant one for
tensor parallelism is the **all-reduce** — every GPU contributes a partial result and all receive the
summed total. Its traffic is set by the activation size, not the model size, and it happens *twice
per transformer layer* (after attention, after the MLP).

That puts a third resource into the race from Chapter 2. Now the GPU can be idle waiting on **compute**,
**memory**, *or* **the network** — and on a multi-GPU deployment the network is frequently the one
that's pinned.

### Worked example: tensor parallelism over the right wire vs the wrong one

Take a 70B-class model (`hidden = 8192`, `80 layers`, BF16) split with **tensor parallelism** across
GPUs, prefilling a 4,096-token batch. Each all-reduce moves `~2 * B * H * bytes`; two per layer,
across 80 layers:

- Per all-reduce: **134 MB** · per layer: **268 MB** · whole prefill: **21.5 GB** of cross-GPU traffic.

Run that same traffic over different wires:

| Wire | Time spent in communication |
|------|----------------------------:|
| NVLink 5 (1,800 GB/s) | **12 ms** |
| NVLink 4 (900 GB/s) | **24 ms** |
| PCIe 5.0 (64 GB/s) | **336 ms** |
| InfiniBand, 1 port (50 GB/s) | **430 ms** |

!!! key "This is why tensor parallelism must live inside an NVLink domain"
    On NVLink the 24 ms of communication overlaps with hundreds of ms of compute — nearly free. Over
    PCIe the *same split* spends 336 ms shuffling data, often dwarfing the compute and leaving the
    GPUs starved. **Tensor parallelism is so communication-heavy it is only viable across NVLink (or
    NVL72).** If your instance type wires its GPUs together with PCIe instead of NVLink, do not use
    tensor parallelism on it — the wire will eat the win.

### Matching parallelism to the wire

This is the bridge to Chapter 5. Each parallelism strategy has a communication appetite, and you
place it on a wire that can feed it:

| Strategy (Ch. 5) | Communicates | Appetite | Lives on |
|------------------|--------------|----------|----------|
| **Tensor (TP)** | every layer, twice (all-reduce) | very high | NVLink only (intra-node / NVL72) |
| **Expert (EP)** | token routing (all-to-all) | high | NVLink, or fast intra-node |
| **Pipeline (PP)** | once per stage boundary (activations) | low | tolerates PCIe / InfiniBand (scale-out) |
| **Data / replicas** | nothing between replicas | ~none | any — separate nodes entirely |

!!! key "The placement rule"
    Put the **chattiest** split on the **fastest** wire. Tensor-parallel a model *within* an 8-GPU
    NVLink node; pipeline-parallel *across* nodes over InfiniBand; replicate whole nodes behind a load
    balancer with no interconnect between them at all. Get this backwards — TP across nodes, replicas
    fighting over NVLink — and you waste the hardware you paid a premium for.

## The host link and why it bites cold starts

PCIe is also the road between **host RAM/SSD and the GPU**. It rarely limits steady-state inference
(weights already live in HBM), but it dominates two moments: **loading a model** (a 140 GB model over
64 GB/s PCIe is a hard ~2-second floor *before* any HBM-bandwidth or filesystem overhead) and
**offloading** (spilling weights or KV to host memory when HBM is full — possible, but you've just
demoted part of your model to a 14× slower tier, so latency craters). Grace-Hopper's **NVLink-C2C**
(900 GB/s CPU↔GPU) exists precisely to make host memory a usable overflow tier instead of a cliff.
Cold-start latency (Chapter 7) is largely a PCIe-and-storage story.

---

You now have all three ceilings — compute, memory, and interconnect — and the numbers to predict each.
The last section spends them: given a model and a latency target, which accelerator, how many, and
wired how — without overpaying.

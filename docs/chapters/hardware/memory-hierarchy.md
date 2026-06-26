# The Memory Hierarchy

The single most common mistake in GPU inference is reasoning about compute when the bottleneck is
memory. Chapter 2 proved that decode is memory-bound; this section shows you the physical structure
that makes it so, with the real capacities and bandwidths you'll plug into every sizing calculation.

The governing fact is the **memory wall**: over the last two decades, compute (FLOPS) has grown far
faster than memory bandwidth. The calculator keeps getting wider; the hose barely keeps up. So the
GPU is built as a *hierarchy* of memories — a few tiny, blistering-fast ones near the math units and
one enormous, comparatively slow one far away — and the entire art of a fast kernel is keeping data
in the fast tiers and minimizing trips to the slow one.

## The five tiers

From closest-and-fastest to farthest-and-largest, for an **H100**:

| Tier | Capacity | Bandwidth | Relative latency | Managed by |
|------|----------|-----------|------------------|------------|
| **Registers** | ~256 KB / SM (~32 MB chip) | ~tens of TB/s aggregate | ~1× (1 cycle) | compiler |
| **Shared mem / L1** | up to 228 KB / SM | ~tens of TB/s | ~20–30× | the kernel (software) |
| **L2 cache** | 50 MB (whole chip) | ~several TB/s | ~10× of L1 | hardware |
| **HBM (VRAM)** | 80 GB HBM3 | **3.35 TB/s** | ~hundreds of × | you (allocations) |
| **Host RAM** (over PCIe/NVLink-C2C) | 100s of GB–TBs | 64 GB/s (PCIe5) … 900 GB/s (C2C) | ~1000s of × | you (offload) |

Read the pattern, not the exact numbers: **each step down is roughly ~10–100× more capacity and
~10× less bandwidth (and worse latency).** That ratio is the whole reason for the hierarchy's
existence and the source of every memory optimization in this book.

!!! key "The number that governs decode is HBM bandwidth"
    When you generate one token, you must read **every weight in the model** from HBM (you can't do
    the math without the weights, and they don't fit in the fast tiers). So decode tokens-per-second
    is capped by:
    \[
    \text{TPS}_{\text{single stream}} \approx \frac{\text{HBM bandwidth}}{\text{bytes of weights read per token}}
    \]
    This is why **HBM bandwidth (TB/s), not FLOPS, is the headline inference number** for a
    decode-heavy server — and why the H200, with the *same compute die* as the H100 but 43% more
    bandwidth, decodes ~43% faster (§ 3.3).

## Why the fast tiers are so small (and why that shapes algorithms)

SRAM (registers, shared memory, L2) is built from transistors right on the compute die — fast
because it's physically close and electrically simple, but it eats die area, so there's very little
of it. HBM is **stacked DRAM** mounted next to the GPU on the same package, connected by a very wide
bus; it's vastly denser (gigabytes) but each access is far slower. You cannot have both: fast memory
is small, large memory is slow. That is not a current-engineering limitation, it's physics, and it's
permanent.

This constraint *writes the algorithms*. The clearest example is **FlashAttention** (Chapter 2 derived
its motivation; here's the hardware reason it exists). Naive attention computes the score matrix
`S = QKᵀ`, which is `N×N` for sequence length `N`. At `N = 4096` that's 16M numbers — far too big for
the 228 KB of shared memory — so the naive kernel writes `S` out to HBM and reads it back to apply
softmax, then writes `P` and reads it again for `P*V`. That's the `~8N²` bytes of memory traffic that
makes attention memory-bound. FlashAttention never materializes the full `N×N` matrix: it **tiles**
the computation so each block of the score matrix is produced, softmaxed, and consumed *entirely in
shared memory*, streaming a running result. Same FLOPs, a fraction of the HBM traffic. The algorithm
is shaped by the size of the SRAM scratchpad.

!!! key "Fast kernels are memory-movement strategies, not math"
    A matmul's *math* is fixed — you can't multiply two matrices with fewer multiplies. What a good
    kernel optimizes is the **data movement**: load each tile from HBM once, reuse it across as much
    arithmetic as possible while it sits in shared memory and registers, write back once. "Make it
    faster" almost always means "touch HBM less," because HBM is the slow tier everyone is waiting on.

## The capacity gate: will the model even fit?

Bandwidth sets *speed*; capacity sets *possibility*. Before any performance question, the model and
its working state must physically fit in HBM. Two things consume it:

**1. Weights.** Parameters × bytes-per-parameter. For a 70B model:

| Precision | Bytes/param | Weights |
|-----------|-------------|---------|
| BF16 / FP16 | 2 | **140 GB** |
| INT8 / FP8 | 1 | **70 GB** |
| INT4 | 0.5 | **35 GB** |

A single 80 GB H100 **cannot hold a 70B model in BF16** (140 > 80). Your options are immediate and
concrete: quantize to INT8 (70 GB, barely fits — with almost no room left), quantize to INT4 (35 GB,
comfortable), or split the model across multiple GPUs (Chapter 5, parallelism). This single
inequality — *weights vs HBM capacity* — drives more deployment decisions than any other number.

**2. The KV cache.** Every token you've processed leaves behind cached keys and values for every
layer (Chapter 2). Its size per token:
\[
\text{KV bytes/token} = 2 \times n_{\text{layers}} \times n_{\text{kv-heads}} \times d_{\text{head}} \times \text{bytes}
\]
For Llama-3 70B (`80 layers, 8 KV heads, d_head 128`, BF16): **320 KB per token**. That sounds tiny
until you multiply by context length and batch size:

| Scenario | KV cache |
|----------|----------|
| 1 sequence, 8K context | **2.7 GB** |
| Batch of 32, 8K context | **86 GB** |

!!! key "The KV cache is not a footnote — it rivals the weights"
    At batch 32 / 8K context, the KV cache (86 GB) is *larger than the INT8 weights* (70 GB). This is
    why memory capacity, not compute, caps how many concurrent requests you can serve: every
    in-flight sequence rents HBM for its entire context. It's also why the KV cache is the prime
    target for its own optimizations — paging (PagedAttention), quantization, and GQA, which shrinks
    `n_kv-heads` (Chapter 5). When someone says "we ran out of memory at batch 40," they mean the KV
    cache hit the HBM wall, not the weights.

## Worked example: fit and speed on one page

Put both gates together for **Llama-3 70B** and watch the hardware choice fall out of arithmetic.

**Fit (capacity gate):**

- BF16 weights = 140 GB → needs ≥ 2× 80 GB GPUs, or 1× H200 (141 GB) with *zero* room for KV — useless.
- INT8 weights = 70 GB → fits one 80 GB GPU with ~10 GB spare → enough KV for only a handful of short
  sequences. Fine for a demo, not for a server.
- Realistic single-GPU serving config: **INT4 weights (35 GB)** leaves ~45 GB for KV cache → ~140k
  tokens of KV budget (≈ batch 17 at 8K context). Now it's a server.

**Speed (bandwidth gate)** — single-stream decode TPS ≈ HBM bandwidth ÷ weight bytes read per token:

| GPU | HBM BW | BF16 (140 GB) | INT8 (70 GB) |
|-----|--------|---------------|--------------|
| A100 80 GB | 2.04 TB/s | 15 tok/s | 29 tok/s |
| H100 | 3.35 TB/s | 24 tok/s | 48 tok/s |
| **H200** | 4.8 TB/s | 34 tok/s | 69 tok/s |
| B200 | 8.0 TB/s | 57 tok/s | 114 tok/s |
| MI300X | 5.3 TB/s | 38 tok/s | 76 tok/s |

Two lessons drop straight out of the table:

1. **Halving precision ~doubles decode speed** — because decode is bandwidth-bound and you're moving
   half the bytes. (This is the hardware mechanism behind Chapter 5's quantization wins; it's not
   only about fitting, it's about *speed*.)
2. **Bandwidth, not the brand, sets single-stream latency.** The H200 beats the H100 by exactly its
   bandwidth ratio (4.8/3.35 ≈ 1.43) despite *identical* compute. If you were chasing the compute
   number you'd have called them equal.

!!! note "These are ceilings, not benchmarks"
    The formula gives the *upper bound* (memory-bound, single stream, weights-only). Real engines hit
    60–80% of it and add KV-cache reads on top. Use it to *rank* options and sanity-check vendor
    claims, not as a guaranteed SLA. If a vendor quotes single-stream TPS far above `BW / weight_bytes`,
    they're batching — which raises throughput, not per-stream latency (Chapter 2).

---

You can now answer "will it fit?" and "how fast per stream?" from a spec sheet and a model card. The
remaining ceiling — the one that governs prefill, batching, and image generation — is raw compute,
and it comes wrapped in the most misread number on any datasheet. That's next.

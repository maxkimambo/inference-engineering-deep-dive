# Anatomy of a GPU

Before you can predict what a GPU will do with your model, you need to know what's physically inside
it — and *why* it's built that way. The architecture isn't arbitrary. Every design choice is an
answer to one question: **how do you do the most arithmetic per second on data you've already got?**
A transformer forward pass is, at the bottom, a pile of matrix multiplications. A GPU is a machine
that exists to make matrix multiplications cheap. Once you see it that way, the spec sheet stops
being a list of acronyms and becomes a description of a calculator.

## Why a GPU and not a CPU

Start from first principles. A **CPU** is a *latency* machine: it's built to finish a single chain
of dependent instructions as fast as possible. It spends most of its transistor budget not on
arithmetic but on *making one thread fast* — deep caches, branch predictors, out-of-order
execution, speculative loads. A modern CPU has maybe 8–64 cores, each very clever.

A **GPU** is a *throughput* machine. It assumes you have not one task but *thousands of identical
tasks*, and it asks: if I don't care how long any single one takes, only how many finish per second,
what should I build? The answer is the opposite of a CPU. Strip out the cleverness, spend every
transistor on arithmetic units, and run thousands of dumb threads in lockstep.

!!! key "Latency vs throughput is the whole reason GPUs exist"
    A CPU optimizes **time to finish one thing**. A GPU optimizes **things finished per unit time**.
    A matmul of a 4096×4096 matrix is millions of independent multiply-accumulates — there is no
    dependency chain to make fast, only a mountain of identical work to spread wide. That is the
    perfect throughput workload, and it is exactly what a transformer is made of.

The concrete trade: a CPU might do ~64 floating-point lanes of work at once; an H100 does **tens of
thousands**. It "wastes" no silicon on predicting branches because the work has almost none.

## SIMT: thousands of threads, one instruction

The GPU's core trick is **SIMT — Single Instruction, Multiple Threads**. The hardware groups threads
into bundles of 32 called a **warp**, and every thread in a warp executes the *same instruction* at
the *same time*, each on its own slice of data. One instruction decode drives 32 arithmetic results.
That's how you get arithmetic density: you amortize all the control logic (fetch, decode, schedule)
across 32 lanes instead of paying it per lane like a CPU does.

This is why GPUs love regular, branch-free work. If half the threads in a warp take an `if` and half
take the `else`, the hardware must run *both* sides and mask off the inactive lanes — **warp
divergence**, where you pay for work you throw away. A matmul has no branches, so warps never
diverge. This is also why irregular, branchy code (parsing, tree traversal) runs terribly on GPUs
and beautifully on CPUs: each is built for the opposite shape of problem.

## The Streaming Multiprocessor (SM)

A GPU is not one giant calculator; it's a tiled array of identical small ones called **Streaming
Multiprocessors (SMs)**. The SM is the fundamental unit of a GPU the way a core is the fundamental
unit of a CPU. An H100 has **132 SMs**; an A100 has **108**. Scaling a GPU generation up mostly means
*adding more SMs* (and feeding them with more memory bandwidth).

Inside each SM:

- **CUDA cores** — the general-purpose lanes. Each does one floating-point or integer operation per
  clock (e.g. an FP32 multiply-add). These handle the "everything else": activations, normalization,
  the elementwise math between the big matmuls. Plentiful but, by modern standards, *not* where the
  matmul horsepower is.
- **Tensor Cores** — specialized matrix-multiply units (below). This is where ~95% of a transformer's
  FLOPS actually happen.
- **Register file** — a large, ultra-fast bank of registers, partitioned among the warps currently
  resident. This is the fastest memory on the chip (§ 3.2).
- **Shared memory / L1** — a small scratchpad (up to ~228 KB on Hopper) that threads in a block can
  share, software-managed. The thing FlashAttention exploits to avoid round-trips to main memory.
- **Warp schedulers** — pick which warp runs each cycle. The SM keeps *many* warps resident at once
  and switches between them instantly: when one warp stalls waiting on memory, another runs. This
  **latency hiding** is how a GPU stays busy despite slow memory — it always has other work queued.

!!! note "Occupancy, in one sentence"
    **Occupancy** is how many warps an SM has resident relative to its maximum. High occupancy gives
    the scheduler more warps to hide memory latency with. It's a knob engine authors tune; you'll see
    it in profiler output, and "low occupancy" is a common diagnosis for a kernel that's leaving the
    GPU idle.

## The Tensor Core: a matrix-multiply in one instruction

The single most important piece of hardware for this entire book is the **Tensor Core**. A CUDA core
multiplies two numbers. A Tensor Core multiplies two small *matrices* and accumulates the result —
in one operation, in one clock region of time.

Concretely, a Tensor Core computes `D = A * B + C` where `A`, `B`, `C`, `D` are small tiles (e.g.
4×4, 8×8, or larger depending on precision and generation). One instruction does dozens to hundreds
of multiply-accumulates. Stack thousands of Tensor Cores across 132 SMs and you get the headline
number: an H100 does **989 trillion** FP16 multiply-accumulates per second (dense — we'll dissect
that number in § 3.3).

!!! key "Why the Tensor Core exists, in roofline terms"
    A matmul of two N×N matrices does `~2N³` FLOPs but only touches `~3N²` numbers — its arithmetic
    intensity grows with N. That makes large matmuls *compute-bound*, the one case where throwing
    more math units at the problem actually helps. The Tensor Core is silicon built precisely for the
    high-intensity corner of the roofline. It does nothing for memory-bound work — which is why a
    Tensor Core monster like the H100 still can't make single-stream decode fast (§ 3.2). The fix
    there is batching, to *raise intensity* until the Tensor Cores have enough to chew.

This is also the deep reason **precision matters on hardware, not just in theory**: a Tensor Core
fed 8-bit inputs instead of 16-bit can pack twice as many values into the same datapath and roughly
*doubles* its throughput. The number format isn't an accounting detail — it's a physical setting on
the calculator (§ 3.3, and Chapter 5's quantization).

### The Transformer Engine

From Hopper (H100) onward, NVIDIA wraps the Tensor Cores in a **Transformer Engine**: hardware +
library logic that runs matmuls in **FP8** while watching the numerics, automatically choosing
between the two FP8 layouts (`E4M3`/`E5M2`) and rescaling per layer to keep the dynamic range from
overflowing. The payoff is the FP8 throughput line on the datasheet (≈2× the FP16 number). You don't
program it directly — the inference engine does — but when a deployment guide says "use FP8 on
Hopper," this is the unit doing the work.

## Mapping one matmul onto the machine

Tie it together. Suppose you multiply the activation `X` (shape `[tokens, 4096]`) by a weight matrix
`W` (shape `[4096, 4096]`) — one projection inside one transformer layer. What physically happens:

1. The engine launches a **grid** of thread blocks; each block is assigned a **tile** of the output
   matrix.
2. Each block loads its slice of `X` and `W` from **HBM** → **L2** → **shared memory** (§ 3.2). This
   movement is the expensive part.
3. Warps in the block feed their tiles into the **Tensor Cores**, which stream multiply-accumulates,
   keeping partial sums in **registers**.
4. The finished output tile is written back down the hierarchy to HBM, to become the input of the
   next layer.

!!! key "The two things that can go wrong, and where this chapter goes next"
    Either the **Tensor Cores run out of work** (memory can't deliver tiles fast enough → memory-bound,
    the decode story) or they **have plenty of work but it's still slow** (you need more or faster
    Tensor Cores → compute-bound, the prefill/training story). Step 2 above — the trip through the
    memory hierarchy — is where most inference time actually goes. That's why the next section is
    memory, not compute.

## MIG: slicing one GPU into many

One more architectural feature you'll meet in a cloud console. **Multi-Instance GPU (MIG)** lets you
partition a single A100/H100/H200 into up to **7 isolated instances**, each with a dedicated, fenced
slice of SMs, L2 cache, and HBM. It's the inverse of the rest of this chapter: instead of ganging
GPUs together, you carve one apart.

When it helps: serving a *small* model where a whole 80 GB GPU is overkill, and you want hard
isolation between tenants (each instance can't touch another's memory or steal its compute). When it
doesn't: anything that needs more memory or bandwidth than one slice provides — a MIG slice has a
*fraction* of the GPU's HBM and bandwidth, so a memory-bound decode workload on a slice is
proportionally slower. For most single-model inference servers you'll run the whole GPU; MIG earns
its place in multi-tenant platforms and dev/test fleets.

---

You now know the calculator: thousands of throughput-optimized lanes, organized into SMs, with
Tensor Cores doing the matmul heavy lifting, all gated by how fast data arrives. That last clause is
the entire next section.

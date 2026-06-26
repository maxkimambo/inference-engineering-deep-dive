# Chapter 2 · Models

This is the chapter where inference stops being a black box. By the end you will be able to take
a single token, trace it through every matrix multiply in a transformer, and explain *exactly*
which step is slow and why.

We cover two model families:

- **LLMs** — autoregressive token generators. They produce one token at a time, each conditioned
  on every token before it.
- **Image and video models** — iterative denoisers. They start from noise and refine a whole
  canvas over many steps.

They look unrelated, but at the bottom they're the same machinery — matrix multiplies and
attention — arranged differently. Understanding one makes the other easy.

## Learning objectives

By the end of this chapter you can:

- [x] Explain why a neural network needs **non-linear activations** (and what breaks without them)
- [x] Trace a token: **text → token id → embedding → transformer blocks → logits → next token**
- [x] Describe what the **KV cache** is, why it exists, and what it costs
- [x] State why **prefill is compute-bound** and **decode is memory-bound** — and prove it with
  arithmetic intensity
- [x] Read a model's `config.json` and predict its memory footprint and bottleneck

## The sections

<div class="grid cards" markdown>

-   :material-graph-outline: __[Neural Networks](neural-networks.md)__

    Nodes, layers, matmul, and the one trick (non-linearity) that makes depth worth anything.

-   :material-cog-transfer: __[LLM Inference Mechanics](llm-inference-mechanics.md)__

    Tokenization, embeddings, the transformer block, attention, the KV cache, and the
    prefill/decode split. The core of the book.

-   :material-image-multiple: __[Image & Video Generation](image-video-generation.md)__

    Diffusion, latent space, the VAE, classifier-free guidance, and why a "50-step" image is
    actually 100 forward passes.

-   :material-speedometer: __[Calculating Bottlenecks](bottlenecks.md)__

    Ops:byte ratio, arithmetic intensity, and the roofline model — the math that tells you
    whether to buy compute or bandwidth.

</div>

!!! key "The one idea to hold onto"
    Everything in inference is a fight between two resources: **how fast the GPU can do math**
    (compute) and **how fast it can move numbers in and out of memory** (bandwidth). Every
    technique in Chapter 5 is a move in that fight. This chapter teaches you to see which one
    you're losing.

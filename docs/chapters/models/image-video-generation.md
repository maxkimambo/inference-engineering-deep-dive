# Image & Video Generation

LLMs generate *sequences* one token at a time. Image and video models generate a *whole canvas* by
iterative refinement. Different shape, same underlying machinery — matmuls and attention. This
section builds the mental model so the modality-specific optimizations in Chapter 6 make sense.

## Not a model — a pipeline

An image generator is not one monolithic network. It's a **pipeline of three** networks:

- **Text encoder** — turns the prompt into a representation the rest of the pipeline can condition
  on. Modern pipelines use a full LLM here (e.g. a 7B model), which is why prompt understanding has
  improved so much.
- **Denoising model** — the heart. Iterates from noise toward an image, guided by the encoded
  prompt.
- **Variational Autoencoder (VAE)** — translates between **latent space** (compact, where the
  denoiser works) and **pixel space** (the final image).

Around the base pipeline sit add-ons you'll see in the wild: **LoRAs** (lightweight fine-tunes for
style/quality) and **ControlNets** (steer output to match input edges/poses). Tools like ComfyUI
let you wire these together like a modular synth.

## Latent space: why we don't denoise pixels

An HD image is 1024×1024 = over a million pixels. The denoiser computes attention over the *entire*
canvas every step; doing that in pixel space is infeasible.

- **Latent space** — a compact, lower-dimensional encoding of an image. A latent might be 128×128
  — about **1%** of the pixel count — while preserving the structure that matters.

Recall the encoder/decoder framing from [Neural Networks](neural-networks.md): text inference
*grows* dimensionality (token → big vector); image inference *shrinks* it (million pixels →
small latent). The VAE is the encoder/decoder that moves between the two worlds. The whole
generation happens in latent space; the VAE only runs at the end to expand the final latent back
to pixels.

## Diffusion: refining noise into signal

The denoiser is a **diffusion model**. It starts from a latent of pure random **noise** and, over
many **steps**, nudges it toward a coherent image conditioned on the prompt.

```
 step 0        step 12       step 30       step 50
 ░▒▓ noise ──► faint shape ──► rough image ──► sharp image
   (each step updates the WHOLE latent at once, unlike a token loop)
```

Most models take **30–50 steps** for a high-quality image. Contrast with an LLM: an LLM commits one
token per forward pass and never revises; a diffusion model revises the entire canvas every step
and commits nothing until the end.

### Classifier-free guidance: why "50 steps" is 100 passes

Here's the detail that surprises people and explains image-gen cost. Each step runs the denoiser
**twice**:

1. once **with** the prompt (conditioned)
2. once **without** any prompt (unconditioned)

then combines the two, scaled by a **guidance scale**, to push the result toward the prompt. This
is **classifier-free guidance (CFG)**.

!!! key "The 2× multiplier"
    Because of CFG, a **"50-step" generation is actually ~100 forward passes**. When you reason
    about image-gen latency or cost, double the step count in your head. This is also why
    "few-step" models (Chapter 6) — which cut steps to 4–8 — are 80–90% faster.

### The knobs

Image generation is steered per-request:

| Argument | Effect |
|----------|--------|
| **Prompt** | what should be in the image |
| **Negative prompt** | what should *not* be — styles/objects to suppress |
| **Steps** | quality vs speed; 30–50 typical, fewer = faster/rougher |
| **Guidance scale** | prompt adherence vs creativity; often ~4 |
| **Image size** | chosen from a fixed menu of resolutions/aspect ratios |

## Architecture: the diffusion transformer

Modern denoisers are **diffusion transformers** — the same transformer machinery as LLMs, but
processing image **patches** instead of text tokens. During training, images are cut into
overlapping 2×2 or 4×4 patches and embedded into latent space; inference runs the reverse, ending
with the VAE expanding latents to pixels.

A clean reference pipeline is **SDXL**: noise → base denoiser → refiner denoiser → VAE decode →
1024×1024 image. Newer models keep the shape but scale every component up:

| Component | SDXL (2023) | Qwen-Image (2025) |
|-----------|-------------|-------------------|
| Text encoder | CLIP-based | full 7B VLM |
| Denoiser | <4B params | 20B params |
| VAE | single encoding | dual encoding |

The capability jump (legible text rendering, photorealistic hands/faces) came from **bigger text
encoders and ~5× bigger denoisers** — not a new trick, just scale and richer pipelines.

!!! info "The convergence frontier"
    The newest research blends diffusion-transformer and LLM architectures: anything tokenizable
    can be modeled autoregressively. LLM-style image models (e.g. HunyuanImage-3.0) gain
    variable-length output and single-pass generation, sidestepping diffusion's fixed-size,
    many-pass nature. Expect image and text inference to keep converging.

## Video: add a time axis

Video models are architecturally the same as image models, just **3–5× bigger** and encoding
**10–100×** more information in latent space.

- An image latent encodes two spatial dimensions: **X, Y**.
- A video latent encodes three: **X, Y, T** (time).

The naive approach generates frame-by-frame, feeding each frame in to produce the next. It fails:
small errors compound across frames and the video goes off the rails — **error accumulation**.
Modern models instead hold the **entire video in latent space** and denoise all frames jointly;
every frame attends to every other frame and is updated each step.

!!! warning "Why video runs batch-size-1"
    Attention over a massive spatiotemporal latent is brutally expensive — so expensive that video
    models often run with **batch size 1**: a full node of 8 GPUs working a single request. Same
    ~50 steps as images, but each step is enormous. This is the compute-bound extreme.

Where do video models sit on the roofline? Firmly **compute-bound**, like LLM prefill and image
generation — attention over a huge latent is heavy math per byte. Their current limitations (slow
generation, unrealistic physics, short outputs, maxing out the newest GPUs) mirror where LLMs were
in late 2023, and research (e.g. Self Forcing, which blends a global quality view with
autoregressive generation) is closing the gap fast.

---

You now have both model families in one frame: LLMs as autoregressive token loops, image/video as
iterative latent denoisers, both built from matmuls and attention, both readable on the roofline.

**Next:** [Calculating Bottlenecks →](bottlenecks.md)

# Containerization

**Containerization** packages an application *together with its dependencies* into one portable
artifact that runs anywhere — the end of "it works on my machine." For inference, where the
dependency chain is long and fragile, it's not optional; it's how you preserve a known-good build in
an ecosystem that breaks constantly.

A container is **lightweight** because it shares the host's OS kernel (here a *Linux* kernel — not a
CUDA kernel; the word is overloaded). That makes it ideal for shipping an inference service.

For most teams, containerization means **Docker**. The vocabulary:

- **Image** — an executable package containing everything needed to run a piece of software.
- **Container** — a *running* instance of an image, isolating the app and its dependencies.
- **Dockerfile** — the human-readable, machine-executable recipe for building an image.
- **Registry** — a central repository for storing and distributing images. (Docker Hub is to images
  what Hugging Face is to model weights or PyPI is to Python packages. NVIDIA and the cloud providers
  run their own.)

## Images are layers

A Docker image is a stack of **layers** — a base, plus filesystem changes on top:

```
 ┌──────────────────────────┐
 │ container layer  READ-WRITE│  ← ephemeral; runtime changes, lost on exit
 ├──────────────────────────┤
 │ image layer      READ-ONLY │  ← your app code + config
 │ image layer      READ-ONLY │  ← pip-installed deps (torch, vllm…)
 │ image layer      READ-ONLY │  ← system packages
 ├──────────────────────────┤
 │ base image       READ-ONLY │  ← e.g. Ubuntu, or a vLLM/CUDA base
 └──────────────────────────┘
```

- **Base image** — an OS distro (Ubuntu) or a richer image from a registry (itself many layers).
- **Additional layers** — dependencies, app code, config, each added by a Dockerfile instruction.
- **Container layer** — a thin, writable top layer created at runtime. Changes here (files created,
  edited, deleted) are **lost when the container stops**. State must live elsewhere.

!!! key "Don't build from scratch — start from a proven base"
    Inference engines like vLLM and SGLang publish **official base images** for each release, with a
    known-good CUDA/driver/Python stack already assembled. Starting from one of these saves you from
    re-deriving a fragile dependency chain — and it's exactly why the [GKE
    hands-on](quantization-pipeline-gke.md) builds `FROM pytorch/...` rather than a bare OS.

## 7.1.1 Dependency management

Inference dependency chains are long and fragile, and they're built for a *specific GPU architecture
and model*. A container pins many moving runtime components:

- **CUDA toolkit version** — the exact CUDA, cuDNN, and driver versions compatible with your stack.
- **Python packages** — `torch`, `transformers`, `diffusers`, and friends.
- **Inference engine** — the version of vLLM / SGLang / TensorRT-LLM.
- **System packages** — Linux libs like `ffmpeg` (especially for audio/image/video models).

Two best practices govern all of this:

### Pack light

Inference images run to *many gigabytes*. Every gigabyte is slower to build, push, pull, and —
critically — slower to load during a [cold start](autoscaling.md#722-cold-starts). Include **only**
strictly necessary dependencies. (Like backpacking: carry what you need, nothing more.)

### Pin exact versions

A pinned dependency tree keeps runtime behavior identical across environments and makes builds
*reproducible* — the same inputs always resolve to the same image.

```
requirements:
  transformers          # No  — floats to whatever's newest, build is not reproducible
  transformers>=4.40.0  # No  — still unbounded above; a future release can break you
  transformers==4.56.2  # Yes — exact, repeatable, protected from breaking changes
```

Tools like `uv`, `poetry`, or `pip` flag incompatibilities at build time. Once an image builds with
pinned versions, it resolves to the *same* versions forever.

!!! warning "Day-zero model support is the exception that proves the rule"
    Breaking changes cluster around **new model releases** — when a DeepSeek or Qwen drops, the whole
    ecosystem races for day-zero support. Engineers then deliberately build against *overnight builds
    and pre-releases* of dependencies rather than stable pins, accepting bugs to ship support fast,
    and rebuild on stable releases over the following weeks. Pinning is the default; this is the
    knowing, temporary exception.

## 7.1.2 NIMs

**NVIDIA Inference Microservices (NIMs)** are pre-built Docker containers for popular open models —
the done-for-you end of the spectrum. Two kinds:

- **Multi-LLM NIM** — a flexible container that runs a *family* of models on a supported GPU
  architecture.
- **LLM-specific NIM** — an engine tuned for *one* model on *one* GPU configuration, for maximum
  performance.

A NIM is just a container — usable as a starting point, a reference architecture to learn from, or an
out-of-the-box service.

!!! info "When to skip the NIM"
    NIMs are opinionated. If you want **maximum control** — your own quantization recipe, custom
    kernels, a precise dependency tree — you're generally better off building your own container from
    a *less opinionated* base image than bending a NIM to your will. NIMs trade control for
    convenience; pick the side your team needs.

**Next:** [Autoscaling →](autoscaling.md)

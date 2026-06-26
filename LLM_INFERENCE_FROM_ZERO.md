# How LLM Inference Works — From Zero

This document assumes **no prior knowledge of machine learning or LLMs**. It assumes only that
you can read code, know what a file, an array, a number, and computer memory are. Every
ML-specific term is defined the first time it appears, in **bold**, with an analogy and usually
a tiny worked example.

It is long on purpose — a chapter per concept and per step. Read it top to bottom; each chapter
builds on the previous ones. The companion reference docs (`LLM_INFERENCE_LIFECYCLE.md`,
`PREFIX_CACHING_EXPLAINED.md`, `PREFIX_PROMPT_CACHING_API.md`) cover the same material faster,
for when you no longer need the hand-holding.

> **How to read this:** Part I builds the mental furniture (model, token, vector, matrix,
> training, hardware). Part II onward walks the actual lifecycle — loading the model, turning a
> prompt into numbers, the "prefill" pass, the "decode" loop — using the furniture from Part I.
> If a later chapter uses a word you forgot, it's defined in Part I or the Glossary at the end.

---

# PART I — FOUNDATIONS (the furniture you need first)

## Chapter 0 — What is a language model, *really*?

Strip away the hype and a large language model (LLM) does exactly **one** thing:

> **Given some text, predict the next chunk of text.**

That's it. It is a **next-token predictor**. You give it "The capital of France is" and it
predicts that the most likely next chunk is " Paris". Everything else — chat, code, reasoning —
is this one ability applied over and over: predict the next chunk, append it, predict again.

- **LLM (Large Language Model)** — a very large mathematical function that takes a sequence of
  text-chunks and outputs a *probability* for what the next chunk should be. "Large" because it
  has billions of internal numbers (defined below as *weights*).

A useful framing: an LLM is a function `f(text) → a guess about the next piece of text`. The
rest of this document is about (a) what's inside that function and (b) the machinery that runs
it efficiently.

### Why "probability" and not just "the answer"?

The model never outputs a single certain next word. It outputs a **probability distribution**:
a score for *every possible* next chunk, saying how likely each one is.

- **Probability distribution** — a list of options each with a likelihood, where the
  likelihoods add up to 1 (100%). Example: `{" Paris": 0.81, " a": 0.04, " the": 0.03, …}`.

Picking one option from that list is a separate step called **sampling** (Chapter 16). This
split — *predict a distribution*, then *pick from it* — is important and comes back later.

---

## Chapter 1 — Tokens: turning text into numbers

Computers do math on numbers, not letters. So the very first thing any LLM system does is
convert text into numbers. The unit of that conversion is the **token**.

- **Token** — a small chunk of text: often a word, a piece of a word, or a single character.
  "Paris" might be one token; "tokenization" might split into "token" + "ization". Roughly, 1
  token ≈ 4 characters of English ≈ ¾ of a word.
- **Tokenizer** — the component that chops text into tokens and maps each token to a number.
- **Vocabulary** — the fixed, finite list of every token the model knows, each with a unique
  integer id. A typical vocabulary has 30,000–260,000 entries.
- **Token ID** — the integer index of a token in the vocabulary. This is the "number" the model
  actually works with.

So tokenization is a two-way lookup table:

```
 text  ── encode ─►  token ids        token ids ── decode ─►  text
 "Paris"            [40, 9847]         [40, 9847]             "Paris"
```

The vocabulary and the rules for chopping (called **BPE** — Byte-Pair Encoding — or
SentencePiece, the common algorithms) are decided **before** training and never change. You can
think of the tokenizer as a fixed dictionary that both sides (input and output) agree on.

> **Key point that surprises people:** the model has *no idea* what letters are. By the time it
> runs, "Paris" is just the number `9847`. All the "intelligence" operates on integer ids and
> the vectors they map to (next chapter).

---

## Chapter 2 — Vectors and embeddings: giving numbers *meaning*

A token id like `9847` is just a label — it carries no meaning (id 9848 isn't "one more" than
Paris in any useful sense). To compute with meaning, each token id is converted into a **vector**.

- **Vector** — an ordered list of numbers, e.g. `[0.2, -0.5, 0.1, 0.9]`. Think of it as a point
  in space. A 4-number vector is a point in 4-dimensional space; a 4096-number vector is a point
  in 4096-dimensional space (impossible to picture, same idea).
- **Dimension** — how many numbers are in the vector. LLMs use big vectors: `d_model` (the
  "model dimension") is often 2048–8192.
- **Embedding** — the specific vector assigned to a token. It encodes the token's "meaning" as a
  location in space, learned during training.

The crucial trick: **similar meanings get similar vectors** (nearby points). The embeddings for
"king" and "queen" sit close together; "king" and "banana" sit far apart. Directions can even
carry relationships — the classic example is that the direction from "king" to "queen" is
similar to the direction from "man" to "woman".

How does a token id become its embedding? A simple **table lookup**:

- **Embedding matrix (E)** — a giant table with one row per vocabulary entry. Row `id` is that
  token's embedding vector. Shape: `[vocab_size × d_model]`.

```
 E  (vocab_size rows × d_model columns)
   row 0     [ ... d_model numbers ... ]
   row 1     [ ... ]
   ...
   row 9847  [ 0.2, -0.5, 0.1, 0.9, ... ]   ← embedding for "Paris"
   ...

 lookup:  embedding = E[token_id]      (just "go to that row")
```

So: **token id → look up its row in E → get the embedding vector.** That vector is the form the
token travels in through the rest of the model.

---

## Chapter 3 — Matrices and matrix multiplication: the one operation

Almost everything an LLM computes is **matrix multiplication**. If you understand this one
operation, you understand the mechanics of the whole model.

- **Matrix** — a grid (table) of numbers, with rows and columns. A vector is just a matrix with
  one row.
- **Matrix multiplication** — a way to combine a vector (or matrix) with another matrix to
  produce a new vector (or matrix). Mechanically: each output number is a **dot product** — you
  multiply pairs of numbers and add them up.

### Dot product (the atom)

- **Dot product** — multiply two equal-length lists element by element, then sum:
  `[a,b,c]·[x,y,z] = a·x + b·y + c·z` → a single number.

### A vector times a matrix, concretely

Say the input vector is `x = [0.2, -0.5, 0.1, 0.9]` (4 numbers) and we multiply it by a matrix
`W` with 4 rows and 3 columns. The result is a 3-number vector, where each output number is the
dot product of `x` with one **column** of `W`:

```
 x = [ 0.2, -0.5, 0.1, 0.9 ]

 W (4 rows × 3 cols):
   ┌  1.0   0.0  -1.0 ┐
   │  0.5   1.0   0.0 │
   │  0.0   0.5   1.0 │
   └ -1.0   0.0   0.5 ┘

 out[0] = 0.2·1.0 + (−0.5)·0.5 + 0.1·0.0 + 0.9·(−1.0) = −0.95
 out[1] = 0.2·0.0 + (−0.5)·1.0 + 0.1·0.5 + 0.9·0.0    = −0.45
 out[2] = 0.2·(−1.0)+(−0.5)·0.0 + 0.1·1.0 + 0.9·0.5   =  0.35

 out = [ −0.95, −0.45, 0.35 ]
```

That's the whole operation. The model does this billions of times with different matrices. Why
it's powerful: a matrix is a **learned transformation** — multiplying by `W` reshapes the input
vector in a way the training process found useful (e.g. "rotate the meaning toward whatever
matters for the next step").

- **Tensor** — the general word for "a block of numbers": a single number, a vector, a matrix,
  or a higher-dimensional grid are all tensors. You'll see "tensor" used loosely to mean "the
  numbers involved."

> **The one-sentence summary of an LLM's math:** take the input vector, multiply it by a series
> of learned matrices (with a few simple non-linear tweaks in between), and read off the result.

---

## Chapter 4 — Weights, parameters, and what "the model" is

- **Weights** (a.k.a. **parameters**) — all the numbers inside those matrices (E, and every `W`
  the model multiplies by). These are *the model*. "Gemma 7B" means the model has **7 billion**
  weights; a 70B model has 70 billion.
- **Architecture** — the fixed wiring: how many matrices there are, their shapes, and the order
  operations run in. The architecture is *code*; the weights are *data* that fills it in.

Analogy: the architecture is an empty spreadsheet with labeled, sized cells; the weights are the
specific numbers in those cells. Same spreadsheet (architecture) + different numbers (weights) =
a different model.

### Where do the weights come from? Training (briefly)

- **Training** — the one-time, enormously expensive process that *produced* the weights.
  Starting from random numbers, the model is shown trillions of tokens of text and repeatedly
  adjusted so its next-token predictions get less wrong. The adjusting is done by an algorithm
  called **gradient descent / backpropagation** (you don't need the details). After training,
  the weights are **frozen**.

- **Inference** — *using* the frozen model to answer a prompt. This document is about inference.
  During inference the weights never change; they are read-only constants you multiply by.

```
 TRAINING (once, weeks, thousands of GPUs):  random weights ──learn──► frozen weights  → saved to a file
 INFERENCE (every request):                  load frozen weights ──run──► predictions
```

So when you run a model, you load a file of pre-computed numbers and do a lot of matrix
multiplications with them. Nothing is "learned" at inference time.

---

## Chapter 5 — The hardware: GPU and VRAM, and why memory rules everything

- **GPU (Graphics Processing Unit)** — a chip with thousands of small cores that do many
  multiplications **in parallel**. Matrix multiplication is exactly "lots of multiplications at
  once," so GPUs are ideal for LLMs. (CPUs have a few fast cores; GPUs have thousands of slower
  ones — better for this job.)
- **VRAM (Video RAM)** — the GPU's own memory, where the weights and working data must live for
  the GPU to use them. It's fast but **limited** (e.g. 24 GB on a high-end consumer card, 80 GB
  on a datacenter card).

Two facts that drive every design decision later:

1. **The weights must fit in VRAM.** A 70B model at 2 bytes per weight is 140 GB — too big for
   one GPU, so it's either **quantized** (compressed to fewer bytes per weight, Chapter 7) or
   **split across multiple GPUs**.
2. **Moving numbers in/out of VRAM is often slower than the math itself.** Much of LLM
   performance engineering is about *not re-reading or re-computing* numbers. This is the entire
   reason the **KV cache** (Chapter 14) exists.

- **Quantization** — storing weights with fewer bits each (e.g. 4 bits instead of 16) to save
  memory and bandwidth, at a small cost to quality. "Q4" in an `ollama` model name means 4-bit.

Hold onto fact #2 — "avoid recomputing" is the theme that explains caching, and it's why the
same model can feel fast or slow depending on the engineering around it.

---

### End of Part I — what you now know

You can now state, in plain terms:
- An LLM is a **next-token predictor** built from **weights** (billions of numbers).
- Text becomes **token ids**; each id becomes an **embedding vector** via a lookup in **E**.
- The model transforms vectors by **matrix multiplication** with frozen, learned matrices.
- It runs on a **GPU**, and its data lives in limited **VRAM**, where avoiding recomputation is
  the central performance concern.

Everything below is these ideas, applied step by step.

---

# PART II — LOADING THE MODEL

This is what happens when you run `ollama run gemma` or a server boots a model like Claude Opus.
It happens **once**, before any prompt — the "cold start."

## Chapter 6 — Finding and reading the model files (a real cold start with `gemma4`)

Let's make this concrete with a model that's actually installed locally: `gemma4` in Ollama.
Everything below uses its *real* numbers (read from the machine with `ollama show gemma4`), so
you can see exactly what a "cold start" reads before any prompt runs.

```
 $ ollama show gemma4
   architecture        gemma4
   parameters          8.0B          ← 8 billion weights
   context length      131072        ← can handle up to 131,072 tokens (128k) of input
   embedding length    2560          ← d_model: each token's vector is 2560 numbers wide
   quantization        Q4_K_M        ← weights compressed to ~4 bits each
   capabilities        completion, vision, audio, tools, thinking
   default params      temperature 1, top_k 64, top_p 0.95
   file size on disk   9.6 GB
```

### 6.1 What `ollama run gemma4` actually points at

A model isn't one file; it's a small bundle the runner assembles. Ollama stores models the way
a container registry stores images:

- **Manifest** — a small JSON index for the tag `gemma4:latest`. It lists the pieces that make
  up the model and points at each by a content hash.
- **Blob** — one content-addressed file (named by its SHA-256 hash) holding one piece: the
  weights file, the parameter defaults, the template, the license, etc. Content-addressed means
  "named by a fingerprint of its bytes" — identical pieces are stored once and shared.

```
 ~/.ollama/models/
   manifests/.../gemma4/latest        ← JSON: "weights = blob A, params = blob B, template = blob C"
   blobs/sha256-<A>                    ← the 9.6 GB GGUF weights file
   blobs/sha256-<B>                    ← default sampling params (temperature 1, top_k 64, top_p 0.95)
   blobs/sha256-<C>                    ← the chat template
```

> This is why your two tags `gemma4:latest` and `gemma4:e4b` show the **same ID**
> (`c6eb396dbd59`) and size — they're two names for the same underlying blobs. The `e4b` tag
> denotes the model's "**E4B**" configuration: ~4-billion *effective* (active) parameters even
> though the raw count is 8.0B. Gemma's newer models are *elastic* — a larger network whose
> layers can run in a cheaper "effective" mode — so "8.0B parameters / e4b" is not a typo.

When you type `ollama run gemma4`, the runner reads the manifest, finds the blobs (downloading
them on first use, then cached forever), and hands the GGUF to the inference engine
(llama.cpp under the hood). Nothing has touched the GPU yet — this is pure file resolution.

### 6.2 The GGUF file — one container for weights + config + tokenizer

- **Checkpoint** — the saved file of trained weights. "Loading a checkpoint" = reading those
  numbers off disk.
- **GGUF** — the single-file format llama.cpp/Ollama use. Unlike the hosted world (which splits
  weights into `.safetensors` + a separate `config.json` + tokenizer files), **GGUF packs all of
  it into one file**: a metadata header (the config + tokenizer) followed by the weight tensors.

```
 gemma4 GGUF file (9.6 GB)
 ┌────────────────────────────────────────────────────────────┐
 │ METADATA HEADER (read first, tiny)                          │
 │   gemma4.block_count            = 34                        │
 │   gemma4.embedding_length       = 2560                      │
 │   gemma4.attention.head_count   = 8                         │
 │   gemma4.attention.head_count_kv= 4                         │
 │   gemma4.context_length         = 131072                    │
 │   tokenizer.ggml.tokens         = [ ...262k entries... ]    │
 │   general.quantization          = Q4_K_M                    │
 ├────────────────────────────────────────────────────────────┤
 │ TENSOR DATA (the 9.6 GB of actual weights)                  │
 │   token_embd.weight, blk.0.attn_q.weight, blk.0.ffn_up...   │
 └────────────────────────────────────────────────────────────┘
```

The engine reads the **metadata header first** — it's small and tells the engine the shapes of
everything in the tensor section, so it knows how to interpret the 9.6 GB that follow. This is
the "read the config before the weights" step.

### 6.3 The config, field by field (what each number means)

Ollama shows the headlines; the full config for this Gemma-class architecture (shown here from
the closely-related Gemma 3 4B checkpoint, which shares `gemma4`'s 2560-wide, 128k-context text
stack) looks like this. Every field is a **hyperparameter** — a design choice fixed by the
model's authors, not learned:

```jsonc
"text_config": {
  "num_hidden_layers": 34,        // L — how many transformer layers are stacked (Ch 15)
  "hidden_size": 2560,            // d_model — width of each token's vector (Ch 2)
  "num_attention_heads": 8,       // parallel attention "heads" per layer (Ch 15, attention)
  "num_key_value_heads": 4,       // KV heads — fewer than query heads = GQA (Ch 15, attention)
  "head_dim": 256,                // size of each head's Q/K/V vector
  "intermediate_size": 10240,     // width of the feed-forward network's hidden layer (Ch 15, FFN)
  "vocab_size": 262208,           // how many distinct tokens the model knows (Ch 1)
  "max_position_embeddings": 131072, // longest sequence it can handle (128k)
  "sliding_window": 1024,         // most layers only attend to the last 1024 tokens (see 6.5)
  "sliding_window_pattern": 6,    // 5 local layers : 1 global layer, repeating
  "rope_theta": 1000000.0,        // positional-encoding base for the global layers (Ch 15, attention)
  "rope_local_base_freq": 10000.0,// positional base for the local (sliding) layers
  "rms_norm_eps": 1e-06,          // tiny constant for numerical stability in normalization
  "hidden_activation": "gelu_pytorch_tanh", // the FFN's non-linearity (Ch 15, FFN)
  "torch_dtype": "bfloat16"       // the precision the weights were trained/released in
},
"bos_token_id": 2,                // "beginning of text" special token (Ch 10)
"eos_token_id": 106,              // "end of turn" — when the model emits this, generation stops
"vision_config": { "model_type": "siglip_vision_model", "num_hidden_layers": 27, ... }
```

A few things worth pausing on (real gotchas):

- **`head_dim` is decoupled from `hidden_size`.** Here `head_dim × num_heads = 256 × 8 = 2048`,
  which is *not* equal to `hidden_size` 2560. Many models keep these equal; Gemma deliberately
  doesn't. The lesson: don't assume `head_dim = hidden_size / num_heads` — it's its own field.
- **GQA at 8:4.** There are 8 query heads but only 4 key/value heads (`num_key_value_heads`).
  This halves the KV cache (Chapter 14/15) at almost no quality cost — and directly shrinks the
  memory math in 6.5.
- **Two RoPE bases.** Global and local layers use different positional-encoding settings
  (`rope_theta` vs `rope_local_base_freq`) because they cover different ranges (whole sequence vs
  a 1024-token window).
- **`vision_config` (and audio).** `gemma4` is **multimodal** — its capabilities listed `vision`
  and `audio`. Those are separate encoder towers (e.g. a SigLIP vision encoder) whose weights
  live in the same file and add to its size. For this document we follow the text path; just know
  the file carries extra encoders.

### 6.4 The tokenizer travels in the same file

GGUF's metadata also contains the **tokenizer**: the full 262,208-entry vocabulary and the
merge rules. So a single `gemma4` GGUF is self-contained — weights, architecture config, *and*
the text↔token-id converter (Chapter 1) all in one blob. (In the hosted world these would be
separate files.) The special-token ids the config names — `bos_token_id 2`, `eos_token_id 106` —
index into this same vocabulary.

### 6.5 What the loader computes *from the config* before reading the weights

Reading the config isn't just bookkeeping — the engine uses it to size memory before it loads
anything. Two calculations matter, and they preview Chapters 7–8.

**(a) Weight memory.** 8.0B parameters at **Q4_K_M** quantization:

- **Q4_K_M** — a "K-quant": most weights stored in **4 bits**, a few important ones in 6 bits,
  averaging ~4.8 bits per weight ("_M" = the medium-size variant). Versus 16 bits (bf16) that's
  ~3–4× smaller.
- `8.0e9 weights × ~4.8 bits ÷ 8 ≈ 4.8 GB` for the transformer weights; the embedding/output
  tables (262208 × 2560 ≈ 671M numbers) plus the vision/audio encoders are kept at higher
  precision, pushing the on-disk total to the **9.6 GB** you see. All of it loads into memory.

> On your machine (Apple Silicon, macOS/`arm64`) there's no separate "graphics card memory":
> the CPU and GPU share one **unified memory** pool, and Ollama's Metal backend addresses the
> weights in place — no copy across a PCIe bus like a discrete NVIDIA GPU would need. So "load
> into VRAM" here means "into unified RAM that the GPU can read directly."

**(b) KV cache budget** — and why Gemma's sliding window is a big deal. The KV cache (Chapter 14)
grows per token. Per-token size = `2 (K,V) × layers × kv_heads × head_dim × 2 bytes`:

```
 naive (every layer keeps the FULL context):
   2 × 34 × 4 × 256 × 2 B  ≈ 272 KB / token
   × 131072 tokens (full 128k context) ≈ 34 GB   ← bigger than the weights!
```

That's clearly impractical on a laptop. Gemma's fix (the `sliding_window` / `pattern: 6` fields):
only **1 layer in 6 is "global"** (attends to the whole sequence); the other 5 are **local** —
they only keep the last 1024 tokens of KV. So for the full 128k context:

```
 ~6 global layers  : 2 × 6  × 4 × 256 × 2 B × 131072 ≈ 6.4 GB
 ~28 local layers  : 2 × 28 × 4 × 256 × 2 B ×   1024 ≈ 0.23 GB
 total KV for full 128k ≈ 6.7 GB   (≈ 5× less than naive)
```

This is why the config carries `sliding_window` and `sliding_window_pattern` at all — they let
the engine reserve a far smaller KV pool. In practice Ollama also defaults the *working* context
(`num_ctx`) to something modest (often 4096) unless you raise it, so day-to-day the KV pool is
tiny; the 128k headline is opt-in and costs the memory above.

- **Hyperparameter** — to restate the term that ties this chapter together: every value in 6.3
  is a *hyperparameter*, a fixed design setting (counts, sizes, windows). Contrast with the
  **weights** (6.5a), the learned numbers. The config is read first precisely because the
  hyperparameters tell the loader how to lay out and size everything else.

### 6.6 Where's the *code*? Selecting the architecture (the missing third ingredient)

So far we've read the config (the **shapes**) and we're about to load the weights (the
**numbers**). But neither is the actual *program* that runs the operations in order — "RMSNorm,
then attention, then feed-forward, 34 times." Where is *that*? It is **not in the model files at
all.** It lives in the inference engine. Running a model needs **three** ingredients, held in two
places:

```
 INGREDIENT          WHAT IT IS                                    WHERE IT LIVES
 ─────────────────────────────────────────────────────────────────────────────────────────
 architecture code   the program: which ops, in what order        the ENGINE (Ollama/llama.cpp,
                     (the forward pass)                            vLLM, HF transformers) — NOT the file
 config              numbers that SIZE that code (L, d_model, …)   the model file (GGUF metadata)
 weights             the learned numbers that FILL the matrices    the model file (GGUF tensors)
```

> **A checkpoint contains no code — only config + weights.** The `.gguf`/`.safetensors` file is
> pure data; it cannot "do attention." The knowledge of *what a gemma4 layer does* is C++/Python
> written by humans and shipped inside the engine. (This is also a safety property: the
> `.safetensors` format exists precisely so a weights file is *only* numbers and can't smuggle
> executable code.)

So **loading is a three-step assembly**, and step 1 is the one we hadn't named:

```
 1. SELECT the architecture code by NAME.
      The config carries an identifier:
        GGUF:  general.architecture = "gemma4"
        HF:    "architectures": ["Gemma3ForConditionalGeneration"]
      The engine looks up its built-in code for that name → e.g. build_gemma4(...), which knows
      the op sequence: embed → 34×[norm, GQA attention, norm, GeGLU FFN] → final norm → lm_head.

 2. INSTANTIATE it at the config's sizes.
      Run that code with L=34, d_model=2560, heads=8/4, head_dim=256 → it allocates EMPTY tensors
      of exactly the right shapes (W_Q is 2560×2048, etc.).

 3. FILL the tensors with the weights, matched by NAME.
      The file's tensors are named (blk.0.attn_q.weight, blk.0.ffn_up.weight, …); the engine
      copies each into the matching empty slot of the instantiated architecture.
```

In one line: **the code is the recipe and its step order; the config sizes the recipe; the
weights are poured into the sized slots.**

**The evidence is already on your screen.** Recall `ollama show gemma4`:

```
 architecture   gemma4      ← the NAME that selects which code path runs
 requires       0.20.0      ← you need an Ollama whose binary CONTAINS the gemma4 code
```

That `requires 0.20.0` line *is* the proof that architecture is code: a newer model needed a
newer engine, because older Ollama binaries simply didn't contain the gemma4 forward-pass code.
The weights were downloadable, but without the matching code an older engine literally couldn't
run them. (Same reason you periodically update `transformers`/`llama.cpp` to run a just-released
model.)

**One nuance.** The code is somewhat generic — config *flags* select among options it already
implements: `hidden_activation: "gelu_pytorch_tanh"` picks GELU from a menu; `num_key_value_heads:
4` runs the GQA path with 4 groups; `sliding_window: 1024` applies the windowed mask it already
has. But a genuinely **new** operation the engine has never implemented (a new attention variant,
audio handling) can't be conjured by config — someone must *write new code* and ship a new engine
version. That is exactly what a `requires <version>` bump represents.

> For a closed model like Claude Opus it's the same split, just private: Anthropic's serving
> stack *is* the architecture code; the weights load into it. You never see either file, but the
> rule holds — **engine holds the code; the checkpoint holds config + weights.**

### 6.7 Recap of Chapter 6

```
 ollama run gemma4
   → read MANIFEST for gemma4:latest        (which blobs make up the model)
   → open the GGUF blob (9.6 GB)
   → read its METADATA HEADER first:
       config  → 34 layers, 2560-wide, 8/4 heads, head_dim 256, 128k context, Q4_K_M, sliding window
       tokenizer → 262k-entry vocab + merge rules
   → SELECT the architecture code by name ("gemma4") from inside the engine     (§6.6)
   → INSTANTIATE it at the config's sizes (empty tensors of the right shapes)
   → from the config, SIZE memory: ~9.6 GB weights, a KV pool (≈6.7 GB for full 128k, far less by default)
   → (next chapters) FILL the tensors with weights, reserve the KV pool, warm up
```

Nothing has computed an answer yet — Chapter 6 is entirely "figure out what this model is, *which
code runs it*, and how big its pieces are." Chapters 7–9 actually move the weights into memory,
carve the KV pool, and warm up.

---

## Chapter 7 — Loading weights into VRAM

The weights are copied from disk into the GPU's VRAM so the GPU can multiply by them.

```
 disk (slow, big) ──read & maybe decompress──► VRAM (fast, limited)
```

Things that happen here:

- **Precision / dtype.** Each weight is stored as a number of some size:
  - **fp16 / bf16** — 16-bit (2 bytes) floating-point. Full quality for inference.
  - **Quantized (e.g. 4-bit)** — compressed, ~4× smaller, slight quality loss. Used locally to
    fit big models on small GPUs.
  - **Floating-point** — a way to store fractional numbers (like `0.0042`) with limited
    precision. The "limited precision" detail matters later (Chapter 19, nondeterminism).
- **Splitting across GPUs (big models only).** If the weights don't fit on one GPU, they're
  divided:
  - **Tensor parallelism** — each matrix is cut into pieces on different GPUs; they multiply
    their piece and combine results.
  - **Pipeline parallelism** — different layers live on different GPUs; data flows through them
    in sequence.
  `ollama` on one machine usually doesn't split.

After this step, every matrix from Chapter 4 (E, all the layer matrices, the output matrix) sits
in VRAM as a read-only constant for the whole session.

---

## Chapter 8 — Reserving the KV cache pool

This will only fully make sense after Chapter 14, but it happens here at load time, so note it:

After the weights are loaded, the **leftover VRAM** is reserved as a big pool for the **KV
cache** — temporary per-conversation data the model will produce while answering. The server
divides this pool into fixed-size **blocks** up front.

```
 total VRAM − weights − working scratch − overhead = KV CACHE POOL → cut into blocks
```

Why reserve it now: allocating memory mid-request is slow and risky, so the engine grabs it all
at boot and manages it itself. The size of this pool caps how many conversations (and how much
total text) the server can hold at once. (Full detail in Chapter 14.)

---

## Chapter 9 — Loading the tokenizer and warming up

- **Load the tokenizer.** The vocabulary and chopping rules go into normal memory (not the GPU —
  tokenization is light text work). Now the server can convert text ↔ token ids.
- **Warm up.** The server runs a throwaway prediction to force one-time setup: the GPU math
  libraries pick their fastest internal routines for this model's exact sizes, and a fast
  replay-path for the generation loop is recorded. Without warmup the *first* real request would
  be unusually slow.

After warmup the server reports **READY** and waits. Loading is done. Everything from here
happens **per request**.

> Local nuance: `ollama` keeps the model loaded for a few minutes of idleness, then unloads it to
> free memory. The next call re-pays the loading cost (Chapters 6–9).

---

# PART III — FROM A PROMPT TO NUMBERS

The user types a prompt. Before any "thinking," the system converts it into the model's native
form (token ids) and decides how to process it.

## Chapter 10 — Receiving the request and building the prompt

### The request

The server receives the user's message plus **decoding parameters** — knobs that control how the
output is generated (all explained in Chapter 16):

```
 messages:  the conversation so far (system instructions, user turns, prior replies)
 settings:  temperature, top_p, top_k, max_tokens, stop sequences, seed, ...
```

### The chat template

Chat models are trained on a specific **format** with special marker tokens that label who is
speaking. The raw messages are rendered into that exact format.

- **Special tokens** — reserved vocabulary entries that aren't normal words but structural
  markers: "beginning of text", "start of a user turn", "end of turn", etc. The model learned
  what they mean during training.
- **Chat template** — the rule for assembling messages + special tokens into one text string.
  Each model family has its own; using the wrong one degrades quality.

Gemma's template, for example, wraps a user message like this:

```
 <bos><start_of_turn>user
 What is the capital of France?<end_of_turn>
 <start_of_turn>model
```

The trailing `<start_of_turn>model` line is the signal "now it's your turn to write" — the model
will continue from there.

---

## Chapter 11 — Tokenizing the prompt

The assembled prompt string is run through the tokenizer (Chapter 1) to become token ids:

```
 "<bos><start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
        │  tokenizer.encode
        ▼
 [ 2, 106, 1645, 108, 1841, 603, 573, 6037, 576, 6081, 235336, 107, 108, 106, 2516, 108 ]
```

Now the prompt is a list of integers — the only form the model understands. The number of tokens
here is your **input token count** (what you're often billed on, and what determines how much
work the next phase does).

---

## Chapter 12 — Checking the prefix cache (reuse before compute)

Before doing the expensive work, the engine checks: **have I already computed the early part of
this exact prompt before?** If so, it can reuse that work.

- **Prefix** — the leading portion of the token sequence (e.g. a long system prompt that's the
  same for every user).
- **Prefix cache** — a store of already-computed internal results (KV, Chapter 14) for prompt
  prefixes, so repeated prefixes don't have to be recomputed.

The canonical case: a 2,000-token system prompt shared by thousands of requests. The first
request computes it; the rest **reuse** it and skip straight to the new part. This is what makes
repeated-context apps fast and cheap.

> The mechanics — how a prefix becomes a lookup key (a "hash"), how blocks are reused, and the
> time-based ("TTL") version exposed by APIs — are the subject of the sibling docs
> `PREFIX_CACHING_EXPLAINED.md` and `PREFIX_PROMPT_CACHING_API.md`. For now: **matching prefix →
> reuse its computed KV → skip recomputing it.**

---

## Chapter 13 — Scheduling and batching

A busy server handles many users at once. It can't run one model pass per user — that would
waste the GPU. Instead it **batches** them.

- **Batch** — a group of requests processed together in a single GPU pass, sharing the cost of
  reading the weights.
- **Continuous batching** — requests join and leave the batch *every step*, rather than waiting
  for a fixed group to assemble. Keeps the GPU continuously busy.
- **Scheduler** — the component that decides which requests run this step, allocates their KV
  blocks, and queues or pauses requests when the KV pool is full.

For a local single user, the "batch" is just one request — no contention. For a hosted service,
your request is mixed with others, which (as a side effect) is part of why outputs aren't
bit-for-bit reproducible (Chapter 19).

---

# PART IV — PREFILL: reading and understanding the prompt

Now the actual model runs. The first phase, **prefill**, processes the *entire prompt at once* to
(a) build up the internal state for every prompt token and (b) produce the prediction for the
*first* new token.

- **Prefill** — the pass that ingests the whole prompt in parallel. Compute-heavy. Its duration
  is the **time to first token (TTFT)** — how long you wait before the answer starts appearing.

## Chapter 14 — The KV cache: the idea that makes generation feasible

This is the most important systems concept, so it gets its own chapter before we walk the layers.

When the model processes the prompt, for each token at each layer it computes two vectors called
**K** and **V** (Key and Value — defined precisely in Chapter 15). The reason they matter:

> Generating each new word requires looking back at the K and V of **every previous token**. And
> those K/V values **don't change** once computed.

So if you recomputed them from scratch for every new word, you'd redo the same work thousands of
times. Instead you compute each token's K/V **once** and **store** them.

- **KV cache** — the stored K and V vectors for every token processed so far, kept in VRAM so
  future steps can read them instead of recomputing. This is what fills the pool reserved in
  Chapter 8.

```
 without KV cache:  to write token #1000, recompute K/V for tokens 0..999   (wasteful, every step)
 with KV cache:     K/V for 0..999 already saved → just read them            (the standard approach)
```

The KV cache is **per conversation** and **grows by one token's worth each step**. It can become
very large (gigabytes for long chats), which is why its memory is managed so carefully — and why
the **prefix cache** (Chapter 12) exists to share it across requests with identical prefixes.

Keep this in mind as we now walk what actually happens inside one layer.

---

## Chapter 15 — Inside one transformer layer (the heart of it)

This is the most important chapter, so we'll go slow and **work a full numerical example by
hand**. The model is a stack of identical **layers**; each one takes the current token vectors,
lets the tokens share information, refines them, and passes them up. Understand one layer and
you understand the whole model — the other 33 layers of `gemma4` do exactly the same thing with
their own weights.

- **Transformer** — the neural-network design all modern LLMs use. Its defining feature is
  **attention** (below). "GPT" = Generative Pre-trained **Transformer**.
- **Layer (a.k.a. block)** — one round of: *attention* (tokens look at each other) followed by a
  *feed-forward network* (each token refined on its own). `gemma4` stacks **34** of them. Each
  layer has its **own** set of weight matrices.

The data flow through one layer (we'll execute every box below with real numbers):

```
 input vectors (one per token)
   │  ① RMSNorm
   ▼
 ② ATTENTION  ── tokens exchange information ──►  ③ + add back the input (residual)
   │  ④ RMSNorm
   ▼
 ⑤ FEED-FORWARD ── each token refined on its own ──►  ⑥ + add back (residual)
   │
   ▼
 output vectors  → become the input to the next layer
```

> This is the **pre-norm** arrangement modern models use: `h = x + Attention(Norm(x))`, then
> `out = h + FFN(Norm(h))`. Note the residual (③, ⑥) adds back the *un-normalized* input — the
> normalize is only to feed clean numbers into the heavy math, not to permanently rescale the
> signal. (Gemma actually adds an *extra* norm right after each sub-block too — "sandwich norm" —
> but the core idea is identical; we'll use the standard two norms.)

### 15.0 — Our running example

To keep arithmetic doable, we use toy sizes. The **real `gemma4` sizes** are in the sidebar so
you always see the correspondence.

```
 TOY (this walk)              REAL gemma4 (per layer)
 d_model      = 4             d_model      = 2560
 head_dim     = 2             head_dim     = 256
 heads        = 1             query heads  = 8,  KV heads = 4   (GQA)
 FFN hidden   = 2             FFN hidden   = 10240
 tokens       = 3             tokens       = however long your prompt is
```

Our 3 tokens (already turned into embeddings back in Chapter 2 — these are the layer's input):

```
 t1 "the"  x1 = [2, 0, 0, 0]
 t2 "cat"  x2 = [0, 2, 0, 0]
 t3 "sat"  x3 = [2, 2, 2, 2]
```

We'll follow **token 3 ("sat")** as it attends back over tokens 1–3 (this is exactly what the
last prompt token does in prefill, and what every token does during decode). The model computes
this for *all* positions in parallel during prefill; we trace one to keep the numbers small.

---

### ① Step 1 — RMSNorm (stabilize the vector)

- **Normalization (RMSNorm)** — rescale a vector so its values sit in a stable range before the
  heavy matrix math, so nothing explodes or vanishes across 34 layers. **RMSNorm** divides the
  vector by its root-mean-square, then multiplies by a learned per-dimension **gain** `g`:

```
 RMSNorm(x) = x / rms(x) · g       where   rms(x) = √( mean(xᵢ²) + ε )
```

(`ε` is a tiny constant — `gemma4`'s config: `rms_norm_eps = 1e-6` — that prevents divide-by-zero;
we'll ignore it as negligible. We take gain `g = [1,1,1,1]` for the walk.)

Compute for each token:

```
 x1 = [2,0,0,0]:  mean(xᵢ²) = (4+0+0+0)/4 = 1  → rms = 1   → n1 = [2,0,0,0]   (unchanged)
 x2 = [0,2,0,0]:  mean(xᵢ²) = (0+4+0+0)/4 = 1  → rms = 1   → n2 = [0,2,0,0]   (unchanged)
 x3 = [2,2,2,2]:  mean(xᵢ²) = (4+4+4+4)/4 = 4  → rms = 2   → n3 = [1,1,1,1]   (HALVED)
```

See it working on `t3`: its values were all `2` (rms 2), so dividing by 2 brings them to a tidy
`[1,1,1,1]`. The first two were already unit-scale, so normalization left them alone — that's
the point, it only rescales when needed.

The normalized vectors `n1, n2, n3` now feed into attention.

---

### ② Step 2 — Attention

**Attention** lets a token pull in information from earlier tokens. Mechanically it's three
sub-steps: (a) every token produces a Query, Key, and Value; (b) the querying token scores
itself against every key; (c) it takes a weighted blend of the values. Here's the intuition,
then the math.

> **The lookup analogy.** Think of a search engine. My **Query** is my search terms ("what am I
> looking for?"). Every earlier token published a **Key** (a label: "what I'm about") and a
> **Value** (its actual content). I compare my Query against all the Keys to score relevance,
> turn those scores into percentages, and pull a blend of the Values weighted by relevance.

#### ②a — Make Q, K, V (three matrix multiplies)

Each normalized vector is multiplied by three learned matrices (`W_Q, W_K, W_V`, each `4×2` in
the toy) to produce a Query, Key, and Value (each 2 numbers). These matrices are **weights** —
frozen, learned during training (Chapter 4).

```
 W_Q (4×2)     W_K (4×2)     W_V (4×2)
 ┌ 1 0 ┐       ┌ 1 0 ┐       ┌ 1 1 ┐
 │ 0 1 │       │ 0 1 │       │ 1 0 │
 │ 1 0 │       │ 0 1 │       │ 0 1 │
 └ 0 1 ┘       └ 1 0 ┘       └ 1 1 ┘
```

Recall a vector·matrix = each output is a dot product with a column (Chapter 3). Compute K and V
for **all three** tokens (every token must publish its key/value so later tokens can attend to
it), and Q for our querying token t3:

```
 Query (only need it for t3):
   q3 = n3·W_Q = [1,1,1,1]·W_Q = [ (1+1), (1+1) ] = [2, 2]

 Keys (every token):
   k1 = n1·W_K = [2,0,0,0]·W_K = [2, 0]
   k2 = n2·W_K = [0,2,0,0]·W_K = [0, 2]
   k3 = n3·W_K = [1,1,1,1]·W_K = [2, 2]

 Values (every token):
   v1 = n1·W_V = [2,0,0,0]·W_V = [2, 2]
   v2 = n2·W_V = [0,2,0,0]·W_V = [2, 0]
   v3 = n3·W_V = [1,1,1,1]·W_V = [3, 3]
```

(Worked example of one: `k3` = `[1,1,1,1]·W_K`. Column 0 of `W_K` is `[1,0,0,1]` →
`1·1+1·0+1·0+1·1 = 2`. Column 1 is `[0,1,1,0]` → `1·0+1·1+1·1+1·0 = 2`. So `k3=[2,2]`. ✓)

> **This is the exact moment K and V are written into the KV cache** (Chapter 14). `k1,v1, k2,v2,
> k3,v3` get stored so that when we later generate token 4, 5, … we don't recompute them. Q is
> used right now and thrown away — it's never cached. (That's why it's the *KV* cache, not QKV.)

#### ②b — Score the query against every key

The query token measures how relevant each earlier token is by taking the **dot product** of its
Query with each Key, then dividing by `√head_dim` to keep the numbers from getting too large
(`head_dim = 2` here, so divide by `√2 ≈ 1.414`):

```
 score(3,1) = q3·k1 /√2 = (2·2 + 2·0)/1.414 = 4/1.414 = 2.83
 score(3,2) = q3·k2 /√2 = (2·0 + 2·2)/1.414 = 4/1.414 = 2.83
 score(3,3) = q3·k3 /√2 = (2·2 + 2·2)/1.414 = 8/1.414 = 5.66
```

Higher score = more relevant. Token 3 scores highest against *itself* (5.66) and equally,
modestly against tokens 1 and 2 (2.83 each).

> **Gemma detail.** Gemma doesn't divide by `√head_dim`; it scales queries by
> `1/√query_pre_attn_scalar` (its config lists `query_pre_attn_scalar = 256`, so `1/16`). Same
> idea — a fixed shrink to keep scores in range — just a different constant. We use `√head_dim`
> here, the textbook default.

#### ②c — Causal mask (no peeking at the future)

- **Causal mask** — before turning scores into weights, set the score of any *future* token to
  −∞ so it gets zero weight. A token may attend only to itself and earlier tokens. This is what
  forces left-to-right generation: when producing a token, the model literally cannot see tokens
  that come after it.

For token 3, tokens 1–3 are all in the past/present → nothing is masked. (If we were computing
token 1, scores against 2 and 3 would be masked to −∞; token 1 can only attend to itself.)

```
            attend to →   t1     t2     t3
 query t1                 ok     ✗(−∞) ✗(−∞)
 query t2                 ok     ok     ✗(−∞)
 query t3                 ok     ok     ok        ← our case: all visible
```

> **Gemma detail — the sliding window.** In `gemma4`, 5 of every 6 layers are "local": their mask
> *also* blocks tokens more than `sliding_window = 1024` positions back, so those layers attend
> only to a recent window. This is the trick from Chapter 6 that shrinks the KV cache ~5×. Only
> every 6th layer attends globally.

#### ②d — Softmax (scores → percentages)

- **Softmax** — convert the raw scores into weights that are all positive and sum to 1 (Chapter
  17 covers it again for sampling). Bigger scores get exponentially bigger shares:
  `weightⱼ = e^scoreⱼ / Σ e^score`.

```
 scores            = [2.83, 2.83, 5.66]
 e^score           = [16.95, 16.95, 287.1]
 sum               = 321.0
 weights = e/sum   = [0.053, 0.053, 0.894]      (sums to 1.0)
```

So token 3 will pull **89.4%** of its information from itself, and **5.3%** from each of tokens 1
and 2.

#### ②e — Blend the values

The attention output is the weighted sum of the Value vectors, using those percentages:

```
 out3 = 0.053·v1 + 0.053·v2 + 0.894·v3
      = 0.053·[2,2] + 0.053·[2,0] + 0.894·[3,3]
      = [0.106,0.106] + [0.106,0] + [2.682,2.682]
      = [2.89, 2.79]
```

That `[2.89, 2.79]` is "what token 3 learned by looking at the sentence so far" — here, mostly a
copy of its own value `v3=[3,3]` nudged slightly by tokens 1 and 2. In a real model with meaning
in the vectors, this is where "sat" would pull in that "cat" is its subject.

#### ②f — Project back to model width

Attention produced a `head_dim`-wide vector (2 numbers); the layer needs to output a
`d_model`-wide vector (4 numbers) to match the residual stream. One more matrix, `W_O` (`2×4`):

```
 W_O (2×4)
 ┌ 1 0 1 0 ┐
 └ 0 1 0 1 ┘

 attn_out3 = out3·W_O = [2.89, 2.79]·W_O = [2.89, 2.79, 2.89, 2.79]
```

> **Multi-head & GQA (the real picture).** We used **one** attention head. `gemma4` runs **8**
> heads in parallel, each with its own `W_Q/W_K/W_V` and its own `head_dim=256` slice — every
> head learns a different relationship (one tracks subjects, another tracks punctuation, etc.).
> Their outputs (8 × 256 = 2048 numbers) are concatenated, then `W_O` (`2048 × 2560`) projects
> back to `d_model`. **GQA** means the 8 query heads share only **4** key/value heads (pairs of
> query heads reuse one K/V) — which is why the KV cache is half-size. The mechanics per head are
> exactly the ②a–②f you just did.

---

### ③ Step 3 — Residual add (refine, don't replace)

- **Residual connection** — add the attention output back to the layer's *original* input. The
  block doesn't overwrite the token's vector; it computes a *correction* and adds it. This keeps
  information from being lost across 34 layers and makes training stable.

```
 h3 = x3 + attn_out3 = [2,2,2,2] + [2.89,2.79,2.89,2.79] = [4.89, 4.79, 4.89, 4.79]
```

(Note: we add back `x3`, the un-normalized input — not `n3`. The norm in ① only cleaned up the
numbers feeding attention.) `h3` is the token after the attention half of the layer.

---

### ④–⑥ The feed-forward half

Attention let tokens *share* information. The **feed-forward network** now refines *each token on
its own* — no looking at neighbors. Same residual pattern: normalize, transform, add back.

#### ④ RMSNorm again

```
 h3 = [4.89, 4.79, 4.89, 4.79]
   mean(xᵢ²) = (4.89²+4.79²+4.89²+4.79²)/4 = 23.43  → rms = 4.84
   n′3 = h3 / 4.84 ≈ [1.01, 0.99, 1.01, 0.99]
```

#### ⑤ The feed-forward network (FFN)

- **Feed-forward network (FFN/MLP)** — two matrix multiplications with a non-linear step between.
  This is where much of the model's *stored knowledge* lives (facts, patterns). It widens the
  vector to a big hidden size, applies a non-linearity, then projects back.
- **Gated FFN (GeGLU)** — Gemma's FFN (config `hidden_activation: gelu_pytorch_tanh`) uses a
  *gate*: two parallel projections (`gate` and `up`), a non-linearity on the gate, multiply them
  together, then project down. `gemma4` widens 2560 → 10240 and back; our toy widens 4 → 2 → 4.

- **Activation function (GELU)** — the non-linearity. Without a "bend" between matrices, stacking
  them would collapse into one matrix and the model couldn't represent complex patterns. **GELU**
  smoothly passes large positives and squashes negatives toward 0 (`GELU(1.0) ≈ 0.841`).

```
 W_gate (4×2)   W_up (4×2)    W_down (2×4)
 ┌ 1 0 ┐        ┌ 1 0 ┐        ┌ 1 0 1 0 ┐
 │ 0 1 │        │ 0 1 │        └ 0 1 0 1 ┘
 │ 0 0 │        │ 1 0 │
 └ 0 0 ┘        └ 0 1 ┘

 n′3 = [1.01, 0.99, 1.01, 0.99]

 gate = n′3·W_gate = [1.01, 0.99]
 up   = n′3·W_up   = [1.01+1.01, 0.99+0.99] = [2.02, 1.98]

 GELU(gate) = [GELU(1.01), GELU(0.99)] ≈ [0.85, 0.83]      ← the non-linear "bend"
 gated = GELU(gate) ⊙ up = [0.85·2.02, 0.83·1.98] ≈ [1.72, 1.65]   (⊙ = multiply element-wise)

 ffn_out3 = gated·W_down = [1.72, 1.65]·W_down = [1.72, 1.65, 1.72, 1.65]
```

#### ⑥ Residual add

```
 layer_out3 = h3 + ffn_out3
            = [4.89,4.79,4.89,4.79] + [1.72,1.65,1.72,1.65]
            = [6.61, 6.44, 6.61, 6.44]
```

**That is the layer's output for token 3.** It becomes the *input* to the next layer.

---

### 15.x — Positional information (RoPE), in one paragraph

You may wonder how the model knows token *order* — so far nothing distinguished position 1 from
position 3. The answer is **RoPE (Rotary Positional Embedding)**: just before scoring (between
②a and ②b), each Query and Key vector is *rotated* by an amount proportional to its position.
Two tokens far apart get more dissimilar rotations, so their dot-product score naturally encodes
distance. `gemma4`'s config carries `rope_theta = 1,000,000` (for the global layers) and
`rope_local_base_freq = 10,000` (for the sliding-window layers) — two different rotation scales
for the two layer types. You don't need the trigonometry; just know position enters here, by
rotating Q and K.

---

### 15.y — Stacking 34 layers

Everything above was **one** layer. `layer_out3` feeds into layer 2, which does the identical
①–⑥ with *its own* weights, producing a new vector, which feeds layer 3, … up through all 34.

The important consequence: with each layer a token's vector mixes in more of its context (through
attention). By the lower layers it knows its immediate neighbors; by the top layers, token 3's
vector reflects the **entire preceding prompt**. This is exactly why a token's K/V depends on
everything before it — and therefore why the cache key must be the whole *prefix*, not the single
token (the fact underpinning prefix caching, Chapter 12 and the sibling docs).

After the final (34th) layer, we have one refined vector per token. The next chapter turns the
**last** token's vector into a prediction.

---

### 15.z — Recap of one layer

```
 input x  ─①RMSNorm─►  n
                       │
            ②ATTENTION: make Q,K,V (W_Q,W_K,W_V) → score q·k/√d → mask → softmax →
                        blend values → project (W_O)        [K,V saved to the KV cache]
                       │
 h = x + attn ◄────────③ residual
   │
   ─④RMSNorm─► n′ ─⑤FFN: gate,up (W_gate,W_up) → GELU → multiply → down (W_down)
   │
 out = h + ffn ◄───────⑥ residual ──► to next layer
```

Two matrix-heavy stages (attention mixes tokens; FFN refines each), each wrapped in
normalize-then-add-back. Repeat 34 times. That's a transformer.

---

## Chapter 16 — From the top layer to a prediction (logits)

After the last layer, we have a final vector for the last token. To turn it into a next-token
prediction, multiply it by one more matrix:

- **Output projection / LM head** — a matrix `[d_model × vocab_size]` that maps the final vector
  to one score per vocabulary token.
- **Logits** — the raw, unnormalized scores, one per possible next token. Higher = the model
  thinks this token is more likely. They are *not yet* probabilities (they can be any number,
  negative or positive).

```
 final vector (d_model numbers) ──× LM head──► logits (vocab_size numbers)
   "Paris" → 9.4,  "France" → 3.1,  "a" → 2.7,  ...   (a score for EVERY token in the vocab)
```

During prefill, the model computed final vectors for *all* prompt positions, but only the **last
position's** logits are needed to choose the first new token. Prefill is now done; the KV cache
holds the whole prompt; we have the first distribution to sample from.

---

# PART V — DECODE: generating the answer, one token at a time

- **Decode** — the loop that generates output tokens one by one, each requiring a small model
  pass that reads the whole KV cache. Its per-token speed is the **inter-token latency (ITL)** —
  how fast words stream out.
- **Autoregressive** — each generated token is fed back in as input to generate the next.
  "Auto" (self) + "regressive" (feeding prior outputs back).

```
 logits ─► pick a token ─► append it ─► run model on just that token ─► new logits ─► repeat
```

## Chapter 17 — Turning logits into a token: softmax and sampling

### Softmax — logits to probabilities

- **Softmax** — a function that converts a list of arbitrary scores (logits) into a probability
  distribution: all values become positive and sum to 1, with bigger logits getting
  exponentially bigger shares.

```
 logits  [9.4, 3.1, 2.7, ...]  ──softmax──►  [0.81, 0.04, 0.03, ...]   (now sums to 1)
```

### Sampling — picking one token

- **Sampling** — choosing the actual next token from the probability distribution. Several knobs
  shape this choice:
  - **Greedy (temperature 0)** — always take the single highest-probability token. Deterministic-
    looking, repetitive.
  - **Temperature** — a dial that flattens or sharpens the distribution before picking. Low (<1)
    = more focused/safe; high (>1) = more random/creative. `T=0` = greedy.
  - **Top-k** — only consider the k highest-probability tokens; ignore the rest.
  - **Top-p (nucleus)** — only consider the smallest set of top tokens whose probabilities add up
    to p (e.g. 0.9); a dynamic cutoff.
  - **Repetition / frequency / presence penalties** — reduce the probability of tokens already
    used, to curb loops and repetition.
  - **Seed** — a number that fixes the random choices, so the same seed + same inputs can
    reproduce the same picks.

This sampling step is the *deliberate* source of variety: with `temperature > 0` the same prompt
can yield different answers by design. (The *accidental* source — floating-point/batching — is
Chapter 19.)

---

## Chapter 18 — Feeding the token back, detokenizing, and stopping

### Feed back (and grow the KV cache)

The chosen token id is sent back through the model — but now for **just that one token**, not the
whole prompt:

```
 new token ──embed──► one pass through the L layers (reading ALL cached K/V) ──► next logits
                       │
                       └── this token's own K and V are APPENDED to the KV cache
```

Because the prompt's K/V are already cached, each step only computes one new token's worth of
work and reads the rest. This is the efficiency the KV cache buys (Chapter 14). It's also why
decode is limited by **memory bandwidth**: every single token requires re-reading all the weights
and the whole KV cache from VRAM.

### Detokenize and stream

- **Detokenize** — convert generated token ids back to text using the vocabulary (Chapter 1, in
  reverse).
- **Streaming** — sending each token's text to the user as it's produced (the "typing" effect),
  rather than waiting for the whole answer.

Subtlety: one visible character can span multiple tokens (and a token can be a partial byte
sequence), so detokenization buffers until it has a complete character before emitting.

### Stop

The loop ends when any of these happens:

```
 • the model emits an end-of-turn / end-of-text special token   (it decided it's finished)
 • a user-specified stop sequence appears in the output
 • the max_tokens limit is reached
 • the request is cancelled or times out
```

Then the final pieces are flushed, resources are released (the conversation's KV blocks are
freed; shared-prefix blocks are kept around for the next request), and usage counts (input
tokens, output tokens) are reported.

---

# PART VI — THE REALITIES THAT CUT ACROSS EVERYTHING

## Chapter 19 — Why the same prompt can give different answers

A common confusion: matrix multiplication is exact and the weights are frozen, so why isn't the
output identical every time? Two reasons:

1. **Sampling (on purpose).** With `temperature > 0`, the model draws randomly from the
   distribution (Chapter 17). Set `temperature = 0` to remove this.
2. **Floating-point rounding + batching (subtle).** Computers store fractional numbers with
   limited precision, and adding many of them in a **different order** gives slightly different
   results (`(a+b)+c ≠ a+(b+c)` after rounding). On a GPU, the order depends on how work is
   parallelized — and in a shared server, on **how your request was batched with others**. Tiny
   differences in the logits can, on a near-tie, flip which token is chosen; because generation
   is autoregressive, one flipped token changes everything after it.

So even "greedy" decoding isn't bit-for-bit reproducible across runs unless the engine uses
**batch-invariant** math (kernels written to sum in a fixed order regardless of batch). A `seed`
fixes reason 1, not reason 2.

---

## Chapter 20 — The two speed regimes (and where optimizations apply)

| | **Prefill** (read the prompt) | **Decode** (write the answer) |
|---|---|---|
| Does what | all prompt tokens at once | one token per step |
| Limited by | GPU compute (math speed) | memory bandwidth (reading weights + KV each token) |
| You feel it as | time to first token (TTFT) | typing speed (inter-token latency) |
| Sped up by | prefix caching (skip it) | batching, KV quantization, GQA, speculative decoding |

A few named optimizations and where they fit:

- **FlashAttention** — a faster, memory-light way to *compute* attention (see "Kinds of
  attention" below, and Chapter 15).
- **PagedAttention** — managing the KV cache in fixed blocks so memory isn't wasted (Chapter 8/14).
- **Speculative decoding** — a small fast model guesses several tokens; the big model checks them
  all in one pass, turning several slow decode steps into one (Chapter 18).
- **Prefix caching** — reuse a shared prompt's computed KV (Chapter 12).
- **Quantization** — smaller weights and smaller KV to fit and move faster (Chapter 7).

### Kinds of attention: *how* (kernels) vs *what* (variants)

"Types of attention" mixes up two different axes. They are not alternatives to each other — a
real model picks **one from each** and combines them. `gemma4` runs a **GQA + sliding-window**
*variant* using **FlashAttention** *kernels*.

**Axis A — the kernel: same math, different speed/memory.** From Chapter 15, attention builds an
N×N scores matrix (`QKᵀ`) for N tokens. The naive way writes that whole matrix to slow GPU
memory; since attention is memory-bandwidth-bound, those round-trips are the cost.

| Kernel | Idea | Scores memory | Exact? |
|---|---|---|---|
| **Naive** | build the full N×N scores matrix in slow memory (HBM), softmax, ×V | **O(N²)** | yes |
| **FlashAttention** | never build it: stream K/V in **tiles**, keep partials in fast on-chip SRAM via **online softmax** | **O(N)** | **yes — exact, not an approximation** |
| **PagedAttention** | not a math change — lets the kernel read the **KV cache** from non-contiguous blocks/pages | — | yes |

> FlashAttention's win is **fewer memory round-trips**, not fewer FLOPs: it computes a
> softmax-weighted sum incrementally (running max + running sum) so the full N×N matrix never
> exists. Same answer, O(N²)→O(N) memory, much faster — and it's what makes 128k context feasible.
> PagedAttention is the storage side (it's what `block_hash_cache.go`/`radix_cache.go` model) and
> composes with FlashAttention.

**Axis B — the variant: different math / different K-V sharing, to shrink the KV cache or beat
O(N²).**

| Variant | Q / K-V heads | KV cache | Trade |
|---|---|---|---|
| **MHA** (multi-head) | N query, **N** KV | largest (baseline) | original; best quality, biggest cache |
| **MQA** (multi-query) | N query, **1** KV | smallest (÷N) | big saving, some quality loss |
| **GQA** (grouped-query) | N query, **G** KV groups | ÷(N/G) | modern default — `gemma4`: 8 query / **4** KV → half |
| **MLA** (multi-head latent) | cache a compressed **latent**, rebuild K/V on the fly | much smaller than GQA | DeepSeek-V2/V3; current frontier |
| **Sliding-window / local** | attend only to last **W** tokens | **O(W)** per layer | Gemma local layers (W=1024), Mistral; bounds cache + compute |
| **Sparse / linear / SSM** | attend to a subset, or replace softmax entirely | sub-quadratic / O(N) | long-doc and alt-architecture (Mamba) territory |

So the precise way to read the FlashAttention bullet above: it isn't a "type of attention" you
choose *instead of* GQA — it's the fast, exact *way to run* whatever variant the model defines.
Axis A makes the same computation cheaper; Axis B changes the computation to shrink the dominant
cost (the KV cache). `gemma4` = **GQA 8:4 + sliding-window/global interleave** (Axis B), executed
with **FlashAttention + PagedAttention** kernels (Axis A).

---

## Chapter 21 — The whole journey in one breath

```
 LOAD (once):
   find files → read config (the shapes) → copy weights into VRAM →
   reserve the KV cache pool → load tokenizer → warm up → READY

 PER PROMPT:
   receive request + settings
   → apply chat template (add role/marker tokens)
   → tokenize (text → token ids)
   → check prefix cache (reuse a shared prompt's KV if seen before)
   → schedule into a batch

   PREFILL  (read the prompt, compute-bound, sets TTFT):
     embed token ids → run L transformer layers
       (each: normalize → attention [make Q,K,V; tokens attend; SAVE K,V to KV cache] →
        add residual → feed-forward → add residual)
     → logits for the last token

   DECODE   (write the answer, bandwidth-bound, sets typing speed):
     loop:
       softmax + sample a token (temperature/top-k/top-p/seed)
       → detokenize → stream it to the user
       → feed the token back through the layers (reading ALL cached K/V;
          append this token's K,V to the cache)
       → next logits
     until end-of-turn token / stop sequence / max_tokens

   FINISH:
     free the conversation's KV (keep shared-prefix KV for next time) → report token usage
```

Every advanced trick in real systems — paging, continuous batching, FlashAttention, speculative
decoding, prefix caching, quantization, multi-GPU sharding — is engineering to make those two
loops (**prefill** and **decode**) cheaper, more parallel, or more shareable. The core never
changes:

> **embed → (normalize, attention, feed-forward) × L layers → logits → sample → feed back →
> repeat.**

---

# GLOSSARY (quick reference)

- **Activation function** — a small non-linear step (SwiGLU/ReLU) between matrices that lets the
  model represent complex patterns.
- **Architecture** — the fixed wiring/shape of the model (code); filled in by weights (data).
- **Attention** — the mechanism by which each token gathers information from earlier tokens via
  Query/Key/Value.
- **Autoregressive** — generating one token at a time, feeding each output back as input.
- **Batch / continuous batching** — processing many requests together in one GPU pass; admitting
  and retiring them every step.
- **BPE / SentencePiece** — algorithms for splitting text into tokens.
- **Causal mask** — the rule that a token may only attend to earlier tokens, never future ones.
- **Checkpoint** — the saved file(s) of trained weights.
- **Decode** — the per-token generation loop; bandwidth-bound; sets typing speed.
- **Dimension / d_model** — how many numbers are in a vector; the model's vector width.
- **Dot product** — multiply two lists element-wise and sum; the atom of matrix multiplication.
- **Embedding / E** — the vector representing a token's meaning; looked up by token id in matrix E.
- **Feed-forward network (FFN/MLP)** — per-token processing after attention; holds much of the
  model's knowledge.
- **GPU / VRAM** — the parallel chip that runs the math / its fast but limited memory.
- **GQA** — grouped-query attention; shares K/V across query heads to shrink the KV cache.
- **Greedy** — always pick the highest-probability token (temperature 0).
- **Head (attention head)** — one of several parallel attention computations, each catching a
  different relationship.
- **Hyperparameter** — a design setting (layer count, sizes) fixed by the model's creators, not
  learned.
- **Inference** — running a trained model to answer prompts (what this document covers).
- **KV cache** — stored Key/Value vectors for all processed tokens, so generation doesn't
  recompute them.
- **Layer** — one round of attention + feed-forward; models stack many (L).
- **LM head / output projection** — the final matrix mapping the last vector to per-token scores.
- **Logits** — raw next-token scores before softmax.
- **Normalization (RMSNorm/LayerNorm)** — rescaling vectors to a stable range between steps.
- **Prefill** — the pass that ingests the whole prompt at once; compute-bound; sets TTFT.
- **Prefix / prefix cache** — a leading run of tokens / a store reusing its computed KV across
  requests.
- **Probability distribution** — options each with a likelihood, summing to 1.
- **Quantization** — storing weights/KV with fewer bits to save memory, at slight quality cost.
- **Query / Key / Value (Q/K/V)** — per-token vectors driving attention: what I seek / what I
  offer / what I hand over.
- **Residual connection** — adding a block's input back to its output to refine, not overwrite.
- **Sampling** — choosing the next token from the distribution (temperature, top-k, top-p, seed).
- **Softmax** — converts logits into a probability distribution.
- **Special tokens** — reserved structural markers (begin/end/role) the model learned to read.
- **Tensor** — any block of numbers (scalar, vector, matrix, or higher).
- **Token / token id / vocabulary / tokenizer** — a text chunk / its integer / the full list /
  the converter.
- **Temperature / top-k / top-p** — sampling controls for randomness and candidate set.
- **Training** — the one-time process that produced the frozen weights (not done at inference).
- **Transformer** — the neural-network design behind modern LLMs; built on attention.
- **TTFT / ITL** — time to first token / inter-token latency; the two latency metrics.
- **Weights / parameters** — the billions of learned numbers that *are* the model.

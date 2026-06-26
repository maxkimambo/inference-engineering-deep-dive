# Neural Networks

You cannot reason about inference cost without a mechanical picture of what a neural network
*does* at runtime. This section builds that picture from the smallest unit up. If you already
know what a matmul is and why ReLU exists, skim to [the cost lens](#the-cost-lens-read-every-layer-as-a-matmul)
at the end — that framing is what the rest of the book leans on.

## The node: a tiny program

The fundamental unit of a neural network is a **node**. A node is a very small program: it takes
some input numbers, multiplies each by a **weight**, adds them up, adds a **bias**, and returns the
result.

!!! info "\"Node\" is the friendly name — here are the official ones"
    A single node is the **artificial neuron**, the basic unit of every neural network. Its
    original form — a single neuron that sums weighted inputs and applies a threshold — is the
    **perceptron** (Frank Rosenblatt, 1958), the term you'll meet in textbooks and papers. Modern
    neurons generalize the perceptron (smooth activations instead of a hard threshold), but it's
    the same idea. We say "node" because it's the simplest way to picture it; reach for
    **neuron** / **perceptron** when you read the literature.

- **Weight** — a learned number that says *how much this input matters*. Set during training,
  frozen during inference.
- **Bias** — a learned number added at the end, letting the node shift its output up or down
  independent of the input.

```
inputs        weights
 x1 ──×w1──┐
 x2 ──×w2──┼──(sum)── + bias ──► output
 x3 ──×w3──┘
```

**A worked example.** Give the node three concrete inputs and its learned weights and bias:

```
x = [ 1.0,  2.0,  3.0 ]     ← the inputs
w = [ 0.5, -1.0,  0.25]     ← the learned weights (one per input)
b =   2.0                   ← the learned bias

output = (1.0 × 0.5) + (2.0 × −1.0) + (3.0 × 0.25) + 2.0
       =    0.5      +    −2.0       +    0.75      + 2.0
       =    1.25
```

That single number `1.25` is the node's output — what it hands to the next layer. A node is
*almost useless alone*; the power comes from stacking thousands of them and learning all the
weights.

## Layers and the network

A **layer** is a group of nodes that all read the *same* inputs but have their *own* weights, and
all compute in parallel. Nodes within a layer don't talk to each other; the "network" — the wiring
— happens *between* layers, where one layer's outputs become the next layer's inputs.

Networks behind LLMs have dozens to hundreds of layers, in three roles:

- **Input layer** — accepts and processes the raw input.
- **Hidden layers** — every layer in between, each transforming the representation a little more.
- **Output layer** — produces the final prediction.

Each hidden layer emits a vector called a **hidden state** — the network's internal,
intermediate representation of the data at that depth.

- **Hidden state** — the vector flowing between layers. Not human-readable; it's the model's
  working representation. Its length is the **dimensionality** (often `d_model`, e.g. 4096).

!!! info "Why text representations get *bigger* and image ones get *smaller*"
    Text inference *increases* dimensionality — a token becomes a vector of thousands of numbers
    to capture meaning. Image models do the opposite: they *reduce* a million-pixel image down to
    a compact **latent** of a few thousand numbers. Same goal — a representation that's the right
    size to compute on — approached from opposite directions. We return to this in
    [Image & Video Generation](image-video-generation.md).

### Encoders and decoders

Two jobs show up everywhere:

- **Encoder** — turns an input (text, image, audio) into an internal representation enriched with
  meaning.
- **Decoder** — turns an internal representation into an output (text, an image).

Modern LLMs are **decoder-only**. Encoder-only models (the BERT family of text-embedding models)
are rarer today. Encoder-decoder models persist in other modalities — Whisper encodes audio, then
decodes text tokens.

!!! key "Composability"
    Neural networks are LEGO. You can fuse several into one model, or chain them into a pipeline.
    An image generator is literally three networks (text encoder → denoiser → VAE) bolted
    together. Keep this in mind — "a model" is often several models wearing a trench coat.

## The most important operation: matmul

The single operation that dominates inference is the **matrix multiplication**, or **matmul**. A
matmul takes a vector (a list of numbers) and a matrix (a grid of numbers) and produces a new
vector.

The simplest neural-network layer — a **linear layer** (a.k.a. **dense** or **fully-connected**
layer) — is exactly one matmul plus a bias:

\[
y = xW + b
\]

where \(x\) is the input vector, \(W\) is the weight matrix, \(b\) is the bias vector, and \(y\)
is the output.

Here's the mechanical version. Say \(x\) has 3 numbers and we want \(y\) to have 2. Then \(W\) is
a \(3 \times 2\) grid and each output is a weighted sum of all inputs:

```
x = [x1, x2, x3]

      | w11  w12 |
W  =  | w21  w22 |          y1 = x1·w11 + x2·w21 + x3·w31 + b1
      | w31  w32 |          y2 = x1·w12 + x2·w22 + x3·w32 + b2

y = [y1, y2]
```

**With real numbers**, take the same `x = [1, 2, 3]` from the node example:

```
                | 0.5    0.0 |
x = [1, 2, 3]   | 1.0   -1.0 |   b = [0.5, 0.5]
                | 0.0    2.0 |
                  ↑col1   ↑col2

y1 = 1·0.5 + 2·1.0  + 3·0.0 + 0.5 = 3.0
y2 = 1·0.0 + 2·(−1.0) + 3·2.0 + 0.5 = 4.5

y = [3.0, 4.5]
```

Three numbers went in, two came out — the **shape** of \(W\) did the resizing; the **values** did
the mixing.

!!! key "A matmul *is* a layer of nodes — that's the whole connection"
    Look at the columns of \(W\). **Column 1 `[0.5, 1.0, 0.0]` is one node's weights; column 2
    `[0.0, −1.0, 2.0]` is another node's weights.** Computing `y1` is exactly running node 1;
    computing `y2` is running node 2. A linear layer with 2 outputs is literally 2 nodes stacked
    side by side, and the matmul runs them all at once. So everything from the node section scales
    up by stacking columns — that's all a layer is.

- The **shape** of \(W\) (here \(3 \times 2\)) sets how many numbers go in (rows) and come out
  (columns).
- The values inside \(W\) are weights, learned in training, frozen at inference.
- The weights of *one* linear layer are a small slice of a model's total weights — a real LLM has
  hundreds of these.

!!! key "This is why GPUs"
    A matmul is thousands of independent multiply-then-add operations with no dependencies between
    them. That is the *one* thing GPUs do extravagantly well — thousands of arithmetic units
    running the same operation in lockstep. The entire field of inference hardware exists to feed
    matmuls. Hold that thought for [Bottlenecks](bottlenecks.md).

## Why depth needs non-linearity

Here's a subtle trap that explains a core design choice. Matmuls are **composable**: multiplying a
vector by \(W_1\) then by \(W_2\) is the same as multiplying it once by the single matrix
\(W_3 = W_1 W_2\).

```python
# two linear layers, back to back
y = x @ W1 + b1
z = y @ W2 + b2

# but matrix multiplication is associative, so...
W3 = W1 @ W2          # precompute one matrix
z  = x @ W3 + b3      # ...the two layers collapse into one
```

This is a disaster for deep networks. If every layer is just a matmul, a 100-layer network
collapses into a *single* equivalent layer. All that depth — gone. The network can only ever
represent linear functions (straight lines and flat planes), which can't model anything
interesting.

The fix is to put a **non-linear** function between layers so they can't be merged. That function
is the **activation function**.

- **Activation function** — a non-linear function applied element-wise to a layer's output. It
  (1) breaks linearity so layers don't collapse, and (2) is differentiable (or nearly so) so the
  network can be trained by gradient descent.

The classic is **ReLU** (Rectified Linear Unit) — comically simple: keep positives, zero out
negatives.

\[
\text{ReLU}(x) = \max(0, x)
\]

```
output
  10 |                          /
     |                        /
   5 |                      /
     |                    /
   0 |________________ /________________
     -10      -5      0      5      10   input
```

Negatives become zero; positives pass through. That single kink is enough non-linearity to stop
the collapse. Modern LLMs use smoother cousins — **SiLU**, **Swish** (named for resembling the
Nike swoosh), and **SwiGLU** — but the pattern is the same: squash negatives toward zero, mostly
preserve positives, stay (mostly) differentiable, run fast.

!!! key "The takeaway, in one line"
    **Linear layers do the work; activations make depth meaningful.** Stack `(matmul → activation)`
    many times and you get a function expressive enough to predict the next token. Remove the
    activations and your 70-billion-parameter model is algebraically one matrix.

## The cost lens: read every layer as a matmul

This is the framing the rest of the book uses, so internalize it:

> **A neural network, at inference time, is a long chain of matmuls separated by cheap
> element-wise functions.**

That single sentence has two consequences you'll use constantly:

1. **The weights are the bulk of the bytes.** Each matmul has a weight matrix sitting in GPU
   memory. To run the matmul you must *read those weights*. For a big model that's reading tens of
   gigabytes — every forward pass. This is the seed of the memory-bandwidth bottleneck.
2. **The activations, norms, and biases are a rounding error.** Compared to the matmuls, the
   element-wise functions cost almost nothing in compute *and* memory. When you optimize
   inference, you optimize matmuls and data movement; you essentially ignore the rest.

```
       a transformer's forward pass, abstracted
   ┌────────────────────────────────────────────────┐
   │  matmul → act → matmul → act → matmul → act ... │
   │   ▲                                              │
   │   └─ reading a weight matrix from memory each    │
   │      time; this read is what you fight to reduce │
   └────────────────────────────────────────────────┘
```

With this lens in hand, the next section traces a real token through a real transformer — and you
will be able to point at each step and say "matmul, big; activation, free."

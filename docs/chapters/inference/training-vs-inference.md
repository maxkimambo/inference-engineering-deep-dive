# Training vs Inference

The fastest way to misunderstand inference is to treat it as a cheaper, simpler version of training —
"same model, just skip the part where it learns." The forward pass *is* shared, but almost everything
around it differs, and those differences are exactly what make serving its own engineering problem.
This section pins down the distinction so the rest of the book has solid ground to stand on.

## Two different shapes of work

Training and inference run the same model, but they run it in opposite *shapes*.

**Training** is a batch job. You have a fixed, enormous dataset; you push large batches through the
model, compute how wrong it was, and propagate that error *backward* to nudge every parameter. Three
passes happen per step — a forward pass, a backward pass, and an optimizer update — and you store the
intermediate activations from the forward pass because the backward pass needs them. It's a closed
system: known inputs, known size, no one waiting. You optimize for one thing — keep the very
expensive cluster as busy as possible until the loss curve flattens.

**Inference** is an online service. Requests arrive when they arrive, each a different length, each
from a user who wants an answer now. There is only a forward pass — no backward, no optimizer, no
stored activations for gradients. But that "just the forward pass" hides the twist that defines
serving: for generative models the forward pass isn't one shot, it's a *loop*.

!!! key "Generation is a loop, and that loop is the whole story"
    A language model doesn't produce its answer in one pass. It produces **one token**, appends that
    token to the input, and runs again to produce the next — over and over until the response is
    done. A 500-token answer is ~500 sequential forward passes, each depending on the last. Training
    sees a whole sequence at once and processes it in parallel; inference must *unroll it one step at
    a time* because it doesn't know token N+1 until it has generated token N. This single fact —
    autoregressive, sequential generation — is the root of nearly every latency problem in this book.

## Why "forward only" doesn't mean "easy"

Dropping the backward pass removes work, but the work that remains is shaped badly for hardware, and
the surrounding system is harder, not easier:

- **The compute is memory-bound, not math-bound.** Because each token is generated alone, every step
  re-reads the entire model from memory to do a comparatively tiny amount of math. The GPU's
  arithmetic units — the thing you paid for — sit mostly idle, starved for data. (Chapter 2 turns this
  into a proven number; for now, hold the surprise: more raw compute often doesn't help inference at
  all.)
- **The batch is dynamic, not fixed.** Training batches are uniform and decided up front. Inference
  batches are assembled *on the fly* from whatever requests happen to be in flight — different
  prompts, different lengths, some finishing while others start. Packing them efficiently without
  making anyone wait is an online scheduling problem training never has to solve.
- **The load is adversarial.** A training run is one steady workload. A serving system faces spikes,
  lulls, and pathological requests, and must hold its latency promise through all of them on a fixed
  fleet of GPUs.

So inference removes a pass and adds a *systems* problem. The hard part moved from "compute the
gradients" to "meet a deadline, under unpredictable load, on hardware that's awkward for the work."

## Stateless service, stateful generation

One subtlety trips up almost everyone, and it sets up several later chapters. An inference *service*
is **stateless** between requests: each call is independent, the server remembers nothing about your
last one. But a single *generation* is deeply **stateful** *within* itself — as it produces each
token, it accumulates internal working memory (the **KV cache**, Chapter 2) that grows with every
token and must be kept for the duration of that response.

!!! key "The server forgets between calls; the model remembers within a call"
    Because the service is stateless, a chat application must resend the *entire* conversation on
    every turn — the model has no memory of previous requests. And because a single generation is
    stateful, the cost of a request grows with its context length (more tokens cached). These two
    facts together are why **prefix caching** and **KV-cache-aware routing** (Chapters 5 and 8) exist:
    they claw back the work of re-processing context the system technically "forgot." Hold this — it's
    the seed of a lot of later optimization.

## The economics flip

The last difference is financial, and it's the reason this discipline gets investment. **Training is
capital expenditure** — a large, one-time (or occasional) cost to produce an asset. **Inference is
cost of goods sold** — a per-request cost you pay every time the asset is used, forever. A model that
took a fortune to train can still be cheap overall if it's rarely used; a model that was cheap to
fine-tune can dominate your bill if it serves billions of tokens a day. The money is in the
*running*, which is why a few percent of inference efficiency is worth chasing hard.

## Try it: feel the statelessness

Prove the "server forgets between calls" point in ten seconds against any OpenAI-compatible endpoint
(a hosted API, or the vLLM servers from the later chapters). Ask it to remember something, then ask
about it in a *separate* call:

```python
from openai import OpenAI
client = OpenAI()   # or point base_url/api_key at your own endpoint

# Call 1 — tell it a fact
client.chat.completions.create(model="MODEL",
    messages=[{"role": "user", "content": "Remember the number 42."}])

# Call 2 — a brand-new request; the server kept nothing
r = client.chat.completions.create(model="MODEL",
    messages=[{"role": "user", "content": "What number did I just tell you?"}])
print(r.choices[0].message.content)   # it has no idea — each call is independent
```

The model can't answer, because nothing carried over — the second call is a blank slate. The only
reason a chat *app* feels like it remembers is that the app resends the whole history every turn. You
just felt why context is re-sent, why long conversations get more expensive per turn, and why caching
that re-sent context is worth a whole technique later.

---

You now know *why* inference is its own problem. The next step is making it measurable: the specific
numbers a serving system is judged by, and which one your workload actually lives or dies by.

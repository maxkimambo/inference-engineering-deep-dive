# Latency, Throughput & the Metrics

"Make it fast" is not a specification. Before you can optimize a serving system you have to say
*fast at what*, measured *how* — and inference has a handful of specific numbers that mean very
different things. Confusing them is the most common way teams optimize the wrong thing. This section
defines the metrics you actually serve to, and shows why one workload's success metric is another's
irrelevance.

## Latency is two numbers, not one

Because generation is a loop (§ 0.1), the time to answer a request isn't a single duration — it has a
beginning and a rhythm. Two numbers capture it:

- **TTFT — Time To First Token.** How long from sending the request until the *first* output token
  appears. This is the "is it alive?" delay: the user has typed, hit enter, and is waiting for
  *anything* to happen. It's dominated by **prefill** — the model reading and processing your whole
  prompt before it can emit a thing (Chapter 2).
- **ITL — Inter-Token Latency** (also called **TPOT**, time per output token). Once tokens start
  flowing, how long between each one. This is the *pace* of generation — how fast the text streams
  out. It's set by the **decode** loop, one token per step.

The total time to a complete answer is just these composed:

\[
\text{total latency} = \text{TTFT} + (N - 1) \times \text{ITL}
\]

for an `N`-token response. A concrete feel for it — TTFT 300 ms, ITL 20 ms, a 500-token reply:

\[
300\,\text{ms} + 499 \times 20\,\text{ms} \approx 10.3\,\text{s}
\]

!!! key "Streaming makes TTFT and ITL matter more than total latency"
    That 10.3 s sounds awful — but the user sees the first word at **300 ms** and then watches text
    stream at 50 tokens/sec, which *reads* as fast and responsive. This is why modern interfaces
    stream tokens as they generate instead of waiting for the whole answer: perceived speed is **TTFT
    (how soon it starts) + ITL (how smoothly it flows)**, not the total. Optimize the two numbers a
    human actually feels, not the sum they never wait for in one piece. A system with great total
    throughput but a 5-second TTFT feels broken; one with a snappy TTFT and steady ITL feels instant
    even if the full answer takes ten seconds.

## Throughput is a system number

Latency is per request, from the user's side. **Throughput** is per system, from the operator's side:
how much work the deployment completes per unit time, usually measured two ways:

- **Requests per second** — how many concurrent users you can serve.
- **Tokens per second (aggregate)** — total tokens generated across *all* in-flight requests; the
  truest measure of a GPU's useful output.

Throughput is what determines how many GPUs you need and therefore what the service costs. A system
serving 10× the throughput on the same hardware is 10× cheaper per token.

## Why the two fight

Here is the tension from the chapter intro, made concrete. The lever for throughput is **batching** —
running many requests through the GPU together so its hardware stays busy (Chapter 2 explains why this
works: it reuses each expensive memory read across more requests). But a bigger batch means each
request shares the GPU with more neighbors, so every individual token comes out a little slower.
Illustratively:

| Batch size | Per-request speed | System throughput |
|-----------:|------------------:|------------------:|
| 1 | 50 tok/s | 50 tok/s |
| 8 | 42 tok/s | 336 tok/s |
| 32 | 30 tok/s | 960 tok/s |
| 64 | 20 tok/s | 1,280 tok/s |

!!! key "Batch size is the dial between latency and throughput"
    Reading down that table: throughput climbs 25× from batch 1 to 64, while per-request speed *falls*
    by more than half. Same GPU, same model — the only change is how many requests you pack together.
    **Low batch = low latency, low throughput, high cost-per-token. High batch = high throughput, low
    cost-per-token, higher latency.** Choosing where to sit on this curve, for your traffic and your
    latency promise, is one of the most consequential knobs you own. There's no universally right
    setting — only the one that fits your SLO.

## Serve to percentiles, not averages

One measurement discipline before you optimize anything: **a latency target is a percentile, not an
average.** "Average TTFT 200 ms" can hide that 1 in 20 users waits 4 seconds — and averages are
exactly the metric that conceals tail pain, because a few fast requests mask many slow ones. Real SLOs
are stated as percentiles:

- **p50** (median) — the typical experience.
- **p95 / p99** — the *unlucky* experience: 95% or 99% of requests are at least this fast. The tail.

The tail is what users remember and what pages your on-call. "p99 TTFT under 500 ms" is a
specification you can engineer toward and verify; "fast" is not. Always ask of a latency requirement:
*at what percentile?*

## Which metric is *your* metric

The same system, judged by different numbers, depending on the job:

| Workload | Lives or dies by | Cares less about |
|----------|------------------|------------------|
| Interactive chat / copilots | **TTFT + ITL** (it must feel instant) | aggregate throughput |
| Batch / offline (summarize a corpus overnight) | **throughput** (total tokens/hour) | per-request latency |
| Agentic / tool-calling pipelines | **total latency** across many hops | single-token pace |
| High-volume API backend | **throughput at a latency SLO** (goodput, § 0.3) | best-case latency |

Naming your metric first is the whole point of this section: an optimization that doubles throughput
is a triumph for a batch job and irrelevant to a chat app that needed a faster TTFT. Know which number
you're paid to move before you touch a knob.

## Try it: measure TTFT and ITL yourself

You can't optimize what you don't measure, so measure. This streams a response from any
OpenAI-compatible endpoint and reports the two latency numbers that matter — first-token time and the
inter-token pace:

```python
import time
from openai import OpenAI
client = OpenAI()   # or point base_url/api_key at your own endpoint

start = time.perf_counter()
ttft, last, gaps = None, None, []
stream = client.chat.completions.create(
    model="MODEL", stream=True,
    messages=[{"role": "user", "content": "Write three sentences about latency."}])
for chunk in stream:
    if not chunk.choices[0].delta.content:
        continue
    now = time.perf_counter()
    if ttft is None:
        ttft = now - start                 # time to FIRST token
    else:
        gaps.append(now - last)            # inter-token gaps
    last = now

itl = sum(gaps) / len(gaps)
print(f"TTFT: {ttft*1000:.0f} ms   ITL: {itl*1000:.1f} ms   "
      f"stream rate: {1/itl:.0f} tok/s")
```

Run it a few times and you'll see TTFT and ITL move independently — a long prompt inflates TTFT
(more prefill) while leaving ITL flat; a busier server inflates ITL (bigger batch) while TTFT holds.
Wrap it in a loop over 100 requests, sort the TTFTs, and read off index 95 to get your **p95** — and
now you're measuring like an operator, not guessing like a tourist.

---

You can now state what "fast enough" means in numbers. The last piece of the framing is what those
numbers cost — because every latency and throughput choice is ultimately a choice about dollars.

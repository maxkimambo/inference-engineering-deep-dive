# Batching

Quantization, speculative decoding, caching, parallelism, disaggregation — the five techniques that
follow are all layered on top of one move so fundamental it's easy to forget it's a *choice*:
**batching**. It's the single difference between a GPU that's economical to serve on and one that burns
money. This section comes first because everything after it assumes a batched serving loop — and
because the way batching works is the source of the most common confusion in inference.

## Why batching exists — the roofline, one more time

Recall the result Chapter 2 proved: decode is **memory-bound**. Generating one token requires reading
the *entire* model's weights from HBM to do a tiny amount of math, so the weight-read dominates and the
Tensor Cores sit mostly idle, starved for work. The GPU you paid for is barely being used.

Batching fixes exactly that. Run `B` requests through the model together and the weight matrix is read
from memory **once** and applied to all `B` requests' math in the same matmul. One expensive read,
`B`× the useful work — arithmetic intensity rises roughly linearly with batch size, and you climb the
diagonal of the roofline toward the compute roof.

!!! key "Batching trades a little latency for a lot of throughput"
    Throughput rises ~linearly with batch size (more useful tokens per weight read) until you hit one
    of two walls: the **compute roof**, or **KV-cache memory** (below). Per-request latency rises only
    modestly — each request now shares the GPU. That asymmetry — large throughput gain, small latency
    cost — is *the* reason GPUs are economical to serve on, and the lever every cost-per-token
    calculation (Chapter 0) ultimately pulls.

## The misconception: a batch shares weights, not reasoning

Here's the worry that brought you to this page, and it's worth stating plainly because almost everyone
has it: *if I push many requests through the model at once, isn't it "thinking about all of them
together" — tangling their reasoning, and overwriting the KV cache between one set of requests and the
next?*

No. And fixing that intuition is the most important idea in this section.

A batch shares the model's **weights** and nothing else. When `B` sequences pass through a matmul
together, the weight matrix is read once and multiplied against a stacked tensor whose **batch
dimension is independent** — lane `b` only ever touches lane `b`'s own data. Each sequence in the batch
carries its *own* state, completely separate from its neighbors:

- its own **activations** (its hidden state flowing through the layers),
- its own **KV cache** (the keys and values from *its* tokens, and only its tokens),
- its own **attention mask**, so a query in sequence A attends only to A's keys — *never* B's.

```
        shared, read once from HBM            per-sequence, never mixed
        ┌───────────────────────┐      ┌──────────────────────────────────┐
seq A ──┤                       ├──►   A's activations · A's KV · A's mask
seq B ──┤   model weights  W    ├──►   B's activations · B's KV · B's mask
seq C ──┤  (one memory read)    ├──►   C's activations · C's KV · C's mask
        └───────────────────────┘      └──────────────────────────────────┘
            the efficiency win              the independence (why it's safe)
```

The GPU is not performing one tangled reasoning over all the queries. It's running `B` **independent**
reasonings down parallel lanes that happen to be fed by the same weight read. The shared read is the
entire efficiency win; the per-lane independence is why batching is correct.

!!! key "There is no 'the KV cache' — every request has its own, born and freed with it"
    Your specific worry — *"I won't have the same KV cache for the next set of requests"* — dissolves
    once you see the KV cache is **per-sequence, not per-batch**. Each request's KV cache is allocated
    when its prefill runs, grows by one entry per token it generates, and is **freed the instant that
    request finishes**. The batch is just a temporary grouping of independent sequences for one forward
    pass; it owns no shared state to overwrite. The next request gets its own fresh KV cache — or
    deliberately reuses a *prefix* of another via prefix caching (§5.3), but only when you ask for it,
    never by accident. **Batching never mixes, shares, or clobbers one request's cache with another's.**

## Static batching, and why it isn't enough

The naive way to batch is at the **request** level: collect `N` requests, run them through the model
together, return all `N` when done. It works, but it wastes the GPU in two ways:

- **Head-of-line blocking.** The batch runs until its *longest* sequence finishes. A request that
  needed 10 tokens is stuck waiting behind one generating 1,000 — its slot in the batch can't be reused
  until the whole batch completes.
- **Draining.** As sequences finish at different lengths, their lanes go empty but stay reserved, so
  the batch shrinks over time and the GPU runs progressively emptier.
- **No mid-flight admission.** A request that arrives one millisecond after the batch starts must wait
  for the entire batch to finish before it can even begin.

```
static:   [A B C D] all start together ──► finish together (everyone waits for slowest)
          A done···· (idle) ·············┘   ← A's slot wasted until the batch ends
```

## Continuous batching — the technique that matters

The fix, and the default in every serious engine, is to batch at the **iteration** (token step) level
instead of the request level. This is **continuous batching** (vLLM/SGLang; TensorRT-LLM calls it
**in-flight batching**; the idea comes from the Orca paper).

After *every* decode step, the scheduler rebuilds the batch: any sequence that just finished **leaves**
and frees its slot, and any waiting request **joins** immediately. The batch's membership changes every
single iteration, so the GPU stays packed regardless of how wildly request lengths differ. Each
sequence's KV cache simply persists across iterations until that sequence finishes — independent of
who else is in the batch.

```
continuous:  step1 [A B C D]    step2 [A B C D]    step3 [E B C D]  ← A finished, E joined
             step4 [E B F D] ...   slots refill the instant they free; GPU never drains
```

A concrete trace makes it click. Take three requests with different prompt and output lengths — **A**
(4-token prompt, 3 out), **B** (6-token prompt, 5 out), **C** (2-token prompt, 2 out) — sharing one
GPU. Each pass reads the weights *once* and advances every **active** lane by one token; a lane that
hits its limit finishes and frees its KV cache and its slot, without disturbing the others. Step
through it:

<div id="cb-widget" style="margin:1.2em 0;font-size:14px;" aria-label="Step-through of three requests A, B and C flowing through continuous batching.">
<style>
#cb-widget button{border:1px solid var(--md-default-fg-color--lighter);background:transparent;color:var(--md-default-fg-color);border-radius:6px;padding:5px 11px;font:inherit;font-size:13px;cursor:pointer;line-height:1.3;}
#cb-widget button:hover:not(:disabled){background:var(--md-default-fg-color--lightest);}
#cb-widget button:disabled{opacity:.35;cursor:default;}
</style>
<div style="background:var(--md-code-bg-color);border-radius:10px;padding:12px 14px;margin-bottom:12px;">
<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
<div style="display:flex;align-items:center;justify-content:center;min-width:112px;height:44px;background:var(--md-default-bg-color);border:1px solid var(--md-default-fg-color--lightest);border-radius:7px;text-align:center;line-height:1.25;font-size:12px;">model weights<br><span style="color:var(--md-default-fg-color--light);">read once / pass</span></div>
<div style="font-size:20px;color:var(--md-default-fg-color--lighter);">&rarr;</div>
<div style="flex:1;min-width:210px;">
<div style="font-size:13px;color:var(--md-default-fg-color--light);"><span id="cb-pass" style="font-weight:600;color:var(--md-default-fg-color);">arrive</span> &middot; applied to <span id="cb-n">0</span> active lane(s)</div>
<div id="cb-status" style="font-size:13px;color:var(--md-default-fg-color--light);line-height:1.5;margin-top:3px;">Three requests arrive with different prompts. Nothing has run yet.</div>
</div>
</div>
</div>
<div id="cb-lanes" style="display:flex;flex-direction:column;gap:9px;"></div>
<div style="display:flex;align-items:center;gap:7px;margin-top:13px;">
<button id="cb-prev" aria-label="previous step">&lsaquo;</button>
<button id="cb-next" style="min-width:78px;">step &rsaquo;</button>
<button id="cb-play">&#9654; play</button>
<button id="cb-reset" aria-label="reset">&#8635;</button>
<span id="cb-count" style="margin-left:auto;font-size:12px;color:var(--md-default-fg-color--lighter);"></span>
</div>
<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;font-size:12px;color:var(--md-default-fg-color--lighter);">
<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;border:1px solid var(--md-default-fg-color--lighter);vertical-align:-1px;"></span> prompt token (shared weights)</span>
<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#108a66;vertical-align:-1px;"></span> generated token (own lane)</span>
<span>KV = private cache per request</span>
</div>
<script>
(function(){
var P={A:4,B:6,C:2},MAXKV=11,ids=['A','B','C'];
var col={A:{s:'#108a66',t:'rgba(16,138,102,.16)'},B:{s:'#6b46d9',t:'rgba(107,70,217,.16)'},C:{s:'#b9740f',t:'rgba(185,116,15,.16)'}};
var F=[
{p:'arrive',r:{A:{g:0,s:'queued'},B:{g:0,s:'queued'},C:{g:0,s:'queued'}},x:'Three requests arrive with different prompts. Nothing has run yet.'},
{p:'prefill',r:{A:{g:1,s:'active'},B:{g:1,s:'active'},C:{g:1,s:'active'}},x:'One shared forward pass reads every prompt and emits each request’s first token. Each now has its own KV cache.'},
{p:'decode 1',r:{A:{g:2,s:'active'},B:{g:2,s:'active'},C:{g:2,s:'done'}},x:'Each active request generates one more token in a single shared pass. C hits its limit, finishes, and its KV cache is freed.'},
{p:'decode 2',r:{A:{g:3,s:'done'},B:{g:3,s:'active'},C:{g:2,s:'done'}},x:'C’s slot is now free. A finishes this step; B keeps generating.'},
{p:'decode 3',r:{A:{g:3,s:'done'},B:{g:4,s:'active'},C:{g:2,s:'done'}},x:'Only B is left in the batch — it has all the lanes to itself now.'},
{p:'decode 4',r:{A:{g:3,s:'done'},B:{g:5,s:'done'},C:{g:2,s:'done'}},x:'B reaches its limit and finishes. Every KV cache is now freed.'},
{p:'done',r:{A:{g:3,s:'done'},B:{g:5,s:'done'},C:{g:2,s:'done'}},x:'All three ran independently down parallel lanes — sharing only the weights, never each other’s tokens or KV cache.'}];
var i=0,timer=null;
var $=function(id){return document.getElementById(id);};
function pills(n,kind,c){var o='';for(var k=0;k<n;k++){var st=kind==='p'?'border:1px solid var(--md-default-fg-color--lighter);':'background:'+c.s+';';o+='<span style="display:inline-block;width:13px;height:13px;border-radius:3px;margin:1px;'+st+'"></span>';}return o;}
function render(){
var f=F[i];
$('cb-pass').textContent=f.p;
$('cb-status').textContent=f.x;
$('cb-n').textContent=ids.filter(function(d){return f.r[d].s==='active';}).length;
$('cb-count').textContent='step '+(i+1)+' of '+F.length;
$('cb-lanes').innerHTML=ids.map(function(d){
var r=f.r[d],c=col[d],done=r.s==='done',q=r.s==='queued',kv=P[d]+r.g,w=Math.round(kv/MAXKV*100);
var badge='<div style="display:flex;align-items:center;justify-content:center;width:29px;height:29px;border-radius:50%;background:'+c.s+';color:#fff;font-weight:600;font-size:14px;flex:none;">'+d+'</div>';
var tag=q?'<span style="font-size:12px;color:var(--md-default-fg-color--lighter);">queued</span>':done?'<span style="font-size:12px;color:var(--md-default-fg-color--light);">✓ done</span>':'<span style="font-size:12px;color:'+c.s+';font-weight:600;">▶ active</span>';
var kvt=done?'<span style="font-size:12px;color:var(--md-default-fg-color--lighter);">KV freed</span>':'<span style="font-size:12px;color:var(--md-default-fg-color--light);">KV '+kv+'</span>';
var bar=done?'<div style="height:6px;border-radius:3px;background:var(--md-default-fg-color--lightest);"></div>':'<div style="height:6px;border-radius:3px;background:var(--md-default-fg-color--lightest);"><div style="height:6px;border-radius:3px;width:'+w+'%;background:'+c.s+';"></div></div>';
return '<div style="display:flex;align-items:center;gap:11px;background:var(--md-code-bg-color);border:1px solid var(--md-default-fg-color--lightest);border-radius:9px;padding:9px 11px;'+(done?'opacity:.72;':'')+'"><div>'+badge+'</div><div style="flex:1;min-width:0;"><div style="margin-bottom:5px;line-height:1;">'+pills(P[d],'p')+pills(r.g,'g',c)+'<span style="font-size:11px;color:var(--md-default-fg-color--lighter);margin-left:6px;">prompt '+P[d]+' + gen '+r.g+'</span></div><div style="display:flex;align-items:center;gap:8px;"><div style="flex:1;">'+bar+'</div>'+kvt+'</div></div><div style="flex:none;width:56px;text-align:right;">'+tag+'</div></div>';
}).join('');
$('cb-prev').disabled=(i===0);$('cb-next').disabled=(i===F.length-1);
}
function step(d){i=Math.max(0,Math.min(F.length-1,i+d));render();}
function stop(){if(timer){clearInterval(timer);timer=null;$('cb-play').innerHTML='▶ play';}}
$('cb-next').onclick=function(){stop();step(1);};
$('cb-prev').onclick=function(){stop();step(-1);};
$('cb-reset').onclick=function(){stop();i=0;render();};
$('cb-play').onclick=function(){if(timer){stop();return;}if(i===F.length-1){i=0;render();}$('cb-play').innerHTML='❙❙ pause';timer=setInterval(function(){if(i>=F.length-1){stop();}else{step(1);}},1400);};
render();
})();
</script>
</div>

C finishes first and frees its slot mid-flight; A finishes next; B runs on alone — and at no point did
any lane read another's tokens or KV cache. The batch is simply *who is in this pass*, and that
membership changes every step. (`✔` = hit its token limit; `KV` counts prompt + generated tokens, the
private cache each lane carries until it's done.)

??? note "Prefer it static — the same trace as a table"
    ```
    pass       lane A (own KV)   lane B (own KV)   lane C (own KV)   weights
    ─────────  ───────────────   ───────────────   ───────────────   ───────
    prefill    gen 1   KV  5     gen 1   KV  7     gen 1   KV  3     read 1×
    decode 1   gen 2   KV  6     gen 2   KV  8     gen 2   KV  4 ✔   read 1×
    decode 2   gen 3   KV  7 ✔   gen 3   KV  9      — slot free —    read 1×
    decode 3    — done —         gen 4   KV 10      — done —         read 1×
    decode 4    — done —         gen 5   KV 11 ✔    — done —         read 1×
    ```

!!! key "Continuous batching is the baseline, not an optimization you add"
    Turning it off is the unusual choice. It's why a single GPU can sustain dozens of concurrent users
    at a stable token rate, and why "throughput at batch 1" benchmarks are meaningless for real
    serving. Every later technique in this chapter assumes a continuously batched loop underneath it.

## Prefill and decode compete — and chunked prefill referees

One subtlety shapes several knobs. **Prefill** (processing a whole prompt at once) is compute-bound and
bursty; **decode** (one token per step) is memory-bound and steady. In a single engine they share the
same forward passes — so a long prompt's prefill entering the batch can **stall every ongoing
decode**, producing a visible stutter in everyone else's token stream (an ITL spike).

**Chunked prefill** is the referee: split a long prefill into bounded chunks and interleave them with
decode steps, so decodes keep flowing while the big prompt is ingested. The cost is a slightly higher
TTFT for that one large prompt; the benefit is that one user's 8K-token paste doesn't freeze everyone
else. (The heavier-duty version of this idea — separate hardware for prefill and decode — is
disaggregation, §5.5.)

## The knobs

What you actually tune (vLLM names shown; other engines have equivalents):

| Knob | What it controls | Push it up → | Push it down → |
|------|------------------|--------------|----------------|
| **continuous batching** | iteration- vs request-level | (leave on) | only for niche offline runs |
| **`max_num_seqs`** | max concurrent sequences in a batch | more throughput, more KV memory, higher ITL | lower latency, less throughput |
| **`max_num_batched_tokens`** | token budget per forward pass | bigger prefill chunks / more decode | smoother interleaving |
| **`enable_chunked_prefill`** + chunk size | how prefill interleaves with decode | protects ITL under long prompts | simpler, but prefill can stall decode |
| **`gpu_memory_utilization`** | share of HBM given to the KV cache | room for a bigger batch | safety margin |
| **scheduling / preemption** | what happens under KV pressure | — | recompute vs swap-to-host (below) |

## Memory is the real ceiling

Batch size is rarely limited by compute — it's limited by **KV-cache memory**. Every concurrent
sequence rents HBM for its entire context (Chapter 3, §3.2): a Llama-3-70B sequence at 8K context is
~2.7 GB of KV cache, so even after quantizing weights to INT4 (leaving ~45 GB on an 80 GB GPU) you fit
only ~16 such sequences before you're out of memory.

Two mechanisms stretch that ceiling:

- **PagedAttention** stores each sequence's KV cache as non-contiguous fixed-size *pages* (like virtual
  memory), eliminating the fragmentation that otherwise wastes a large fraction of HBM — so you pack
  more sequences into the same card.
- **Preemption.** When new requests would exceed KV memory, the scheduler evicts a running sequence and
  resumes it later — either by **recomputing** its KV (drop it, redo prefill on return) or **swapping**
  it to host memory. Which one is a knob; recompute is usually cheaper than the PCIe round-trip.

This is why quantizing the *KV cache* (§5.1) is so synergistic: smaller per-token KV → more sequences
fit → bigger batches → more throughput. Memory bought is batch size bought.

## Worked example: the throughput/latency curve

Sweeping batch size on one GPU and model traces out the curve you're choosing a point on:

| Batch size | Per-request speed | System throughput |
|-----------:|------------------:|------------------:|
| 1 | 50 tok/s | 50 tok/s |
| 8 | 42 tok/s | 336 tok/s |
| 32 | 30 tok/s | 960 tok/s |
| 64 | 20 tok/s | 1,280 tok/s |

Throughput climbs ~25× from batch 1 to 64 while per-request speed more than halves. There is no
"correct" row — only the largest batch whose per-request latency still clears your SLO. That point is
**maximum goodput** (Chapter 0): the most *useful* tokens per GPU-hour, not the most tokens.

## How to optimize batching — for whichever metric you chase

!!! key "Set batch size to your SLO, then let the other four techniques buy back what it cost"
    - **Latency-critical (chat, copilots):** cap `max_num_seqs` lower, **enable chunked prefill** so big
      prompts don't stall decodes, and if you're at scale, **disaggregate** (§5.5) so prefill never
      shares a GPU with decode.
    - **Throughput/cost-critical (batch jobs, high-volume APIs):** push `max_num_seqs` high, raise
      `gpu_memory_utilization`, and **quantize the KV cache** (§5.1) plus rely on PagedAttention to fit
      more concurrent sequences. Accept the higher ITL.
    - **Mind the interactions:** a bigger batch *starves speculative decoding* (§5.2) of the spare
      compute it feeds on; quantization frees memory for bigger batches; disaggregation lets prefill and
      decode pick their batch sizes independently. These are knobs to balance, not boxes to check.

## Try it: watch continuous batching work

Two probes against any vLLM (or OpenAI-compatible) endpoint. First, the throughput/latency curve —
fire rising concurrency and watch aggregate tokens/s climb while per-request latency grows:

```python
import asyncio, time
from openai import AsyncOpenAI
client = AsyncOpenAI()   # point base_url/api_key at your endpoint

async def one():
    t = time.perf_counter()
    r = await client.chat.completions.create(model="MODEL", max_tokens=128,
            messages=[{"role": "user", "content": "Count slowly to fifty."}])
    return time.perf_counter() - t, r.usage.completion_tokens

async def sweep(n):
    t = time.perf_counter()
    res = await asyncio.gather(*[one() for _ in range(n)])
    wall = time.perf_counter() - t
    toks = sum(c for _, c in res)
    lat  = sorted(l for l, _ in res)
    print(f"concurrency {n:>3}: {toks/wall:6.0f} tok/s aggregate | "
          f"p50 {lat[len(lat)//2]:.2f}s  p95 {lat[int(len(lat)*0.95)-1]:.2f}s")

for n in (1, 4, 16, 64):
    asyncio.run(sweep(n))
```

You'll see aggregate throughput rise far faster than concurrency while latency creeps up — the curve
above, on your hardware. For the second probe, fire a mix of `max_tokens=8` and `max_tokens=512`
requests together: the short ones return quickly *without* waiting for the long ones — continuous
batching freeing slots mid-flight, exactly the head-of-line blocking that static batching couldn't
avoid.

---

Batching is the floor everything else stands on. With a continuously batched loop in place, the five
techniques that follow each move a *different* corner of the roofline — and you now know the one they
all assume, and why pushing more requests through the model never tangles their reasoning or their
caches.

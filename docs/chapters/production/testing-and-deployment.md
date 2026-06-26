# Testing and Deployment

Beyond the replica-level benchmarking you did configuring the engine, a production system must be
tested **end-to-end** before deploying — and deployed in a way that can't take down live traffic.

## Testing strategies

- **Manual testing** — scripts (or button clicks) sending *synthetic* traffic to the service.
- **Load testing** — automatically sending a *large volume* to test scaling and sustained
  performance.
- **Shadow traffic** — *copying live traffic* to a deployment to measure performance under real-world
  conditions, without affecting users.

Testing inference is **expensive** — engineering time to build/measure it, plus GPUs to serve the
test traffic. Minimize the cost: e.g. start shadow testing with a *random sample* of production
traffic, followed by a *shorter* load test. And remember AI usage fluctuates on **daily and weekly
cycles** — a test at 3 a.m. Sunday tells you little about Monday's peak.

## 7.4.1 Zero-downtime deployment

The traditional high-availability deploy is **blue-green**: two identical environments — the live
*blue* and the updated *green*. When green is ready, all traffic cuts over at once; blue stays ready
for rollback.

!!! warning "Blue-green doesn't fit large-scale inference"
    It **doubles your GPUs**. If blue runs 100 GPUs, green needs another 100 *before* you can cut
    over — the same capacity/cost problem that makes inference testing hard. For GPU-heavy services,
    blue-green is usually unaffordable.

The GPU-frugal alternative is a **canary deployment** (named for the canaries that warned coal
miners) — catch errors on a *small* slice of traffic before they hit everyone:

```
            ┌──► OLD DEPLOYMENT  (most traffic)
 TRAFFIC ───┤
            └──► NEW DEPLOYMENT  (small, growing %)   ← watch it here
```

1. Build the new deployment and get it ready for requests.
2. Route a **small percentage** of live traffic to it.
3. Monitor — is it handling traffic correctly? **Revert** if not.
4. Gradually raise its share, monitoring, until it serves **100%**.

Canary can ramp in minutes or roll out slowly for stability. And unlike blue-green, **it barely adds
cost at scale**: shifting traffic *off* the old deployment scales *it* down, so you're not paying for
two full fleets.

!!! key "Keep the canary warm"
    With autoscaling, a brand-new deployment starts at *minimum* replicas. During the ramp, make sure
    the new deployment always has enough active replicas for the traffic you're sending it — otherwise
    its requests queue behind cold starts and users see a latency spike. Ramp the replicas alongside
    the traffic.

## 7.4.2 Cost estimation

Moving from a public API to dedicated GPUs changes how you think about cost — the whole point is to
**escape per-token pricing and own your unit economics**, but it makes estimation harder.

**Public API cost is simple** — a linear function of usage:

```python
# per-token API
total_input_tokens  = 1000   # millions
total_output_tokens = 500    # millions
price_per_million_in  = 1.25
price_per_million_out = 10

input_cost  = total_input_tokens  * price_per_million_in    # 1250
output_cost = total_output_tokens * price_per_million_out   # 5000
total_cost  = input_cost + output_cost                      # $6,250
```

**Dedicated cost** is a function of many variables — batch sizing (latency vs throughput tuning),
traffic patterns (are GPUs saturated or idle?), and sequence lengths (input/output tokens, average
*and* outlier). Rather than reverse-engineer a per-token price from your GPU bill, convert the *other*
way — turn your token usage into a total and compare:

```python
# dedicated deployment
total_gpu_hours    = 1600
price_per_gpu_hour = 3.50
total_cost = total_gpu_hours * price_per_gpu_hour           # $5,600
```

Here dedicated ($5,600) beats the API ($6,250) — but only at this usage level and utilization. Below
some volume the API wins; the crossover *is* the "are we ready for dedicated infra?" decision.

!!! key "Use a long horizon, and count engineering time (TCO)"
    Estimate over **at least a week** to smooth daily/weekly cycles — a single day misleads. And the
    GPU bill isn't the whole story: the **engineering time** to build and maintain the inference
    system is a real cost. Add it to the GPU spend for true **total cost of ownership (TCO)**. Dedicated
    inference buys reliability, security, and control — but those have a payroll line, not just a
    hardware line.

## 7.4.3 Observability

Inference is mission-critical, so monitor it like any mission-critical component — **alerting, logs,
and observability at the right level of abstraction.**

What to measure:

| Metric | What it tells you |
|--------|-------------------|
| **Total volume** | requests a deployment is receiving |
| **Request/response sizes** | input and output sequence lengths |
| **Response codes** | counts of 2XX / 4XX / 5XX from the model server |
| **Latency** | TTFT, TPS, end-to-end — at **p50, p90, p99** |
| **Replica count** | instances serving + instances starting up |
| **Utilization** | CPU, host memory, GPU, GPU memory |
| **Queue depth** | requests enqueued and waiting (for async traffic) |

!!! key "Metrics are only useful together"
    They're interdependent — a latency spike *could* be request volume, or it *could* be a few
    long-input-sequence requests. Seeing the metrics **side by side** is what turns "*what* is
    happening" into "*why*." A p99 latency alert next to a flat volume graph but a spiking
    input-size graph tells the whole story at a glance.

When things break, you need **logs** — both server logs and **audit logs** (who changed the inference
service, and when) — delivered in real time.

!!! info "Don't silo inference observability"
    Build it with **deep integration into your existing tooling** — Grafana, Datadog, PagerDuty,
    Sentry — so inference metrics sit in context next to the rest of the application. An inference
    dashboard nobody looks at because it's in a separate system is worse than no dashboard.

**Next:** [Client Code →](client-code.md)

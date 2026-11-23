# Part A — Reference-Assisted Tutorial and Pre-Coding Design Record

- **Course:** CSP502 Software Lab-I (M.Tech. CSE W26)
- **Practical:** Practical 8 — From Resilience to Reliability: Testing and Observing a Concurrent Service in Go
- **Group Members:** Gaurav Acharya (`MT26MCS024`), Sandeep (`MT26MCS032`)
- **Timestamp (Pre-Coding Freeze):** `2026-11-18T14:15:00+05:30`

---

## 1. Technical References Consulted (Cited Sections)

1. **Go Testing Tutorial** (`https://go.dev/doc/tutorial/add-a-test`)
   - *Sections used:* "Write a test", "Run the tests", "Add a failing test".
   - *Takeaways:* Standard `testing` package (`*testing.T`), naming convention `TestXxx`, deterministic assertions (`t.Fatalf` vs `t.Errorf`), running targeted and full suites via `go test ./...` and `go test -v`.
2. **Data Race Detector** (`https://go.dev/doc/articles/race_detector`)
   - *Sections used:* "Introduction", "Usage", "Report Format", "Typical Data Races", "Runtime Overheads".
   - *Takeaways:* ThreadSanitizer-based dynamic happens-before analysis (`go test -race ./...`); detects only unsynchronized concurrent memory accesses along *executed* paths; imposes 5–10x memory and 2–20x CPU overhead, making race-instrumented timings invalid as performance benchmarks.
3. **Package `context` Documentation** (`https://pkg.go.dev/context`)
   - *Sections used:* "Overview", `WithCancel`, `WithTimeout`, `WithDeadline`, `Context.Done`, `Context.Err`.
   - *Takeaways:* Cooperative cancellation propagation across goroutine trees; always `defer cancel()` to release timer/child resources; distinguish `context.Canceled` from `context.DeadlineExceeded`.
4. **Go Concurrency Patterns: Pipelines and Cancellation** (`https://go.dev/blog/pipelines`)
   - *Sections used:* "Fan-out, fan-in", "Stopping short", "Explicit cancellation", "Guidelines for pipeline construction".
   - *Takeaways:* Senders close outbound channels once all sends finish; receivers abort via an explicit `done`/context channel in `select` to prevent orphaned blocked goroutines (goroutine leaks).

---

## 2. Claim Analysis (A1)

| # | Claim | Classification | Reasoning |
|---|---|---|---|
| 1 | A single successful run proves that no request can ever be lost. | **Unjustified** | Concurrent schedules are nondeterministic; one run exercises one goroutine interleaving under one timing profile. Buffer drops, race windows, or shutdown losses can appear only under different scheduling or load bursts. Testing demonstrates the presence of bugs, never their absence. |
| 2 | A race-detector pass proves the service cannot deadlock. | **Unjustified** | Go's `-race` detector checks for unsynchronized concurrent read/write memory accesses (safety/data-race property). Deadlocks are liveness failures (circular wait on locks or permanently blocked channels) that can occur in completely race-free code. Furthermore, `-race` inspects only executed branches. |
| 3 | Retrying a request is harmless when its result is recorded only once. | **Conditionally Justified** | Recording a result once (idempotent commit) prevents duplicate terminal records *only if* (a) the underlying work/downstream side effects are themselves idempotent (no duplicate fee charges or certificate issuance), and (b) retries are bounded by backoff and a retry budget so they do not trigger retry storms that exhaust worker capacity. |
| 4 | A clean shutdown can require rejecting new requests while finishing accepted ones. | **Justified** | Clean/graceful shutdown requires two phases: closing the intake gate (rejecting new arrivals with an explicit `Rejected` status) so the work queue becomes finite, and draining already-accepted requests within a bounded grace period. Keeping intake open under continuous traffic prevents the queue from ever draining. |
| 5 | A low mean response time guarantees acceptable tail latency. | **Unjustified** | Arithmetic means mask heavy-tailed or bimodal distributions. If 95% of requests finish in 8 ms and 5% stall behind slow workers for 500 ms, the mean is ~32.6 ms while p95/p99 reaches 500 ms. Tail latency governs user-perceived SLA compliance. |

---

## 3. Testable Invariants (A2)

1. **I1 — Request Accounting Conservation (Safety):**
   $$N_{\text{generated}} = N_{\text{success}} + N_{\text{failure}} + N_{\text{timeout}} + N_{\text{cancelled}} + N_{\text{rejected}}$$
   Every uniquely generated request ID must map to a terminal count; zero stranded or lost requests.
2. **I2 — Unique Terminal Outcome (Safety / Idempotency):**
   Each unique `RequestID` must commit *exactly one* terminal outcome from the mutually exclusive set `{Success, Failure, Timeout, Cancelled, Rejected}`. Duplicate deliveries or repeated downstream acknowledgements must not create duplicate terminal records or mutate an already-committed terminal outcome.
3. **I3 — Monotonic Lifecycle Ordering (Safety):**
   Lifecycle timestamps per request must satisfy causal ordering:
   - If rejected at intake: `t_generated <= t_rejected == t_terminal` (no `started` event).
   - If admitted and processed: `t_generated <= t_admitted <= t_started <= t_terminal`.
   - If admitted and cancelled while queued: `t_generated <= t_admitted <= t_terminal` (no `started` event).
4. **I4 — Bounded Shutdown & Goroutine Liveness (Liveness):**
   Initiating shutdown closes intake immediately (subsequent arrivals return `Rejected`), terminates all workers within $T_{\text{grace}} + \epsilon$, leaves `outstanding_work == 0`, and leaks zero goroutines (`DeltaGoroutines <= 0` after settle).
5. **I5 — Fault Isolation (Liveness / Containment):**
   Deterministic worker errors on a configured subset of request IDs must yield `Failure` only for those IDs without crashing workers or preventing unrelated requests from completing with `Success`.
6. **I6 — Bounded Cancellation & Timeout Enforcement (Liveness):**
   Slow workers exceeding the configured SLA deadline or active cancellations must unblock workers and drain queued items promptly rather than blocking indefinitely.

### What Tests Can and Cannot Prove
- **Can prove:** Violations of invariants on specific deterministic seeds, fault schedules, and concurrency levels; presence of data races along exercised code paths (`-race`); regression protection for fixed defects; empirical latency/throughput envelopes on the test hardware.
- **Cannot prove:** Absence of all data races on untested code paths; absence of rare interleaving deadlocks; resilience to unmodeled production failures (process SIGKILL, kernel OOM killer, network partitions, disk full, clock skew, GC pauses under multi-hour heap fragmentation).

---

## 4. Pre-Coding Predictions

1. **Slow Workers:** In the baseline (no per-request timeout, unbounded queue), injecting 10% slow workers (140 ms vs 5–15 ms normal) under surge load will cause Head-of-Line (HOL) blocking across the 8-worker pool, growing queue depth past 150–300 requests, pushing p95 response time into hundreds of milliseconds, and cutting throughput dramatically. With SLA timeouts and bounded queues, slow requests will abort at ~40 ms, freeing workers and keeping p95 bounded.
2. **Cancellation:** Cancelling mid-run while requests are buffered in the channel will expose accounting violations in a naive baseline if workers exit immediately without draining queued items as `Cancelled`. A proper cancellation drain will account for 100% of queued items as `Cancelled` and terminate within the grace period.
3. **Duplicate Delivery:** Injecting 5% duplicate Request IDs into a naive baseline without intake deduplication will cause duplicate processing and duplicate terminal records (violating I2). Adding an intake/result idempotency ledger will detect and reject/deduplicate repeats with `Duplicate terminal outcomes = 0`.

---

## 5. Pre-Coding Test Matrix

| Test ID | Target Scenario | Fault Parameters | Invariants Verified | Expected Baseline Result | Expected Improved Result |
|---|---|---|---|---|---|
| `T0` | Normal workload (3 runs) | `N=1200, W=8, seed={101,202,303}`, 0 faults | I1, I2, I3 | PASS | PASS |
| `T1` | Duplicate-ID injection | `N=200`, 20 duplicate IDs re-submitted concurrently | I1, I2 | FAIL (duplicate terminal records) | PASS (0 duplicate terminal records) |
| `T2` | Mid-flight cancellation | `N=300, W=4`, cancel at 80 ms while queue active | I1, I4, I6 | FAIL (stranded queued requests) | PASS (all queued drained as `Cancelled`) |
| `T3` | Slow-worker injection | `N=200, W=4`, 15% slow (150 ms), SLA 40 ms | I3, I6 | FAIL (no timeout, unbounded queue/p95) | PASS (timed out at ~40 ms, queue bounded) |
| `T4` | Deterministic worker failures | `N=200, W=4`, every 10th ID fails deterministically | I1, I5 | PASS (unrelated requests succeed) | PASS (unrelated requests succeed) |
| `T5` | Active shutdown & leak check | `N=400, W=4`, shutdown at 100 ms, grace 120 ms | I1, I4 | FAIL (shutdown > grace / stranded work) | PASS (`outstanding=0`, 0 leaked goroutines) |
| `T6` | Race detector (`-race`) | Full test suite under `-race` | Thread safety | Zero data races (after channel-close guard) | Zero data races |


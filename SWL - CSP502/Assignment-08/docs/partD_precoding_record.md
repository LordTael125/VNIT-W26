# Part D — Instructor-Issued Reliability Change: Pre-Coding Design Record

- **Course:** CSP502 Software Lab-I (M.Tech. CSE W26)
- **Practical:** Practical 8 — From Resilience to Reliability: Testing and Observing a Concurrent Service in Go
- **Group Members:** Gaurav Acharya (`MT26MCS024`), Sandeep (`MT26MCS032`)
- **Timestamp (Recorded Before Implementing Part E):** `2026-11-18T15:25:00+05:30`

---

## 1. Instructor-Issued Requirement (Recorded Verbatim)

> "A downstream acknowledgement may be delayed or repeated. The service must preserve a single terminal result per request while allowing safe recovery after an acknowledgement timeout. Students must specify whether their guarantee is at-most-once processing, at-least-once attempts, or exactly-once recorded outcome; these are not interchangeable."

---

## 2. Affected Baseline Assumptions

1. **Instantaneous, Synchronous Result Collection:** The Part B baseline treated `ResultCollector.Record(req)` as an infallible, zero-latency, single-invocation local call. In reality, downstream acknowledgement behaves like an asynchronous RPC/sink subject to variable latency, timeouts, packet drops, and duplicate delivery.
2. **One-to-One Correspondence Between Worker Execution and Terminal Recording:** Baseline assumed each admitted request invokes result recording once. Under delayed acknowledgements, a worker may time out waiting for Attempt 1 and issue Attempt 2 while Attempt 1's acknowledgement is still in flight—producing two acknowledgements for one request.
3. **Unprotected Terminal State Mutation:** Baseline appended every recorded result unconditionally to a slice; repeated or late acknowledgements therefore produced duplicate terminal entries and inflated terminal counts above $N_{\text{generated}}$, violating Invariants I1 and I2.
4. **Fixed Worker Occupancy Duration:** Worker holding time now equals `ProcessDuration + AckWaitDuration` (including retry backoff). Unbounded retries would amplify worker occupancy and trigger retry storms under downstream slowdowns.

---

## 3. Explicit Reliability Guarantee Specification

Our improved service provides:
- **At-Least-Once Acknowledgement Delivery Attempts** (bounded by per-request `MaxAckRetries` and a global `RetryBudget`), combined with
- **Exactly-Once Recorded Outcome** inside the service's `ResultLedger` (keyed by `RequestID`).

We explicitly distinguish the three non-interchangeable guarantees:
1. **At-most-once processing:** The request is attempted at most once; if the acknowledgement times out, the system fails fast without retrying. Safe against duplicate side effects, but suffers false failures on transient ack delays.
2. **At-least-once attempts:** On acknowledgement timeout, delivery is retried until acknowledged or retries are exhausted. Recovers from transient delays/drops, but downstream receivers can observe duplicate delivery attempts.
3. **Exactly-once recorded outcome:** The service's terminal ledger atomically commits *at most one* and *exactly one* terminal outcome per `RequestID` (first-commit-wins via mutex-protected compare-and-commit). Late or duplicate acknowledgements are safely deduplicated and logged without mutating the committed terminal record or skewing accounting totals. Note: this does **not** guarantee *exactly-once physical execution* in an external non-idempotent downstream without distributed two-phase commit or downstream idempotency keys.

---

## 4. Design Options Considered

### Option A — Fail-Fast At-Most-Once Acknowledgement (Single Attempt + Tombstone Map)
- **Mechanism:** Send downstream ack once and wait up to `AckTimeout`. If it times out, commit `Timeout` into an idempotent tombstone map and discard any late/repeated acks.
- **Advantages:** Simplest implementation; strictly bounded latency (`<= ProcessTime + AckTimeout`); zero retry traffic amplification; cannot cause duplicate downstream delivery.
- **Disadvantages:** Every transient ack delay (> `AckTimeout`) becomes a permanent user-visible failure even when downstream succeeded; lowers goodput significantly under 15% ack delay.

### Option B (Selected) — Bounded At-Least-Once Ack Recovery + Idempotent Commit Ledger
- **Mechanism:**
  1. Separate **business processing** from **downstream acknowledgement** so retrying a timed-out acknowledgement re-sends the already-computed result payload carrying `RequestID` as an idempotency token.
  2. Enforce a per-request retry limit (`MaxAckRetries = 3`), exponential backoff with deterministic jitter (`2ms, 4ms, 8ms`), and a service-wide **Retry Budget** (`RetryBudgetRatio * TotalRequests`) so systemic downstream slowdowns cannot trigger retry storms.
  3. Maintain a thread-safe **Idempotent Result Ledger** (`map[uint64]TerminalRecord`). A helper `CommitOnce(req, status, reason)` atomically commits the terminal outcome only if `req.ID` is uncommitted; any late ack from an earlier attempt or repeated downstream ack hits the already-committed entry, increments `DuplicateAcksIgnored`, logs `ack_duplicate_ignored`, and returns without modifying the ledger or accounting totals.
  4. Combine with **Explicit Admission Control** (bounded queue), **Per-Request SLA Deadlines**, and **Graceful Bounded Shutdown** (two-phase admission close + bounded worker drain).
- **Advantages:** Recovers cleanly from transient ack delays and drops (restoring high goodput), strictly preserves Invariants I1 and I2 (`0` duplicate terminal outcomes, `0` accounting violations), and prevents retry storms via the global retry budget.
- **Disadvantages:** Slightly higher mean/p95 latency on delayed-ack requests (waiting for timeout + retry); modest memory overhead for the `ResultLedger` map; slightly higher code complexity requiring careful lock-ordering and race verification.

---

## 5. Expected Trade-Offs and Pre-Coding Predictions

1. **Duplicate Terminal Outcomes:** Baseline will record significant duplicate terminal outcomes (~15–20% of requests under `ack_faults.json` due to 15% repeated acks + 5% duplicate submissions + delayed late acks). Improved will reduce duplicate terminal outcomes to **strictly 0**.
2. **Terminal Accounting Violations:** Baseline will suffer accounting violations whenever duplicate acks inflate `len(records) > TotalGenerated` or dropped acks leave requests unrecorded. Improved will maintain **0 accounting violations** across all 4 conditions and all 3 repetitions.
3. **Goodput vs. Latency Trade-Off under `ack_faults`:** Because Improved retries timed-out/dropped acks (with backoff) instead of blindly accepting or failing immediately, goodput will remain high (~90%+ success excluding deterministic worker failures & duplicate intake rejections), while p95 latency under `ack_faults` will increase moderately (~30–60 ms) compared to normal steady state (~15 ms), reflecting the honest cost of timeout detection and retry recovery.
4. **Overload & Shutdown:** Under `overload_slow` and `failure_cancel`, Improved's explicit admission control, SLA timeouts, and two-phase bounded shutdown will slash p95 latency and shutdown completion time drastically compared to Baseline, at the deliberate cost of fast-rejecting excess traffic (`Rejected > 0`) and timing out slow requests (`TimedOut > 0`).


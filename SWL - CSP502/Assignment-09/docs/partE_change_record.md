# Part E — Instructor-Issued Constraint and Pre-Coding Redesign Record

- **Course:** CSP502 Software Lab-I (M.Tech. CSE W26)
- **Lab Exercise:** 9 — Resource-Aware, Fault-Tolerant and Priority-Based Event Processing in Go
- **Group Members:** Gaurav Acharya (`MT26MCS024`), Sandeep (`MT26MCS032`)
- **Timestamp (Recorded Before Implementing Part E/F):** `2026-11-25T15:10:00+05:30`

---

## 1. Operating Constraint Released After Part D

> **Constrained Operating Envelope:** Reduce worker pool capacity to **$W = 2$ workers** (down from $W = 4$), enforce a **strict finite queue limit of $K = 64$ events**, and cap **maximum acceptable normal-priority waiting time at $\le 60\text{ ms}$** under peak disturbances (D1 Burst, D2 Slow Tasks, D3 Transient Failures, D4 Priority Pressure).

---

## 2. Invalidated Baseline Assumptions

1. **Unbounded Queue Absorption:** The Part B/C/D baseline used a large/unbounded queue ($K = 4096$) that absorbed bursts without rejecting events. With $K = 64$, bursts (D1) and slow-worker backlogs (D2) will saturate the queue and require explicit admission rejection.
2. **Abundant Worker Capacity ($W = 4$):** Halving workers to $W = 2$ cuts sustainable throughput in half, making HOL blocking from 5x slow tasks (D2) and starvation from 70% high-priority pressure (D4) acute.
3. **Zero-Cost Strict Priority:** Under $W=2$ and D4 (70% `LOGIN_FAILED`), strict priority starves normal events well beyond 60 ms.
4. **No Retry / No Timeout Isolation:** Without bounded retries, D3 (5% transient first-attempt failures) leaves 5% of valid events failed; without per-task timeouts, slow tasks monopolize both workers.

---

## 3. Alternatives Considered and Selected Mechanisms

1. **Scheduling / Fairness Mechanism:**
   - *Alternative 1:* Weighted Round-Robin (fixed 3:1 high:normal ratio) — predictable, but still delays normal events if high-priority queue is deep and arrivals are bursty.
   - *Selected — **Priority Aging + Weighted Cap**:* Serve `highQ` by default, but promote `normalQ` whenever the head normal event has waited $\ge 20\text{ ms}$ (`AgingThresholdMs = 20`) **or** after 4 consecutive high-priority dispatches (`MaxConsecutiveHigh = 4`). Directly bounds `MaxNormalWait` under D4 without sacrificing high-priority responsiveness during normal load.
2. **Overload / Failure Mechanism:**
   - *Alternative 1:* Drop-oldest from queue on overflow — complicates event accounting and hides load shedding.
   - *Selected — **Finite Queue Admission Control ($K=64$) + Per-Task Context Timeout ($25\text{ ms}$) + Bounded Retries (max 2 attempts, no side effects on failed attempts)**:* Fast-rejects excess arrivals at intake (`OutcomeRejected`), aborts stalled slow tasks cleanly (`OutcomeTimedOut`) preventing late aggregate updates, and recovers 5% transient D3 failures (`MaxRetries = 1`, i.e., 2 total attempts) while recording attempt counts separately.

---

## 4. Predicted Benefits and Potential New Failure Modes

- **Predicted Benefits:**
  - Zero normal-priority starvation under D4 (`MaxNormalWait` drops sharply below 60 ms).
  - Bounded memory and fast recovery after D1 bursts (queue capped at $K=64$, draining rapidly).
  - Near-100% recovery of D3 transient failures without duplicate aggregate updates.
  - Exact accounting: `Generated = Malformed + Successful + Failed + TimedOut + Cancelled + Rejected`.
- **Potential New Failure Modes:**
  - *Admission Rejection Spike:* Under D1 Burst with $W=2, K=64$, bounding the queue will reject excess events (`Rejected > 0`), trading raw completion count for bounded latency.
  - *Retry Amplification under Slowdowns:* Retrying failed attempts consumes extra worker time; capped to 1 retry (2 attempts total) and skipped if task context is already cancelled/timed out.
  - *Timeout False Positives:* Tight task timeouts can time out legitimate slow tasks (`TimedOut > 0`) under D2.

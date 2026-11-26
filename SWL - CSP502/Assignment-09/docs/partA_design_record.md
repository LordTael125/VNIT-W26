# Part A — Independent Reasoning and Pre-Coding Design Record

- **Course:** CSP502 Software Lab-I (M.Tech. CSE W26)
- **Lab Exercise:** 9 — Resource-Aware, Fault-Tolerant and Priority-Based Event Processing in Go
- **Group Members:** Gaurav Acharya (`MT26MCS024`), Sandeep (`MT26MCS032`)
- **Timestamp (Pre-Coding Design Freeze):** `2026-11-25T14:00:00+05:30`

---

## 1. Claim Classification and Justification

| # | Claim | Classification | Technical Justification |
|---|---|---|---|
| (i) | Priority queues always improve service quality. | **Unjustified** | Priority scheduling improves latency for high-priority events (`LOGIN_FAILED`) only at the expense of normal-priority events (`FILE_ACCESS`, `LOGIN`, etc.). Under sustained high-priority bursts or overloaded capacity, strict priority queues starve normal-priority events indefinitely and add scheduling/synchronization overhead. |
| (ii) | More workers eliminate overload. | **Unjustified** | Overload occurs when arrival rate exceeds sustainable system capacity ($\lambda > \mu_{\max}$). Adding goroutines beyond physical CPU cores or past shared-state lock contention limits increases context switching, mutex contention, and memory usage without raising $\mu_{\max}$. |
| (iii) | Retries are harmless. | **Unjustified** | Unbounded or unbudgeted retries amplify effective arrival rate ($\lambda_{\text{eff}} = \lambda(1+r)$) during overload/slowdowns, worsening queue depth and causing retry storms. Furthermore, retrying non-idempotent aggregate updates corrupts counts unless failed attempts commit zero side effects. |
| (iv) | A faster average response implies better service for every class. | **Unjustified** | A lower aggregate mean can hide severe class-level degradation or tail-latency spikes. Serving 70% high-priority events in 1 ms while delaying 30% normal events by 500 ms yields a low overall mean (~150 ms) while violating normal-class SLAs. |

---

## 2. Queue Data-Structure Comparison

| Structure | Enqueue / Dequeue Complexity | Memory Cost | Tie-Breaking (Within Class) | Starvation Risk |
|---|---|---|---|---|
| **Single FIFO Queue** (ring buffer / slice / channel) | $O(1)$ enqueue, $O(1)$ dequeue | $O(K)$ contiguous pointers; lowest overhead | Strict global arrival order | None (every admitted item advances in $O(K)$ steps) |
| **Binary Heap** (`container/heap`) | $O(\log K)$ push, $O(\log K)$ pop | $O(K)$ slice + sequence counter per entry | Requires explicit `(priority, arrivalSeq)` comparator to preserve FIFO within class | High for normal class under sustained high-priority arrivals unless aging is integrated |
| **Separate Priority Queues** (`highQ` + `normalQ` ring buffers) | $O(1)$ enqueue, $O(1)$ dequeue | $O(K)$ split across 2 buffers; zero heapify swaps | **Automatic** strict FIFO within each priority class by construction | High for normal class under strict priority; **easily mitigated** via aging or weighted round-robin |

**Choice for Part C:** Separate FIFO queues (`highQ` and `normalQ`) protected by a single scheduler mutex + `sync.Cond` (or notification channel), because $O(1)$ operations and automatic intra-class FIFO tie-breaking avoid $O(\log K)$ heap overhead and comparator complexity for two discrete priority classes.

---

## 3. Counting Complexity: Linear Search vs. Hash-Based Counting ($n$ events, $u$ unique IPs)

- **Repeated Linear Searches** (scanning a slice of IP counters per `LOGIN_FAILED` event):
  - *Expected time:* $O(n \cdot u)$ total ($O(u)$ per lookup).
  - *Worst-case time:* $O(n^2)$ when $u = \Theta(n)$ (every failed login from a distinct IP).
  - *Space:* $O(u)$.
- **Hash-Based Counting** (Go `map[string]int`):
  - *Expected time:* $O(n)$ total ($O(1)$ amortized lookup/insert per event).
  - *Worst-case time:* $O(n \cdot u)$ or $O(n^2)$ only under adversarial hash-collision attacks (mitigated in Go's runtime by per-map randomized hash seeds).
  - *Space:* $O(u)$ plus map bucket load-factor overhead (~1.5x–2x).

---

## 4. Pipeline Architecture, Ownership, and Termination

```
[Input Lines] --> [Validator/Parser] --(malformed)--> [Malformed Counter]
                         | (valid, unique EventID 1..N)
                         v
                [Admission & Scheduler]  (FIFO or Separate Priority Queues)
                         |
                         v
                [Worker Pool (W goroutines)] --(timeout / retry / fail)--> [Result Status]
                         | (success only, exactly once)
                         v
                [Single-Owner / Mutex-Guarded Aggregator]
                         |
                         v
                [Final Counts, Flagged IPs (>=5), Terminal Statuses]
```

- **Shared State Ownership:**
  - `Admission/Scheduler` owns `highQ` / `normalQ` under `schedMu`.
  - `Aggregator / Ledger` owns `ipFailedCounts`, `fileAccessCounts`, `otherEventCounts`, `malformedCount`, and `terminalStatus[eventID]` under `aggMu` (single-writer lock / atomic commit).
- **Termination Conditions:**
  1. Intake closes after all $N$ lines are submitted (or when shutdown/drain is triggered).
  2. Workers drain remaining queued items (or mark them `Cancelled` if context/grace expires) and exit when queues are empty and closed.
  3. Aggregator returns final counts, sorted flagged IPs (`count >= 5`), and verifies `Generated == Malformed + Successful + Failed + TimedOut + Cancelled + Rejected`.

---

## 5. Pre-Coding Predictions and Measurable Criteria

- **Skewed IP activity:** Increases cache locality in hash maps (`ipFailedCounts`) with negligible CPU impact, but can cause lock contention if a single global mutex guards all counters; flagging (`>= 5`) remains exact regardless of completion order.
- **Burst of high-priority events (D1 / D4):** Under strict priority, high-priority latency stays low while normal-priority events experience severe queue wait spikes and potential starvation (`MaxNormalWait` grows proportionally to burst length).
- **Slow workers (D2):** 10% slow tasks (5x cost) occupy workers, causing Head-of-Line (HOL) blocking in FIFO and queue growth under paced load.
- **Reduced CPU / worker availability:** Decreases sustainable throughput $\mu_{\max}$ linearly; queues saturate faster under D1/D4.
- **Success & Recovery Criteria:**
  - Zero accounting violations (`Generated == Malformed + TerminalOutcomes`), zero duplicate terminal updates, zero goroutine leaks.
  - Recovery threshold after D1 burst: queue depth returns to $\le 5$ within $\le 250\text{ ms}$ after burst ends.
  - Normal-priority fairness target: `MaxNormalWait` bounded (e.g., $\le 60\text{ ms}$ under constrained workers $W=2$, queue $K=64$).

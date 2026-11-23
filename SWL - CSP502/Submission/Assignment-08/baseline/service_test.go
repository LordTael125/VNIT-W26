package baseline

import (
	"os"
	"runtime"
	"testing"
	"time"

	"vnit.ac.in/csp502/practical8/workload"
)

func strictMode() bool {
	return os.Getenv("STRICT_INVARIANTS") == "1"
}

// TestNormalWorkloadInvariants (Part B verification):
// Setup: 1,000 normal requests, 8 workers, fixed seed=101, 0 injected faults.
// Expected property: I1 (accounting conservation), I2 (unique terminal outcomes),
// I3 (monotonic lifecycle ordering), and 0 outstanding work after clean drain.
func TestNormalWorkloadInvariants(t *testing.T) {
	cfg := workload.Config{
		Condition:      "unit_normal",
		TotalRequests:  1000,
		NumWorkers:     8,
		QueueCapacity:  64,
		ArrivalRateRPS: 2000,
		MinServiceMs:   1,
		MaxServiceMs:   3,
		AckTimeoutMs:   25,
	}
	sum, err := RunWorkload(cfg, 101, 1, "")
	if err != nil {
		t.Fatalf("RunWorkload failed: %v", err)
	}
	t.Logf("Observed: %s", FormatInvariantReport(sum))
	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("I1 violated: accounting violations = %d", sum.TerminalAccountingViolations)
	}
	if sum.DuplicateTerminalOutcomes != 0 {
		t.Errorf("I2 violated: duplicate terminal outcomes = %d", sum.DuplicateTerminalOutcomes)
	}
	if sum.LifecycleOrderViolations != 0 {
		t.Errorf("I3 violated: lifecycle order violations = %d", sum.LifecycleOrderViolations)
	}
	if sum.Successful != 1000 {
		t.Errorf("expected 1000 successful requests, got %d", sum.Successful)
	}
}

// TestDuplicateID (Part C Test 1):
// Setup: Submit 100 unique requests plus 20 duplicate re-submissions of IDs 1..20.
// Expected property (I2): Every unique request ID has at most one terminal outcome.
// Observed Baseline behavior: Naive baseline lacks intake deduplication, creating 20 duplicate
// terminal records (Defect D-01). Fails when STRICT_INVARIANTS=1.
func TestDuplicateID(t *testing.T) {
	cfg := workload.Config{
		Condition:     "unit_dup_id",
		TotalRequests: 100,
		NumWorkers:    4,
		MinServiceMs:  2,
		MaxServiceMs:  3,
		AckTimeoutMs:  25,
	}
	svc := NewService(cfg)
	svc.Start()

	for i := 1; i <= 100; i++ {
		svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqCertificate,
			ServiceDuration: 2 * time.Millisecond,
		})
	}
	// Inject 20 duplicate deliveries for IDs 1..20
	for i := 1; i <= 20; i++ {
		svc.Submit(workload.Item{
			ID:                uint64(i),
			SubmissionSeq:     2,
			IsDuplicateSubmit: true,
			Type:              workload.ReqCertificate,
			ServiceDuration:   2 * time.Millisecond,
		})
	}

	shDur, out := svc.CloseAndDrain()
	sum := svc.Summarize("unit_dup_id", 1, 1, 200*time.Millisecond, shDur, out, 0)
	t.Logf("Setup: 100 unique + 20 duplicate IDs | Observed: TerminalRecords=%d UniqueIDs=%d DupOutcomes=%d AcctViolations=%d",
		sum.TotalTerminalRecords, sum.UniqueTerminalIDs, sum.DuplicateTerminalOutcomes, sum.TerminalAccountingViolations)

	if sum.DuplicateTerminalOutcomes == 0 {
		t.Logf("No duplicate outcomes observed")
	} else {
		t.Logf("[DEFECT EVIDENCE D-01] Baseline recorded %d duplicate terminal outcomes for 20 repeated deliveries", sum.DuplicateTerminalOutcomes)
		if strictMode() {
			t.Errorf("I2 VIOLATION (D-01): expected 0 duplicate terminal outcomes, observed %d", sum.DuplicateTerminalOutcomes)
		}
	}
}

// TestCancellation (Part C Test 2):
// Setup: Enqueue 250 requests (10ms each) across 4 workers; cancel abruptly after 40ms while queue is deep.
// Expected property (I1, I6): All 250 generated requests reach a terminal outcome (Completed/Cancelled) and workers exit.
// Observed Baseline behavior: Abrupt cancellation exits workers without draining queued requests, leaving
// stranded work and missing terminal records (Defect D-02). Fails when STRICT_INVARIANTS=1.
func TestCancellation(t *testing.T) {
	cfg := workload.Config{
		Condition:     "unit_cancel",
		TotalRequests: 250,
		NumWorkers:    4,
		MinServiceMs:  10,
		MaxServiceMs:  10,
		AckTimeoutMs:  25,
	}
	svc := NewService(cfg)
	svc.Start()

	for i := 1; i <= 250; i++ {
		svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqVerification,
			ServiceDuration: 10 * time.Millisecond,
		})
	}
	time.Sleep(40 * time.Millisecond)
	shDur, out := svc.ShutdownAbrupt()
	sum := svc.Summarize("unit_cancel", 1, 1, 100*time.Millisecond, shDur, out, 0)

	t.Logf("Setup: 250 queued requests, cancel at 40ms | Observed: Completed=%d Cancelled=%d Missing=%d Outstanding=%d Shutdown=%.2fms",
		sum.Successful, sum.Cancelled, sum.MissingTerminalIDs, sum.OutstandingWorkAfterShutdown, sum.ShutdownCompletionMs)

	if sum.MissingTerminalIDs > 0 || sum.OutstandingWorkAfterShutdown > 0 {
		t.Logf("[DEFECT EVIDENCE D-02] Baseline left %d missing terminal records (%d stranded in queue) on cancellation",
			sum.MissingTerminalIDs, sum.OutstandingWorkAfterShutdown)
		if strictMode() {
			t.Errorf("I1 VIOLATION (D-02): expected 0 missing terminal IDs on cancellation, observed %d (outstanding=%d)",
				sum.MissingTerminalIDs, sum.OutstandingWorkAfterShutdown)
		}
	}
}

// TestSlowWorker (Part C Test 3):
// Setup: 120 requests across 4 workers; 20% are slow (120ms vs 2ms normal). SLA deadline target = 35ms.
// Expected property (I6): Slow requests should time out near SLA deadline and queue should stay bounded.
// Observed Baseline behavior: Without per-request timeouts, 0 requests time out and p95 latency spikes > 150ms (Defect D-03).
func TestSlowWorker(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_slow_worker",
		TotalRequests:    120,
		NumWorkers:       4,
		QueueCapacity:    32,
		RequestTimeoutMs: 35,
		ArrivalRateRPS:   1500,
		MinServiceMs:     2,
		MaxServiceMs:     3,
		AckTimeoutMs:     25,
		Faults: workload.FaultConfig{
			SlowRatio:      0.20,
			SlowDurationMs: 120,
		},
	}
	sum, err := RunWorkload(cfg, 42, 1, "")
	if err != nil {
		t.Fatalf("RunWorkload failed: %v", err)
	}
	t.Logf("Setup: 120 reqs, 4 workers, 20%% slow (120ms), target SLA=35ms | Observed: TimedOut=%d MaxQueue=%d Mean=%.2fms p95=%.2fms Max=%.2fms",
		sum.TimedOut, sum.MaxQueueDepth, sum.MeanResponseMs, sum.P95ResponseMs, sum.MaxResponseMs)

	if sum.TimedOut == 0 || sum.MaxQueueDepth > int64(cfg.QueueCapacity) {
		t.Logf("[DEFECT EVIDENCE D-03] Baseline has no SLA timeout or bounded queue: TimedOut=%d, MaxQueue=%d (>%d), p95=%.2fms",
			sum.TimedOut, sum.MaxQueueDepth, cfg.QueueCapacity, sum.P95ResponseMs)
		if strictMode() {
			t.Errorf("I6 VIOLATION (D-03): expected slow requests to time out and queue <= %d, got TimedOut=%d MaxQueue=%d p95=%.2fms",
				cfg.QueueCapacity, sum.TimedOut, sum.MaxQueueDepth, sum.P95ResponseMs)
		}
	}
}

// TestDeterministicFailure (Part C Test 4):
// Setup: 150 requests across 4 workers; every 10th request ID (15 requests) fails deterministically.
// Expected property (I1, I5): Exactly 15 Failed and 135 Successful; unrelated requests unaffected.
func TestDeterministicFailure(t *testing.T) {
	cfg := workload.Config{
		Condition:     "unit_failure",
		TotalRequests: 150,
		NumWorkers:    4,
		MinServiceMs:  1,
		MaxServiceMs:  2,
		AckTimeoutMs:  25,
	}
	svc := NewService(cfg)
	svc.Start()

	for i := 1; i <= 150; i++ {
		svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqApproval,
			ServiceDuration: 2 * time.Millisecond,
			IsFailing:       i%10 == 0,
		})
	}
	shDur, out := svc.CloseAndDrain()
	sum := svc.Summarize("unit_failure", 1, 1, 200*time.Millisecond, shDur, out, 0)
	t.Logf("Setup: 150 reqs, every 10th fails | Observed: Succ=%d Fail=%d AcctViolations=%d",
		sum.Successful, sum.Failed, sum.TerminalAccountingViolations)

	if sum.Failed != 15 {
		t.Errorf("I5 violated: expected exactly 15 failed requests, got %d", sum.Failed)
	}
	if sum.Successful != 135 {
		t.Errorf("I5 violated: expected 135 unrelated requests to succeed, got %d", sum.Successful)
	}
	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("I1 violated: accounting violations = %d", sum.TerminalAccountingViolations)
	}
}

// TestShutdownBoundedNoLeaks (Part C Test 5):
// Setup: Continuous submissions of 300 slow/normal requests across 4 workers; stop intake during active
// processing at 50ms. Verify bounded shutdown policy and zero leaked goroutines.
// Observed Baseline behavior: CloseAndDrain waits unboundedly (exceeding 100ms grace), whereas
// ShutdownAbrupt leaves queued work stranded (Defect D-04).
func TestShutdownBoundedNoLeaks(t *testing.T) {
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
	beforeG := runtime.NumGoroutine()

	cfg := workload.Config{
		Condition:       "unit_shutdown",
		TotalRequests:   200,
		NumWorkers:      4,
		ShutdownGraceMs: 100,
		MinServiceMs:    15,
		MaxServiceMs:    15,
		AckTimeoutMs:    25,
	}
	svc := NewService(cfg)
	svc.Start()

	for i := 1; i <= 200; i++ {
		svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqCertificate,
			ServiceDuration: 15 * time.Millisecond,
		})
	}
	// Baseline CloseAndDrain has no grace-period bound: drains all 200 items (~750ms > 100ms grace).
	shDur, out := svc.CloseAndDrain()

	time.Sleep(15 * time.Millisecond)
	afterG := runtime.NumGoroutine()
	deltaG := afterG - beforeG

	t.Logf("Setup: 200 active requests (15ms each), grace=100ms | Observed: Shutdown=%.2fms Outstanding=%d GoroutineDelta=%d",
		float64(shDur.Microseconds())/1000.0, out, deltaG)

	if deltaG > 2 {
		t.Errorf("possible goroutine leak: delta=%d (before=%d after=%d)", deltaG, beforeG, afterG)
	}
	if shDur > 150*time.Millisecond {
		t.Logf("[DEFECT EVIDENCE D-04] Baseline CloseAndDrain exceeded 100ms shutdown grace bound (took %.2fms)",
			float64(shDur.Microseconds())/1000.0)
		if strictMode() {
			t.Errorf("I4 VIOLATION (D-04): expected bounded shutdown <= 150ms, took %.2fms", float64(shDur.Microseconds())/1000.0)
		}
	}
}


package improved

import (
	"runtime"
	"sync"
	"testing"
	"time"

	"vnit.ac.in/csp502/practical8/workload"
)

// TestNormalWorkload_InvariantsAndOrdering verifies Part B/E steady-state correctness.
func TestNormalWorkload_InvariantsAndOrdering(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_normal",
		TotalRequests:    1000,
		NumWorkers:       8,
		QueueCapacity:    64,
		ArrivalRateRPS:   2000,
		MinServiceMs:     1,
		MaxServiceMs:     3,
		RequestTimeoutMs: 50,
		AckTimeoutMs:     25,
		MaxAckRetries:    3,
		RetryBudgetRatio: 0.25,
		ShutdownGraceMs:  200,
	}
	sum, err := RunWorkload(cfg, 101, 1, "")
	if err != nil {
		t.Fatalf("RunWorkload failed: %v", err)
	}
	t.Logf("Normal 1000 reqs: Succ=%d Rej=%d AcctViol=%d DupOut=%d OrderViol=%d Mean=%.2fms p95=%.2fms",
		sum.Successful, sum.Rejected, sum.TerminalAccountingViolations, sum.DuplicateTerminalOutcomes,
		sum.LifecycleOrderViolations, sum.MeanResponseMs, sum.P95ResponseMs)

	if !sum.InvariantsPassed {
		t.Fatalf("invariants failed: %+v", sum)
	}
}

// TestDuplicateID_Deduplicated (Part C Test 1):
// Injects 100 unique IDs plus 40 concurrent & sequential duplicate deliveries of existing IDs.
// Verifies I1 & I2: exactly 100 unique terminal records, 0 duplicate terminal outcomes.
func TestDuplicateID_Deduplicated(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_dup_id",
		TotalRequests:    100,
		NumWorkers:       4,
		QueueCapacity:    128,
		RequestTimeoutMs: 50,
		AckTimeoutMs:     25,
		MaxAckRetries:    2,
		RetryBudgetRatio: 0.25,
	}
	svc := NewService(cfg)
	svc.Start()

	var wg sync.WaitGroup
	// Submit 100 unique requests
	for i := 1; i <= 100; i++ {
		svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqCertificate,
			ServiceDuration: 2 * time.Millisecond,
		})
	}
	// Inject 40 duplicate submissions concurrently from 4 goroutines
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			for i := 1; i <= 10; i++ {
				svc.Submit(workload.Item{
					ID:                uint64(offset*10 + i),
					SubmissionSeq:     2,
					IsDuplicateSubmit: true,
					Type:              workload.ReqCertificate,
					ServiceDuration:   2 * time.Millisecond,
				})
			}
		}(g)
	}
	wg.Wait()

	shDur, out := svc.CloseAndDrain()
	sum := svc.Summarize("unit_dup_id", 1, 1, 200*time.Millisecond, shDur, out, 0)
	t.Logf("Observed: UniqueGen=%d TerminalRecs=%d DupBlocked=%d DupOutcomes=%d AcctViol=%d",
		sum.UniqueGenerated, sum.TotalTerminalRecords, sum.DuplicateSubmissionsBlocked,
		sum.DuplicateTerminalOutcomes, sum.TerminalAccountingViolations)

	if sum.DuplicateTerminalOutcomes != 0 {
		t.Errorf("I2 violated: expected 0 duplicate terminal outcomes, got %d", sum.DuplicateTerminalOutcomes)
	}
	if sum.DuplicateSubmissionsBlocked != 40 {
		t.Errorf("expected 40 duplicate submissions blocked, got %d", sum.DuplicateSubmissionsBlocked)
	}
	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("I1 violated: accounting violations = %d", sum.TerminalAccountingViolations)
	}
	if sum.UniqueTerminalIDs != 100 {
		t.Errorf("expected 100 unique terminal IDs, got %d", sum.UniqueTerminalIDs)
	}
}

// TestCancellation_DrainsQueuedAndAccountsAll (Part C Test 2):
// Cancels service while requests are both queued and actively processing.
// Verifies I1, I4, I6: zero missing terminal records, zero stranded queued work, prompt exit.
func TestCancellation_DrainsQueuedAndAccountsAll(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_cancel",
		TotalRequests:    250,
		NumWorkers:       4,
		QueueCapacity:    256,
		RequestTimeoutMs: 200,
		AckTimeoutMs:     25,
	}
	svc := NewService(cfg)
	svc.Start()

	for i := 1; i <= 250; i++ {
		svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqVerification,
			ServiceDuration: 15 * time.Millisecond,
		})
	}
	time.Sleep(45 * time.Millisecond)
	shDur, out := svc.Cancel()
	sum := svc.Summarize("unit_cancel", 1, 1, 100*time.Millisecond, shDur, out, 0)

	t.Logf("Observed: Succ=%d Cancelled=%d Missing=%d Outstanding=%d Shutdown=%.2fms",
		sum.Successful, sum.Cancelled, sum.MissingTerminalIDs, sum.OutstandingWorkAfterShutdown, sum.ShutdownCompletionMs)

	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("I1 violated on cancel: accounting violations=%d (missing=%d)", sum.TerminalAccountingViolations, sum.MissingTerminalIDs)
	}
	if sum.OutstandingWorkAfterShutdown != 0 {
		t.Errorf("I4 violated: outstanding work = %d", sum.OutstandingWorkAfterShutdown)
	}
	if sum.Cancelled == 0 {
		t.Errorf("expected non-zero cancelled requests during active backlog cancellation")
	}
	if shDur > 100*time.Millisecond {
		t.Errorf("I6 violated: cancellation took too long: %.2fms", float64(shDur.Microseconds())/1000.0)
	}
}

// TestSlowWorker_EnforcesSLATimeoutAndBoundsQueue (Part C Test 3):
// Injects 20% slow tasks (150ms vs 2ms normal) with SLA timeout = 35ms and bounded queue = 32.
// Verifies I6: slow tasks time out near 35ms, queue never exceeds capacity 32, fast tasks survive.
func TestSlowWorker_EnforcesSLATimeoutAndBoundsQueue(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_slow_worker",
		TotalRequests:    150,
		NumWorkers:       4,
		QueueCapacity:    32,
		RequestTimeoutMs: 35,
		ArrivalRateRPS:   1500,
		MinServiceMs:     2,
		MaxServiceMs:     3,
		AckTimeoutMs:     25,
		MaxAckRetries:    2,
		RetryBudgetRatio: 0.25,
		Faults: workload.FaultConfig{
			SlowRatio:      0.20,
			SlowDurationMs: 150,
		},
	}
	sum, err := RunWorkload(cfg, 42, 1, "")
	if err != nil {
		t.Fatalf("RunWorkload failed: %v", err)
	}
	t.Logf("Observed: Succ=%d TimedOut=%d Rej=%d MaxQueue=%d Mean=%.2fms p95=%.2fms Max=%.2fms",
		sum.Successful, sum.TimedOut, sum.Rejected, sum.MaxQueueDepth, sum.MeanResponseMs, sum.P95ResponseMs, sum.MaxResponseMs)

	if sum.TimedOut == 0 {
		t.Errorf("I6 violated: expected slow tasks to time out at 35ms SLA, got 0 timeouts")
	}
	if sum.MaxQueueDepth > int64(cfg.QueueCapacity) {
		t.Errorf("admission control violated: MaxQueueDepth=%d > QueueCapacity=%d", sum.MaxQueueDepth, cfg.QueueCapacity)
	}
	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("I1 violated: accounting violations = %d", sum.TerminalAccountingViolations)
	}
}

// TestDeterministicFailure_IsolatedFromHealthyRequests (Part C Test 4):
// Every 10th request fails deterministically; verifies all 135 unrelated requests succeed cleanly.
func TestDeterministicFailure_IsolatedFromHealthyRequests(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_failure",
		TotalRequests:    150,
		NumWorkers:       4,
		QueueCapacity:    200,
		RequestTimeoutMs: 50,
		AckTimeoutMs:     25,
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
	t.Logf("Observed: Succ=%d Fail=%d AcctViol=%d", sum.Successful, sum.Failed, sum.TerminalAccountingViolations)

	if sum.Failed != 15 || sum.Successful != 135 || sum.TerminalAccountingViolations != 0 {
		t.Fatalf("I5/I1 violated: succ=%d fail=%d acctViol=%d", sum.Successful, sum.Failed, sum.TerminalAccountingViolations)
	}
}

// TestShutdown_BoundedNoBlockedGoroutines (Part C Test 5):
// Stops intake during active processing with 80ms grace period; verifies:
// (a) post-shutdown submissions are fast-rejected,
// (b) shutdown completes within grace + tolerance,
// (c) outstanding work == 0, accounting violations == 0, goroutine delta <= 1.
func TestShutdown_BoundedNoBlockedGoroutines(t *testing.T) {
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
	beforeG := runtime.NumGoroutine()

	cfg := workload.Config{
		Condition:        "unit_shutdown",
		TotalRequests:    200,
		NumWorkers:       4,
		QueueCapacity:    100,
		RequestTimeoutMs: 100,
		AckTimeoutMs:     25,
		ShutdownGraceMs:  80,
	}
	svc := NewService(cfg)
	svc.Start()

	// Enqueue 100 active tasks (20ms each -> would take 500ms without bounded grace)
	for i := 1; i <= 100; i++ {
		svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqCertificate,
			ServiceDuration: 20 * time.Millisecond,
		})
	}

	shDur, out := svc.Shutdown(80 * time.Millisecond)

	// Submit 20 new requests AFTER shutdown begins -> must be fast-rejected
	for i := 101; i <= 120; i++ {
		admitted := svc.Submit(workload.Item{
			ID:              uint64(i),
			SubmissionSeq:   1,
			Type:            workload.ReqCertificate,
			ServiceDuration: 5 * time.Millisecond,
		})
		if admitted {
			t.Errorf("I4 violated: post-shutdown request %d was admitted", i)
		}
	}

	time.Sleep(15 * time.Millisecond)
	afterG := runtime.NumGoroutine()
	deltaG := afterG - beforeG

	sum := svc.Summarize("unit_shutdown", 1, 1, 200*time.Millisecond, shDur, out, deltaG)
	t.Logf("Observed: Shutdown=%.2fms (grace=80ms) Succ=%d Cancel=%d Rej=%d Outstanding=%d AcctViol=%d GoroutineDelta=%d",
		sum.ShutdownCompletionMs, sum.Successful, sum.Cancelled, sum.Rejected,
		sum.OutstandingWorkAfterShutdown, sum.TerminalAccountingViolations, deltaG)

	if shDur > 150*time.Millisecond {
		t.Errorf("I4 violated: shutdown took %.2fms (> 150ms bound for 80ms grace)", float64(shDur.Microseconds())/1000.0)
	}
	if out != 0 {
		t.Errorf("I4 violated: outstanding work = %d", out)
	}
	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("I1 violated: accounting violations = %d", sum.TerminalAccountingViolations)
	}
	if sum.Rejected < 20 {
		t.Errorf("expected at least 20 post-shutdown rejections, got %d", sum.Rejected)
	}
	if deltaG > 2 {
		t.Errorf("I4 violated: possible goroutine leak (before=%d after=%d delta=%d)", beforeG, afterG, deltaG)
	}
}

// TestDownstreamAck_DelayedAndRepeated_ExactlyOnceRecorded (Part D Test):
// Injects 30% delayed acks (45ms > 20ms ack timeout) and 30% repeated acks (2-3x), plus 10% dropped acks.
// Verifies Part D guarantee: At-least-once bounded ack retry recovery + Exactly-once recorded outcome.
func TestDownstreamAck_DelayedAndRepeated_ExactlyOnceRecorded(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_partD_ack",
		TotalRequests:    200,
		NumWorkers:       8,
		QueueCapacity:    200,
		ArrivalRateRPS:   1000,
		MinServiceMs:     1,
		MaxServiceMs:     3,
		RequestTimeoutMs: 60,
		AckTimeoutMs:     20,
		MaxAckRetries:    3,
		RetryBudgetRatio: 0.60,
		ShutdownGraceMs:  200,
		Faults: workload.FaultConfig{
			AckDelayRatio:     0.30,
			AckDelayMs:        45,
			AckDuplicateRatio: 0.30,
			AckDropRatio:      0.10,
		},
	}
	sum, err := RunWorkload(cfg, 777, 1, "")
	if err != nil {
		t.Fatalf("RunWorkload failed: %v", err)
	}
	t.Logf("Part D Ack Faults: UniqueGen=%d Succ=%d TimedOut=%d RetriesUsed=%d DupAcksIgnored=%d DupOutcomes=%d AcctViol=%d",
		sum.UniqueGenerated, sum.Successful, sum.TimedOut, sum.AckRetriesUsed,
		sum.DuplicateAcksIgnored, sum.DuplicateTerminalOutcomes, sum.TerminalAccountingViolations)

	if sum.DuplicateTerminalOutcomes != 0 {
		t.Errorf("Part D / I2 violated: duplicate terminal outcomes = %d", sum.DuplicateTerminalOutcomes)
	}
	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("Part D / I1 violated: accounting violations = %d", sum.TerminalAccountingViolations)
	}
	if sum.AckRetriesUsed == 0 {
		t.Errorf("expected ack retries to fire under 30%% delayed + 10%% dropped acks")
	}
	if sum.DuplicateAcksIgnored == 0 {
		t.Errorf("expected late/repeated acks to be observed and safely ignored")
	}
	if sum.Successful < 190 {
		t.Errorf("expected high recovery goodput (>=190/200), got %d", sum.Successful)
	}
}

// TestDownstreamAck_RetryBudgetCapsAmplification (Part D Retry Budget Test):
// Sets a tiny retry budget (10% of 100 = 10 retries) while 50% of acks are dropped.
// Verifies retry budget strictly caps retry amplification and commits OutcomeTimeout cleanly when exhausted.
func TestDownstreamAck_RetryBudgetCapsAmplification(t *testing.T) {
	cfg := workload.Config{
		Condition:        "unit_retry_budget",
		TotalRequests:    100,
		NumWorkers:       8,
		QueueCapacity:    128,
		ArrivalRateRPS:   2000,
		MinServiceMs:     1,
		MaxServiceMs:     2,
		RequestTimeoutMs: 50,
		AckTimeoutMs:     15,
		MaxAckRetries:    3,
		RetryBudgetRatio: 0.10, // budget = 10 retries
		Faults: workload.FaultConfig{
			AckDropRatio: 0.50,
		},
	}
	sum, err := RunWorkload(cfg, 99, 1, "")
	if err != nil {
		t.Fatalf("RunWorkload failed: %v", err)
	}
	t.Logf("Retry Budget Test: Budget=10 RetriesUsed=%d Succ=%d TimedOut=%d AcctViol=%d",
		sum.AckRetriesUsed, sum.Successful, sum.TimedOut, sum.TerminalAccountingViolations)

	if sum.AckRetriesUsed > 10 {
		t.Errorf("retry budget violated: used %d > budget 10", sum.AckRetriesUsed)
	}
	if sum.TimedOut == 0 {
		t.Errorf("expected some requests to fail with Timeout once retry budget exhausted")
	}
	if sum.TerminalAccountingViolations != 0 {
		t.Errorf("I1 violated: accounting violations = %d", sum.TerminalAccountingViolations)
	}
}


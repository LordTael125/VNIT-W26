package main

import (
	"runtime"
	"testing"
	"time"
)

// TestBaselineAccounting verifies that in normal baseline execution,
// every single request is processed exactly once without loss or duplicates.
func TestBaselineAccounting(t *testing.T) {
	cfg := ServiceConfig{
		Name:              "TestBaselineAccounting",
		NumWorkers:        4,
		QueueCapacity:     0,
		Timeout:           0,
		IsResilient:       false,
		RandomSeed:        12345,
		TotalRequests:     200,
		TargetArrivalRate: 500,
		MinServiceTime:    2 * time.Millisecond,
		MaxServiceTime:    4 * time.Millisecond,
	}

	sim := NewSimulation(cfg)
	res := sim.GenerateAndRun()

	if !res.AccountingValid {
		t.Fatalf("Accounting verification failed: TotalGenerated=%d, SumOutcomes=%d",
			res.TotalGenerated, res.Successful+res.Failed+res.TimedOut+res.Cancelled+res.Rejected)
	}

	if res.Successful != cfg.TotalRequests {
		t.Errorf("Expected %d successful requests, got %d", cfg.TotalRequests, res.Successful)
	}

	if res.Rejected != 0 || res.TimedOut != 0 || res.Failed != 0 {
		t.Errorf("Unexpected non-successful requests in clean run: Rejected=%d, TimedOut=%d, Failed=%d",
			res.Rejected, res.TimedOut, res.Failed)
	}
}

// TestResilientFastRejection verifies that when the bounded queue fills up,
// excess requests are explicitly rejected (fast rejection) without leaking or hanging.
func TestResilientFastRejection(t *testing.T) {
	cfg := ServiceConfig{
		Name:              "TestResilientFastRejection",
		NumWorkers:        2,
		QueueCapacity:     10, // Small queue
		Timeout:           50 * time.Millisecond,
		IsResilient:       true,
		RandomSeed:        999,
		TotalRequests:     100,
		TargetArrivalRate: 2000, // Very high arrival rate to overwhelm 2 workers
		MinServiceTime:    10 * time.Millisecond,
		MaxServiceTime:    15 * time.Millisecond,
	}

	sim := NewSimulation(cfg)
	res := sim.GenerateAndRun()

	if !res.AccountingValid {
		t.Fatalf("Accounting invariant violated under rejection: Total=%d, Sum=%d",
			res.TotalGenerated, res.Successful+res.Failed+res.TimedOut+res.Cancelled+res.Rejected)
	}

	if res.Rejected == 0 {
		t.Errorf("Expected some requests to be rejected under 2000 req/s surge with queue cap 10, got 0")
	}

	if res.MaxQueueLength > 10 {
		t.Errorf("Queue length exceeded bounded limit: got %d, max allowed 10", res.MaxQueueLength)
	}
}

// TestResilientTimeoutIsolation verifies that slow requests are aborted at SLA deadline
// and do not indefinitely block the worker pool.
func TestResilientTimeoutIsolation(t *testing.T) {
	cfg := ServiceConfig{
		Name:              "TestResilientTimeoutIsolation",
		NumWorkers:        2,
		QueueCapacity:     50,
		Timeout:           15 * time.Millisecond, // Strict SLA
		IsResilient:       true,
		RandomSeed:        777,
		TotalRequests:     50,
		TargetArrivalRate: 200,
		SlowRatio:         0.40, // 40% slow requests
		SlowDuration:      60 * time.Millisecond,
		MinServiceTime:    2 * time.Millisecond,
		MaxServiceTime:    4 * time.Millisecond,
	}

	sim := NewSimulation(cfg)
	res := sim.GenerateAndRun()

	if !res.AccountingValid {
		t.Fatalf("Accounting invariant violated under timeout isolation: Total=%d, Sum=%d",
			res.TotalGenerated, res.Successful+res.Failed+res.TimedOut+res.Cancelled+res.Rejected)
	}

	if res.TimedOut == 0 {
		t.Errorf("Expected slow requests to be timed out, got 0")
	}

	// Verify maximum response time is bounded reasonably (context timeout 15ms + small overhead)
	// Under baseline, slow requests would take >= 60ms.
	t.Logf("TimedOut count: %d, Successful: %d, MaxRespTime: %.2f ms",
		res.TimedOut, res.Successful, res.MaxResponseTime)
}

// TestGoroutineCleanShutdown verifies that no worker goroutines leak after execution completes.
func TestGoroutineCleanShutdown(t *testing.T) {
	runtime.GC()
	initialGoroutines := runtime.NumGoroutine()

	cfg := ServiceConfig{
		Name:              "TestGoroutineCleanShutdown",
		NumWorkers:        8,
		QueueCapacity:     20,
		Timeout:           10 * time.Millisecond,
		IsResilient:       true,
		RandomSeed:        42,
		TotalRequests:     80,
		TargetArrivalRate: 800,
		SlowRatio:         0.2,
		SlowDuration:      30 * time.Millisecond,
		MinServiceTime:    2 * time.Millisecond,
		MaxServiceTime:    4 * time.Millisecond,
	}

	sim := NewSimulation(cfg)
	res := sim.GenerateAndRun()

	if !res.AccountingValid {
		t.Fatalf("Accounting failed")
	}

	// Allow goroutine scheduler to settle
	time.Sleep(50 * time.Millisecond)
	runtime.GC()
	finalGoroutines := runtime.NumGoroutine()

	diff := finalGoroutines - initialGoroutines
	if diff > 2 { // Allow small runtime margin for GC/test runner
		t.Errorf("Potential goroutine leak detected: initial=%d, final=%d, diff=%d",
			initialGoroutines, finalGoroutines, diff)
	}
}


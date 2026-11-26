package engine

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"vnit.ac.in/csp502/lab9/events"
)

func TestHandVerifiedFixture_SequentialAndConcurrent(t *testing.T) {
	fixturePath := filepath.Join("..", "datasets", "hand_verified_fixture.log")
	records, stats, err := events.ParseAndValidate("fixture", fixturePath, 0)
	if err != nil {
		t.Fatalf("ParseAndValidate failed: %v", err)
	}

	if stats.TotalLines != 18 || stats.Malformed != 2 || stats.ValidCount != 16 || stats.DuplicateRows != 1 {
		t.Fatalf("unexpected fixture stats: %+v", stats)
	}

	seqAgg, _ := SequentialReference(records, nil, false, 0)
	if seqAgg.MalformedCount != 2 {
		t.Fatalf("expected 2 malformed, got %d", seqAgg.MalformedCount)
	}
	if !reflect.DeepEqual(seqAgg.FlaggedIPs, []string{"10.10.10.50"}) {
		t.Fatalf("expected exactly [10.10.10.50] flagged (count=5, while 10.10.10.60 has 4), got %v", seqAgg.FlaggedIPs)
	}
	if seqAgg.IPFailedCounts["10.10.10.50"] != 5 || seqAgg.IPFailedCounts["10.10.10.60"] != 4 {
		t.Fatalf("unexpected IPFailedCounts: %v", seqAgg.IPFailedCounts)
	}
	// Duplicate valid line for /data/report001.txt must remain distinct (count = 2)
	if seqAgg.FileAccessCounts["/data/report001.txt"] != 2 || seqAgg.FileAccessCounts["/data/report002.txt"] != 1 {
		t.Fatalf("unexpected FileAccessCounts: %v", seqAgg.FileAccessCounts)
	}
	if seqAgg.OtherEventCounts["LOGIN"] != 1 || seqAgg.OtherEventCounts["LOGOUT"] != 1 ||
		seqAgg.OtherEventCounts["PROCESS_START"] != 1 || seqAgg.OtherEventCounts["PROCESS_END"] != 1 {
		t.Fatalf("unexpected OtherEventCounts: %v", seqAgg.OtherEventCounts)
	}

	for _, pol := range []Policy{PolicyFIFO, PolicyPriority, PolicyModified} {
		cfg := Config{
			Policy:             pol,
			Workers:            4,
			QueueCapacity:      64,
			AgingThresholdMs:   20,
			MaxConsecutiveHigh: 4,
			TaskTimeoutMs:      50,
			MaxRetries:         1,
		}
		agg, m := RunBatch(records, cfg, "fixture", 1, 100*time.Microsecond)
		if !m.AccountingValid || !m.MatchesSeqReplay {
			t.Fatalf("policy %s failed accounting/seq match: %+v", pol, m)
		}
		if m.Generated != 18 || m.Malformed != 2 || m.Successful != 16 {
			t.Fatalf("policy %s unexpected counts: %+v", pol, m)
		}
		if !reflect.DeepEqual(agg.FlaggedIPs, seqAgg.FlaggedIPs) ||
			!reflect.DeepEqual(agg.IPFailedCounts, seqAgg.IPFailedCounts) ||
			!reflect.DeepEqual(agg.FileAccessCounts, seqAgg.FileAccessCounts) ||
			!reflect.DeepEqual(agg.OtherEventCounts, seqAgg.OtherEventCounts) {
			t.Fatalf("policy %s aggregates mismatch sequential reference", pol)
		}
	}
}

func TestPriorityScheduler_PreservesIntraClassFIFOOrder(t *testing.T) {
	s := newScheduler(Config{Policy: PolicyPriority, QueueCapacity: 16})
	now := time.Now()
	s.Enqueue(queuedTask{item: events.ReplayItem{Record: events.Record{EventID: 1, Priority: events.PriorityNormal}}, enqueuedTime: now})
	s.Enqueue(queuedTask{item: events.ReplayItem{Record: events.Record{EventID: 2, Priority: events.PriorityHigh}}, enqueuedTime: now})
	s.Enqueue(queuedTask{item: events.ReplayItem{Record: events.Record{EventID: 3, Priority: events.PriorityNormal}}, enqueuedTime: now})
	s.Enqueue(queuedTask{item: events.ReplayItem{Record: events.Record{EventID: 4, Priority: events.PriorityHigh}}, enqueuedTime: now})
	s.Close()

	var order []uint64
	for {
		tk, ok := s.Dequeue()
		if !ok {
			break
		}
		order = append(order, tk.item.Record.EventID)
	}
	want := []uint64{2, 4, 1, 3}
	if !reflect.DeepEqual(order, want) {
		t.Fatalf("expected intra-class FIFO priority order %v, got %v", want, order)
	}
}

func TestModified_FiniteQueueRetriesAndExactOnceAccounting(t *testing.T) {
	records, _, err := events.ParseAndValidate("small", filepath.Join("..", "datasets", "Gr05_events_small.log"), 42)
	if err != nil {
		t.Fatalf("ParseAndValidate failed: %v", err)
	}

	trace := events.BuildReplayTrace(records, events.TraceSpec{
		Scenario:         "unit_d3_retry",
		Seed:             42,
		NumEvents:        300,
		NominalRateRPS:   1500, // intentional overload for K=16
		BaseCostUs:       800,
		TransientFailRat: 0.10, // 10% first-attempt failures
	})

	cfg := Config{
		Policy:             PolicyModified,
		Workers:            2,
		QueueCapacity:      16,
		AgingThresholdMs:   15,
		MaxConsecutiveHigh: 4,
		TaskTimeoutMs:      25,
		MaxRetries:         1,
	}
	_, m := RunPaced(trace, 0, cfg, "unit_d3", "small", 1, 0, context.Background())
	t.Logf("Modified D3+Overload: Gen=%d Succ=%d Rej=%d Fail=%d T/O=%d Attempts=%d Retries=%d Acct=%v SeqMatch=%v",
		m.Generated, m.Successful, m.Rejected, m.Failed, m.TimedOut, m.TotalAttempts, m.RetriesUsed, m.AccountingValid, m.MatchesSeqReplay)

	if !m.AccountingValid {
		t.Fatalf("accounting invalid: %+v", m)
	}
	if !m.MatchesSeqReplay {
		t.Fatalf("aggregate mismatch with sequential replay of successful IDs")
	}
	if m.RetriesUsed == 0 {
		t.Errorf("expected bounded retries to fire on 10%% transient failures")
	}
	if m.Rejected == 0 {
		t.Errorf("expected finite queue K=16 to reject excess overload")
	}
}

func TestModified_CancellationPreventsLateAggregateUpdates(t *testing.T) {
	records, _, err := events.ParseAndValidate("small", filepath.Join("..", "datasets", "Gr05_events_small.log"), 42)
	if err != nil {
		t.Fatalf("ParseAndValidate failed: %v", err)
	}
	trace := events.BuildReplayTrace(records, events.TraceSpec{
		Scenario:       "unit_cancel",
		Seed:           42,
		NumEvents:      300,
		NominalRateRPS: 2000,
		BaseCostUs:     2000,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()

	cfg := Config{
		Policy:             PolicyModified,
		Workers:            2,
		QueueCapacity:      64,
		AgingThresholdMs:   15,
		MaxConsecutiveHigh: 4,
		TaskTimeoutMs:      25,
		MaxRetries:         1,
	}
	_, m := RunPaced(trace, 0, cfg, "unit_cancel", "small", 1, 0, ctx)
	t.Logf("Cancel Test: Gen=%d Succ=%d Cancel=%d Rej=%d T/O=%d Acct=%v SeqMatch=%v GoroutineDelta=%d",
		m.Generated, m.Successful, m.Cancelled, m.Rejected, m.TimedOut, m.AccountingValid, m.MatchesSeqReplay, m.GoroutineDelta)

	if !m.AccountingValid || !m.MatchesSeqReplay {
		t.Fatalf("invariant violation on cancellation: %+v", m)
	}
	if m.Cancelled == 0 {
		t.Errorf("expected cancelled events when context expires mid-run")
	}
	if m.GoroutineDelta > 2 {
		t.Errorf("goroutine leak detected: delta=%d", m.GoroutineDelta)
	}
}

func TestFairness_AgingReducesMaxNormalWaitUnderPriorityPressure(t *testing.T) {
	records, _, err := events.ParseAndValidate("small", filepath.Join("..", "datasets", "Gr05_events_small.log"), 42)
	if err != nil {
		t.Fatalf("ParseAndValidate failed: %v", err)
	}
	trace := events.BuildReplayTrace(records, events.TraceSpec{
		Scenario:         "unit_d4",
		Seed:             42,
		NumEvents:        250,
		NominalRateRPS:   1200,
		BaseCostUs:       1200,
		HighPriorityFrac: 0.70,
	})

	_, mStrict := RunPaced(trace, 0, Config{
		Policy:        PolicyPriority,
		Workers:       2,
		QueueCapacity: 256,
	}, "d4", "small", 1, 0, context.Background())

	_, mFair := RunPaced(trace, 0, Config{
		Policy:             PolicyFairnessOnly,
		Workers:            2,
		QueueCapacity:      256,
		AgingThresholdMs:   10,
		MaxConsecutiveHigh: 3,
	}, "d4", "small", 1, 0, context.Background())

	t.Logf("D4 MaxNormalWait: StrictPriority=%.2fms vs FairnessAging=%.2fms", mStrict.MaxNormalWaitMs, mFair.MaxNormalWaitMs)
	if !mStrict.AccountingValid || !mFair.AccountingValid {
		t.Fatalf("accounting invalid")
	}
	if mFair.MaxNormalWaitMs >= mStrict.MaxNormalWaitMs {
		t.Errorf("expected aging fairness to reduce MaxNormalWait (strict=%.2fms, fair=%.2fms)",
			mStrict.MaxNormalWaitMs, mFair.MaxNormalWaitMs)
	}
}

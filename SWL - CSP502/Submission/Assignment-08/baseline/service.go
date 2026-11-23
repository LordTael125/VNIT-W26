package baseline

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"vnit.ac.in/csp502/practical8/workload"
)

// Service implements the Part B Baseline concurrent student-request service.
// It uses an unbounded intake buffer, no per-request SLA timeout, no intake
// deduplication, and naive non-idempotent downstream acknowledgement recording.
type Service struct {
	cfg       workload.Config
	startTime time.Time

	intakeMu     sync.RWMutex
	intakeClosed bool
	workQueue    chan *inFlightReq

	// Abrupt cancellation channel used by naive baseline shutdown/cancel.
	stopOnce sync.Once
	stopCh   chan struct{}

	wgWorkers sync.WaitGroup
	wgAcks    sync.WaitGroup

	// Instrumentation counters
	eventSeq      atomic.Uint64
	queueDepth    atomic.Int64
	maxQueueDepth atomic.Int64
	activeWorkers atomic.Int64
	peakWorkers   atomic.Int64

	eventsMu sync.Mutex
	events   []workload.LifecycleEvent

	recordsMu sync.Mutex
	records   []workload.TerminalRecord

	uniqueGenerated atomic.Int64
	totalSubmits    atomic.Int64
	dupSubmits      atomic.Int64
}

type inFlightReq struct {
	item         workload.Item
	arrivalTime  time.Time
	enqueuedTime time.Time
}

// NewService constructs a configured Baseline service instance.
func NewService(cfg workload.Config) *Service {
	bufCap := cfg.TotalRequests * 2
	if bufCap < 4096 {
		bufCap = 4096
	}
	return &Service{
		cfg:       cfg,
		startTime: time.Now(),
		workQueue: make(chan *inFlightReq, bufCap),
		stopCh:    make(chan struct{}),
		events:    make([]workload.LifecycleEvent, 0, cfg.TotalRequests*4),
		records:   make([]workload.TerminalRecord, 0, cfg.TotalRequests),
	}
}

func (s *Service) logEvent(reqID uint64, subSeq int, event string, outcome workload.Outcome, reason string, workerID int) {
	seq := s.eventSeq.Add(1)
	now := time.Now()
	ev := workload.LifecycleEvent{
		Seq:           seq,
		TimestampNs:   now.UnixNano(),
		ElapsedMs:     float64(now.Sub(s.startTime).Microseconds()) / 1000.0,
		RequestID:     reqID,
		SubmissionSeq: subSeq,
		Event:         event,
		Outcome:       outcome,
		Reason:        reason,
		WorkerID:      workerID,
		ActiveWorkers: s.activeWorkers.Load(),
		QueueDepth:    s.queueDepth.Load(),
	}
	s.eventsMu.Lock()
	s.events = append(s.events, ev)
	s.eventsMu.Unlock()
}

func (s *Service) incQueue() {
	v := s.queueDepth.Add(1)
	for {
		m := s.maxQueueDepth.Load()
		if v <= m || s.maxQueueDepth.CompareAndSwap(m, v) {
			break
		}
	}
}

func (s *Service) decQueue() {
	s.queueDepth.Add(-1)
}

func (s *Service) incActive() {
	v := s.activeWorkers.Add(1)
	for {
		m := s.peakWorkers.Load()
		if v <= m || s.peakWorkers.CompareAndSwap(m, v) {
			break
		}
	}
}

func (s *Service) decActive() {
	s.activeWorkers.Add(-1)
}

// Start launches the configured pool of baseline worker goroutines.
func (s *Service) Start() {
	s.startTime = time.Now()
	for w := 1; w <= s.cfg.NumWorkers; w++ {
		s.wgWorkers.Add(1)
		go s.worker(w)
	}
}

// Submit accepts a request item into the baseline service.
// Baseline performs NO duplicate-ID checking and NO bounded-queue admission control.
func (s *Service) Submit(item workload.Item) {
	now := time.Now()
	s.totalSubmits.Add(1)
	if item.IsDuplicateSubmit {
		s.dupSubmits.Add(1)
	} else {
		s.uniqueGenerated.Add(1)
	}

	s.logEvent(item.ID, item.SubmissionSeq, "generated", "", "", 0)

	s.intakeMu.RLock()
	if s.intakeClosed {
		s.intakeMu.RUnlock()
		end := time.Now()
		s.logEvent(item.ID, item.SubmissionSeq, "rejected", workload.OutcomeRejected, "intake closed during shutdown", 0)
		s.logEvent(item.ID, item.SubmissionSeq, "terminal", workload.OutcomeRejected, "intake closed during shutdown", 0)
		s.recordNaive(workload.TerminalRecord{
			RequestID:     item.ID,
			SubmissionSeq: item.SubmissionSeq,
			Type:          item.Type,
			Outcome:       workload.OutcomeRejected,
			Reason:        "intake closed during shutdown",
			ArrivalTime:   now,
			EndTime:       end,
			ResponseTime:  end.Sub(now),
			Admitted:      false,
		})
		return
	}

	req := &inFlightReq{
		item:         item,
		arrivalTime:  now,
		enqueuedTime: time.Now(),
	}
	s.incQueue()
	// Log admitted strictly before channel send to guarantee happens-before ordering with worker's "started" event.
	s.logEvent(item.ID, item.SubmissionSeq, "admitted", "", "", 0)
	s.workQueue <- req
	s.intakeMu.RUnlock()
}

func (s *Service) recordNaive(rec workload.TerminalRecord) {
	s.recordsMu.Lock()
	s.records = append(s.records, rec)
	s.recordsMu.Unlock()
}

func (s *Service) worker(workerID int) {
	defer s.wgWorkers.Done()

	for {
		select {
		case <-s.stopCh:
			// Naive baseline exits immediately on stop signal without draining remaining queued items.
			return
		case req, ok := <-s.workQueue:
			if !ok {
				return
			}
			s.decQueue()
			s.processRequest(workerID, req)
		}
	}
}

func (s *Service) processRequest(workerID int, req *inFlightReq) {
	s.incActive()
	defer s.decActive()

	started := time.Now()
	queueWait := started.Sub(req.enqueuedTime)
	s.logEvent(req.item.ID, req.item.SubmissionSeq, "started", "", "", workerID)

	// Baseline has no per-request SLA timeout: sleeps full service duration
	// (respecting only abrupt stopCh so tests can still terminate).
	timer := time.NewTimer(req.item.ServiceDuration)
	select {
	case <-s.stopCh:
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		end := time.Now()
		s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", workload.OutcomeCancelled, "abrupt shutdown", workerID)
		s.recordNaive(workload.TerminalRecord{
			RequestID:     req.item.ID,
			SubmissionSeq: req.item.SubmissionSeq,
			Type:          req.item.Type,
			Outcome:       workload.OutcomeCancelled,
			Reason:        "abrupt shutdown",
			WorkerID:      workerID,
			ArrivalTime:   req.arrivalTime,
			EnqueuedTime:  req.enqueuedTime,
			StartedTime:   started,
			EndTime:       end,
			QueueWait:     queueWait,
			ServiceTime:   end.Sub(started),
			ResponseTime:  end.Sub(req.arrivalTime),
			AckAttempts:   0,
			Admitted:      true,
		})
		return
	case <-timer.C:
	}

	outcome := workload.OutcomeSuccess
	reason := "ok"
	if req.item.IsFailing {
		outcome = workload.OutcomeFailure
		reason = "deterministic worker error"
	}

	// Downstream acknowledgement phase (naive, non-idempotent, no retry recovery).
	s.sendDownstreamAckNaive(workerID, req, started, queueWait, outcome, reason)
}

func (s *Service) sendDownstreamAckNaive(
	workerID int,
	req *inFlightReq,
	started time.Time,
	queueWait time.Duration,
	outcome workload.Outcome,
	reason string,
) {
	ackTimeout := time.Duration(s.cfg.AckTimeoutMs) * time.Millisecond
	if ackTimeout <= 0 {
		ackTimeout = 25 * time.Millisecond
	}

	// Fast path when no downstream ack faults affect this request.
	if !req.item.AckDelayed && !req.item.AckDropped && req.item.AckDuplicates == 0 {
		end := time.Now()
		s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", outcome, reason, workerID)
		s.recordNaive(workload.TerminalRecord{
			RequestID:     req.item.ID,
			SubmissionSeq: req.item.SubmissionSeq,
			Type:          req.item.Type,
			Outcome:       outcome,
			Reason:        reason,
			WorkerID:      workerID,
			ArrivalTime:   req.arrivalTime,
			EnqueuedTime:  req.enqueuedTime,
			StartedTime:   started,
			EndTime:       end,
			QueueWait:     queueWait,
			ServiceTime:   end.Sub(started),
			ResponseTime:  end.Sub(req.arrivalTime),
			AckAttempts:   1,
			Admitted:      true,
		})
		return
	}

	// Simulate unreliable downstream acknowledgement channel.
	ackCh := make(chan struct{}, 1+req.item.AckDuplicates)
	s.wgAcks.Add(1)
	go func() {
		defer s.wgAcks.Done()
		if req.item.AckDropped {
			// Acknowledgement lost in transit; nothing sent.
			return
		}
		if req.item.AckDelayed {
			time.Sleep(req.item.AckDelayDuration)
		}
		// Primary ack delivery
		ackCh <- struct{}{}
		// Repeated/duplicate downstream ack deliveries
		for d := 0; d < req.item.AckDuplicates; d++ {
			time.Sleep(500 * time.Microsecond)
			ackCh <- struct{}{}
		}
	}()

	timer := time.NewTimer(ackTimeout)
	defer timer.Stop()

	select {
	case <-ackCh:
		end := time.Now()
		s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", outcome, reason, workerID)
		s.recordNaive(workload.TerminalRecord{
			RequestID:     req.item.ID,
			SubmissionSeq: req.item.SubmissionSeq,
			Type:          req.item.Type,
			Outcome:       outcome,
			Reason:        reason,
			WorkerID:      workerID,
			ArrivalTime:   req.arrivalTime,
			EnqueuedTime:  req.enqueuedTime,
			StartedTime:   started,
			EndTime:       end,
			QueueWait:     queueWait,
			ServiceTime:   end.Sub(started),
			ResponseTime:  end.Sub(req.arrivalTime),
			AckAttempts:   1,
			Admitted:      true,
		})
	case <-timer.C:
		// Ack timed out: naive baseline records Timeout immediately without retry.
		end := time.Now()
		s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", workload.OutcomeTimeout, "downstream ack timeout", workerID)
		s.recordNaive(workload.TerminalRecord{
			RequestID:     req.item.ID,
			SubmissionSeq: req.item.SubmissionSeq,
			Type:          req.item.Type,
			Outcome:       workload.OutcomeTimeout,
			Reason:        "downstream ack timeout",
			WorkerID:      workerID,
			ArrivalTime:   req.arrivalTime,
			EnqueuedTime:  req.enqueuedTime,
			StartedTime:   started,
			EndTime:       end,
			QueueWait:     queueWait,
			ServiceTime:   end.Sub(started),
			ResponseTime:  end.Sub(req.arrivalTime),
			AckAttempts:   1,
			Admitted:      true,
		})
	}

	// Naive baseline also processes late/repeated ack callbacks arriving on ackCh
	// without idempotency checking, causing duplicate terminal outcomes.
	if req.item.AckDelayed || req.item.AckDuplicates > 0 {
		s.wgAcks.Add(1)
		go func() {
			defer s.wgAcks.Done()
			extraWait := req.item.AckDelayDuration + 10*time.Millisecond
			if extraWait < 15*time.Millisecond {
				extraWait = 15 * time.Millisecond
			}
			deadline := time.NewTimer(extraWait)
			defer deadline.Stop()
			for {
				select {
				case <-ackCh:
					lateEnd := time.Now()
					s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", outcome, "late_or_duplicate_ack", workerID)
					s.recordNaive(workload.TerminalRecord{
						RequestID:     req.item.ID,
						SubmissionSeq: req.item.SubmissionSeq,
						Type:          req.item.Type,
						Outcome:       outcome,
						Reason:        "late_or_duplicate_ack",
						WorkerID:      workerID,
						ArrivalTime:   req.arrivalTime,
						EnqueuedTime:  req.enqueuedTime,
						StartedTime:   started,
						EndTime:       lateEnd,
						QueueWait:     queueWait,
						ServiceTime:   lateEnd.Sub(started),
						ResponseTime:  lateEnd.Sub(req.arrivalTime),
						AckAttempts:   1,
						Admitted:      true,
					})
				case <-deadline.C:
					return
				}
			}
		}()
	}
}

// CloseAndDrain performs normal end-of-workload shutdown by closing the work queue
// and waiting unboundedly for all queued requests to finish.
func (s *Service) CloseAndDrain() (time.Duration, int) {
	t0 := time.Now()
	s.intakeMu.Lock()
	if !s.intakeClosed {
		s.intakeClosed = true
		close(s.workQueue)
	}
	s.intakeMu.Unlock()

	s.wgWorkers.Wait()
	s.wgAcks.Wait()
	outstanding := int(s.queueDepth.Load())
	return time.Since(t0), outstanding
}

// ShutdownAbrupt models naive baseline cancellation/shutdown: closes intake and
// immediately signals workers to stop without draining buffered queued items.
func (s *Service) ShutdownAbrupt() (time.Duration, int) {
	t0 := time.Now()
	s.intakeMu.Lock()
	if !s.intakeClosed {
		s.intakeClosed = true
		close(s.workQueue)
	}
	s.intakeMu.Unlock()

	s.stopOnce.Do(func() {
		close(s.stopCh)
	})

	s.wgWorkers.Wait()
	s.wgAcks.Wait()
	outstanding := len(s.workQueue)
	return time.Since(t0), outstanding
}

// Cancel triggers abrupt cancellation of workers (used by Part C cancellation test).
func (s *Service) Cancel() {
	s.stopOnce.Do(func() {
		close(s.stopCh)
	})
}

// RunWorkload executes a full deterministic workload end-to-end and returns metrics.
func RunWorkload(cfg workload.Config, seed int64, rep int, eventLogPath string) (workload.RunSummary, error) {
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
	baseGoroutines := runtime.NumGoroutine()

	items := workload.Generate(cfg, seed)
	svc := NewService(cfg)
	svc.Start()
	wallStart := time.Now()

	var shutdownDur time.Duration
	var outstanding int
	var shutdownDone sync.WaitGroup
	var shutdownTriggered atomic.Bool

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Optional mid-run shutdown / cancellation trigger
	if cfg.Faults.ShutdownAfterMs > 0 {
		shutdownDone.Add(1)
		go func() {
			defer shutdownDone.Done()
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Duration(cfg.Faults.ShutdownAfterMs) * time.Millisecond):
				shutdownTriggered.Store(true)
				shutdownDur, outstanding = svc.ShutdownAbrupt()
			}
		}()
	} else if cfg.Faults.CancelAfterMs > 0 {
		shutdownDone.Add(1)
		go func() {
			defer shutdownDone.Done()
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Duration(cfg.Faults.CancelAfterMs) * time.Millisecond):
				shutdownTriggered.Store(true)
				shutdownDur, outstanding = svc.ShutdownAbrupt()
			}
		}()
	}

	for _, item := range items {
		svc.Submit(item)
		if item.InterArrival > 0 {
			// When intake is already closed during mid-run shutdown, fast-forward
			// remaining scheduled arrivals so they still register as rejected promptly.
			if !shutdownTriggered.Load() {
				time.Sleep(item.InterArrival)
			}
		}
	}

	cancel()
	shutdownDone.Wait()

	if !shutdownTriggered.Load() {
		shutdownDur, outstanding = svc.CloseAndDrain()
	}

	wallDur := time.Since(wallStart)

	time.Sleep(15 * time.Millisecond)
	goroutineDelta := runtime.NumGoroutine() - baseGoroutines
	if goroutineDelta < 0 {
		goroutineDelta = 0
	}

	summary := svc.Summarize(cfg.Condition, seed, rep, wallDur, shutdownDur, outstanding, goroutineDelta)

	if eventLogPath != "" {
		if err := svc.WriteEventsJSONL(eventLogPath); err != nil {
			return summary, err
		}
	}
	return summary, nil
}

// Events returns a snapshot copy of all recorded lifecycle events.
func (s *Service) Events() []workload.LifecycleEvent {
	s.eventsMu.Lock()
	defer s.eventsMu.Unlock()
	cp := make([]workload.LifecycleEvent, len(s.events))
	copy(cp, s.events)
	return cp
}

// Records returns a snapshot copy of all terminal records.
func (s *Service) Records() []workload.TerminalRecord {
	s.recordsMu.Lock()
	defer s.recordsMu.Unlock()
	cp := make([]workload.TerminalRecord, len(s.records))
	copy(cp, s.records)
	return cp
}

// WriteEventsJSONL persists all raw lifecycle events as newline-delimited JSON.
func (s *Service) WriteEventsJSONL(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	events := s.Events()
	for _, ev := range events {
		b, err := json.Marshal(ev)
		if err != nil {
			return err
		}
		if _, err := w.Write(b); err != nil {
			return err
		}
		if err := w.WriteByte('\n'); err != nil {
			return err
		}
	}
	return w.Flush()
}

// Summarize evaluates all Part B/E metrics and runs formal invariant verification.
func (s *Service) Summarize(
	condition string,
	seed int64,
	rep int,
	wallDur time.Duration,
	shutdownDur time.Duration,
	outstanding int,
	goroutineDelta int,
) workload.RunSummary {
	recs := s.Records()
	events := s.Events()

	sum := workload.RunSummary{
		Architecture:                 "Baseline",
		Condition:                    condition,
		Seed:                         seed,
		Repetition:                   rep,
		UniqueGenerated:              int(s.uniqueGenerated.Load()),
		TotalSubmissions:             int(s.totalSubmits.Load()),
		DuplicateSubmissions:         int(s.dupSubmits.Load()),
		TotalTerminalRecords:         len(recs),
		MaxQueueDepth:                s.maxQueueDepth.Load(),
		PeakActiveWorkers:            s.peakWorkers.Load(),
		WallDurationSec:              wallDur.Seconds(),
		ShutdownCompletionMs:         float64(shutdownDur.Microseconds()) / 1000.0,
		OutstandingWorkAfterShutdown: outstanding,
		GoroutineDelta:               goroutineDelta,
	}

	seenCounts := make(map[uint64]int, sum.UniqueGenerated)
	admittedRespMs := make([]float64, 0, len(recs))
	overallRespMs := make([]float64, 0, len(recs))
	var totalQueueWaitMs float64
	admittedN := 0

	for _, r := range recs {
		seenCounts[r.RequestID]++
		switch r.Outcome {
		case workload.OutcomeSuccess:
			sum.Successful++
		case workload.OutcomeFailure:
			sum.Failed++
		case workload.OutcomeTimeout:
			sum.TimedOut++
		case workload.OutcomeCancelled:
			sum.Cancelled++
		case workload.OutcomeRejected:
			sum.Rejected++
		}
		respMs := float64(r.ResponseTime.Microseconds()) / 1000.0
		overallRespMs = append(overallRespMs, respMs)
		if r.Admitted {
			admittedRespMs = append(admittedRespMs, respMs)
			totalQueueWaitMs += float64(r.QueueWait.Microseconds()) / 1000.0
			admittedN++
		}
	}

	sum.UniqueTerminalIDs = len(seenCounts)
	for _, c := range seenCounts {
		if c > 1 {
			sum.DuplicateTerminalOutcomes += (c - 1)
		}
	}

	missing := 0
	for id := uint64(1); id <= uint64(sum.UniqueGenerated); id++ {
		if seenCounts[id] == 0 {
			missing++
		}
	}
	sum.MissingTerminalIDs = missing

	// Accounting violation = missing terminal outcomes + duplicate terminal outcomes
	sum.TerminalAccountingViolations = missing + sum.DuplicateTerminalOutcomes
	sum.LifecycleOrderViolations = workload.VerifyLifecycleOrder(events)

	if admittedN > 0 {
		sum.MeanQueueWaitMs = totalQueueWaitMs / float64(admittedN)
		sum.MeanResponseMs, sum.P50ResponseMs, sum.P95ResponseMs, sum.P99ResponseMs, sum.MaxResponseMs = workload.ComputeLatencies(admittedRespMs)
	}
	if len(overallRespMs) > 0 {
		sum.OverallMeanResponseMs, _, _, _, _ = workload.ComputeLatencies(overallRespMs)
	}

	if wallDur.Seconds() > 0 {
		sum.ThroughputRPS = float64(sum.UniqueTerminalIDs) / wallDur.Seconds()
		sum.GoodputRPS = float64(sum.Successful) / wallDur.Seconds()
	}

	sum.InvariantsPassed = (sum.TerminalAccountingViolations == 0 &&
		sum.DuplicateTerminalOutcomes == 0 &&
		sum.MissingTerminalIDs == 0 &&
		sum.OutstandingWorkAfterShutdown == 0 &&
		sum.LifecycleOrderViolations == 0)

	return sum
}

// FormatInvariantReport returns a human-readable verification report for Part B.
func FormatInvariantReport(sum workload.RunSummary) string {
	return fmt.Sprintf(
		"[%s | %s | seed=%d] Generated=%d TerminalRecords=%d UniqueIDs=%d (Succ=%d Fail=%d TimeOut=%d Cancel=%d Rej=%d) | AcctViolations=%d DupOutcomes=%d Missing=%d LifecycleViol=%d Outstanding=%d | Throughput=%.1f req/s Goodput=%.1f req/s Mean=%.2fms p95=%.2fms Shutdown=%.2fms | PASS=%v",
		sum.Architecture, sum.Condition, sum.Seed,
		sum.UniqueGenerated, sum.TotalTerminalRecords, sum.UniqueTerminalIDs,
		sum.Successful, sum.Failed, sum.TimedOut, sum.Cancelled, sum.Rejected,
		sum.TerminalAccountingViolations, sum.DuplicateTerminalOutcomes, sum.MissingTerminalIDs,
		sum.LifecycleOrderViolations, sum.OutstandingWorkAfterShutdown,
		sum.ThroughputRPS, sum.GoodputRPS, sum.MeanResponseMs, sum.P95ResponseMs, sum.ShutdownCompletionMs,
		sum.InvariantsPassed,
	)
}

package improved

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

// ResultLedger enforces Invariant I2 (Exactly-Once Recorded Outcome) via
// atomic compare-and-commit keyed by RequestID.
type ResultLedger struct {
	mu        sync.Mutex
	committed map[uint64]workload.TerminalRecord
	order     []workload.TerminalRecord
}

func NewResultLedger(capacity int) *ResultLedger {
	if capacity < 64 {
		capacity = 64
	}
	return &ResultLedger{
		committed: make(map[uint64]workload.TerminalRecord, capacity),
		order:     make([]workload.TerminalRecord, 0, capacity),
	}
}

// CommitOnce records rec only if RequestID has not yet committed a terminal outcome.
// Returns true if this call committed the terminal outcome, false if already committed.
func (l *ResultLedger) CommitOnce(rec workload.TerminalRecord) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, exists := l.committed[rec.RequestID]; exists {
		return false
	}
	l.committed[rec.RequestID] = rec
	l.order = append(l.order, rec)
	return true
}

// IsCommitted reports whether RequestID already has a terminal outcome.
func (l *ResultLedger) IsCommitted(reqID uint64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	_, exists := l.committed[reqID]
	return exists
}

// All returns a snapshot of all committed terminal records in commit order.
func (l *ResultLedger) All() []workload.TerminalRecord {
	l.mu.Lock()
	defer l.mu.Unlock()
	cp := make([]workload.TerminalRecord, len(l.order))
	copy(cp, l.order)
	return cp
}

// Service implements the Part E Improved concurrent student-request service:
//   1. Part D: At-least-once bounded ack recovery + exactly-once recorded outcome (ResultLedger)
//   2. Idempotent request intake deduplication
//   3. Explicit bounded-queue admission control (load shedding)
//   4. Per-request SLA context deadlines & cooperative cancellation drain
//   5. Two-phase graceful bounded shutdown with zero stranded work & zero leaked goroutines
type Service struct {
	cfg       workload.Config
	startTime time.Time

	svcCtx    context.Context
	svcCancel context.CancelFunc

	ackStopOnce sync.Once
	ackStopCh   chan struct{}

	intakeMu     sync.Mutex
	intakeClosed bool
	seenIntake   map[uint64]struct{}
	workQueue    chan *inFlightReq

	ledger *ResultLedger

	wgWorkers sync.WaitGroup
	wgAcks    sync.WaitGroup

	// Global retry budget (prevents retry storms under systemic downstream stalls)
	retryBudgetRemaining atomic.Int64
	ackRetriesUsed       atomic.Int64
	dupAcksIgnored       atomic.Int64
	dupSubmitsBlocked    atomic.Int64

	// Telemetry counters
	eventSeq      atomic.Uint64
	queueDepth    atomic.Int64
	maxQueueDepth atomic.Int64
	activeWorkers atomic.Int64
	peakWorkers   atomic.Int64

	eventsMu sync.Mutex
	events   []workload.LifecycleEvent

	uniqueGenerated atomic.Int64
	totalSubmits    atomic.Int64
	dupSubmits      atomic.Int64
}

type inFlightReq struct {
	item         workload.Item
	arrivalTime  time.Time
	enqueuedTime time.Time
}

// NewService constructs a configured Improved service instance.
func NewService(cfg workload.Config) *Service {
	qCap := cfg.QueueCapacity
	if qCap <= 0 {
		qCap = 64
	}
	budget := int64(float64(cfg.TotalRequests) * cfg.RetryBudgetRatio)
	if budget < 10 {
		budget = 10
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &Service{
		cfg:        cfg,
		startTime:  time.Now(),
		svcCtx:     ctx,
		svcCancel:  cancel,
		ackStopCh:  make(chan struct{}),
		seenIntake: make(map[uint64]struct{}, cfg.TotalRequests),
		workQueue:  make(chan *inFlightReq, qCap),
		ledger:     NewResultLedger(cfg.TotalRequests),
		events:     make([]workload.LifecycleEvent, 0, cfg.TotalRequests*5),
	}
	s.retryBudgetRemaining.Store(budget)
	return s
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

func (s *Service) tryConsumeRetryBudget() bool {
	for {
		cur := s.retryBudgetRemaining.Load()
		if cur <= 0 {
			return false
		}
		if s.retryBudgetRemaining.CompareAndSwap(cur, cur-1) {
			return true
		}
	}
}

// Start launches the configured pool of worker goroutines.
func (s *Service) Start() {
	s.startTime = time.Now()
	for w := 1; w <= s.cfg.NumWorkers; w++ {
		s.wgWorkers.Add(1)
		go s.worker(w)
	}
}

// Submit enforces idempotent intake deduplication and explicit bounded-queue admission control.
// Returns true if admitted to the worker queue, false if deduplicated or rejected.
func (s *Service) Submit(item workload.Item) bool {
	now := time.Now()
	s.totalSubmits.Add(1)
	if item.IsDuplicateSubmit {
		s.dupSubmits.Add(1)
	}

	s.logEvent(item.ID, item.SubmissionSeq, "generated", "", "", 0)

	s.intakeMu.Lock()
	// 1. Idempotent intake deduplication: never re-enqueue or overwrite an already-seen RequestID.
	if _, seen := s.seenIntake[item.ID]; seen {
		s.intakeMu.Unlock()
		s.dupSubmitsBlocked.Add(1)
		s.logEvent(item.ID, item.SubmissionSeq, "duplicate_ignored", "", "duplicate request ID deduplicated at intake", 0)
		return false
	}
	s.seenIntake[item.ID] = struct{}{}
	s.uniqueGenerated.Add(1)

	// 2. Shutdown admission gate: fast-reject arrivals once shutdown has begun.
	if s.intakeClosed {
		s.intakeMu.Unlock()
		end := time.Now()
		s.logEvent(item.ID, item.SubmissionSeq, "rejected", workload.OutcomeRejected, "service shutting down", 0)
		if s.ledger.CommitOnce(workload.TerminalRecord{
			RequestID:     item.ID,
			SubmissionSeq: item.SubmissionSeq,
			Type:          item.Type,
			Outcome:       workload.OutcomeRejected,
			Reason:        "service shutting down",
			ArrivalTime:   now,
			EndTime:       end,
			ResponseTime:  end.Sub(now),
			Admitted:      false,
		}) {
			s.logEvent(item.ID, item.SubmissionSeq, "terminal", workload.OutcomeRejected, "service shutting down", 0)
		}
		return false
	}

	// 3. Explicit bounded-queue admission control: check capacity under intakeMu so
	// we can log "admitted" strictly before channel send (preserving causal happens-before order).
	if len(s.workQueue) < cap(s.workQueue) {
		req := &inFlightReq{
			item:         item,
			arrivalTime:  now,
			enqueuedTime: time.Now(),
		}
		s.incQueue()
		s.logEvent(item.ID, item.SubmissionSeq, "admitted", "", "", 0)
		s.workQueue <- req
		s.intakeMu.Unlock()
		return true
	}

	s.intakeMu.Unlock()
	end := time.Now()
	s.logEvent(item.ID, item.SubmissionSeq, "rejected", workload.OutcomeRejected, "admission control: queue saturated", 0)
	if s.ledger.CommitOnce(workload.TerminalRecord{
		RequestID:     item.ID,
		SubmissionSeq: item.SubmissionSeq,
		Type:          item.Type,
		Outcome:       workload.OutcomeRejected,
		Reason:        "admission control: queue saturated",
		ArrivalTime:   now,
		EndTime:       end,
		ResponseTime:  end.Sub(now),
		Admitted:      false,
	}) {
		s.logEvent(item.ID, item.SubmissionSeq, "terminal", workload.OutcomeRejected, "admission control: queue saturated", 0)
	}
	return false
}

func (s *Service) worker(workerID int) {
	defer s.wgWorkers.Done()

	// Always drain workQueue to completion once closed so zero queued items are ever stranded.
	for req := range s.workQueue {
		s.decQueue()

		// If service context is already cancelled (grace period expired or explicit cancel),
		// immediately commit Cancelled without burning CPU on stale queued work.
		if s.svcCtx.Err() != nil {
			end := time.Now()
			qWait := end.Sub(req.enqueuedTime)
			if s.ledger.CommitOnce(workload.TerminalRecord{
				RequestID:     req.item.ID,
				SubmissionSeq: req.item.SubmissionSeq,
				Type:          req.item.Type,
				Outcome:       workload.OutcomeCancelled,
				Reason:        "cancelled while queued",
				WorkerID:      workerID,
				ArrivalTime:   req.arrivalTime,
				EnqueuedTime:  req.enqueuedTime,
				EndTime:       end,
				QueueWait:     qWait,
				ResponseTime:  end.Sub(req.arrivalTime),
				Admitted:      true,
			}) {
				s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", workload.OutcomeCancelled, "cancelled while queued", workerID)
			}
			continue
		}

		s.processRequest(workerID, req)
	}
}

func (s *Service) processRequest(workerID int, req *inFlightReq) {
	s.incActive()
	defer s.decActive()

	started := time.Now()
	queueWait := started.Sub(req.enqueuedTime)
	s.logEvent(req.item.ID, req.item.SubmissionSeq, "started", "", "", workerID)

	// Per-request SLA deadline derived from service context.
	reqTimeout := time.Duration(s.cfg.RequestTimeoutMs) * time.Millisecond
	var reqCtx context.Context
	var reqCancel context.CancelFunc
	if reqTimeout > 0 {
		reqCtx, reqCancel = context.WithTimeout(s.svcCtx, reqTimeout)
	} else {
		reqCtx, reqCancel = context.WithCancel(s.svcCtx)
	}
	defer reqCancel()

	timer := time.NewTimer(req.item.ServiceDuration)
	select {
	case <-reqCtx.Done():
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		end := time.Now()
		var outcome workload.Outcome
		var reason string
		if s.svcCtx.Err() != nil {
			outcome = workload.OutcomeCancelled
			reason = "service context cancelled during processing"
		} else {
			outcome = workload.OutcomeTimeout
			reason = fmt.Sprintf("SLA processing timeout (%dms)", s.cfg.RequestTimeoutMs)
		}
		if s.ledger.CommitOnce(workload.TerminalRecord{
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
			Admitted:      true,
		}) {
			s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", outcome, reason, workerID)
		}
		return
	case <-timer.C:
	}

	outcome := workload.OutcomeSuccess
	reason := "ok"
	if req.item.IsFailing {
		outcome = workload.OutcomeFailure
		reason = "deterministic worker error"
	}

	// Part D: Resilient downstream acknowledgement with bounded retry budget & idempotent commit ledger.
	s.sendDownstreamAckResilient(workerID, req, started, queueWait, outcome, reason)
}

func (s *Service) sendDownstreamAckResilient(
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
	maxRetries := s.cfg.MaxAckRetries
	if maxRetries < 0 {
		maxRetries = 0
	}

	commitAck := func(attempt int, src string) bool {
		end := time.Now()
		committed := s.ledger.CommitOnce(workload.TerminalRecord{
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
			AckAttempts:   attempt,
			Admitted:      true,
		})
		if committed {
			s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", outcome, reason, workerID)
			return true
		}
		// Late or repeated downstream acknowledgement: safely deduplicated without mutating state!
		s.dupAcksIgnored.Add(1)
		s.logEvent(req.item.ID, req.item.SubmissionSeq, "ack_duplicate_ignored", "", src, workerID)
		return false
	}

	// Fast path when no downstream ack disturbance affects this request.
	if !req.item.AckDelayed && !req.item.AckDropped && req.item.AckDuplicates == 0 {
		commitAck(1, "direct")
		return
	}

	// Attempt loop: At-Least-Once acknowledgement delivery (bounded) + Exactly-Once Recorded Outcome.
	totalAttempts := 1 + maxRetries
	for attempt := 1; attempt <= totalAttempts; attempt++ {
		if s.ledger.IsCommitted(req.item.ID) {
			return
		}
		if s.svcCtx.Err() != nil {
			end := time.Now()
			if s.ledger.CommitOnce(workload.TerminalRecord{
				RequestID:     req.item.ID,
				SubmissionSeq: req.item.SubmissionSeq,
				Type:          req.item.Type,
				Outcome:       workload.OutcomeCancelled,
				Reason:        "cancelled during ack wait",
				WorkerID:      workerID,
				ArrivalTime:   req.arrivalTime,
				EnqueuedTime:  req.enqueuedTime,
				StartedTime:   started,
				EndTime:       end,
				QueueWait:     queueWait,
				ServiceTime:   end.Sub(started),
				ResponseTime:  end.Sub(req.arrivalTime),
				AckAttempts:   attempt,
				Admitted:      true,
			}) {
				s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", workload.OutcomeCancelled, "cancelled during ack wait", workerID)
			}
			return
		}

		if attempt > 1 {
			if !s.tryConsumeRetryBudget() {
				break
			}
			s.ackRetriesUsed.Add(1)
			backoff := time.Duration(1<<(attempt-2)) * 2 * time.Millisecond
			if backoff > 10*time.Millisecond {
				backoff = 10 * time.Millisecond
			}
			s.logEvent(req.item.ID, req.item.SubmissionSeq, "ack_retry", "", fmt.Sprintf("attempt=%d backoff=%v", attempt, backoff), workerID)
			bt := time.NewTimer(backoff)
			select {
			case <-s.svcCtx.Done():
				bt.Stop()
				continue
			case <-bt.C:
			}
			if s.ledger.IsCommitted(req.item.ID) {
				// A delayed acknowledgement from attempt 1 arrived during backoff and committed cleanly!
				return
			}
		}

		ackNotify := make(chan struct{}, 1)
		s.wgAcks.Add(1)
		go func(att int) {
			defer s.wgAcks.Done()

			// Attempt 1 experiences the configured downstream fault (drop/delay/duplicates).
			// Retry attempts experience recovered delivery (still guarded against duplicates).
			if att == 1 && req.item.AckDropped {
				return
			}
			if att == 1 && req.item.AckDelayed {
				dt := time.NewTimer(req.item.AckDelayDuration)
				select {
				case <-s.ackStopCh:
					dt.Stop()
					return
				case <-dt.C:
				}
			}

			commitAck(att, fmt.Sprintf("downstream_ack_attempt_%d", att))
			select {
			case ackNotify <- struct{}{}:
			default:
			}

			// Simulate repeated/duplicate downstream acknowledgements arriving shortly after primary ack.
			if att == 1 && req.item.AckDuplicates > 0 {
				for d := 0; d < req.item.AckDuplicates; d++ {
					dt := time.NewTimer(500 * time.Microsecond)
					select {
					case <-s.ackStopCh:
						dt.Stop()
						return
					case <-dt.C:
					}
					commitAck(att, fmt.Sprintf("downstream_duplicate_ack_%d", d+1))
				}
			}
		}(attempt)

		at := time.NewTimer(ackTimeout)
		select {
		case <-ackNotify:
			if !at.Stop() {
				select {
				case <-at.C:
				default:
				}
			}
			return
		case <-s.svcCtx.Done():
			if !at.Stop() {
				select {
				case <-at.C:
				default:
				}
			}
			continue
		case <-at.C:
			// Timed out waiting for this attempt's ack; proceed to bounded retry if budget permits.
		}
	}

	// Exhausted retry limit or global retry budget without receiving an ack.
	end := time.Now()
	if s.ledger.CommitOnce(workload.TerminalRecord{
		RequestID:     req.item.ID,
		SubmissionSeq: req.item.SubmissionSeq,
		Type:          req.item.Type,
		Outcome:       workload.OutcomeTimeout,
		Reason:        "downstream ack retry limit/budget exhausted",
		WorkerID:      workerID,
		ArrivalTime:   req.arrivalTime,
		EnqueuedTime:  req.enqueuedTime,
		StartedTime:   started,
		EndTime:       end,
		QueueWait:     queueWait,
		ServiceTime:   end.Sub(started),
		ResponseTime:  end.Sub(req.arrivalTime),
		AckAttempts:   totalAttempts,
		Admitted:      true,
	}) {
		s.logEvent(req.item.ID, req.item.SubmissionSeq, "terminal", workload.OutcomeTimeout, "downstream ack retry limit/budget exhausted", workerID)
	}
}

func (s *Service) closeIntake() {
	s.intakeMu.Lock()
	if !s.intakeClosed {
		s.intakeClosed = true
		close(s.workQueue)
	}
	s.intakeMu.Unlock()
}

// Shutdown executes two-phase graceful bounded shutdown:
//  Phase 1: Close intake gate (new arrivals fast-rejected with OutcomeRejected).
//  Phase 2: Wait up to grace for workers to drain accepted work; on grace expiry,
//           cancel service context so workers abort stragglers and drain remaining
//           queued items as OutcomeCancelled with zero stranded work and zero leaks.
func (s *Service) Shutdown(grace time.Duration) (time.Duration, int) {
	t0 := time.Now()
	s.closeIntake()

	workersDone := make(chan struct{})
	go func() {
		s.wgWorkers.Wait()
		close(workersDone)
	}()

	if grace <= 0 {
		s.svcCancel()
		<-workersDone
	} else {
		gt := time.NewTimer(grace)
		select {
		case <-workersDone:
			if !gt.Stop() {
				select {
				case <-gt.C:
				default:
				}
			}
		case <-gt.C:
			s.svcCancel()
			<-workersDone
		}
	}

	// Stop any lingering delayed-ack simulator timers cleanly after workers exit
	// so bounded shutdown never leaks goroutines or stalls indefinitely.
	s.ackStopOnce.Do(func() {
		close(s.ackStopCh)
	})
	s.wgAcks.Wait()

	outstanding := len(s.workQueue) + int(s.activeWorkers.Load())
	return time.Since(t0), outstanding
}

// CloseAndDrain cleanly finishes all accepted requests and lets in-flight delayed acks
// arrive and be deduplicated by the idempotent ledger before stopping.
func (s *Service) CloseAndDrain() (time.Duration, int) {
	t0 := time.Now()
	s.closeIntake()
	s.wgWorkers.Wait()
	// Allow any in-flight delayed/duplicate downstream acks to settle & verify deduplication.
	s.wgAcks.Wait()
	s.svcCancel()
	outstanding := len(s.workQueue) + int(s.activeWorkers.Load())
	return time.Since(t0), outstanding
}

// Cancel immediately cancels the service context while safely closing intake and draining queued items.
func (s *Service) Cancel() (time.Duration, int) {
	return s.Shutdown(0)
}

// RunWorkload executes a deterministic workload end-to-end and returns comprehensive reliability metrics.
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

	runCtx, runCancel := context.WithCancel(context.Background())
	defer runCancel()

	grace := time.Duration(cfg.ShutdownGraceMs) * time.Millisecond
	if grace <= 0 {
		grace = 150 * time.Millisecond
	}

	if cfg.Faults.ShutdownAfterMs > 0 {
		shutdownDone.Add(1)
		go func() {
			defer shutdownDone.Done()
			select {
			case <-runCtx.Done():
				return
			case <-time.After(time.Duration(cfg.Faults.ShutdownAfterMs) * time.Millisecond):
				shutdownTriggered.Store(true)
				shutdownDur, outstanding = svc.Shutdown(grace)
			}
		}()
	} else if cfg.Faults.CancelAfterMs > 0 {
		shutdownDone.Add(1)
		go func() {
			defer shutdownDone.Done()
			select {
			case <-runCtx.Done():
				return
			case <-time.After(time.Duration(cfg.Faults.CancelAfterMs) * time.Millisecond):
				shutdownTriggered.Store(true)
				shutdownDur, outstanding = svc.Cancel()
			}
		}()
	}

	for _, item := range items {
		svc.Submit(item)
		if item.InterArrival > 0 {
			if !shutdownTriggered.Load() {
				time.Sleep(item.InterArrival)
			}
		}
	}

	runCancel()
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

// Records returns a snapshot of all committed terminal records.
func (s *Service) Records() []workload.TerminalRecord {
	return s.ledger.All()
}

// WriteEventsJSONL persists raw lifecycle events as newline-delimited JSON.
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

// Summarize evaluates all Part E metrics and checks all invariants.
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
		Architecture:                 "Improved",
		Condition:                    condition,
		Seed:                         seed,
		Repetition:                   rep,
		UniqueGenerated:              int(s.uniqueGenerated.Load()),
		TotalSubmissions:             int(s.totalSubmits.Load()),
		DuplicateSubmissions:         int(s.dupSubmits.Load()),
		DuplicateSubmissionsBlocked:  int(s.dupSubmitsBlocked.Load()),
		TotalTerminalRecords:         len(recs),
		AckRetriesUsed:               int(s.ackRetriesUsed.Load()),
		DuplicateAcksIgnored:         int(s.dupAcksIgnored.Load()),
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

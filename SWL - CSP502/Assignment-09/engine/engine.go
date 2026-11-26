package engine

import (
	"context"
	"math"
	"reflect"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"vnit.ac.in/csp502/lab9/events"
)

// Policy identifies the scheduling & resilience configuration.
type Policy string

const (
	PolicyFIFO           Policy = "fifo"
	PolicyPriority       Policy = "priority"
	PolicyFairnessOnly   Policy = "fairness_only"
	PolicyResilienceOnly Policy = "resilience_only"
	PolicyModified       Policy = "modified"
)

// TerminalStatus defines the mutually exclusive per-event outcome.
type TerminalStatus string

const (
	StatusSuccessful TerminalStatus = "Successful"
	StatusFailed     TerminalStatus = "Failed"
	StatusTimedOut   TerminalStatus = "TimedOut"
	StatusCancelled  TerminalStatus = "Cancelled"
	StatusRejected   TerminalStatus = "Rejected"
)

// Config controls worker pool, queue bounds, fairness, and fault handling.
type Config struct {
	Policy             Policy
	Workers            int
	QueueCapacity      int // 0 means unbounded (large enough for all events)
	AgingThresholdMs   float64
	MaxConsecutiveHigh int
	TaskTimeoutMs      float64
	MaxRetries         int // 0 = single attempt; 1 = up to 2 attempts total
	RecoveryThreshold  int // queue depth threshold after D1 burst (default 5)
}

// AggregateResult holds the domain outputs required by Lab 9.
type AggregateResult struct {
	IPFailedCounts   map[string]int
	FlaggedIPs       []string // sorted IPs with failed count >= 5
	FileAccessCounts map[string]int
	OtherEventCounts map[string]int
	MalformedCount   int
}

// RunMetrics captures all Part B-F performance, accounting, and fairness metrics.
type RunMetrics struct {
	Scenario              string  `json:"scenario"`
	Policy                string  `json:"policy"`
	Dataset               string  `json:"dataset"`
	Repetition            int     `json:"repetition"`
	Workers               int     `json:"workers"`
	QueueCapacity         int     `json:"queue_capacity"`
	Generated             int     `json:"generated"`
	Malformed             int     `json:"malformed"`
	Successful            int     `json:"successful"`
	Failed                int     `json:"failed"`
	TimedOut              int     `json:"timed_out"`
	Cancelled             int     `json:"cancelled"`
	Rejected              int     `json:"rejected"`
	TotalAttempts         int     `json:"total_attempts"`
	RetriesUsed           int     `json:"retries_used"`
	AccountingValid       bool    `json:"accounting_valid"`
	DuplicateOutcomes     int     `json:"duplicate_outcomes"`
	MissingIDs            int     `json:"missing_ids"`
	MatchesSeqReplay      bool    `json:"matches_seq_replay"`
	FlaggedIPCount        int     `json:"flagged_ip_count"`
	ElapsedSec            float64 `json:"elapsed_sec"`
	ThroughputRPS         float64 `json:"throughput_rps"`
	PeakQueueLen          int     `json:"peak_queue_len"`
	PeakHeapMB            float64 `json:"peak_heap_mb"`
	HighAccepted          int     `json:"high_accepted"`
	HighSuccessful        int     `json:"high_successful"`
	HighSuccessRate       float64 `json:"high_success_rate"`
	HighMedianLatMs       float64 `json:"high_median_lat_ms"`
	HighP95LatMs          float64 `json:"high_p95_lat_ms"`
	HighSuccMedianLatMs   float64 `json:"high_succ_median_lat_ms"`
	HighSuccP95LatMs      float64 `json:"high_succ_p95_lat_ms"`
	NormAccepted          int     `json:"norm_accepted"`
	NormSuccessful        int     `json:"norm_successful"`
	NormSuccessRate       float64 `json:"norm_success_rate"`
	NormMedianLatMs       float64 `json:"norm_median_lat_ms"`
	NormP95LatMs          float64 `json:"norm_p95_lat_ms"`
	NormSuccMedianLatMs   float64 `json:"norm_succ_median_lat_ms"`
	NormSuccP95LatMs      float64 `json:"norm_succ_p95_lat_ms"`
	MaxNormalWaitMs       float64 `json:"max_normal_wait_ms"`
	BurstRecoveryMs       float64 `json:"burst_recovery_ms"`
	GoroutineDelta        int     `json:"goroutine_delta"`
}

// SequentialReference processes valid records sequentially in arrival order.
// If allowedIDs is non-nil, only events whose EventID is in allowedIDs are aggregated.
func SequentialReference(records []events.Record, allowedIDs map[uint64]bool, simulateWork bool, baseCost time.Duration) (AggregateResult, time.Duration) {
	t0 := time.Now()
	res := AggregateResult{
		IPFailedCounts:   make(map[string]int),
		FileAccessCounts: make(map[string]int),
		OtherEventCounts: make(map[string]int),
	}

	for _, r := range records {
		if !r.Valid {
			res.MalformedCount++
			continue
		}
		if allowedIDs != nil && !allowedIDs[r.EventID] {
			continue
		}
		if simulateWork && baseCost > 0 {
			busyOrSleep(baseCost)
		}
		applyAggregate(&res, r)
	}

	res.FlaggedIPs = extractFlaggedIPs(res.IPFailedCounts)
	return res, time.Since(t0)
}

func applyAggregate(res *AggregateResult, r events.Record) {
	switch r.Class {
	case events.ClassLoginFailed:
		res.IPFailedCounts[r.IP]++
	case events.ClassFileAccess:
		res.FileAccessCounts[r.Resource]++
	default:
		res.OtherEventCounts[r.EventType]++
	}
}

func extractFlaggedIPs(counts map[string]int) []string {
	var flagged []string
	for ip, c := range counts {
		if c >= 5 {
			flagged = append(flagged, ip)
		}
	}
	sort.Strings(flagged)
	return flagged
}

// busyOrSleep simulates deterministic processing cost using time.Sleep for longer
// costs or calibrated work for sub-100us batch costs so batch timing is accurate.
func busyOrSleep(d time.Duration) {
	if d <= 0 {
		return
	}
	if d >= 50*time.Microsecond {
		time.Sleep(d)
		return
	}
	target := time.Now().Add(d)
	for time.Now().Before(target) {
	}
}

type queuedTask struct {
	item         events.ReplayItem
	arrivalTime  time.Time
	enqueuedTime time.Time
}

// scheduler implements O(1) FIFO or Separate Priority Queues with optional Aging/Weighted fairness.
type scheduler struct {
	mu              sync.Mutex
	cond            *sync.Cond
	policy          Policy
	capacity        int
	agingThresh     time.Duration
	maxConsecHigh   int
	closed          bool
	fifoQ           []queuedTask
	highQ           []queuedTask
	normQ           []queuedTask
	consecHighCount int
	peakLen         int
}

func newScheduler(cfg Config) *scheduler {
	s := &scheduler{
		policy:        cfg.Policy,
		capacity:      cfg.QueueCapacity,
		agingThresh:   time.Duration(cfg.AgingThresholdMs * float64(time.Millisecond)),
		maxConsecHigh: cfg.MaxConsecutiveHigh,
	}
	s.cond = sync.NewCond(&s.mu)
	return s
}

func (s *scheduler) currentLenLocked() int {
	if s.policy == PolicyFIFO {
		return len(s.fifoQ)
	}
	return len(s.highQ) + len(s.normQ)
}

func (s *scheduler) CurrentLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.currentLenLocked()
}

func (s *scheduler) PeakLen() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.peakLen
}

func (s *scheduler) Enqueue(t queuedTask) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	cur := s.currentLenLocked()
	if s.capacity > 0 && cur >= s.capacity {
		return false
	}
	if s.policy == PolicyFIFO {
		s.fifoQ = append(s.fifoQ, t)
	} else if t.item.Record.Priority == events.PriorityHigh {
		s.highQ = append(s.highQ, t)
	} else {
		s.normQ = append(s.normQ, t)
	}
	if cur+1 > s.peakLen {
		s.peakLen = cur + 1
	}
	s.cond.Signal()
	return true
}

func (s *scheduler) Dequeue() (queuedTask, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.currentLenLocked() == 0 && !s.closed {
		s.cond.Wait()
	}
	if s.currentLenLocked() == 0 && s.closed {
		return queuedTask{}, false
	}

	if s.policy == PolicyFIFO {
		t := s.fifoQ[0]
		s.fifoQ = s.fifoQ[1:]
		return t, true
	}

	// Priority-based policies (separate FIFO queues per priority class)
	hasHigh := len(s.highQ) > 0
	hasNorm := len(s.normQ) > 0

	useFairness := s.policy == PolicyFairnessOnly || s.policy == PolicyModified
	if hasHigh && hasNorm && useFairness {
		promoteNorm := false
		if s.maxConsecHigh > 0 && s.consecHighCount >= s.maxConsecHigh {
			promoteNorm = true
		} else if s.agingThresh > 0 && time.Since(s.normQ[0].enqueuedTime) >= s.agingThresh {
			promoteNorm = true
		}
		if promoteNorm {
			t := s.normQ[0]
			s.normQ = s.normQ[1:]
			s.consecHighCount = 0
			return t, true
		}
	}

	if hasHigh {
		t := s.highQ[0]
		s.highQ = s.highQ[1:]
		s.consecHighCount++
		return t, true
	}

	t := s.normQ[0]
	s.normQ = s.normQ[1:]
	s.consecHighCount = 0
	return t, true
}

func (s *scheduler) Close() {
	s.mu.Lock()
	s.closed = true
	s.cond.Broadcast()
	s.mu.Unlock()
}

type eventOutcome struct {
	eventID  uint64
	priority events.Priority
	status   TerminalStatus
	accepted bool
	qWaitMs  float64
	latMs    float64
}

// ledger enforces exactly-once terminal outcome and aggregate updates.
type ledger struct {
	mu          sync.Mutex
	agg         AggregateResult
	committed   map[uint64]TerminalStatus
	outcomes    []eventOutcome
	dupAttempts int
}

func newLedger(expected int) *ledger {
	return &ledger{
		agg: AggregateResult{
			IPFailedCounts:   make(map[string]int),
			FileAccessCounts: make(map[string]int),
			OtherEventCounts: make(map[string]int),
		},
		committed: make(map[uint64]TerminalStatus, expected),
		outcomes:  make([]eventOutcome, 0, expected),
	}
}

func (l *ledger) Commit(rec events.Record, status TerminalStatus, accepted bool, qWait, lat time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, exists := l.committed[rec.EventID]; exists {
		l.dupAttempts++
		return
	}
	l.committed[rec.EventID] = status
	if status == StatusSuccessful {
		applyAggregate(&l.agg, rec)
	}
	l.outcomes = append(l.outcomes, eventOutcome{
		eventID:  rec.EventID,
		priority: rec.Priority,
		status:   status,
		accepted: accepted,
		qWaitMs:  float64(qWait.Microseconds()) / 1000.0,
		latMs:    float64(lat.Microseconds()) / 1000.0,
	})
}

// RunBatch executes a fault-free batch workload (Part B/C) without pacing sleeps.
func RunBatch(records []events.Record, cfg Config, datasetName string, rep int, baseCost time.Duration) (AggregateResult, RunMetrics) {
	items := make([]events.ReplayItem, 0, len(records))
	malformed := 0
	for _, r := range records {
		if !r.Valid {
			malformed++
			continue
		}
		items = append(items, events.ReplayItem{
			Record:     r,
			BaseCostNs: int64(baseCost),
		})
	}
	return runTraceInternal(items, malformed, cfg, "batch", datasetName, rep, false, 0, context.Background())
}

// RunPaced executes a paced replay trace (Part C/D/F) with optional external cancellation context.
func RunPaced(items []events.ReplayItem, malformed int, cfg Config, scenario, datasetName string, rep int, burstEndOffsetNs int64, ctx context.Context) (AggregateResult, RunMetrics) {
	return runTraceInternal(items, malformed, cfg, scenario, datasetName, rep, true, burstEndOffsetNs, ctx)
}

func runTraceInternal(
	items []events.ReplayItem,
	malformed int,
	cfg Config,
	scenario, datasetName string,
	rep int,
	paced bool,
	burstEndOffsetNs int64,
	parentCtx context.Context,
) (AggregateResult, RunMetrics) {
	if parentCtx == nil {
		parentCtx = context.Background()
	}

	runtime.GC()
	var memBefore, memAfter runtime.MemStats
	runtime.ReadMemStats(&memBefore)
	baseGoroutines := runtime.NumGoroutine()

	sched := newScheduler(cfg)
	ldg := newLedger(len(items))
	ldg.agg.MalformedCount = malformed

	var totalAttempts atomic.Int64
	var retriesUsed atomic.Int64
	var peakHeapBytes atomic.Uint64
	peakHeapBytes.Store(memBefore.HeapAlloc)

	useResilience := cfg.Policy == PolicyResilienceOnly || cfg.Policy == PolicyModified
	taskTimeout := time.Duration(cfg.TaskTimeoutMs * float64(time.Millisecond))
	maxRetries := cfg.MaxRetries
	if !useResilience {
		taskTimeout = 0
		maxRetries = 0
	}

	var wg sync.WaitGroup
	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				task, ok := sched.Dequeue()
				if !ok {
					return
				}
				if parentCtx.Err() != nil {
					now := time.Now()
					ldg.Commit(task.item.Record, StatusCancelled, true, now.Sub(task.enqueuedTime), now.Sub(task.arrivalTime))
					continue
				}

				qWait := time.Since(task.enqueuedTime)
				var tCtx context.Context
				var tCancel context.CancelFunc
				if taskTimeout > 0 {
					tCtx, tCancel = context.WithTimeout(parentCtx, taskTimeout)
				} else {
					tCtx, tCancel = context.WithCancel(parentCtx)
				}

				status := executeTask(tCtx, task.item, maxRetries, &totalAttempts, &retriesUsed)
				tCancel()
				lat := time.Since(task.arrivalTime)
				ldg.Commit(task.item.Record, status, true, qWait, lat)
			}
		}()
	}

	// Track recovery after D1 burst and peak heap memory
	recThresh := cfg.RecoveryThreshold
	if recThresh <= 0 {
		recThresh = 5
	}
	var burstEndTime time.Time
	var recoveredTime time.Time
	var burstEnded atomic.Bool
	var recovered atomic.Bool

	stopMonitor := make(chan struct{})
	var monWg sync.WaitGroup
	monWg.Add(1)
	go func() {
		defer monWg.Done()
		tk := time.NewTicker(5 * time.Millisecond)
		defer tk.Stop()
		for {
			select {
			case <-stopMonitor:
				return
			case now := <-tk.C:
				var ms runtime.MemStats
				runtime.ReadMemStats(&ms)
				for {
					cur := peakHeapBytes.Load()
					if ms.HeapAlloc <= cur || peakHeapBytes.CompareAndSwap(cur, ms.HeapAlloc) {
						break
					}
				}
				if burstEnded.Load() && !recovered.Load() {
					if sched.CurrentLen() <= recThresh {
						recoveredTime = now
						recovered.Store(true)
					}
				}
			}
		}
	}()

	start := time.Now()
	for _, it := range items {
		if paced && it.ArrivalOffsetNs > 0 {
			target := start.Add(time.Duration(it.ArrivalOffsetNs))
			if d := time.Until(target); d > 0 {
				t := time.NewTimer(d)
				select {
				case <-parentCtx.Done():
					t.Stop()
				case <-t.C:
				}
			}
		}
		if burstEndOffsetNs > 0 && !burstEnded.Load() && it.ArrivalOffsetNs >= burstEndOffsetNs {
			burstEndTime = time.Now()
			burstEnded.Store(true)
		}

		now := time.Now()
		if parentCtx.Err() != nil {
			ldg.Commit(it.Record, StatusCancelled, false, 0, 0)
			continue
		}
		if !sched.Enqueue(queuedTask{item: it, arrivalTime: now, enqueuedTime: now}) {
			ldg.Commit(it.Record, StatusRejected, false, 0, time.Since(now))
		}
	}

	if burstEndOffsetNs > 0 && !burstEnded.Load() {
		burstEndTime = time.Now()
		burstEnded.Store(true)
	}

	sched.Close()
	wg.Wait()
	elapsed := time.Since(start)
	if burstEnded.Load() && !recovered.Load() {
		recoveredTime = time.Now()
		recovered.Store(true)
	}
	close(stopMonitor)
	monWg.Wait()

	runtime.ReadMemStats(&memAfter)
	if memAfter.HeapAlloc > peakHeapBytes.Load() {
		peakHeapBytes.Store(memAfter.HeapAlloc)
	}
	goroutineDelta := runtime.NumGoroutine() - baseGoroutines
	if goroutineDelta < 0 {
		goroutineDelta = 0
	}

	ldg.agg.FlaggedIPs = extractFlaggedIPs(ldg.agg.IPFailedCounts)

	// Verify exact match against sequential replay of successful EventIDs
	succIDs := make(map[uint64]bool, len(items))
	var validRecords []events.Record
	for _, it := range items {
		validRecords = append(validRecords, it.Record)
	}
	for id, st := range ldg.committed {
		if st == StatusSuccessful {
			succIDs[id] = true
		}
	}
	seqCheck, _ := SequentialReference(validRecords, succIDs, false, 0)
	matchesSeq := reflect.DeepEqual(ldg.agg.IPFailedCounts, seqCheck.IPFailedCounts) &&
		reflect.DeepEqual(ldg.agg.FlaggedIPs, seqCheck.FlaggedIPs) &&
		reflect.DeepEqual(ldg.agg.FileAccessCounts, seqCheck.FileAccessCounts) &&
		reflect.DeepEqual(ldg.agg.OtherEventCounts, seqCheck.OtherEventCounts)

	metrics := buildMetrics(
		scenario, cfg, datasetName, rep, len(items), malformed,
		ldg, int(totalAttempts.Load()), int(retriesUsed.Load()),
		sched.PeakLen(), peakHeapBytes.Load(), elapsed,
		burstEndTime, recoveredTime, burstEnded.Load(),
		matchesSeq, goroutineDelta,
	)

	return ldg.agg, metrics
}

func executeTask(
	ctx context.Context,
	item events.ReplayItem,
	maxRetries int,
	totalAttempts, retriesUsed *atomic.Int64,
) TerminalStatus {
	attempts := 1 + maxRetries
	cost := time.Duration(item.BaseCostNs)
	for att := 1; att <= attempts; att++ {
		if ctx.Err() != nil {
			if errorsIsDeadline(ctx) {
				return StatusTimedOut
			}
			return StatusCancelled
		}
		totalAttempts.Add(1)
		if att > 1 {
			retriesUsed.Add(1)
		}

		// Simulate processing duration cooperatively with context cancellation/timeout
		if cost >= 50*time.Microsecond {
			t := time.NewTimer(cost)
			select {
			case <-ctx.Done():
				if !t.Stop() {
					select {
					case <-t.C:
					default:
					}
				}
				if errorsIsDeadline(ctx) {
					return StatusTimedOut
				}
				return StatusCancelled
			case <-t.C:
			}
		} else {
			busyOrSleep(cost)
			if ctx.Err() != nil {
				if errorsIsDeadline(ctx) {
					return StatusTimedOut
				}
				return StatusCancelled
			}
		}

		// Deterministic first-attempt transient failure (D3)
		if att == 1 && item.FailFirstAttempt {
			// Failed attempt commits NO side effects; retry if budget allows
			continue
		}
		return StatusSuccessful
	}
	return StatusFailed
}

func errorsIsDeadline(ctx context.Context) bool {
	return ctx.Err() == context.DeadlineExceeded
}

func buildMetrics(
	scenario string,
	cfg Config,
	datasetName string,
	rep int,
	validGen int,
	malformed int,
	ldg *ledger,
	totalAttempts, retriesUsed int,
	peakQ int,
	peakHeap uint64,
	elapsed time.Duration,
	burstEnd, recTime time.Time,
	hasBurst bool,
	matchesSeq bool,
	goroutineDelta int,
) RunMetrics {
	m := RunMetrics{
		Scenario:         scenario,
		Policy:           string(cfg.Policy),
		Dataset:          datasetName,
		Repetition:       rep,
		Workers:          cfg.Workers,
		QueueCapacity:    cfg.QueueCapacity,
		Generated:        validGen + malformed,
		Malformed:        malformed,
		TotalAttempts:    totalAttempts,
		RetriesUsed:      retriesUsed,
		DuplicateOutcomes: ldg.dupAttempts,
		PeakQueueLen:     peakQ,
		PeakHeapMB:       float64(peakHeap) / (1024.0 * 1024.0),
		ElapsedSec:       elapsed.Seconds(),
		FlaggedIPCount:   len(ldg.agg.FlaggedIPs),
		MatchesSeqReplay: matchesSeq,
		GoroutineDelta:   goroutineDelta,
	}

	var highAcc, highSucc, normAcc, normSucc []float64
	var maxNormWait float64

	for _, o := range ldg.outcomes {
		switch o.status {
		case StatusSuccessful:
			m.Successful++
		case StatusFailed:
			m.Failed++
		case StatusTimedOut:
			m.TimedOut++
		case StatusCancelled:
			m.Cancelled++
		case StatusRejected:
			m.Rejected++
		}
		if o.priority == events.PriorityHigh {
			if o.accepted {
				m.HighAccepted++
				highAcc = append(highAcc, o.latMs)
				if o.status == StatusSuccessful {
					m.HighSuccessful++
					highSucc = append(highSucc, o.latMs)
				}
			}
		} else {
			if o.accepted {
				m.NormAccepted++
				normAcc = append(normAcc, o.latMs)
				if o.qWaitMs > maxNormWait {
					maxNormWait = o.qWaitMs
				}
				if o.status == StatusSuccessful {
					m.NormSuccessful++
					normSucc = append(normSucc, o.latMs)
				}
			}
		}
	}

	m.MissingIDs = validGen - len(ldg.committed)
	m.AccountingValid = (m.Generated == m.Malformed+m.Successful+m.Failed+m.TimedOut+m.Cancelled+m.Rejected) &&
		m.MissingIDs == 0 && m.DuplicateOutcomes == 0

	if elapsed.Seconds() > 0 {
		m.ThroughputRPS = float64(m.Successful) / elapsed.Seconds()
	}
	if m.HighAccepted > 0 {
		m.HighSuccessRate = float64(m.HighSuccessful) / float64(m.HighAccepted)
		m.HighMedianLatMs, m.HighP95LatMs = medianAndP95(highAcc)
	}
	if len(highSucc) > 0 {
		m.HighSuccMedianLatMs, m.HighSuccP95LatMs = medianAndP95(highSucc)
	}
	if m.NormAccepted > 0 {
		m.NormSuccessRate = float64(m.NormSuccessful) / float64(m.NormAccepted)
		m.NormMedianLatMs, m.NormP95LatMs = medianAndP95(normAcc)
	}
	if len(normSucc) > 0 {
		m.NormSuccMedianLatMs, m.NormSuccP95LatMs = medianAndP95(normSucc)
	}
	m.MaxNormalWaitMs = maxNormWait

	if hasBurst && !recTime.IsZero() && !burstEnd.IsZero() {
		d := recTime.Sub(burstEnd)
		if d < 0 {
			d = 0
		}
		m.BurstRecoveryMs = float64(d.Microseconds()) / 1000.0
	}

	return m
}

func medianAndP95(vals []float64) (float64, float64) {
	if len(vals) == 0 {
		return 0, 0
	}
	cp := make([]float64, len(vals))
	copy(cp, vals)
	sort.Float64s(cp)
	med := cp[len(cp)/2]
	idx95 := int(math.Ceil(0.95*float64(len(cp)))) - 1
	if idx95 < 0 {
		idx95 = 0
	}
	if idx95 >= len(cp) {
		idx95 = len(cp) - 1
	}
	return med, cp[idx95]
}

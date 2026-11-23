package workload

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"
)

// RequestType classifies incoming university student service requests.
type RequestType string

const (
	ReqCertificate  RequestType = "Certificate"
	ReqVerification RequestType = "Verification"
	ReqApproval     RequestType = "Approval"
)

// Outcome defines the mutually exclusive set of terminal request outcomes.
type Outcome string

const (
	OutcomeSuccess   Outcome = "Success"
	OutcomeFailure   Outcome = "Failure"
	OutcomeTimeout   Outcome = "Timeout"
	OutcomeCancelled Outcome = "Cancelled"
	OutcomeRejected  Outcome = "Rejected"
)

// FaultConfig isolates fault-injection parameters from nominal service parameters.
type FaultConfig struct {
	SlowRatio            float64 `json:"slow_ratio"`
	SlowDurationMs       int     `json:"slow_duration_ms"`
	FailureRatio         float64 `json:"failure_ratio"`
	DuplicateSubmitRatio float64 `json:"duplicate_submit_ratio"`
	CancelAfterMs        int     `json:"cancel_after_ms"`
	ShutdownAfterMs      int     `json:"shutdown_after_ms"`
	AckDelayRatio        float64 `json:"ack_delay_ratio"`
	AckDelayMs           int     `json:"ack_delay_ms"`
	AckDuplicateRatio    float64 `json:"ack_duplicate_ratio"`
	AckDropRatio         float64 `json:"ack_drop_ratio"`
}

// Config captures complete reproducible experiment configuration.
type Config struct {
	Condition        string      `json:"condition"`
	Description      string      `json:"description"`
	TotalRequests    int         `json:"total_requests"`
	NumWorkers       int         `json:"num_workers"`
	QueueCapacity    int         `json:"queue_capacity"`
	ArrivalRateRPS   float64     `json:"arrival_rate_rps"`
	MinServiceMs     int         `json:"min_service_ms"`
	MaxServiceMs     int         `json:"max_service_ms"`
	RequestTimeoutMs int         `json:"request_timeout_ms"`
	AckTimeoutMs     int         `json:"ack_timeout_ms"`
	MaxAckRetries    int         `json:"max_ack_retries"`
	RetryBudgetRatio float64     `json:"retry_budget_ratio"`
	ShutdownGraceMs  int         `json:"shutdown_grace_ms"`
	Seeds            []int64     `json:"seeds"`
	Faults           FaultConfig `json:"faults"`
}

// LoadConfig reads a JSON configuration file from disk.
func LoadConfig(path string) (Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// Item represents one deterministic request delivery event in a workload stream.
type Item struct {
	ID                uint64
	SubmissionSeq     int // 1 = original, 2 = injected duplicate delivery
	IsDuplicateSubmit bool
	Type              RequestType
	ServiceDuration   time.Duration
	IsSlow            bool
	IsFailing         bool
	InterArrival      time.Duration
	AckDelayed        bool
	AckDelayDuration  time.Duration
	AckDuplicates     int // number of extra repeated downstream acks (0, 1, or 2)
	AckDropped        bool
}

// LifecycleEvent is emitted as structured JSONL for full post-hoc observability.
type LifecycleEvent struct {
	Seq           uint64  `json:"seq"`
	TimestampNs   int64   `json:"ts_ns"`
	ElapsedMs     float64 `json:"elapsed_ms"`
	RequestID     uint64  `json:"request_id"`
	SubmissionSeq int     `json:"submission_seq"`
	Event         string  `json:"event"`
	Outcome       Outcome `json:"outcome,omitempty"`
	Reason        string  `json:"reason,omitempty"`
	WorkerID      int     `json:"worker_id,omitempty"`
	ActiveWorkers int64   `json:"active_workers"`
	QueueDepth    int64   `json:"queue_depth"`
}

// TerminalRecord represents one recorded terminal outcome entry.
type TerminalRecord struct {
	RequestID     uint64
	SubmissionSeq int
	Type          RequestType
	Outcome       Outcome
	Reason        string
	WorkerID      int
	ArrivalTime   time.Time
	EnqueuedTime  time.Time
	StartedTime   time.Time
	EndTime       time.Time
	QueueWait     time.Duration
	ServiceTime   time.Duration
	ResponseTime  time.Duration
	AckAttempts   int
	Admitted      bool
}

// RunSummary captures all Part B and Part E metrics plus invariant verification checks.
type RunSummary struct {
	Architecture                string  `json:"architecture"`
	Condition                   string  `json:"condition"`
	Seed                        int64   `json:"seed"`
	Repetition                  int     `json:"repetition"`
	UniqueGenerated             int     `json:"unique_generated"`
	TotalSubmissions            int     `json:"total_submissions"`
	DuplicateSubmissions        int     `json:"duplicate_submissions"`
	DuplicateSubmissionsBlocked int     `json:"duplicate_submissions_blocked"`
	TotalTerminalRecords        int     `json:"total_terminal_records"`
	UniqueTerminalIDs           int     `json:"unique_terminal_ids"`
	DuplicateTerminalOutcomes   int     `json:"duplicate_terminal_outcomes"`
	MissingTerminalIDs          int     `json:"missing_terminal_ids"`
	TerminalAccountingViolations int    `json:"terminal_accounting_violations"`
	LifecycleOrderViolations    int     `json:"lifecycle_order_violations"`
	Successful                  int     `json:"successful"`
	Failed                      int     `json:"failed"`
	TimedOut                    int     `json:"timed_out"`
	Cancelled                   int     `json:"cancelled"`
	Rejected                    int     `json:"rejected"`
	AckRetriesUsed              int     `json:"ack_retries_used"`
	DuplicateAcksIgnored        int     `json:"duplicate_acks_ignored"`
	MaxQueueDepth               int64   `json:"max_queue_depth"`
	PeakActiveWorkers           int64   `json:"peak_active_workers"`
	WallDurationSec             float64 `json:"wall_duration_sec"`
	ThroughputRPS               float64 `json:"throughput_rps"`
	GoodputRPS                  float64 `json:"goodput_rps"`
	MeanQueueWaitMs             float64 `json:"mean_queue_wait_ms"`
	MeanResponseMs              float64 `json:"mean_response_ms"`
	P50ResponseMs               float64 `json:"p50_response_ms"`
	P95ResponseMs               float64 `json:"p95_response_ms"`
	P99ResponseMs               float64 `json:"p99_response_ms"`
	MaxResponseMs               float64 `json:"max_response_ms"`
	OverallMeanResponseMs       float64 `json:"overall_mean_response_ms"`
	ShutdownCompletionMs        float64 `json:"shutdown_completion_ms"`
	OutstandingWorkAfterShutdown int    `json:"outstanding_work_after_shutdown"`
	GoroutineDelta              int     `json:"goroutine_delta"`
	InvariantsPassed            bool    `json:"invariants_passed"`
}

// Generate creates a completely deterministic sequence of workload items for a given seed.
// Both Baseline and Improved services consume this exact same stream for apples-to-apples comparison.
func Generate(cfg Config, seed int64) []Item {
	rng := rand.New(rand.NewSource(seed))
	types := []RequestType{ReqCertificate, ReqVerification, ReqApproval}

	var baseInterArrival time.Duration
	if cfg.ArrivalRateRPS > 0 {
		baseInterArrival = time.Duration(float64(time.Second) / cfg.ArrivalRateRPS)
	}

	items := make([]Item, 0, int(float64(cfg.TotalRequests)*1.15))
	minMs := cfg.MinServiceMs
	maxMs := cfg.MaxServiceMs
	if maxMs < minMs {
		maxMs = minMs
	}
	spanUs := int64(maxMs-minMs) * 1000

	for i := 1; i <= cfg.TotalRequests; i++ {
		reqType := types[rng.Intn(len(types))]
		isSlow := cfg.Faults.SlowRatio > 0 && rng.Float64() < cfg.Faults.SlowRatio
		isFailing := cfg.Faults.FailureRatio > 0 && rng.Float64() < cfg.Faults.FailureRatio

		var svcDur time.Duration
		if isSlow {
			svcDur = time.Duration(cfg.Faults.SlowDurationMs) * time.Millisecond
		} else if spanUs > 0 {
			svcDur = time.Duration(minMs)*time.Millisecond + time.Duration(rng.Int63n(spanUs))*time.Microsecond
		} else {
			svcDur = time.Duration(minMs) * time.Millisecond
		}

		var inter time.Duration
		if baseInterArrival > 0 {
			// Deterministic jitter in [0.8x, 1.2x] of nominal inter-arrival
			inter = time.Duration(float64(baseInterArrival) * (0.8 + 0.4*rng.Float64()))
		}

		ackDelayed := cfg.Faults.AckDelayRatio > 0 && rng.Float64() < cfg.Faults.AckDelayRatio
		ackDelayDur := time.Duration(cfg.Faults.AckDelayMs) * time.Millisecond
		ackDups := 0
		if cfg.Faults.AckDuplicateRatio > 0 && rng.Float64() < cfg.Faults.AckDuplicateRatio {
			ackDups = 1 + rng.Intn(2) // 1 or 2 extra duplicate acks
		}
		ackDropped := cfg.Faults.AckDropRatio > 0 && rng.Float64() < cfg.Faults.AckDropRatio

		item := Item{
			ID:                uint64(i),
			SubmissionSeq:     1,
			IsDuplicateSubmit: false,
			Type:              reqType,
			ServiceDuration:   svcDur,
			IsSlow:            isSlow,
			IsFailing:         isFailing,
			InterArrival:      inter,
			AckDelayed:        ackDelayed,
			AckDelayDuration:  ackDelayDur,
			AckDuplicates:     ackDups,
			AckDropped:        ackDropped,
		}
		items = append(items, item)

		// Inject deterministic duplicate delivery of a recent request ID shortly after original
		if i > 5 && cfg.Faults.DuplicateSubmitRatio > 0 && rng.Float64() < cfg.Faults.DuplicateSubmitRatio {
			dupOffset := 1 + rng.Intn(5)
			dupID := uint64(i - dupOffset)
			dupItem := item
			dupItem.ID = dupID
			dupItem.SubmissionSeq = 2
			dupItem.IsDuplicateSubmit = true
			dupItem.InterArrival = 100 * time.Microsecond
			dupItem.AckDelayed = false
			dupItem.AckDuplicates = 0
			dupItem.AckDropped = false
			items = append(items, dupItem)
		}
	}
	return items
}

// NearestRankPercentile computes the exact nearest-rank p-th percentile (0 < p <= 100).
// Formula: index = ceil((p/100) * N) - 1 on sorted values.
func NearestRankPercentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[n-1]
	}
	idx := int(math.Ceil((p/100.0)*float64(n))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	return sorted[idx]
}

// ComputeLatencies computes mean, p50, p95, p99, max in milliseconds.
func ComputeLatencies(vals []float64) (mean, p50, p95, p99, maxVal float64) {
	n := len(vals)
	if n == 0 {
		return 0, 0, 0, 0, 0
	}
	cp := make([]float64, n)
	copy(cp, vals)
	sort.Float64s(cp)
	var sum float64
	for _, v := range cp {
		sum += v
	}
	mean = sum / float64(n)
	p50 = NearestRankPercentile(cp, 50)
	p95 = NearestRankPercentile(cp, 95)
	p99 = NearestRankPercentile(cp, 99)
	maxVal = cp[n-1]
	return
}

// VerifyLifecycleOrder validates causal timestamp monotonicity per RequestID.
// Returns the number of requests with any lifecycle ordering violation.
func VerifyLifecycleOrder(events []LifecycleEvent) int {
	type stamp struct {
		hasGen   bool
		genSeq   uint64
		hasAdm   bool
		admSeq   uint64
		hasRej   bool
		rejSeq   uint64
		hasStart bool
		startSeq uint64
		hasTerm  bool
		termSeq  uint64
		bad      bool
	}
	byID := make(map[uint64]*stamp)
	for _, ev := range events {
		if ev.SubmissionSeq > 1 {
			continue // duplicate delivery events tracked separately
		}
		st, ok := byID[ev.RequestID]
		if !ok {
			st = &stamp{}
			byID[ev.RequestID] = st
		}
		switch ev.Event {
		case "generated":
			if !st.hasGen {
				st.hasGen = true
				st.genSeq = ev.Seq
			}
		case "admitted":
			if !st.hasGen || ev.Seq < st.genSeq {
				st.bad = true
			}
			st.hasAdm = true
			st.admSeq = ev.Seq
		case "rejected":
			if !st.hasGen || ev.Seq < st.genSeq {
				st.bad = true
			}
			st.hasRej = true
			st.rejSeq = ev.Seq
		case "started":
			if !st.hasAdm || ev.Seq < st.admSeq {
				st.bad = true
			}
			st.hasStart = true
			st.startSeq = ev.Seq
		case "terminal":
			if !st.hasGen {
				st.bad = true
			}
			if st.hasStart && ev.Seq < st.startSeq {
				st.bad = true
			}
			if st.hasAdm && ev.Seq < st.admSeq {
				st.bad = true
			}
			st.hasTerm = true
			st.termSeq = ev.Seq
		}
	}
	violations := 0
	for _, st := range byID {
		if st.bad || !st.hasTerm {
			violations++
		}
	}
	return violations
}

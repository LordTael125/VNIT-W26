package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// RequestType represents the academic request classification
type RequestType string

const (
	ReqCertificate     RequestType = "Certificate"
	ReqVerification   RequestType = "Verification"
	ReqAdministrative  RequestType = "Approval"
)

// RequestStatus represents the lifecycle state of a request
type RequestStatus string

const (
	StatusAccepted   RequestStatus = "Accepted"
	StatusProcessing RequestStatus = "Processing"
	StatusCompleted  RequestStatus = "Completed"
	StatusFailed      RequestStatus = "Failed"
	StatusTimedOut   RequestStatus = "TimedOut"
	StatusCancelled  RequestStatus = "Cancelled"
	StatusRejected   RequestStatus = "Rejected"
)

// Request encapsulates all attributes mandated by Practical 7
type Request struct {
	ID                uint64
	Type              RequestType
	ArrivalTime       time.Time
	EstimatedDuration time.Duration
	Status            RequestStatus
	IsSlow            bool
	IsFailing         bool
	EnqueuedTime      time.Time
	DequeuedTime      time.Time
	StartTime         time.Time
	EndTime           time.Time
	QueueWaitDuration time.Duration
	ProcessDuration   time.Duration
	ResponseTime      time.Duration
	WorkerID          int
	ErrMessage        string
}

// ServiceConfig defines parameters for service deployment
type ServiceConfig struct {
	Name            string
	NumWorkers      int
	QueueCapacity   int           // 0 indicates unbounded queue mode
	Timeout         time.Duration // 0 indicates no processing timeout
	IsResilient     bool
	RandomSeed      int64
	TotalRequests   int
	TargetArrivalRate float64     // requests per second
	SlowRatio       float64       // fraction of slow requests (0.0 to 1.0)
	SlowDuration    time.Duration
	FailureRatio    float64       // fraction of failing requests (0.0 to 1.0)
	MinServiceTime  time.Duration
	MaxServiceTime  time.Duration
}

// QueueMetrics tracks atomic queue depth
type QueueMetrics struct {
	currentDepth int64
	maxDepth     int64
}

func (q *QueueMetrics) Inc() int64 {
	val := atomic.AddInt64(&q.currentDepth, 1)
	for {
		curMax := atomic.LoadInt64(&q.maxDepth)
		if val <= curMax || atomic.CompareAndSwapInt64(&q.maxDepth, curMax, val) {
			break
		}
	}
	return val
}

func (q *QueueMetrics) Dec() int64 {
	return atomic.AddInt64(&q.currentDepth, -1)
}

func (q *QueueMetrics) Max() int64 {
	return atomic.LoadInt64(&q.maxDepth)
}

func (q *QueueMetrics) Current() int64 {
	return atomic.LoadInt64(&q.currentDepth)
}

// ResultRecord holds consolidated metrics from a simulation run
type ResultRecord struct {
	ScenarioName      string
	Architecture      string
	TotalGenerated    int
	Accepted          int
	Rejected          int
	Successful        int
	Failed            int
	TimedOut          int
	Cancelled         int
	MaxQueueLength    int64
	Throughput        float64 // completed/s
	Goodput           float64 // successful/s
	MeanResponseTime  float64 // ms
	MedianResponseTime float64 // ms
	P90ResponseTime   float64 // ms
	P99ResponseTime   float64 // ms
	MaxResponseTime   float64 // ms
	MeanQueueWaitTime float64 // ms
	MeanServiceTime   float64 // ms
	TotalDuration     float64 // s
	AccountingValid   bool
}

// TimeSeriesSample records transient queue behaviour during recovery tests
type TimeSeriesSample struct {
	TimestampSec float64
	QueueDepth   int64
	LatencyMs    float64
	Phase        string
}

// ResultCollector manages thread-safe aggregation of finished requests
type ResultCollector struct {
	mu           sync.Mutex
	records      []*Request
	requestsSeen map[uint64]bool
	duplicateErr bool
}

func NewResultCollector(capacity int) *ResultCollector {
	return &ResultCollector{
		records:      make([]*Request, 0, capacity),
		requestsSeen: make(map[uint64]bool, capacity),
	}
}

func (rc *ResultCollector) Record(req *Request) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	if rc.requestsSeen[req.ID] {
		rc.duplicateErr = true
	}
	rc.requestsSeen[req.ID] = true
	rc.records = append(rc.records, req)
}

func (rc *ResultCollector) AllRecords() []*Request {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	copied := make([]*Request, len(rc.records))
	copy(copied, rc.records)
	return copied
}

// ServiceSimulation orchestrates the 3-stage service execution
type ServiceSimulation struct {
	cfg       ServiceConfig
	queueM    QueueMetrics
	collector *ResultCollector
	workQueue chan *Request
	wgWorkers sync.WaitGroup
	rng       *rand.Rand
	samples   []TimeSeriesSample
	sampleMu  sync.Mutex
}

func NewSimulation(cfg ServiceConfig) *ServiceSimulation {
	sim := &ServiceSimulation{
		cfg:       cfg,
		collector: NewResultCollector(cfg.TotalRequests),
		rng:       rand.New(rand.NewSource(cfg.RandomSeed)),
	}

	if cfg.QueueCapacity > 0 {
		sim.workQueue = make(chan *Request, cfg.QueueCapacity)
	} else {
		// Unbounded simulation queue: buffer adequately large for experiment
		sim.workQueue = make(chan *Request, cfg.TotalRequests*2)
	}
	return sim
}

// StartWorkers launches the configured worker pool
func (s *ServiceSimulation) StartWorkers() {
	for w := 1; w <= s.cfg.NumWorkers; w++ {
		s.wgWorkers.Add(1)
		go s.workerRoutine(w)
	}
}

func (s *ServiceSimulation) workerRoutine(workerID int) {
	defer s.wgWorkers.Done()

	for req := range s.workQueue {
		s.queueM.Dec()
		req.DequeuedTime = time.Now()
		req.QueueWaitDuration = req.DequeuedTime.Sub(req.EnqueuedTime)
		req.StartTime = req.DequeuedTime
		req.WorkerID = workerID
		req.Status = StatusProcessing

		if !s.cfg.IsResilient || s.cfg.Timeout == 0 {
			// Baseline execution: no timeout, workers block on slow execution
			time.Sleep(req.EstimatedDuration)
			req.EndTime = time.Now()
			req.ProcessDuration = req.EndTime.Sub(req.StartTime)
			req.ResponseTime = req.EndTime.Sub(req.ArrivalTime)

			if req.IsFailing {
				req.Status = StatusFailed
				req.ErrMessage = "internal verification failed"
			} else {
				req.Status = StatusCompleted
			}
		} else {
			// Resilient execution: enforce context deadline and cancellation
			ctx, cancel := context.WithTimeout(context.Background(), s.cfg.Timeout)
			doneCh := make(chan struct{})

			go func(d time.Duration) {
				time.Sleep(d)
				close(doneCh)
			}(req.EstimatedDuration)

			select {
			case <-ctx.Done():
				// SLA Timeout expired! Fast release worker
				req.EndTime = time.Now()
				req.ProcessDuration = req.EndTime.Sub(req.StartTime)
				req.ResponseTime = req.EndTime.Sub(req.ArrivalTime)
				req.Status = StatusTimedOut
				req.ErrMessage = fmt.Sprintf("SLA timeout exceeded (%v)", s.cfg.Timeout)
			case <-doneCh:
				req.EndTime = time.Now()
				req.ProcessDuration = req.EndTime.Sub(req.StartTime)
				req.ResponseTime = req.EndTime.Sub(req.ArrivalTime)
				if req.IsFailing {
					req.Status = StatusFailed
					req.ErrMessage = "service error"
				} else {
					req.Status = StatusCompleted
				}
			}
			cancel()
		}

		s.collector.Record(req)
	}
}

// Ingest handles admission control at Request Intake
func (s *ServiceSimulation) Ingest(req *Request) {
	req.ArrivalTime = time.Now()

	if !s.cfg.IsResilient || s.cfg.QueueCapacity == 0 {
		// Baseline Admission: blindly accept everything into unbounded queue
		req.Status = StatusAccepted
		req.EnqueuedTime = time.Now()
		s.queueM.Inc()
		s.workQueue <- req
		return
	}

	// Resilient Admission Control: fast-reject if queue capacity is saturated
	select {
	case s.workQueue <- req:
		req.Status = StatusAccepted
		req.EnqueuedTime = time.Now()
		s.queueM.Inc()
	default:
		// Queue full! Fast reject with explicit rejection status
		req.Status = StatusRejected
		req.EndTime = time.Now()
		req.QueueWaitDuration = 0
		req.ProcessDuration = 0
		req.ResponseTime = req.EndTime.Sub(req.ArrivalTime)
		req.ErrMessage = "admission rejected: queue capacity saturated"
		s.collector.Record(req)
	}
}

// GenerateAndRun executes the configured workload
func (s *ServiceSimulation) GenerateAndRun() ResultRecord {
	startTime := time.Now()
	s.StartWorkers()

	// Inter-arrival calculation
	var interArrival time.Duration
	if s.cfg.TargetArrivalRate > 0 {
		interArrival = time.Duration(float64(time.Second) / s.cfg.TargetArrivalRate)
	} else {
		interArrival = 2 * time.Millisecond
	}

	// Request types pool
	types := []RequestType{ReqCertificate, ReqVerification, ReqAdministrative}

	// Start monitor goroutine for time-series logging
	stopMonitor := make(chan struct{})
	var monitorWg sync.WaitGroup
	monitorWg.Add(1)
	go func() {
		defer monitorWg.Done()
		ticker := time.NewTicker(50 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopMonitor:
				return
			case t := <-ticker.C:
				depth := s.queueM.Current()
				elapsed := t.Sub(startTime).Seconds()
				s.sampleMu.Lock()
				s.samples = append(s.samples, TimeSeriesSample{
					TimestampSec: elapsed,
					QueueDepth:   depth,
				})
				s.sampleMu.Unlock()
			}
		}
	}()

	// Dispatch requests
	for i := 1; i <= s.cfg.TotalRequests; i++ {
		reqType := types[s.rng.Intn(len(types))]

		isSlow := s.cfg.SlowRatio > 0 && s.rng.Float64() < s.cfg.SlowRatio
		isFailing := s.cfg.FailureRatio > 0 && s.rng.Float64() < s.cfg.FailureRatio

		var estDuration time.Duration
		if isSlow {
			estDuration = s.cfg.SlowDuration
		} else {
			span := int64(s.cfg.MaxServiceTime - s.cfg.MinServiceTime)
			if span <= 0 {
				estDuration = s.cfg.MinServiceTime
			} else {
				estDuration = s.cfg.MinServiceTime + time.Duration(s.rng.Int63n(span))
			}
		}

		req := &Request{
			ID:                uint64(i),
			Type:              reqType,
			EstimatedDuration: estDuration,
			IsSlow:            isSlow,
			IsFailing:         isFailing,
		}

		s.Ingest(req)

		// Controlled inter-arrival delay with small random jitter
		if interArrival > 0 {
			jitter := time.Duration(float64(interArrival) * (0.8 + 0.4*s.rng.Float64()))
			time.Sleep(jitter)
		}
	}

	// Close queue to signal workers no more requests are incoming
	close(s.workQueue)

	// Wait for workers to cleanly finish processing all accepted items
	s.wgWorkers.Wait()

	close(stopMonitor)
	monitorWg.Wait()

	totalDuration := time.Since(startTime)
	records := s.collector.AllRecords()

	return s.evaluateResults(records, totalDuration)
}

func (s *ServiceSimulation) evaluateResults(records []*Request, duration time.Duration) ResultRecord {
	rec := ResultRecord{
		ScenarioName:   s.cfg.Name,
		TotalGenerated: s.cfg.TotalRequests,
		MaxQueueLength: s.queueM.Max(),
		TotalDuration:  duration.Seconds(),
	}

	if s.cfg.IsResilient {
		rec.Architecture = "Improved"
	} else {
		rec.Architecture = "Baseline"
	}

	var totalResponseTime time.Duration
	var totalQueueTime time.Duration
	var totalServiceTime time.Duration
	var acceptedCount int
	var responseTimes []float64

	for _, r := range records {
		switch r.Status {
		case StatusCompleted:
			rec.Successful++
			rec.Accepted++
			acceptedCount++
		case StatusFailed:
			rec.Failed++
			rec.Accepted++
			acceptedCount++
		case StatusTimedOut:
			rec.TimedOut++
			rec.Accepted++
			acceptedCount++
		case StatusCancelled:
			rec.Cancelled++
			rec.Accepted++
			acceptedCount++
		case StatusRejected:
			rec.Rejected++
		}

		respMs := float64(r.ResponseTime.Nanoseconds()) / 1e6
		responseTimes = append(responseTimes, respMs)
		totalResponseTime += r.ResponseTime
		totalQueueTime += r.QueueWaitDuration
		totalServiceTime += r.ProcessDuration
	}

	n := len(records)
	if n > 0 {
		rec.MeanResponseTime = float64(totalResponseTime.Nanoseconds()) / float64(n) / 1e6
		sort.Float64s(responseTimes)
		rec.MedianResponseTime = responseTimes[n/2]
		rec.P90ResponseTime = responseTimes[int(float64(n)*0.90)]
		rec.P99ResponseTime = responseTimes[int(float64(n)*0.99)]
		rec.MaxResponseTime = responseTimes[n-1]
	}

	if acceptedCount > 0 {
		rec.MeanQueueWaitTime = float64(totalQueueTime.Nanoseconds()) / float64(acceptedCount) / 1e6
		rec.MeanServiceTime = float64(totalServiceTime.Nanoseconds()) / float64(acceptedCount) / 1e6
	}

	durSec := duration.Seconds()
	if durSec > 0 {
		// Throughput: completed tasks per second
		completedTotal := rec.Successful + rec.Failed + rec.TimedOut
		rec.Throughput = float64(completedTotal) / durSec
		// Goodput: successful tasks per second
		rec.Goodput = float64(rec.Successful) / durSec
	}

	// Exact accounting verification invariant
	sumOutcomes := rec.Successful + rec.Failed + rec.TimedOut + rec.Cancelled + rec.Rejected
	rec.AccountingValid = (sumOutcomes == s.cfg.TotalRequests) &&
		(rec.Accepted == s.cfg.TotalRequests-rec.Rejected) &&
		!s.collector.duplicateErr &&
		(len(records) == s.cfg.TotalRequests)

	return rec
}

// RunRecoveryExperiment executes a 3-phase surge-and-recovery test
func RunRecoveryExperiment(isResilient bool) ([]TimeSeriesSample, ResultRecord) {
	cfg := ServiceConfig{
		Name:            "RecoveryDynamics",
		NumWorkers:      4,
		QueueCapacity:   0,
		Timeout:         0,
		IsResilient:     isResilient,
		RandomSeed:      99,
		TotalRequests:   1200,
		MinServiceTime:  3 * time.Millisecond,
		MaxServiceTime:  5 * time.Millisecond,
	}

	if isResilient {
		cfg.QueueCapacity = 40
		cfg.Timeout = 25 * time.Millisecond
	}

	sim := NewSimulation(cfg)
	sim.StartWorkers()
	startTime := time.Now()

	// Phase 1: 0 to 400 requests -> Normal Rate (400 req/s => 2.5ms delay)
	// Phase 2: 401 to 800 requests -> Surge Rate (1600 req/s => 0.62ms delay)
	// Phase 3: 801 to 1200 requests -> Return to Normal Rate (400 req/s => 2.5ms delay)

	var samples []TimeSeriesSample
	var sampleMu sync.Mutex

	stopMon := make(chan struct{})
	var monWg sync.WaitGroup
	monWg.Add(1)
	go func() {
		defer monWg.Done()
		ticker := time.NewTicker(25 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopMon:
				return
			case t := <-ticker.C:
				d := sim.queueM.Current()
				elapsed := t.Sub(startTime).Seconds()
				phase := "Normal-1"
				if elapsed >= 1.0 && elapsed < 2.0 {
					phase = "Surge"
				} else if elapsed >= 2.0 {
					phase = "Post-Surge"
				}
				sampleMu.Lock()
				samples = append(samples, TimeSeriesSample{
					TimestampSec: elapsed,
					QueueDepth:   d,
					Phase:        phase,
				})
				sampleMu.Unlock()
			}
		}
	}()

	for i := 1; i <= cfg.TotalRequests; i++ {
		var delay time.Duration
		isSlow := false
		if i <= 350 {
			delay = 3 * time.Millisecond
		} else if i <= 750 {
			delay = 700 * time.Microsecond // surge
			if i%10 == 0 {
				isSlow = true // occasional slow request during surge
			}
		} else {
			delay = 3 * time.Millisecond // recovered arrival rate
		}

		dur := 4 * time.Millisecond
		if isSlow {
			dur = 50 * time.Millisecond
		}

		req := &Request{
			ID:                uint64(i),
			Type:              ReqVerification,
			EstimatedDuration: dur,
			IsSlow:            isSlow,
		}

		sim.Ingest(req)
		time.Sleep(delay)
	}

	close(sim.workQueue)
	sim.wgWorkers.Wait()
	close(stopMon)
	monWg.Wait()

	totalDur := time.Since(startTime)
	res := sim.evaluateResults(sim.collector.AllRecords(), totalDur)
	return samples, res
}

func main() {
	runAll := flag.Bool("all", true, "Run all benchmark experiments")
	csvOutput := flag.String("csv", "benchmark_results.csv", "Path to CSV output file")
	flag.Parse()

	fmt.Println("==========================================================================")
	fmt.Println("  CSP502 Practical 7: Resilient Concurrent Service Evaluation Engine     ")
	fmt.Println("  Students: Gaurav Acharya [MT26MCS024] & Sandeep [MT26MCS032]            ")
	fmt.Println("  Institution: Visvesvaraya National Institute of Technology (VNIT)       ")
	fmt.Println("==========================================================================")

	if !*runAll {
		fmt.Println("Use --all to execute standard test suite.")
		return
	}

	// Define benchmark experimental matrix
	scenarios := []ServiceConfig{
		// 1. Part B: Baseline Normal Operating Workload
		{
			Name:              "Baseline_Normal",
			NumWorkers:        4,
			QueueCapacity:     0, // unbounded
			Timeout:           0, // no timeout
			IsResilient:       false,
			RandomSeed:        42,
			TotalRequests:     1000,
			TargetArrivalRate: 350, // req/sec (well within capacity of 4 * 250 = 1000)
			SlowRatio:         0.0,
			SlowDuration:      0,
			FailureRatio:      0.0,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 2. Part C: Experiment C1 - Arrival Rate Surge (Overload on Baseline)
		{
			Name:              "Baseline_C1_Surge",
			NumWorkers:        4,
			QueueCapacity:     0, // unbounded
			Timeout:           0,
			IsResilient:       false,
			RandomSeed:        42,
			TotalRequests:     1200,
			TargetArrivalRate: 1400, // exceeds processing capacity ~1000 req/s
			SlowRatio:         0.0,
			SlowDuration:      0,
			FailureRatio:      0.0,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 3. Part C: Experiment C2 - Slow Processing (Head-of-Line Blocking on Baseline)
		{
			Name:              "Baseline_C2_SlowTasks",
			NumWorkers:        4,
			QueueCapacity:     0,
			Timeout:           0,
			IsResilient:       false,
			RandomSeed:        42,
			TotalRequests:     1000,
			TargetArrivalRate: 350,
			SlowRatio:         0.08, // 8% slow tasks
			SlowDuration:      60 * time.Millisecond, // 15x normal
			FailureRatio:      0.0,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 4. Part C: Experiment C3 - Processing Failures on Baseline
		{
			Name:              "Baseline_C3_Failures",
			NumWorkers:        4,
			QueueCapacity:     0,
			Timeout:           0,
			IsResilient:       false,
			RandomSeed:        42,
			TotalRequests:     1000,
			TargetArrivalRate: 350,
			SlowRatio:         0.0,
			SlowDuration:      0,
			FailureRatio:      0.15, // 15% fail
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 5. Part E & F: Improved Design under Normal Load
		{
			Name:              "Improved_Normal",
			NumWorkers:        4,
			QueueCapacity:     50, // bounded queue
			Timeout:           25 * time.Millisecond, // SLA deadline
			IsResilient:       true,
			RandomSeed:        42,
			TotalRequests:     1000,
			TargetArrivalRate: 350,
			SlowRatio:         0.0,
			SlowDuration:      0,
			FailureRatio:      0.0,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 6. Part E & F: Improved Design under C1 Surge (Admission Control Fast-Rejects)
		{
			Name:              "Improved_C1_Surge",
			NumWorkers:        4,
			QueueCapacity:     50, // bounded queue drops excess
			Timeout:           25 * time.Millisecond,
			IsResilient:       true,
			RandomSeed:        42,
			TotalRequests:     1200,
			TargetArrivalRate: 1400,
			SlowRatio:         0.0,
			SlowDuration:      0,
			FailureRatio:      0.0,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 7. Part E & F: Improved Design under C2 Slow Tasks (Context Timeout Cancels)
		{
			Name:              "Improved_C2_SlowTasks",
			NumWorkers:        4,
			QueueCapacity:     50,
			Timeout:           25 * time.Millisecond, // aborts 60ms task at 25ms
			IsResilient:       true,
			RandomSeed:        42,
			TotalRequests:     1000,
			TargetArrivalRate: 350,
			SlowRatio:         0.08,
			SlowDuration:      60 * time.Millisecond,
			FailureRatio:      0.0,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 8. Part E & F: Combined Stress Disturbance - Surge + Slow + Failures (Baseline Meltdown)
		{
			Name:              "Baseline_CombinedStress",
			NumWorkers:        4,
			QueueCapacity:     0,
			Timeout:           0,
			IsResilient:       false,
			RandomSeed:        42,
			TotalRequests:     1200,
			TargetArrivalRate: 1200,
			SlowRatio:         0.08,
			SlowDuration:      60 * time.Millisecond,
			FailureRatio:      0.10,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},

		// 9. Part E & F: Combined Stress Disturbance - Improved Design Resilience
		{
			Name:              "Improved_CombinedStress",
			NumWorkers:        4,
			QueueCapacity:     50,
			Timeout:           25 * time.Millisecond,
			IsResilient:       true,
			RandomSeed:        42,
			TotalRequests:     1200,
			TargetArrivalRate: 1200,
			SlowRatio:         0.08,
			SlowDuration:      60 * time.Millisecond,
			FailureRatio:      0.10,
			MinServiceTime:    3 * time.Millisecond,
			MaxServiceTime:    5 * time.Millisecond,
		},
	}

	var results []ResultRecord

	for idx, sc := range scenarios {
		fmt.Printf("\n[%d/%d] Executing: %s (%s Architecture)...\n", idx+1, len(scenarios), sc.Name, map[bool]string{false: "Baseline", true: "Improved"}[sc.IsResilient])
		sim := NewSimulation(sc)
		res := sim.GenerateAndRun()
		results = append(results, res)

		fmt.Printf("   Generated: %d | Accepted: %d | Success: %d | Failed: %d | TimedOut: %d | Rejected: %d\n",
			res.TotalGenerated, res.Accepted, res.Successful, res.Failed, res.TimedOut, res.Rejected)
		fmt.Printf("   Mean Resp: %.2f ms | Max Resp: %.2f ms | P99: %.2f ms | Max Queue: %d | Throughput: %.1f req/s\n",
			res.MeanResponseTime, res.MaxResponseTime, res.P99ResponseTime, res.MaxQueueLength, res.Throughput)
		fmt.Printf("   Accounting Valid: %v (No lost/duplicate requests)\n", res.AccountingValid)
	}

	// Execute Recovery Benchmark
	fmt.Printf("\n[10/%d] Executing Recovery Dynamics Experiment (Baseline vs Improved)...\n", len(scenarios)+1)
	baseSamples, baseRec := RunRecoveryExperiment(false)
	baseRec.ScenarioName = "Baseline_RecoveryDynamics"
	results = append(results, baseRec)

	impSamples, impRec := RunRecoveryExperiment(true)
	impRec.ScenarioName = "Improved_RecoveryDynamics"
	results = append(results, impRec)

	// Export Results to CSV
	writeResultsCSV(*csvOutput, results)
	writeRecoveryCSV("recovery_timeseries.csv", baseSamples, impSamples)

	// Print Summary Table
	printSummaryTable(results)
}

func writeResultsCSV(filename string, results []ResultRecord) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating CSV: %v\n", err)
		return
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	header := []string{
		"Scenario", "Architecture", "TotalGenerated", "Accepted", "Rejected",
		"Successful", "Failed", "TimedOut", "MaxQueueDepth", "Throughput_req_s",
		"Goodput_req_s", "MeanResponseTime_ms", "MedianResponseTime_ms",
		"P90ResponseTime_ms", "P99ResponseTime_ms", "MaxResponseTime_ms",
		"MeanQueueWaitTime_ms", "MeanServiceTime_ms", "Duration_s", "AccountingValid",
	}
	w.Write(header)

	for _, r := range results {
		row := []string{
			r.ScenarioName,
			r.Architecture,
			strconv.Itoa(r.TotalGenerated),
			strconv.Itoa(r.Accepted),
			strconv.Itoa(r.Rejected),
			strconv.Itoa(r.Successful),
			strconv.Itoa(r.Failed),
			strconv.Itoa(r.TimedOut),
			strconv.FormatInt(r.MaxQueueLength, 10),
			fmt.Sprintf("%.2f", r.Throughput),
			fmt.Sprintf("%.2f", r.Goodput),
			fmt.Sprintf("%.2f", r.MeanResponseTime),
			fmt.Sprintf("%.2f", r.MedianResponseTime),
			fmt.Sprintf("%.2f", r.P90ResponseTime),
			fmt.Sprintf("%.2f", r.P99ResponseTime),
			fmt.Sprintf("%.2f", r.MaxResponseTime),
			fmt.Sprintf("%.2f", r.MeanQueueWaitTime),
			fmt.Sprintf("%.2f", r.MeanServiceTime),
			fmt.Sprintf("%.2f", r.TotalDuration),
			strconv.FormatBool(r.AccountingValid),
		}
		w.Write(row)
	}
	fmt.Printf("\n[+] Benchmark metrics successfully written to: %s\n", filename)
}

func writeRecoveryCSV(filename string, baseSamples, impSamples []TimeSeriesSample) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating recovery CSV: %v\n", err)
		return
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	w.Write([]string{"Architecture", "TimestampSec", "QueueDepth", "Phase"})

	for _, s := range baseSamples {
		w.Write([]string{
			"Baseline",
			fmt.Sprintf("%.3f", s.TimestampSec),
			strconv.FormatInt(s.QueueDepth, 10),
			s.Phase,
		})
	}
	for _, s := range impSamples {
		w.Write([]string{
			"Improved",
			fmt.Sprintf("%.3f", s.TimestampSec),
			strconv.FormatInt(s.QueueDepth, 10),
			s.Phase,
		})
	}
	fmt.Printf("[+] Transient recovery samples successfully written to: %s\n", filename)
}

func printSummaryTable(results []ResultRecord) {
	fmt.Println("\n=======================================================================================================================")
	fmt.Println("                                           COMPARATIVE BENCHMARK SUMMARY                                              ")
	fmt.Println("=======================================================================================================================")
	fmt.Printf("%-24s %-10s %-7s %-7s %-7s %-7s %-8s %-10s %-10s %-10s\n",
		"Scenario", "Arch", "Gen", "Succ", "Fail", "TimeOut", "Reject", "MaxQueue", "MeanResp", "Throughput")
	fmt.Println("-----------------------------------------------------------------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-24s %-10s %-7d %-7d %-7d %-7d %-8d %-10d %-10.2f %-10.1f\n",
			r.ScenarioName, r.Architecture, r.TotalGenerated, r.Successful, r.Failed, r.TimedOut, r.Rejected,
			r.MaxQueueLength, r.MeanResponseTime, r.Throughput)
	}
	fmt.Println("=======================================================================================================================")
}


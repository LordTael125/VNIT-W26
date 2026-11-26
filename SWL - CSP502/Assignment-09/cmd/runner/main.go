package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"time"

	"vnit.ac.in/csp502/lab9/engine"
	"vnit.ac.in/csp502/lab9/events"
)

func main() {
	mode := flag.String("mode", "all", "Execution mode: validate, batch, disturbances, partf, all")
	flag.Parse()

	if err := os.MkdirAll("data/validation", 0755); err != nil {
		panic(err)
	}
	if err := os.MkdirAll("data/batch", 0755); err != nil {
		panic(err)
	}
	if err := os.MkdirAll("data/disturbances", 0755); err != nil {
		panic(err)
	}
	if err := os.MkdirAll("data/comparison", 0755); err != nil {
		panic(err)
	}

	switch *mode {
	case "validate":
		runValidation()
	case "batch":
		runBatchBenchmarks()
	case "disturbances":
		runDisturbances()
	case "partf":
		runPartFComparison()
	case "all":
		runValidation()
		runBatchBenchmarks()
		runDisturbances()
		runPartFComparison()
	default:
		fmt.Fprintf(os.Stderr, "unknown mode: %s\n", *mode)
		os.Exit(1)
	}
}

type datasetEntry struct {
	name string
	path string
	seed int64
}

var allDatasets = []datasetEntry{
	{"small_1k", "datasets/Gr05_events_small.log", 42},
	{"medium_10k", "datasets/Gr05_events_medium.log", 42},
	{"large_100k", "datasets/Gr05_events_large.log", 42},
	{"large_variant_100k", "datasets/Gr05_events_large_variant.log", 42},
}

func runValidation() {
	fmt.Println("=== Part B: Validating Datasets & Environment ===")
	envInfo := fmt.Sprintf("OS=%s ARCH=%s Go=%s LogicalCPUs=%d GOMAXPROCS=%d",
		runtime.GOOS, runtime.GOARCH, runtime.Version(), runtime.NumCPU(), runtime.GOMAXPROCS(0))
	fmt.Println(envInfo)

	f, err := os.Create("data/validation/validation_summary.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"dataset", "seed", "sha256", "total_lines", "valid", "malformed", "duplicates", "unique_users", "unique_ips", "login_failed", "file_access", "other_events"})

	for _, ds := range allDatasets {
		recs, st, err := events.ParseAndValidate(ds.name, ds.path, ds.seed)
		if err != nil {
			panic(err)
		}
		lf := st.EventCounts["LOGIN_FAILED"]
		fa := st.EventCounts["FILE_ACCESS"]
		oth := st.ValidCount - lf - fa
		fmt.Println(" ", events.FormatStats(st))
		_ = w.Write([]string{
			st.Name, strconv.FormatInt(st.Seed, 10), st.SHA256,
			strconv.Itoa(st.TotalLines), strconv.Itoa(st.ValidCount), strconv.Itoa(st.Malformed),
			strconv.Itoa(st.DuplicateRows), strconv.Itoa(st.UniqueUsers), strconv.Itoa(st.UniqueIPs),
			strconv.Itoa(lf), strconv.Itoa(fa), strconv.Itoa(oth),
		})
		// Verify sequential reference runs cleanly
		seqAgg, _ := engine.SequentialReference(recs, nil, false, 0)
		if len(seqAgg.FlaggedIPs) == 0 {
			fmt.Printf("  warning: %s has 0 flagged IPs\n", ds.name)
		}
	}
	w.Flush()
}

func runBatchBenchmarks() {
	fmt.Println("=== Part B/C: Fault-Free Batch Benchmarks (Parsing Separated from Processing) ===")
	f, err := os.Create("data/batch/batch_raw.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"dataset", "policy", "workers", "rep", "elapsed_sec", "throughput_rps", "peak_heap_mb", "accounting_valid", "matches_seq"})

	workerCounts := []int{1, 2, 4, 8, 32} // 32 = oversubscribed (2x 16 logical CPUs)
	baseCost := 5 * time.Microsecond

	for _, ds := range allDatasets {
		// Parse outside timed benchmark section
		recs, _, err := events.ParseAndValidate(ds.name, ds.path, ds.seed)
		if err != nil {
			panic(err)
		}

		// Sequential reference (3 reps)
		var seqTimes []float64
		for rep := 1; rep <= 3; rep++ {
			runtime.GC()
			var msBefore, msAfter runtime.MemStats
			runtime.ReadMemStats(&msBefore)
			_, dur := engine.SequentialReference(recs, nil, true, baseCost)
			runtime.ReadMemStats(&msAfter)
			heapMB := float64(msAfter.HeapAlloc) / (1024 * 1024)
			seqTimes = append(seqTimes, dur.Seconds())
			_ = w.Write([]string{
				ds.name, "sequential", "1", strconv.Itoa(rep),
				fmt.Sprintf("%.5f", dur.Seconds()),
				fmt.Sprintf("%.1f", float64(len(recs))/dur.Seconds()),
				fmt.Sprintf("%.3f", heapMB),
				"true", "true",
			})
		}
		sort.Float64s(seqTimes)
		seqMed := seqTimes[1]
		fmt.Printf("  [%s | sequential] median=%.4fs\n", ds.name, seqMed)

		// Concurrent FIFO & Priority across worker counts
		for _, pol := range []engine.Policy{engine.PolicyFIFO, engine.PolicyPriority} {
			for _, wc := range workerCounts {
				var times []float64
				for rep := 1; rep <= 3; rep++ {
					cfg := engine.Config{
						Policy:        pol,
						Workers:       wc,
						QueueCapacity: 0, // unbounded for fault-free batch
					}
					_, m := engine.RunBatch(recs, cfg, ds.name, rep, baseCost)
					times = append(times, m.ElapsedSec)
					_ = w.Write([]string{
						ds.name, string(pol), strconv.Itoa(wc), strconv.Itoa(rep),
						fmt.Sprintf("%.5f", m.ElapsedSec),
						fmt.Sprintf("%.1f", m.ThroughputRPS),
						fmt.Sprintf("%.3f", m.PeakHeapMB),
						strconv.FormatBool(m.AccountingValid),
						strconv.FormatBool(m.MatchesSeqReplay),
					})
				}
				sort.Float64s(times)
				med := times[1]
				speedup := seqMed / med
				eff := speedup / float64(wc)
				fmt.Printf("  [%s | %s | W=%d] median=%.4fs speedup=%.2fx eff=%.2f\n",
					ds.name, pol, wc, med, speedup, eff)
			}
		}
	}
	w.Flush()
}

func buildScenarioSpec(scenario string) events.TraceSpec {
	// 1,200 valid events per paced experiment; sustainable capacity with W=4, 1.2ms cost is ~3300 req/s,
	// or with 2.0ms cost is ~2000 req/s. Using BaseCostUs=1800 and NominalRateRPS=950 (~50% util)
	// makes D1 (3x = 2850 req/s) and D2 (10% 5x = 9ms slow tasks) clearly saturate W=4 and W=2!
	spec := events.TraceSpec{
		Scenario:       scenario,
		Seed:           2026,
		NumEvents:      1200,
		NominalRateRPS: 950.0,
		BaseCostUs:     1800,
		SlowMultiplier: 5,
	}
	switch scenario {
	case "D0_Normal":
	case "D1_Burst":
		spec.BurstMiddle3x = true
	case "D2_SlowTasks":
		spec.SlowRatio = 0.10
	case "D3_TransientFail":
		spec.TransientFailRat = 0.05
	case "D4_PriorityPressure":
		spec.HighPriorityFrac = 0.70
	}
	return spec
}

func burstEndOffsetFor(trace []events.ReplayItem, scenario string) int64 {
	if scenario != "D1_Burst" || len(trace) == 0 {
		return 0
	}
	idx := (2 * len(trace)) / 3
	if idx >= len(trace) {
		idx = len(trace) - 1
	}
	return trace[idx].ArrivalOffsetNs
}

func runDisturbances() {
	fmt.Println("=== Part D: Controlled Disturbances on FIFO & Priority Baselines (W=4, Unbounded) ===")
	recs, _, err := events.ParseAndValidate("medium", "datasets/Gr05_events_medium.log", 42)
	if err != nil {
		panic(err)
	}

	f, err := os.Create("data/disturbances/disturbances_raw.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	writeMetricsHeader(w)

	scenarios := []string{"D0_Normal", "D1_Burst", "D2_SlowTasks", "D3_TransientFail", "D4_PriorityPressure"}
	for _, sc := range scenarios {
		spec := buildScenarioSpec(sc)
		trace := events.BuildReplayTrace(recs, spec)
		_ = events.SaveReplayTrace(filepath.Join("configs", "trace_"+sc+".json"), trace)
		burstEnd := burstEndOffsetFor(trace, sc)

		for _, pol := range []engine.Policy{engine.PolicyFIFO, engine.PolicyPriority} {
			cfg := engine.Config{
				Policy:            pol,
				Workers:           4,
				QueueCapacity:     0,
				RecoveryThreshold: 5,
			}
			_, m := engine.RunPaced(trace, 0, cfg, sc, "medium_10k", 1, burstEnd, context.Background())
			writeMetricsRow(w, m)
			fmt.Printf("  [%s | %s] Succ=%d Fail=%d PeakQ=%d Thru=%.1f HighMed/p95=%.2f/%.2f NormMed/p95=%.2f/%.2f MaxNormWait=%.2fms Rec=%.2fms\n",
				sc, pol, m.Successful, m.Failed, m.PeakQueueLen, m.ThroughputRPS,
				m.HighMedianLatMs, m.HighP95LatMs, m.NormMedianLatMs, m.NormP95LatMs,
				m.MaxNormalWaitMs, m.BurstRecoveryMs)
		}
	}
	w.Flush()
}

func runPartFComparison() {
	fmt.Println("=== Part F: Comparative Evaluation Under Constrained Envelope (W=2, K=64, 3 Reps) ===")
	recs, _, err := events.ParseAndValidate("medium", "datasets/Gr05_events_medium.log", 42)
	if err != nil {
		panic(err)
	}

	f, err := os.Create("data/comparison/partf_raw.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	writeMetricsHeader(w)

	scenarios := []string{"D0_Normal", "D1_Burst", "D2_SlowTasks", "D3_TransientFail", "D4_PriorityPressure"}
	policies := []engine.Policy{
		engine.PolicyFIFO,
		engine.PolicyPriority,
		engine.PolicyFairnessOnly,
		engine.PolicyResilienceOnly,
		engine.PolicyModified,
	}

	for _, sc := range scenarios {
		spec := buildScenarioSpec(sc)
		trace := events.BuildReplayTrace(recs, spec)
		burstEnd := burstEndOffsetFor(trace, sc)

		for _, pol := range policies {
			for rep := 1; rep <= 3; rep++ {
				cfg := engine.Config{
					Policy:             pol,
					Workers:            2,  // Part E constraint: reduced from 4 to 2
					QueueCapacity:      64, // Part E constraint: strict finite queue K=64
					AgingThresholdMs:   20,
					MaxConsecutiveHigh: 4,
					TaskTimeoutMs:      15,
					MaxRetries:         1,
					RecoveryThreshold:  5,
				}
				_, m := engine.RunPaced(trace, 0, cfg, sc, "medium_10k", rep, burstEnd, context.Background())
				writeMetricsRow(w, m)
				if !m.AccountingValid || !m.MatchesSeqReplay {
					fmt.Fprintf(os.Stderr, "ERROR: invariant failed for %s/%s rep %d\n", sc, pol, rep)
					os.Exit(1)
				}
			}
		}
		fmt.Printf("  Completed %s (5 policies x 3 reps = 15 runs, all invariants valid)\n", sc)
	}
	w.Flush()
}

func writeMetricsHeader(w *csv.Writer) {
	_ = w.Write([]string{
		"scenario", "policy", "dataset", "rep", "workers", "queue_capacity",
		"generated", "malformed", "successful", "failed", "timed_out", "cancelled", "rejected",
		"total_attempts", "retries_used", "accounting_valid", "duplicate_outcomes", "missing_ids", "matches_seq_replay",
		"flagged_ip_count", "elapsed_sec", "throughput_rps", "peak_queue_len", "peak_heap_mb",
		"high_accepted", "high_successful", "high_success_rate", "high_median_lat_ms", "high_p95_lat_ms", "high_succ_median_lat_ms", "high_succ_p95_lat_ms",
		"norm_accepted", "norm_successful", "norm_success_rate", "norm_median_lat_ms", "norm_p95_lat_ms", "norm_succ_median_lat_ms", "norm_succ_p95_lat_ms",
		"max_normal_wait_ms", "burst_recovery_ms", "goroutine_delta",
	})
}

func writeMetricsRow(w *csv.Writer, m engine.RunMetrics) {
	_ = w.Write([]string{
		m.Scenario, m.Policy, m.Dataset, strconv.Itoa(m.Repetition), strconv.Itoa(m.Workers), strconv.Itoa(m.QueueCapacity),
		strconv.Itoa(m.Generated), strconv.Itoa(m.Malformed), strconv.Itoa(m.Successful), strconv.Itoa(m.Failed), strconv.Itoa(m.TimedOut), strconv.Itoa(m.Cancelled), strconv.Itoa(m.Rejected),
		strconv.Itoa(m.TotalAttempts), strconv.Itoa(m.RetriesUsed), strconv.FormatBool(m.AccountingValid), strconv.Itoa(m.DuplicateOutcomes), strconv.Itoa(m.MissingIDs), strconv.FormatBool(m.MatchesSeqReplay),
		strconv.Itoa(m.FlaggedIPCount), fmt.Sprintf("%.5f", m.ElapsedSec), fmt.Sprintf("%.2f", m.ThroughputRPS), strconv.Itoa(m.PeakQueueLen), fmt.Sprintf("%.3f", m.PeakHeapMB),
		strconv.Itoa(m.HighAccepted), strconv.Itoa(m.HighSuccessful), fmt.Sprintf("%.4f", m.HighSuccessRate), fmt.Sprintf("%.3f", m.HighMedianLatMs), fmt.Sprintf("%.3f", m.HighP95LatMs), fmt.Sprintf("%.3f", m.HighSuccMedianLatMs), fmt.Sprintf("%.3f", m.HighSuccP95LatMs),
		strconv.Itoa(m.NormAccepted), strconv.Itoa(m.NormSuccessful), fmt.Sprintf("%.4f", m.NormSuccessRate), fmt.Sprintf("%.3f", m.NormMedianLatMs), fmt.Sprintf("%.3f", m.NormP95LatMs), fmt.Sprintf("%.3f", m.NormSuccMedianLatMs), fmt.Sprintf("%.3f", m.NormSuccP95LatMs),
		fmt.Sprintf("%.3f", m.MaxNormalWaitMs), fmt.Sprintf("%.3f", m.BurstRecoveryMs), strconv.Itoa(m.GoroutineDelta),
	})
}

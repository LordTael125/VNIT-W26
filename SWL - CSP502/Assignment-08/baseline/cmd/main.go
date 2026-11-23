package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"

	"vnit.ac.in/csp502/practical8/baseline"
	"vnit.ac.in/csp502/practical8/workload"
)

func main() {
	cfgPath := flag.String("config", "configs/normal.json", "Path to workload JSON configuration")
	seed := flag.Int64("seed", 0, "Override random seed (0 uses first seed in config)")
	rep := flag.Int("rep", 1, "Repetition number")
	eventsOut := flag.String("events-out", "", "Optional output path for raw JSONL lifecycle event log")
	csvOut := flag.String("csv-out", "", "Optional output path to append CSV summary row")
	jsonOut := flag.Bool("json", false, "Print full JSON summary to stdout")
	flag.Parse()

	cfg, err := workload.LoadConfig(*cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config %s: %v\n", *cfgPath, err)
		os.Exit(1)
	}
	useSeed := *seed
	if useSeed == 0 {
		if len(cfg.Seeds) > 0 {
			useSeed = cfg.Seeds[0]
		} else {
			useSeed = 101
		}
	}

	sum, err := baseline.RunWorkload(cfg, useSeed, *rep, *eventsOut)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run failed: %v\n", err)
		os.Exit(1)
	}

	if *csvOut != "" {
		if err := appendSummaryCSV(*csvOut, sum); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write CSV %s: %v\n", *csvOut, err)
			os.Exit(1)
		}
	}

	if *jsonOut {
		b, _ := json.MarshalIndent(sum, "", "  ")
		fmt.Println(string(b))
	} else {
		fmt.Println(baseline.FormatInvariantReport(sum))
	}
}

var csvHeader = []string{
	"architecture", "condition", "seed", "repetition",
	"unique_generated", "total_submissions", "duplicate_submissions", "duplicate_submissions_blocked",
	"total_terminal_records", "unique_terminal_ids", "duplicate_terminal_outcomes", "missing_terminal_ids",
	"terminal_accounting_violations", "lifecycle_order_violations",
	"successful", "failed", "timed_out", "cancelled", "rejected",
	"ack_retries_used", "duplicate_acks_ignored",
	"max_queue_depth", "peak_active_workers", "wall_duration_sec",
	"throughput_rps", "goodput_rps",
	"mean_queue_wait_ms", "mean_response_ms", "p50_response_ms", "p95_response_ms", "p99_response_ms", "max_response_ms",
	"overall_mean_response_ms", "shutdown_completion_ms", "outstanding_work_after_shutdown", "goroutine_delta", "invariants_passed",
}

func appendSummaryCSV(path string, s workload.RunSummary) error {
	needHeader := false
	if _, err := os.Stat(path); os.IsNotExist(err) {
		needHeader = true
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	if needHeader {
		if err := w.Write(csvHeader); err != nil {
			return err
		}
	}
	row := []string{
		s.Architecture, s.Condition, strconv.FormatInt(s.Seed, 10), strconv.Itoa(s.Repetition),
		strconv.Itoa(s.UniqueGenerated), strconv.Itoa(s.TotalSubmissions), strconv.Itoa(s.DuplicateSubmissions), strconv.Itoa(s.DuplicateSubmissionsBlocked),
		strconv.Itoa(s.TotalTerminalRecords), strconv.Itoa(s.UniqueTerminalIDs), strconv.Itoa(s.DuplicateTerminalOutcomes), strconv.Itoa(s.MissingTerminalIDs),
		strconv.Itoa(s.TerminalAccountingViolations), strconv.Itoa(s.LifecycleOrderViolations),
		strconv.Itoa(s.Successful), strconv.Itoa(s.Failed), strconv.Itoa(s.TimedOut), strconv.Itoa(s.Cancelled), strconv.Itoa(s.Rejected),
		strconv.Itoa(s.AckRetriesUsed), strconv.Itoa(s.DuplicateAcksIgnored),
		strconv.FormatInt(s.MaxQueueDepth, 10), strconv.FormatInt(s.PeakActiveWorkers, 10), fmt.Sprintf("%.4f", s.WallDurationSec),
		fmt.Sprintf("%.2f", s.ThroughputRPS), fmt.Sprintf("%.2f", s.GoodputRPS),
		fmt.Sprintf("%.3f", s.MeanQueueWaitMs), fmt.Sprintf("%.3f", s.MeanResponseMs), fmt.Sprintf("%.3f", s.P50ResponseMs),
		fmt.Sprintf("%.3f", s.P95ResponseMs), fmt.Sprintf("%.3f", s.P99ResponseMs), fmt.Sprintf("%.3f", s.MaxResponseMs),
		fmt.Sprintf("%.3f", s.OverallMeanResponseMs), fmt.Sprintf("%.3f", s.ShutdownCompletionMs),
		strconv.Itoa(s.OutstandingWorkAfterShutdown), strconv.Itoa(s.GoroutineDelta), strconv.FormatBool(s.InvariantsPassed),
	}
	if err := w.Write(row); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}


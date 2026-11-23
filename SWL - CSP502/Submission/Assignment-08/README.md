# CSP502 Practical 8 — From Resilience to Reliability: Testing and Observing a Concurrent Service in Go

**Course:** CSP502 Software Lab-I (M.Tech. CSE W26), VNIT Nagpur  
**Faculty Guide:** Dr. A. S. Mokhade  
**Group Members:** Gaurav Acharya (`MT26MCS024`), Sandeep (`MT26MCS032`)

---

## Directory Structure

```
Assignment-08/
├── Assignment-08.pdf                         # Final compiled consolidated report (Parts A–F)
├── README.md                                 # Reproduction guide and layout overview
├── go.mod                                    # Go module definition
├── workload/
│   └── workload.go                           # Deterministic workload generator, config schema, p95 & lifecycle verifier
├── baseline/                                 # Part B & Part C preserved baseline service
│   ├── service.go                            # Configurable worker-pool service + JSONL instrumentation + invariant checker
│   ├── service_test.go                       # Part C automated tests (Duplicate-ID, Cancel, Slow, Failure, Shutdown)
│   └── cmd/main.go                           # CLI runner for baseline experiments
├── improved/                                 # Part D & Part E improved reliable service
│   ├── service.go                            # Idempotent commit ledger, bounded ack retries + budget, admission control, SLA deadlines, graceful bounded shutdown
│   ├── service_test.go                       # Full correctness, fault-injection, Part D ack-fault, retry-budget & leak test suite
│   └── cmd/main.go                           # CLI runner for improved experiments
├── configs/                                  # Reproducible JSON experiment configurations (seeds, rates, faults)
│   ├── normal.json
│   ├── overload_slow.json
│   ├── failure_cancel.json
│   └── ack_faults.json
├── docs/                                     # Timestamped pre-coding design records
│   ├── partA_design_record.md
│   └── partD_precoding_record.md
├── scripts/
│   ├── run_experiments.sh                    # End-to-end automated test & benchmark harness
│   └── generate_plots.py                     # Generates summary tables and comparative PDF/PNG graphs
├── data/                                     # Measured outputs, raw JSONL logs, CSV tables, and test evidence
│   ├── baseline_runs/                        # Part B: 3 normal runs (raw JSONL event logs + summary CSV)
│   ├── tests/                                # Normal, race-detector, and strict-baseline failure logs
│   └── comparison/                           # Part E: 24-run raw CSV, aggregated summary CSV, sample Part D JSONL logs
└── report/
    ├── Assignment-08.tex                     # Master LaTeX report source
    ├── VNIT_logo.jpg                         # Institute logo
    └── figures/                              # Generated publication-grade comparative PDF/PNG figures
```

---

## Exact Commands to Reproduce All Results

### 1. Run Automated Unit Tests and Race Detector (Parts C & E)

```bash
# Standard test suite across baseline and improved packages
go test -v ./...

# Race-detector verification (Note: race-instrumented timings are NOT used as performance benchmarks)
go test -race -v ./...

# Reproduce Part C honest baseline defect failures (D-01 Duplicate-ID, D-02 Cancel accounting, D-03 Slow worker, D-04 Unbounded shutdown)
STRICT_INVARIANTS=1 go test -v ./baseline/...
```

### 2. Run Full Experiment Harness (Part B 3-Run Baseline + Part E 24-Run Comparison)

```bash
./scripts/run_experiments.sh
```

Or invoke individual runs manually:

```bash
# Part B Baseline Normal Run 1 (seed=101, N=1200, W=8)
go run ./baseline/cmd -config configs/normal.json -seed 101 -rep 1 \
  -events-out data/baseline_runs/run1_events.jsonl \
  -csv-out data/baseline_runs/baseline_summary.csv

# Part E Improved Service under Part D Downstream Ack Faults
go run ./improved/cmd -config configs/ack_faults.json -seed 101 -rep 1 \
  -events-out data/comparison/improved_ack_faults_run1_events.jsonl
```

### 3. Generate Summary Tables and Comparative Figures

```bash
python3 scripts/generate_plots.py
```


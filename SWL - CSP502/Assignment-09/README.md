# CSP502 Lab Exercise 9 — Resource-Aware, Fault-Tolerant and Priority-Based Event Processing in Go

**Course:** CSP502 Software Lab-I (M.Tech. CSE W26), VNIT Nagpur  
**Faculty Guide:** Dr. A. S. Mokhade  
**Group Members:** Gaurav Acharya (`MT26MCS024`), Sandeep (`MT26MCS032`)

---

## Directory Structure

```
Assignment-09/
├── Assignment-09.pdf                 # Final consolidated report (Parts A–G)
├── README.md                         # Exact commands to reproduce all results
├── go.mod                            # Go module definition
├── datasets/                         # Lab 4A synthetic logs + hand-verified fixture
│   ├── Gr05_events_small.log         # 1,000 records
│   ├── Gr05_events_medium.log        # 10,000 records
│   ├── Gr05_events_large.log         # 100,000 records
│   ├── Gr05_events_large_variant.log # 100,000 records (variant)
│   └── hand_verified_fixture.log     # 18-line fixture (2 malformed, dup line, IP w/ 5 failures)
├── events/events.go                  # Parser, dataset validator, deterministic replay trace builder
├── engine/
│   ├── engine.go                     # Sequential reference, FIFO, Priority, Fairness/Resilience/Modified engines
│   └── engine_test.go                # Automated correctness, fixture, cancellation, fairness & -race tests
├── cmd/runner/main.go                # CLI runner for validation, batch, disturbances, and Part F evaluation
├── configs/                          # Saved deterministic replay traces (trace_D0_Normal.json .. trace_D4_PriorityPressure.json)
├── docs/                             # Dated pre-coding design & change records
│   ├── partA_design_record.md
│   └── partE_change_record.md
├── scripts/
│   ├── run_experiments.sh            # Full automated test + experiment pipeline
│   └── generate_plots.py             # Generates summary CSVs and comparative figures
├── data/                             # Measured raw CSVs, summaries, validation & test logs
└── report/                           # LaTeX source (Assignment-09.tex), VNIT_logo.jpg, and figures/
```

---

## Exact Commands to Reproduce All Results

```bash
# 1. Run automated correctness tests and race-detector verification
go test -count=1 -v ./...
go test -count=1 -race -v ./...

# 2. Run the full experiment pipeline (validation, batch, disturbances, Part F, plots)
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh

# 3. Compile the LaTeX report
cd report
pdflatex -interaction=nonstopmode Assignment-09.tex
pdflatex -interaction=nonstopmode Assignment-09.tex
cp Assignment-09.pdf ../Assignment-09.pdf
```

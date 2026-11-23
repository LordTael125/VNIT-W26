#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Cleaning previous generated data ==="
rm -rf data/baseline_runs/* data/tests/* data/comparison/*
mkdir -p data/baseline_runs data/tests data/comparison

echo "=== 1. Running Part C Automated Tests (Strict Baseline Audit -> Honest Failure Evidence) ==="
STRICT_INVARIANTS=1 go test -v ./baseline/... > data/tests/baseline_strict_fail.log 2>&1 || true
echo "Saved data/tests/baseline_strict_fail.log"

echo "=== 2. Running Normal & Race-Detector Tests (Baseline & Improved) ==="
go test -v ./baseline/... > data/tests/baseline_test.log 2>&1
go test -race -v ./baseline/... > data/tests/baseline_race.log 2>&1
go test -v ./improved/... > data/tests/improved_test.log 2>&1
go test -race -v ./improved/... > data/tests/improved_race.log 2>&1
echo "Saved all 4 test logs in data/tests/"

echo "=== 3. Running Part B Baseline Normal Workload (3 Repetitions, 1200 reqs each) ==="
SEEDS=(101 202 303)
for idx in 0 1 2; do
  rep=$((idx + 1))
  seed="${SEEDS[$idx]}"
  go run ./baseline/cmd \
    -config configs/normal.json \
    -seed "$seed" \
    -rep "$rep" \
    -events-out "data/baseline_runs/run${rep}_events.jsonl" \
    -csv-out data/baseline_runs/baseline_summary.csv
done

echo "=== 4. Running Part E Controlled Evaluation (4 Conditions x 3 Reps x 2 Architectures = 24 runs) ==="
CONFIGS=(normal overload_slow failure_cancel ack_faults)
for cond in "${CONFIGS[@]}"; do
  cfg="configs/${cond}.json"
  for idx in 0 1 2; do
    rep=$((idx + 1))
    seed="${SEEDS[$idx]}"
    ev_base=""
    ev_imp=""
    if [[ "$cond" == "ack_faults" && "$rep" == "1" ]]; then
      ev_base="data/comparison/baseline_ack_faults_run1_events.jsonl"
      ev_imp="data/comparison/improved_ack_faults_run1_events.jsonl"
    fi
    go run ./baseline/cmd \
      -config "$cfg" \
      -seed "$seed" \
      -rep "$rep" \
      -events-out "$ev_base" \
      -csv-out data/comparison/comparison_raw.csv
    go run ./improved/cmd \
      -config "$cfg" \
      -seed "$seed" \
      -rep "$rep" \
      -events-out "$ev_imp" \
      -csv-out data/comparison/comparison_raw.csv
  done
done

echo "=== All experiments complete! ==="


#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== 1. Running Unit Tests & Race Detector ==="
go test -count=1 -v ./... > data/tests/unit_test.log 2>&1
go test -count=1 -race -v ./... > data/tests/race_test.log 2>&1
echo "Saved data/tests/unit_test.log and data/tests/race_test.log"

echo "=== 2. Running Dataset Validation ==="
go run ./cmd/runner -mode=validate > data/validation/validation_report.txt

echo "=== 3. Running Batch Benchmarks (Part B/C) ==="
go run ./cmd/runner -mode=batch > data/batch/batch_summary.txt

echo "=== 4. Running Paced Disturbances (Part D) ==="
go run ./cmd/runner -mode=disturbances > data/disturbances/disturbances_summary.txt

echo "=== 5. Running Part F Comparative Evaluation ==="
go run ./cmd/runner -mode=partf > data/comparison/partf_summary.txt

echo "=== 6. Generating Summary CSVs & Comparative Plots ==="
python3 scripts/generate_plots.py

echo "=== All Lab 9 experiments complete! ==="

#!/usr/bin/env bash
# NetFlow-Forecaster runner for Linux/macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Show help if requested
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat << 'EOF'
NetFlow-Forecaster Runner

USAGE:
    ./runners/run.sh <mode> [options]

MODES:
    synthetic              Generate synthetic data and train (default)
    kaggle                 Download and train on Kaggle dataset
    public_benchmark       Download CICIDS2017 and train
    benchmark              Search for best candidate (with meta-policy)
    train                  Train a single model on existing data
    visualize              Generate dashboards from completed run

OPTIONS:
    --mode MODE            Execution mode (default: synthetic)
    --samples N            Synthetic data samples (default: 720)
    --epochs N             Training epochs (default: 130)
    --target-quality Q     Quality gate for benchmark (default: 90)
    --max-attempts N       Max attempts in benchmark (default: 24)
    --skip-install         Skip pip install
    --help                 Show this message

EXAMPLES:
    ./runners/run.sh synthetic --epochs 60
    ./runners/run.sh benchmark --target-quality 90
    ./runners/run.sh public_benchmark --samples 5000
EOF
    exit 0
fi

echo "[$(date +'%H:%M:%S')] NetFlow-Forecaster"
echo "[$(date +'%H:%M:%S')] Python: $(python3 --version 2>&1)"

cd "$PROJECT_DIR"
python3 "$SCRIPT_DIR/run.py" "$@"
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[$(date +'%H:%M:%S')] ✓ Completed successfully"
else
    echo "[$(date +'%H:%M:%S')] ✗ Process exited with code $EXIT_CODE" >&2
fi

exit $EXIT_CODE

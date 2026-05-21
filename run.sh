#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$PROJECT_DIR/ml"

MODE="synthetic"
SAMPLES=72
INTERVAL=5
EPOCHS=130
SKIP_INSTALL=0

usage() {
  cat <<EOF
AI-driven network design pipeline

Usage:
  bash run.sh [synthetic|live|deploy|destroy] [options]

Modes:
  synthetic          Generate synthetic telemetry, train, visualize (default)
  live               Collect telemetry from deployed ContainerLab, train, visualize
  deploy             Deploy ContainerLab topology
  destroy            Destroy ContainerLab topology

Options:
  --samples N        Live sample count, or synthetic hours (default: 72)
  --interval SEC     Seconds between live samples (default: 5)
  --epochs N         Training epochs (default: 130)
  --skip-install     Do not install Python dependencies
EOF
}

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [[ $# -gt 0 && "$1" != --* ]]; then
  MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples) SAMPLES="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PROJECT_DIR/runs/${RUN_STAMP}_${MODE}"
DATA_FILE="$RUN_DIR/raw_data/telemetry.csv"

install_deps() {
  if [[ "$SKIP_INSTALL" -eq 1 ]]; then
    return
  fi
  log "Installing Python dependencies"
  python -m pip install -r "$PROJECT_DIR/requirements.txt" --quiet --break-system-packages 2>/dev/null \
    || python -m pip install -r "$PROJECT_DIR/requirements.txt" --quiet
}

run_ml() {
  log "Training LSTM"
  cd "$ML_DIR"
  python enhanced_train.py --data "$DATA_FILE" --epochs "$EPOCHS" --output-dir "$RUN_DIR"
  log "Building dashboard"
  python visualize.py --data "$RUN_DIR/raw_data/telemetry.csv" --output-dir "$RUN_DIR"
  log "Evaluating model"
  python evaluate_model.py --run-dir "$RUN_DIR"
  log "Exporting readable model report"
  python export_model_report.py --run-dir "$RUN_DIR"
  log "Cleaning empty run folders"
  cd "$PROJECT_DIR"
  python scripts/cleanup_runs.py
  log "Done"
  printf 'Run folder:\n  %s\nArtifacts:\n  %s\n  %s\n  %s\n  %s\n  %s\nBinary model weights:\n  %s\n' \
    "$RUN_DIR" "$RUN_DIR/raw_data/telemetry.csv" "$RUN_DIR/images/traffic_prediction_dashboard.png" "$RUN_DIR/images/model_evaluation_dashboard.png" "$RUN_DIR/json/model_metadata.json" "$RUN_DIR/model/model_readable_report.md" "$RUN_DIR/model/lstm_model.pth"
}

case "$MODE" in
  synthetic)
    install_deps
    mkdir -p "$RUN_DIR/raw_data"
    log "Generating synthetic telemetry"
    cd "$ML_DIR"
    python generate_data.py --hours "$SAMPLES" --output "$DATA_FILE" --seed 7
    run_ml
    ;;
  live)
    install_deps
    mkdir -p "$RUN_DIR/raw_data"
    log "Collecting live ContainerLab telemetry"
    cd "$PROJECT_DIR"
    python scripts/collect_telemetry.py --mode live --samples "$SAMPLES" --interval "$INTERVAL" --output "$DATA_FILE"
    run_ml
    ;;
  deploy)
    log "Deploying ContainerLab topology"
    cd "$PROJECT_DIR/containerlab"
    sudo containerlab deploy -t topology.clab.yml
    ;;
  destroy)
    log "Destroying ContainerLab topology"
    cd "$PROJECT_DIR/containerlab"
    sudo containerlab destroy -t topology.clab.yml
    ;;
  *)
    echo "Unknown mode: $MODE"
    usage
    exit 1
    ;;
esac

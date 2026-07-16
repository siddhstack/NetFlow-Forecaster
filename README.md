NetFlow-Forecaster is a reproducible benchmarking framework for short-term
network telemetry forecasting (traffic, latency, packet loss), built around
two things: (1) a hybrid attention-LSTM + Gradient Boosting forecaster, and
(2) a lightweight, persistent, bandit-guided meta-policy that automatically
selects which training configuration to try next, gated on domain-specific
spike-detection criteria rather than plain error metrics alone.

The forecasting architecture itself (attention-LSTM + Gradient Boosting
ensembling) follows established patterns in the hybrid time-series
forecasting literature. What this project contributes is the automated,
experience-driven candidate-selection loop and its application, with
transparent quality gates, to joint network telemetry forecasting.

# NetFlow-Forecaster

Applied network telemetry forecasting with automated, self-improving
candidate selection.

## Highlights

- Forecasts `traffic_mbps`, `latency_ms`, and `packet_loss_pct` jointly.
- Hybrid attention-LSTM + Gradient Boosting ensemble with a spike-weighted
  training loss.
- A persistent, UCB-style meta-policy (ml/meta_policy.py) that ranks
  candidate training configurations using cross-run experience, rather than
  a fixed schedule or grid search.
- Ablation studies quantifying the meta-policy's and the spike-weighted
  loss's actual contribution (see "Ablation Studies" below) instead of
  asserting them.
- Statistical significance testing (paired t-test, Diebold-Mariano) on every
  evaluated run, not just point-estimate MAE comparisons.
- Evaluated on synthetic, local CSV, external NetFlow CSV, and CICIDS2017
  (public benchmark) data.

## Current Evidence

| Dataset | Rows | Quality | MAE vs Persistence | Traffic Spike F1 | Status |
|---|---:|---:|---:|---:|---|
| Synthetic | 2000 | 89.8% | +9.2% | 0.874 | Best checked-in demo |
| Generic `ml/telemetry.csv` | 3000 | 81.8% | +18.3% | 0.810 | Local CSV demo |
| External NetFlow CSV | 120000 | 73.0% | +34.7% | 0.887 | Large CSV trial |
| CICIDS2017 (public) | <PLACEHOLDER — DO NOT INVENT: run ml/load_public_benchmark.py + training pipeline> | | | | |

The requested `>=90%` quality gate has not yet been reached by these
checked-in runs. See "Ablation Studies" below for the automated search that
targets this gate, and its measured effect.

Statistical significance (paired t-test and Diebold-Mariano test, model vs.
persistence baseline) is now computed automatically for every run; see
`results/significance_tests.csv` inside each run directory.

## Ablation Studies

Two ablations isolate the contributions this project actually claims:

### 1. Candidate-selection strategy (ml/ablation_selection.py)

Compares three ways of choosing the next training configuration to try:
a fixed default order, a random order, and the persisted meta-policy
(ml/meta_policy.py). Measures attempts-to-reach-90%-quality-gate and final
quality for each, per dataset.

Run:
```
python ml/ablation_selection.py --data ml/telemetry.csv --output-dir runs/ablation_selection
```

Results: `<output-dir>/ablation_selection_summary.json`

<PLACEHOLDER — DO NOT INVENT: fill in attempts-to-gate and quality numbers
per strategy per dataset only after running the script above.>

### 2. Spike-weighted loss configuration (ml/ablation_spike_loss.py)

Compares three loss configurations, each run across 3 seeds: no spike
weighting (plain MSE), uniform spike weighting (the current shipped
default), and differentiated per-feature spike weighting (the value the
code's own docstring recommends but never benchmarks).

Run:
```
python ml/ablation_spike_loss.py --data ml/telemetry.csv --output-dir runs/ablation_spike_loss
```

Results: `<output-dir>/ablation_spike_loss_summary.json`

<PLACEHOLDER — DO NOT INVENT: fill in mean +/- std MAE and spike F1 per
condition per feature only after running the script above.>

## How It Works

All data paths normalize into the same schema:

```text
timestamp,traffic_mbps,latency_ms,packet_loss_pct
```

The pipeline then follows this flow:

```text
telemetry CSV
  -> feature transforms and chronological split
  -> LSTM-based sequence model
  -> Gradient Boosting model over lag and rolling features
  -> validation-tuned blend and residual baseline check
  -> dashboards, evaluation summaries, and model artifacts
```

The LSTM component learns recent sequence behavior from lookback windows. The
Gradient Boosting component handles tabular lag, rolling-window, and abrupt
spike features. The evaluation layer checks whether the result actually beats
simple baselines.

## Repository Layout

```text
.
├── runners/                  # Cross-platform and shell-specific entrypoints
│   ├── run.py                # Primary runner
│   ├── run.ps1               # Windows PowerShell wrapper
│   └── run.sh                # Linux/WSL wrapper
├── ml/                       # Training, data loading, evaluation, dashboards
├── scripts/                  # Live telemetry collection and cleanup helpers
├── containerlab/             # ContainerLab topology
├── configs/                  # FRR router configs
├── docs/
│   ├── images/               # Checked-in dashboard evidence
│   └── results/              # Checked-in evaluation evidence
├── tests/                    # Pytest coverage
└── .github/                  # CI and community templates
```

## Setup

PowerShell:

```powershell
cd "C:\path\to\NetFlow-Forecaster"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install -r requirements.txt
```

Bash:

```bash
cd NetFlow-Forecaster
python -m pip install -r requirements.txt
```

If ONNX-related installation fails on Windows, install the exporter dependency
directly:

```powershell
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org onnx onnxscript
```

## Quick Start

Cross-platform Python runner:

```powershell
python runners\run.py synthetic --samples 2000 --epochs 40
```

Windows wrapper:

```powershell
.\runners\run.ps1 synthetic -Samples 2000 -Epochs 40
```

Linux/WSL wrapper:

```bash
bash runners/run.sh synthetic --samples 2000 --epochs 40
```

Fast smoke run:

```powershell
.\runners\run.ps1 synthetic -Samples 300 -Epochs 3 -SkipInstall
```

Universal benchmark on `ml/telemetry.csv`:

```powershell
.\runners\run.ps1 benchmark -TargetQuality 90 -MaxAttempts 12
```

Public benchmark dataset:

```powershell
.\runners\run.ps1 public_benchmark --samples 5000 --epochs 60
```

## Common Workflows

### Synthetic Trial

```powershell
.\runners\run.ps1 synthetic -Samples 2000 -Epochs 40
```

### Kaggle Trial

```powershell
.\runners\run.ps1 kaggle -Samples 5000 -Epochs 60
```

### Local CSV Training

Place a CSV at `ml/telemetry.csv` with:

```text
timestamp,traffic_mbps,latency_ms,packet_loss_pct
```

Then run:

```powershell
.\runners\run.ps1 train -Epochs 80
```

### Manual End-To-End Trial

```powershell
python ml\enhanced_train.py --data ml\telemetry.csv --output-dir runs\sample_hybrid_trial --epochs 40 --device auto
python ml\visualize.py --data runs\sample_hybrid_trial\raw_data\telemetry.csv --output-dir runs\sample_hybrid_trial --sensitivity 1.3
python ml\evaluate_model.py --run-dir runs\sample_hybrid_trial
python ml\export_model_report.py --run-dir runs\sample_hybrid_trial
```

### Dataset-Optimized Tree Model

```powershell
.\runners\run.ps1 dataset_opt -SkipInstall
```

### Sequence Model Ablation

```powershell
python ml\compare_sequence_models.py --data ml\telemetry.csv --output-dir runs\sequence_comparison --epochs 40
```

This compares LSTM, GRU, mean-pooling LSTM, and LSTM-with-attention variants on
the same chronological split.

### Live ContainerLab Workflow

ContainerLab is Linux-focused. Use Linux, WSL2, or a Linux VM with Docker and
ContainerLab installed.

PowerShell:

```powershell
.\runners\run.ps1 deploy
.\runners\run.ps1 live -Samples 120 -Interval 10 -Epochs 80
.\runners\run.ps1 destroy
```

Bash:

```bash
bash runners/run.sh deploy
bash runners/run.sh live --samples 120 --interval 10 --epochs 80
bash runners/run.sh destroy
```

## Run Outputs

Each complete run is written under `runs/` and usually contains:

```text
raw_data/telemetry.csv
results/predictions.csv
results/actuals.csv
results/prediction_intervals.csv
results/train_losses.csv
results/evaluation_comparison.csv
results/evaluation_baselines.csv
results/evaluation_spikes.csv
results/significance_tests.csv
json/metrics.json
json/evaluation_summary.json
json/model_metadata.json
json/model_readable_summary.json
images/traffic_prediction_dashboard.png
images/model_evaluation_dashboard.png
model/lstm_model.pth
model/gb_model.joblib
model/model_readable_report.md
```

The `runs/` directory is ignored by git because it can contain large generated
models and experiment artifacts. The curated evidence copied into `docs/` is
tracked.

## Reproducibility

- Source and all evaluation code are under the MIT License (see LICENSE).
- Continuous integration compiles the ML scripts, runs the full pytest
  suite, and executes an auto-benchmark smoke test on every push (see
  .github/workflows/ci.yml).
- See CITATION.cff for citation metadata. This repository is archived on
  Zenodo: <PLACEHOLDER — DO NOT INVENT: DOI, filled in after Zenodo
  archival by the human author>.
- See CHANGELOG.md for a dated history of methodology changes, including
  the spike false-positive fix pass and the additions in this document.

## Important Limitations

- This is not a foundation model, transformer forecaster, or autonomous
  network controller.
- The attention model is not a Transformer or Temporal Fusion Transformer.
- The hybrid LSTM + Gradient Boosting architecture follows established
  patterns in the forecasting literature; it is not claimed as a novel
  architecture. The candidate-selection meta-policy and its measured effect
  (see Ablation Studies) are the project's differentiating contribution.
- Prediction intervals are approximate (residual-normal 95% bands), not
  full probabilistic/conformal forecasts.
- The `packet_loss_pct` column derived from CICIDS2017 is a proxy built
  from retransmission/flag-based fields, not a directly measured network
  loss statistic — see ml/load_public_benchmark.py docstring.
- Synthetic and public-benchmark performance do not guarantee live-network
  performance. The >=90% quality gate has not yet been reached on any
  checked-in real-data run.
- No human/operator evaluation has been performed; only automated metrics.

## Testing

```powershell
python -m compileall ml scripts runners
python -m pytest
```

CI also compiles the ML scripts, runs tests, and executes an auto-benchmark
smoke check.

## Troubleshooting

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If ONNX export fails with `ModuleNotFoundError: No module named 'onnxscript'`:

```powershell
python -m pip install onnx onnxscript
```

If a run fails after training but before dashboards, rerun the final steps:

```powershell
python ml\visualize.py --data runs\<run_folder>\raw_data\telemetry.csv --output-dir runs\<run_folder>
python ml\evaluate_model.py --run-dir runs\<run_folder>
python ml\export_model_report.py --run-dir runs\<run_folder>
```

## Community

- See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development workflow.
- See [.github/CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for community expectations.
- See [.github/SECURITY.md](.github/SECURITY.md) for vulnerability reporting.
- See [.github/SUPPORT.md](.github/SUPPORT.md) for help and support channels.
- This project is released under the [MIT License](LICENSE).

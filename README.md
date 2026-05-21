# AI-Driven Network Design and Traffic Prediction

This project builds a network telemetry prediction pipeline around synthetic,
Kaggle, and live ContainerLab data. The current default trainer is a hybrid
ensemble: a stacked LSTM learns temporal sequence behavior, while Gradient
Boosting learns engineered lag, rolling-window, and spike features. The two
models are blended into one forecast for traffic, latency, and packet loss.

The pipeline can run fully offline with synthetic telemetry, train from a local
CSV, pull a Kaggle network dataset, or collect live telemetry from a deployed
ContainerLab spine-leaf topology.

## What Is Included

- Synthetic telemetry generator for offline trials.
- Kaggle loader for `crawford/computer-network-traffic`.
- Live ContainerLab collector for traffic, latency, and packet loss.
- Hybrid ensemble trainer in `ml/enhanced_train.py`.
- Original LSTM trainer in `ml/train_model.py` kept as a reference.
- Dataset-optimized tree trainer in `ml/train_kaggle_model.py`.
- ONNX export for the LSTM component.
- Dashboards for forecasts, training loss, errors, correlations, and spikes.
- Model evaluation reports, readable summaries, and baseline comparisons.

## Current Default Model

`run.ps1 synthetic`, `run.ps1 kaggle`, `run.ps1 live`, and `run.ps1 train`
currently call `ml/enhanced_train.py`.

That trainer creates:

- `model/lstm_model.pth`: PyTorch LSTM sequence model.
- `model/lstm_model.onnx`: ONNX export of the LSTM component.
- `model/gb_model.joblib`: Gradient Boosting spike/tabular component.
- `results/predictions.csv`: blended ensemble predictions.
- `results/actuals.csv`: matching actual values.
- `results/prediction_intervals.csv`: residual-based 95% forecast bands.
- `results/train_losses.csv`: LSTM training and validation losses.
- `json/metrics.json`: training settings and model metrics.
- `json/scaler_params.json`: input and target scaler metadata.

The trainer starts from `65%` Gradient Boosting and `35%` LSTM, then tunes the
blend on a chronological validation slice. This leans on Gradient Boosting for
spike capture while still using the LSTM for time-window behavior.

The current neural component is a stacked LSTM with temporal attention,
LayerNorm, and a residual connection from the final hidden state.
Older run metadata may contain `attention_lstm` or `hybrid_lstm_gradient_boosting`
from earlier experiments, but new hybrid runs are labeled
`hybrid_attention_lstm_gradient_boosting`.
The trainer uses CUDA automatically when PyTorch can see a GPU, and otherwise
falls back to CPU.

## How The Pieces Connect

The project has three main data paths:

- Synthetic data: `ml/generate_data.py` creates offline telemetry for demos and
  repeatable experiments.
- Kaggle data: `ml/load_kaggle_data.py` loads external network traffic rows for
  larger dataset-style trials.
- Live data: `scripts/collect_telemetry.py` samples a running ContainerLab
  topology.

Those paths all produce the same core columns:

```text
timestamp,traffic_mbps,latency_ms,packet_loss_pct
```

The runner scripts connect the pipeline:

- `run.ps1` is the main Windows entrypoint.
- `run.sh` is the Linux/WSL entrypoint.
- `ml/enhanced_train.py` trains the current hybrid model.
- `ml/visualize.py` creates the prediction dashboard and spike summary.
- `ml/evaluate_model.py` compares the model against simple baselines.
- `ml/export_model_report.py` writes human-readable model summaries.
- `ml/compare_sequence_models.py` runs controlled LSTM/GRU/attention ablations.

Why each part matters:

- The LSTM branch learns temporal sequence behavior from lookback windows.
- The Gradient Boosting branch uses lag and rolling-window features that are
  often strong for abrupt tabular spike patterns.
- The ensemble weight search uses validation data to choose the blend instead
  of assuming one fixed weight is always best.
- The dashboards make model behavior inspectable instead of hiding everything
  behind one score.
- The ablation runner helps prove whether attention or other sequence changes
  actually help on the current dataset.

## Current Limitations

- The neural model has temporal attention, but it is not a Transformer or a
  Temporal Fusion Transformer.
- This is an experimentation pipeline, not a proven production deployment.
- Quality depends on the dataset, split, spike frequency, and dashboard
  sensitivity. Trust the generated evaluation artifacts for each run rather
  than any fixed README quality claim.
- Prediction intervals are residual-based approximations, not a full
  probabilistic forecasting model.
- The LSTM scalers are fit on the chronological training slice only; validation
  and test windows are transformed with those fitted scalers.

## Project Layout

```text
ai_network_project/
  containerlab/topology.clab.yml
  configs/frr/*/frr.conf
  scripts/collect_telemetry.py
  scripts/cleanup_runs.py
  ml/generate_data.py
  ml/enhanced_train.py
  ml/train_model.py
  ml/train_kaggle_model.py
  ml/load_kaggle_data.py
  ml/visualize.py
  ml/evaluate_model.py
  ml/export_model_report.py
  run.ps1
  run.sh
  requirements.txt
```

## Setup

From PowerShell:

```powershell
cd "C:\Users\siddh\Downloads\ai_network_project (1)\ai_network_project"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install -r requirements.txt
```

If pip has certificate issues on Windows, install the ONNX exporter dependency
with trusted PyPI hosts:

```powershell
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org onnxscript
```

## Quick Trials

Synthetic trial with 2000 generated samples and 40 LSTM epochs:

```powershell
.\run.ps1 synthetic -Samples 2000 -Epochs 40
```

Kaggle trial with 5000 rows and 60 epochs:

```powershell
.\run.ps1 kaggle -Samples 5000 -Epochs 60
```

Fast smoke test:

```powershell
.\run.ps1 synthetic -Samples 300 -Epochs 3
```

Skip dependency installation after packages are already installed:

```powershell
.\run.ps1 synthetic -Samples 2000 -Epochs 40 -SkipInstall
```

## Sample End-To-End Trial

This trial uses the checked-in sample data at `ml/telemetry.csv`, trains the
hybrid model, creates graphs, and writes evaluation tables.

```powershell
python ml\enhanced_train.py --data ml\telemetry.csv --output-dir runs\sample_hybrid_trial --epochs 40 --device auto
python ml\visualize.py --data runs\sample_hybrid_trial\raw_data\telemetry.csv --output-dir runs\sample_hybrid_trial --sensitivity 1.3
python ml\evaluate_model.py --run-dir runs\sample_hybrid_trial
python ml\export_model_report.py --run-dir runs\sample_hybrid_trial
```

Inspect these outputs:

- Data used: `runs/sample_hybrid_trial/raw_data/telemetry.csv`
- Prediction graph: `runs/sample_hybrid_trial/images/traffic_prediction_dashboard.png`
- Evaluation graph: `runs/sample_hybrid_trial/images/model_evaluation_dashboard.png`
- Metrics: `runs/sample_hybrid_trial/json/evaluation_summary.json`
- Baseline comparison: `runs/sample_hybrid_trial/results/evaluation_baselines.csv`
- Spike metrics: `runs/sample_hybrid_trial/results/evaluation_spikes.csv`
- Prediction intervals: `runs/sample_hybrid_trial/results/prediction_intervals.csv`

Do not treat one trial as proof of model superiority. Compare multiple runs and
use the baseline/ablation outputs before making claims about improvement.

Every run creates a timestamped folder under `runs/`, for example:

```text
runs/20260521_103546_synthetic/
```

The `runs/` folder is ignored by git because it can contain large generated
models, images, and experiment outputs.

## Training From Local Telemetry

Save a CSV at `ml/telemetry.csv` with these columns:

```text
timestamp,traffic_mbps,latency_ms,packet_loss_pct
```

Then run:

```powershell
.\run.ps1 train -Epochs 80
```

Manual equivalent:

```powershell
python ml\enhanced_train.py --data ml\telemetry.csv --output-dir runs\hybrid_manual --epochs 80
python ml\visualize.py --data runs\hybrid_manual\raw_data\telemetry.csv --output-dir runs\hybrid_manual --sensitivity 1.3
python ml\evaluate_model.py --run-dir runs\hybrid_manual
python ml\export_model_report.py --run-dir runs\hybrid_manual
```

## Outputs

Each complete run contains:

- `raw_data/telemetry.csv`
- `results/predictions.csv`
- `results/actuals.csv`
- `results/prediction_intervals.csv`
- `results/train_losses.csv`
- `results/evaluation_comparison.csv`
- `results/evaluation_baselines.csv`
- `results/evaluation_spikes.csv`
- `json/metrics.json`
- `json/spike_summary.json`
- `json/scaler_params.json`
- `json/evaluation_summary.json`
- `json/model_metadata.json`
- `json/model_readable_summary.json`
- `images/traffic_prediction_dashboard.png`
- `images/model_evaluation_dashboard.png`
- `model/lstm_model.pth`
- `model/lstm_model.onnx`
- `model/gb_model.joblib`
- `model/model_readable_report.md`
- `model/model_weights_summary.csv`
- `model/model_gate_summary.csv`

Binary files such as `.pth`, `.onnx`, and `.joblib` are not meant to be read in
a text editor. Use the JSON, CSV, dashboard PNGs, and Markdown report for
inspection.

## Model Notes

The hybrid trainer uses:

- LSTM lookback sequence length: `96`
- LSTM hidden size: `128`
- LSTM layers: `2`
- Batch size: `32`
- Temporal attention: additive attention over LSTM time steps
- Stability: residual final-state connection plus `LayerNorm`
- LSTM epochs: controlled by `-Epochs` or `--epochs`
- Chronological split: `--train-ratio 0.70`, validation until `--test-ratio 0.82`, test after that
- Early stopping: validation-loss patience coordinated with the LR scheduler
- Learning-rate scheduler: `ReduceLROnPlateau`, factor `0.5`, patience `8`, min LR `1e-5`
- Gradient Boosting estimators: `300`
- Gradient Boosting learning rate: `0.05`
- Gradient Boosting max depth: `5`
- Ensemble weights: validation-tuned from a `0.65` Gradient Boosting, `0.35` LSTM starting point
- Packet loss transform: `log1p` during neural training, `expm1` after neural prediction
- GPU support: automatic CUDA use when available, or explicit `--device cpu` / `--device cuda`
- Uncertainty output: residual-normal 95% intervals in `results/prediction_intervals.csv`

The dashboard spike thresholds are computed as:

```text
training mean + sensitivity * training standard deviation
```

Change sensitivity when visualizing:

```powershell
python ml\visualize.py --data runs\hybrid_manual\raw_data\telemetry.csv --output-dir runs\hybrid_manual --sensitivity 1.3
```

## Baseline Comparison

Use the ablation runner before claiming an architecture improvement. It trains
the same split across LSTM, GRU, mean-pooling LSTM, and attention-LSTM variants,
then writes MSE, MAE, SMAPE, spike F1, and epoch counts.

```powershell
python ml\compare_sequence_models.py --data ml\telemetry.csv --output-dir runs\sequence_comparison --epochs 40
```

Outputs:

- `results/sequence_model_comparison.csv`
- `json/sequence_model_comparison.json`

## Kaggle And Dataset Modes

Standard Kaggle hybrid run:

```powershell
.\run.ps1 kaggle -Samples 5000 -Epochs 60
```

Dataset-optimized Gradient Boosting-only run:

```powershell
.\run.ps1 kaggle_opt -Samples 5000 -SkipInstall
```

For an arbitrary local CSV:

```powershell
.\run.ps1 dataset_opt -SkipInstall
```

`dataset_opt` expects `ml/telemetry.csv` to exist.

## Live ContainerLab Workflow

ContainerLab is Linux-focused. On Windows, use WSL2, a Linux VM, or a Linux host
with Docker and ContainerLab installed.

From PowerShell:

```powershell
.\run.ps1 deploy
.\run.ps1 live -Samples 120 -Interval 10 -Epochs 80
.\run.ps1 destroy
```

From Linux or WSL:

```bash
bash run.sh deploy
bash run.sh live --samples 120 --interval 10 --epochs 80
bash run.sh destroy
```

The live collector:

1. Checks for running `clab-ai-traffic-lab-*` containers.
2. Reads interface byte counters from `/proc/net/dev`.
3. Sends ping probes to create measurable traffic.
4. Measures latency and packet loss.
5. Writes run-ready telemetry into `runs/<timestamp>_live/raw_data/telemetry.csv`.

## Troubleshooting

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If ONNX export fails with `ModuleNotFoundError: No module named 'onnxscript'`:

```powershell
python -m pip install onnx onnxscript
```

If pip has certificate errors:

```powershell
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org onnx onnxscript
```

If a run fails after training but before dashboards, rerun the last steps on the
same run folder:

```powershell
python ml\visualize.py --data runs\<run_folder>\raw_data\telemetry.csv --output-dir runs\<run_folder>
python ml\evaluate_model.py --run-dir runs\<run_folder>
python ml\export_model_report.py --run-dir runs\<run_folder>
```

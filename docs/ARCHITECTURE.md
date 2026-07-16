# NetFlow-Forecaster Architecture & Data Flow

## Quick Overview

**Purpose**: Reproducible benchmarking framework for network telemetry forecasting with automated candidate selection.

**Key Components**:
- **Data Input**: Synthetic, CSV, Kaggle, CICIDS2017 public benchmark
- **Preprocessing**: Feature engineering, chronological splits, spike detection
- **Training**: Hybrid LSTM + Gradient Boosting with spike-weighted loss
- **Meta-Policy**: Persistent bandit-style candidate selection across runs
- **Evaluation**: Dashboard generation, statistical significance testing
- **Output**: Predictions, metrics, model artifacts, evaluation summaries

---

## Data Flow Pipeline

### Step 1: Data Ingestion

| Source | Handler | Output | Rows |
|--------|---------|--------|------|
| **Synthetic** | `ml/generate_data.py` | `runs/<timestamp>_synthetic/raw_data/telemetry.csv` | 720 (configurable) |
| **Local CSV** | `ml/telemetry.csv` (checked-in) | Copied to run dir | 3000 |
| **Kaggle Flow** | `ml/load_kaggle_data.py` | `runs/<timestamp>_kaggle/raw_data/telemetry.csv` | 8000+ |
| **CICIDS2017** | `ml/load_public_benchmark.py` | `runs/<timestamp>_public_benchmark/raw_data/telemetry.csv` | 5000 (configurable) |

**Schema** (immutable across all sources):
```
timestamp (datetime),
traffic_mbps (float),
latency_ms (float),
packet_loss_pct (float)
```

**Key Constraint**: All timestamps are sorted ascending, chronological splits never mixed.

---

### Step 2: Feature Preparation & Splitting

**Handler**: `ml/train_model.py::create_sequences()`; Gradient Boosting alignment is handled in `ml/enhanced_train.py`.

**Process**:
1. Read telemetry CSV (timestamp order preserved)
2. Split chronologically: 70% train | 15% validation | 15% test
3. Generate LSTM sequences (lookback window, default 20 timesteps)
4. Create Gradient Boosting features (lag values, rolling stats)
5. Compute spike thresholds per feature (used for loss weighting & detection)

**Output Format**:
- LSTM: `(seq_len=20, features=3)` sequences
- GB: Lag + rolling features for each timestep
- Spikes: Boolean masks for anomalies

---

### Step 3: Training (Hybrid Ensemble)

**Handlers**:
- **LSTM**: `ml/enhanced_train.py` (attention-based sequence model)
- **Gradient Boosting**: Fused with LSTM embeddings
- **Candidate Selection**: `ml/trainer_tournament.py` (generates configs)

**Candidates** (10 variants per tournament):
- `hybrid_aggressive` - high LSTM regularization
- `hybrid_low_quantile` - low error percentile focus
- `hybrid_default` - balanced
- `hybrid_gb_heavy` - GB-weighted ensemble
- `hybrid_r2_recovery` - R² optimization
- `hybrid_short_seq` - shorter lookback
- `gb_spike`, `gb_spike_deep` - GB-only variants
- `specialist` models (per-feature)

**Loss Function**:
```
spike_weighted_loss = MSE + spike_weight × (feature_spike_multipliers × anomaly_penalty)
```

**Meta-Policy Integration** (`ml/meta_policy.py`):
- Reads persistent experience store (`runs/.experience/memory.jsonl`)
- Ranks candidates using UCB exploration
- Selects next candidate if quality gate not reached
- Updates experience with new run results

**Quality Gate**:
Passes if: `(MAE_traffic < X%) AND (MAE_latency < Y%) AND (spike_f1_traffic > Z%)`

---

### Step 4: Evaluation & Testing

**Handlers**:
- `ml/visualize.py` - Creates prediction/evaluation dashboards (PNG)
- `ml/evaluate_model.py` - Computes metrics vs. persistence baseline
- `ml/significance_tests.py` - Statistical rigor (Diebold-Mariano, paired t-test)
- `ml/export_model_report.py` - Human-readable model summary (MD)

**Output Files** (per `runs/<run_id>/`):
```
results/
  predictions.csv          # Model predictions
  actuals.csv             # Ground truth  
  prediction_intervals.csv # 95% confidence bands
  train_losses.csv        # Per-epoch losses
  evaluation_comparison.csv # vs. baseline (persistence)
  evaluation_baselines.csv # Baseline stats
  evaluation_spikes.csv    # Spike metrics (precision, recall, F1)
  significance_tests.csv   # Statistical p-values (DM, t-test)

images/
  traffic_prediction_dashboard.png
  model_evaluation_dashboard.png

json/
  metrics.json
  evaluation_summary.json
  model_metadata.json
  model_readable_summary.json
  spike_summary.json

model/
  lstm_model.pth           # PyTorch LSTM weights
  gb_model.joblib          # Scikit-learn GB model
  model_readable_report.md # Human summary
```

---

### Step 5: Ablation Studies (Isolated Contribution Measurement)

#### Ablation 1: Candidate-Selection Strategy
**File**: `ml/ablation_selection.py`

Measures: How much does the meta-policy help vs. fixed/random ordering?

**Strategies**:
1. `fixed_order` - Always try candidates in default sequence
2. `random_order` - Shuffle candidate order each run
3. `meta_policy` - Use persisted UCB rankings (default)

**Output**: `<output-dir>/ablation_selection_summary.json`
```json
{
  "strategy": {
    "synthetic": {
      "attempts_to_gate": N,
      "best_quality": X%,
      "mean_quality": Y%,
      "std_quality": Z%
    }
  }
}
```

#### Ablation 2: Spike-Weighted Loss Configuration
**File**: `ml/ablation_spike_loss.py`

Measures: Does spike weighting actually improve spike detection?

**Conditions** (3 seeds each):
1. `no_spike_weighting` - Plain MSE (baseline)
2. `uniform_spike_weighting` - Fixed spike weight (current default)
3. `differentiated_spike_weighting` - Per-feature multipliers (0.0, 1.8, 2.5)

**Output**: `<output-dir>/ablation_spike_loss_summary.json`
```json
{
  "condition": {
    "traffic_mbps": {
      "mean_mae": X,
      "std_mae": Y,
      "mean_spike_f1": A,
      "std_spike_f1": B
    }
  }
}
```

---

## Directory Structure

```
ai_network_project/
├── ml/
│   ├── generate_data.py              # Synthetic data creation
│   ├── load_kaggle_data.py           # Kaggle dataset loader
│   ├── load_public_benchmark.py      # CICIDS2017 loader (+ proxy derivation)
│   ├── enhanced_train.py             # LSTM + GB training
│   ├── trainer_tournament.py         # Candidate generation + random_order()
│   ├── auto_benchmark.py             # Benchmark loop + meta-policy calls
│   ├── meta_policy.py                # rank_candidates() with UCB + experience store
│   ├── experience_store.py           # Persistent memory (memory.jsonl)
│   ├── evaluate_model.py             # Evaluation + significance tests auto-call
│   ├── significance_tests.py         # DM + t-test implementations
│   ├── predict.py                    # Single-step forecasting
│   ├── metrics_utils.py              # Quality gates, spike detection
│   ├── spike_loss.py                 # Spike-weighted MSE loss
│   ├── visualize.py                  # Dashboard generation
│   ├── export_model_report.py        # Readable model summary
│   ├── ablation_selection.py         # Ablation 1: candidate ordering
│   ├── ablation_spike_loss.py        # Ablation 2: loss configuration
│   ├── telemetry.csv                 # Local benchmark dataset (3000 rows)
│   └── telemetry_*.csv               # Trial/profile CSVs
├── tests/
│   ├── test_significance_tests.py    # Diebold-Mariano + t-test tests
│   ├── test_ablation_selection.py    # Policy isolation verification
│   ├── test_ablation_spike_loss.py   # Ablation output structure
│   ├── test_load_public_benchmark.py # CICIDS2017 column mapping
│   └── (20+ other test files)        # Feature-level tests
├── runners/
│   ├── run.ps1                       # PowerShell entry point (Windows)
│   ├── run.sh                        # Bash entry point (Linux/macOS)
│   └── run.py                        # Shared Python orchestrator
├── containerlab/
│   ├── topology.clab.yml             # Lab network definition
│   └── clab-ai-traffic-lab/          # Generated lab state
├── docs/
│   ├── ARCHITECTURE.md               # This file
│   ├── results/
│   │   ├── paper_evidence_manifest.json
│   │   ├── ablation_selection_summary.json
│   │   ├── ablation_spike_loss_summary.json
│   │   ├── *_evaluation_summary.json
│   │   └── images/                   # Dashboard screenshots
│   └── netflow_fixes_spec_v2.md      # Historical methodology notes
├── runs/                             # Execution artifacts (git-ignored)
│   └── <timestamp>_<mode>/
│       ├── raw_data/
│       ├── results/
│       ├── images/
│       ├── json/
│       └── model/
├── CITATION.cff                      # Citation metadata for Zenodo
├── CHANGELOG.md                      # Dated methodology history
├── README.md                         # User-facing documentation
├── requirements.txt                  # Pinned dependencies
└── LICENSE                           # MIT License
```

---

## Key Design Principles

### 1. **Chronological Integrity**
- Timestamps always sorted ascending
- Train/val/test split is temporal, never shuffled
- No leakage from future to past

### 2. **Reproducibility**
- All randomness seeded (default: 7)
- Dependencies pinned in requirements.txt
- Zenodo archival with DOI (placeholder: see CITATION.cff)
- CHANGELOG documents methodology changes

### 3. **Transparent Quality Measurement**
- Ablations quantify actual contributions, not asserted
- Significance tests (DM + t-test) on every run
- Quality gates are explicit, not hidden in aggregate metrics

### 4. **Persistent Meta-Policy**
- Experience stored in `runs/.experience/memory.jsonl`
- UCB exploration balances known-good vs. exploration
- Never polluted by ablation test runs (policy isolation rule)

### 5. **Modular Architecture**
- Each script is independently callable (CLI)
- Tests are fast (< 60s total per suite)
- No hard dependencies on external APIs (kagglehub fallback provided)

---

## Workflow Examples

### Quick Start: Synthetic Demo
```bash
./runners/run.sh synthetic --epochs 60
# → Creates 720 synthetic rows
# → Trains hybrid model
# → Generates dashboards & metrics
# → Output: runs/<timestamp>_synthetic/
```

### Benchmark with Meta-Policy
```bash
./runners/run.sh benchmark --target-quality 90 --max-attempts 12
# → Loads ml/telemetry.csv
# → Runs 12 candidate attempts (ordered by meta-policy)
# → Stops when >=90% quality reached
# → Syncs best run to docs/generic_*
```

### Public Benchmark (CICIDS2017)
```bash
./runners/run.sh public_benchmark --samples 5000 --epochs 60
# → Downloads CICIDS2017 via kagglehub
# → Derives packet_loss_pct proxy from flags
# → Trains & evaluates on public data
# → Enables comparison to other published results
```

### Ablation: Does Spike Weighting Help?
```bash
python ml/ablation_spike_loss.py --data ml/telemetry.csv --output-dir runs/ablation_spike_loss
# → Tests 3 loss configurations × 3 seeds
# → Outputs: runs/ablation_spike_loss/
# → Summary: runs/ablation_spike_loss/ablation_spike_loss_summary.json
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Fatal error (missing file, invalid config, etc.) |
| 130 | User interrupted (Ctrl+C) |
| 2+ | Command-specific errors (from subprocesses) |

---

## Testing

**Run all tests**:
```bash
python -m pytest tests/ -v
```

**Expected**: 28 tests, ~45s runtime

**Warnings**: RuntimeWarning from numpy correlation with low-variance features is expected (not an error).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Python was not found" | Install Python 3.10+; re-open terminal |
| "No module named 'torch'" | Run `pip install -r requirements.txt` |
| "No ml/telemetry.csv" | Use synthetic/kaggle modes or add CSV to ml/ |
| "NaN in significance tests" | Update to latest scipy; see test_significance_tests.py |
| "Policy isolation test fails" | Don't write to runs/.experience during ablations |
| "Kaggle download fails" | Provide --local-csv fallback or set kagglehub credentials |

---

## Future Improvements

- [ ] GPU acceleration (CUDA auto-detection)
- [ ] Multi-GPU data parallel training
- [ ] Hyperparameter grid search via meta-policy
- [ ] Live NetFlow ingestion (currently placeholder)
- [ ] Foundation model fine-tuning branch
- [ ] Online learning / streaming predictions

# Changelog

## [Unreleased]
### Added
- Candidate-selection ablation study (ml/ablation_selection.py) comparing the
  meta-policy against fixed-order and random-order baselines.
- Spike-weighted loss ablation (ml/ablation_spike_loss.py), 3 conditions x 3 seeds.
- Statistical significance testing (ml/significance_tests.py): paired t-test and
  Diebold-Mariano test, model vs. persistence baseline, per feature per run.
- Public benchmark dataset loader (ml/load_public_benchmark.py) for CICIDS2017.
- CITATION.cff and this changelog.

## [2026-05-26] - Spike fix pass (from docs/netflow_fixes_spec_v2.md)
### Changed
- Added a false-positive penalty term to optimize_ensemble_weights() in
  ml/enhanced_train.py, heavier for latency/packet-loss than for traffic.
- Made spike boosting in ml/calibrate_predictions.py conditional on not already
  over-firing.
- Adjusted hybrid_r2_recovery candidate multipliers in ml/trainer_tournament.py
  to reduce latency spike over-prediction.
- Replaced the packet-loss log1p/expm1 transform with sqrt/square in
  ml/train_model.py to preserve spike contrast.

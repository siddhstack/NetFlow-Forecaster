"""Statistical significance testing: model vs. persistence baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]


def diebold_mariano(
    errors_model: np.ndarray,
    errors_baseline: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.
    
    errors_model / errors_baseline: 1D arrays of forecast errors
    (actual - predicted), paired at the same timesteps, same length.
    h: forecast horizon (1 for one-step-ahead forecasts used throughout
    this project).
    power: 2 for squared-error loss, 1 for absolute-error loss.
    
    Returns (dm_statistic, p_value) using the Harvey-Leybourne-Newbold
    (1997) small-sample correction and a Student-t reference distribution
    with (n - 1) degrees of freedom.
    """
    e1 = np.asarray(errors_model, dtype=float)
    e2 = np.asarray(errors_baseline, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError("errors_model and errors_baseline must have the same shape")
    n = e1.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 paired observations")
    
    d = (np.abs(e1) ** power) - (np.abs(e2) ** power)
    d_bar = float(d.mean())
    
    # Newey-West style long-run variance with (h - 1) lags.
    var_d = float(np.var(d, ddof=0))
    for lag in range(1, h):
        if n > lag:
            cov = float(np.cov(d[lag:], d[:-lag], ddof=0)[0, 1])
            var_d += 2.0 * (1.0 - lag / h) * cov
    
    var_d_bar = var_d / n
    if var_d_bar <= 0:
        return 0.0, 1.0
    
    dm_stat = d_bar / np.sqrt(var_d_bar)
    
    # Harvey, Leybourne & Newbold (1997) small-sample correction.
    correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm_stat_corrected = dm_stat * correction
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))
    
    return float(dm_stat_corrected), float(p_value)


def paired_t_test(errors_model: np.ndarray, errors_baseline: np.ndarray) -> tuple[float, float]:
    """Paired t-test on absolute errors (model vs. persistence baseline).
    
    If the paired differences have zero variance (e.g., identical distributions),
    returns (0.0, 1.0) indicating no significant difference.
    """
    abs_model = np.abs(np.asarray(errors_model, dtype=float))
    abs_baseline = np.abs(np.asarray(errors_baseline, dtype=float))
    
    # If all differences are zero, no variance => p=1.0 (not significant)
    diff = abs_model - abs_baseline
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    
    statistic, p_value = stats.ttest_rel(abs_model, abs_baseline)
    
    # Handle NaN from degenerate cases
    if np.isnan(p_value):
        return 0.0, 1.0
    
    return float(statistic), float(p_value)


def run_for_dir(run_dir: Path, alpha: float = 0.05) -> list[dict]:
    """Run significance tests for a run directory."""
    predictions_path = run_dir / "results" / "predictions.csv"
    actuals_path = run_dir / "results" / "actuals.csv"
    
    if not predictions_path.exists() or not actuals_path.exists():
        return []
    
    predictions = pd.read_csv(predictions_path)
    actuals = pd.read_csv(actuals_path)
    
    rows: list[dict] = []
    for feature in FEATURES:
        if feature not in actuals.columns or feature not in predictions.columns:
            continue
        
        actual = actuals[feature].to_numpy(dtype=float)
        model_errors = actual - predictions[feature].to_numpy(dtype=float)
        
        # Persistence baseline: shift actual by 1
        persistence_pred = actual.copy()
        persistence_pred[1:] = actual[:-1]
        baseline_errors = actual - persistence_pred
        
        dm_stat, dm_p = diebold_mariano(model_errors, baseline_errors)
        t_stat, t_p = paired_t_test(model_errors, baseline_errors)
        
        rows.append({
            "feature": feature,
            "diebold_mariano_stat": dm_stat,
            "diebold_mariano_p_value": dm_p,
            "dm_significant": int(dm_p < alpha),
            "paired_t_stat": t_stat,
            "paired_t_p_value": t_p,
            "t_significant": int(t_p < alpha),
        })
    
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory to test.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    rows = run_for_dir(run_dir, args.alpha)
    
    out_path = run_dir / "results" / "significance_tests.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

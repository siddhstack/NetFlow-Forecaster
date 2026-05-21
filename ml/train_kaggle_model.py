"""Train a dataset-specific spike-aware model.

This script is intentionally separate from train_model.py. It is for dataset
CSV files that already have the standard telemetry columns. It keeps the data
chronological, builds lag/rolling features, oversamples spike windows, and
trains one regressor per output metric. It writes the same artifact names used
by the dashboard/evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from run_layout import artifact_path, ensure_run_layout
from train_model import FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Kaggle telemetry CSV.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    parser.add_argument("--lookback", type=int, default=24, help="Lag window size.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Chronological train split.")
    parser.add_argument("--spike-std", type=float, default=1.2, help="Mean + std multiplier for spike labels.")
    parser.add_argument("--spike-oversample", type=int, default=0, help="Extra copies for spike training windows.")
    parser.add_argument("--model", choices=["gbr", "rf"], default="gbr", help="Regressor family.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def load_telemetry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def add_features(df: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, list[str]]:
    base = df.copy()
    frames = [base]
    feature_cols: list[str] = []

    if "timestamp" in base.columns:
        ts = pd.to_datetime(base["timestamp"], errors="coerce")
        hour = ts.dt.hour.fillna(0).astype(int)
        day_of_week = ts.dt.dayofweek.fillna(0).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
                    "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
                    "weekday_sin": np.sin(2.0 * np.pi * day_of_week / 7.0),
                    "weekday_cos": np.cos(2.0 * np.pi * day_of_week / 7.0),
                    "is_weekend": (day_of_week >= 5).astype(float),
                }
            )
        )
        feature_cols.extend(["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend"])

    for feature in FEATURES:
        feature_data: dict[str, pd.Series] = {}
        for lag in range(1, lookback + 1):
            col = f"{feature}_lag_{lag}"
            feature_data[col] = base[feature].shift(lag)
            feature_cols.append(col)
        for window in (3, 6, 12, 24):
            if window <= lookback:
                mean_col = f"{feature}_roll_mean_{window}"
                std_col = f"{feature}_roll_std_{window}"
                max_col = f"{feature}_roll_max_{window}"
                shifted = base[feature].shift(1)
                feature_data[mean_col] = shifted.rolling(window).mean()
                feature_data[std_col] = shifted.rolling(window).std()
                feature_data[max_col] = shifted.rolling(window).max()
                feature_cols.extend([mean_col, std_col, max_col])
        delta_col = f"{feature}_delta_1"
        feature_data[delta_col] = base[feature].diff().shift(1)
        feature_cols.append(delta_col)
        frames.append(pd.DataFrame(feature_data))

    work = pd.concat(frames, axis=1).copy()
    work = work.dropna(subset=feature_cols + FEATURES).reset_index(drop=True)
    return work, feature_cols


def spike_thresholds(y_train: np.ndarray, multiplier: float) -> dict[str, float]:
    return {
        feature: float(y_train[:, idx].mean() + multiplier * y_train[:, idx].std(ddof=0))
        for idx, feature in enumerate(FEATURES)
    }


def spike_mask(y: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    mask = np.zeros(len(y), dtype=bool)
    for idx, feature in enumerate(FEATURES):
        mask |= y[:, idx] > thresholds[feature]
    return mask


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    scale = np.maximum(target.std(axis=0, ddof=0), 1e-9)
    return float(np.mean(((prediction - target) / scale) ** 2))


def staged_predictions(model: MultiOutputRegressor, x_test: np.ndarray) -> list[np.ndarray]:
    if not all(hasattr(estimator, "staged_predict") for estimator in model.estimators_):
        return []
    staged_by_target = [list(estimator.staged_predict(x_test)) for estimator in model.estimators_]
    stage_count = min(len(stages) for stages in staged_by_target)
    return [
        np.column_stack([target_stages[stage_idx] for target_stages in staged_by_target])
        for stage_idx in range(stage_count)
    ]


def staged_loss_curve(model: MultiOutputRegressor, x_test: np.ndarray, y_test: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, int]:
    final_predictions = model.predict(x_test)
    if not all(hasattr(estimator, "staged_predict") for estimator in model.estimators_):
        return (
            pd.DataFrame(
                {
                    "epoch": [1],
                    "mse_loss": [normalized_mse(final_predictions, y_test)],
                    "raw_mse_loss": [float(np.mean((final_predictions - y_test) ** 2))],
                }
            ),
            final_predictions,
            1,
        )

    all_stage_predictions = staged_predictions(model, x_test)
    rows = []
    for stage_idx, prediction in enumerate(all_stage_predictions):
        rows.append(
            {
                "epoch": stage_idx + 1,
                "mse_loss": normalized_mse(prediction, y_test),
                "raw_mse_loss": float(np.mean((prediction - y_test) ** 2)),
            }
        )
    loss_df = pd.DataFrame(rows)
    best_idx = int(loss_df["mse_loss"].idxmin())
    return loss_df, all_stage_predictions[best_idx], int(loss_df.loc[best_idx, "epoch"])


def tree_count(model: MultiOutputRegressor) -> int:
    total = 0
    for estimator in model.estimators_:
        if hasattr(estimator, "estimators_"):
            total += int(np.asarray(estimator.estimators_).size)
    return total


def estimator_count(model: MultiOutputRegressor) -> int:
    counts = [int(getattr(estimator, "n_estimators_", getattr(estimator, "n_estimators", 1))) for estimator in model.estimators_]
    return int(round(float(np.mean(counts)))) if counts else 1


def make_model(name: str, seed: int):
    if name == "rf":
        base = RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        base = GradientBoostingRegressor(
            n_estimators=240,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            subsample=0.8,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4,
            random_state=seed,
        )
    return MultiOutputRegressor(base)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    ensure_run_layout(output_dir)

    raw_df = load_telemetry(data_path)
    modeled_df, feature_cols = add_features(raw_df, args.lookback)
    if len(modeled_df) < 50:
        raise ValueError("Not enough rows after lag feature creation. Use more samples or a shorter lookback.")

    x = modeled_df[feature_cols].to_numpy(dtype=float)
    y = modeled_df[FEATURES].to_numpy(dtype=float)
    split = max(1, min(len(x) - 1, int(len(x) * args.train_split)))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    thresholds = spike_thresholds(y_train, args.spike_std)
    spikes = spike_mask(y_train, thresholds)
    if spikes.any() and args.spike_oversample > 0:
        x_spike = np.repeat(x_train[spikes], args.spike_oversample, axis=0)
        y_spike = np.repeat(y_train[spikes], args.spike_oversample, axis=0)
        x_train_aug = np.vstack([x_train, x_spike])
        y_train_aug = np.vstack([y_train, y_spike])
    else:
        x_train_aug, y_train_aug = x_train, y_train

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_aug)
    x_test_scaled = scaler.transform(x_test)
    model = make_model(args.model, args.seed)
    model.fit(x_train_scaled, y_train_aug)
    loss_df, best_predictions, best_stage = staged_loss_curve(model, x_test_scaled, y_test)
    predictions = np.clip(best_predictions, 0.0, None)

    metrics = {
        "training": {
            "model": f"DatasetSpikeAware-{args.model}",
            "lookback": args.lookback,
            "spike_std": args.spike_std,
            "spike_thresholds": thresholds,
            "spike_oversample": args.spike_oversample,
            "feature_count": len(feature_cols),
            "train_samples": int(len(x_train_aug)),
            "base_train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "estimators_per_target": estimator_count(model),
            "best_stage": best_stage,
            "tree_count": tree_count(model),
        }
    }
    print(f"Loaded {len(raw_df)} rows from {data_path}")
    print(f"Feature rows after lookback: {len(modeled_df)}")
    print(f"Training samples: {len(x_train_aug)} | Test samples: {len(x_test)}")
    print("\nTest performance:")
    print(f"{'feature':<18} {'MAE':>10} {'RMSE':>10}")
    for idx, feature in enumerate(FEATURES):
        mae = float(mean_absolute_error(y_test[:, idx], predictions[:, idx]))
        rmse = float(np.sqrt(mean_squared_error(y_test[:, idx], predictions[:, idx])))
        metrics[feature] = {"mae": mae, "rmse": rmse}
        print(f"{feature:<18} {mae:10.3f} {rmse:10.3f}")

    raw_copy = artifact_path(output_dir, "telemetry.csv", "raw_data")
    if data_path.resolve() != raw_copy.resolve():
        shutil.copy2(data_path, raw_copy)
    model_payload = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "target_features": FEATURES,
        "training": metrics["training"],
    }
    model_path = artifact_path(output_dir, "dataset_model.joblib", "model")
    joblib.dump(model_payload, model_path)
    pd.DataFrame(predictions, columns=FEATURES).to_csv(artifact_path(output_dir, "predictions.csv", "results"), index=False)
    pd.DataFrame(y_test, columns=FEATURES).to_csv(artifact_path(output_dir, "actuals.csv", "results"), index=False)
    loss_df.to_csv(
        artifact_path(output_dir, "train_losses.csv", "results"),
        index=False,
    )
    artifact_path(output_dir, "metrics.json", "json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    feature_importance = np.mean(
        np.vstack([est.feature_importances_ for est in model.estimators_ if hasattr(est, "feature_importances_")]),
        axis=0,
    )
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": feature_importance})
    importance_df.sort_values("importance", ascending=False).to_csv(
        artifact_path(output_dir, "feature_importance.csv", "results"),
        index=False,
    )
    print(f"Saved dataset optimized artifacts -> {output_dir}")
    print(f"Saved dataset model -> {model_path}")


if __name__ == "__main__":
    main()

"""Load a trained hybrid ensemble and forecast the next N steps.

Usage
-----
    python ml/predict.py \
        --run-dir runs/best_run \
        --data /tmp/t.csv \
        --forecast-steps 24 \
        --output forecast.csv

This script:
  1. Loads scaler_params.json, lstm_model.pth, and gb_model.joblib from
     the run directory.
  2. Reads the latest rows from --data.
  3. Runs LSTM inference and (if available) GB inference.
  4. Blends using the saved ensemble weights.
  5. Writes a forecast CSV with columns: timestamp, traffic_mbps, latency_ms,
     packet_loss_pct, traffic_mbps_lower_95, latency_ms_lower_95,
     packet_loss_pct_lower_95, traffic_mbps_upper_95, latency_ms_upper_95,
     packet_loss_pct_upper_95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from run_layout import artifact_path
from train_model import (
    FEATURES,
    INPUT_FEATURES,
    TIME_FEATURES,
    MultivariateTrafficLSTM,
    add_time_features,
    inverse_transform_features,
    load_dataset,
    transform_features,
)
from enhanced_train import EnhancedMultivariateTrafficLSTM
from train_kaggle_model import add_features


DEFAULT_SEQUENCE_LENGTH = 96
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_LAYERS = 2


def _load_scaler_params(run_dir: Path) -> dict:
    path = run_dir / "json" / "scaler_params.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler params at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rebuild_scaler(params: dict, key: str):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = np.array(params[key]["mean"], dtype=np.float64)
    scaler.scale_ = np.array(params[key]["scale"], dtype=np.float64)
    scaler.var_ = np.array(params[key]["var"], dtype=np.float64)
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def _load_lstm(run_dir: Path, input_size: int, hidden_size: int, layers: int) -> torch.nn.Module:
    model_path = run_dir / "model" / "lstm_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"lstm_model.pth not found in {run_dir / 'model'}")
    state = torch.load(model_path, map_location="cpu")
    for cls in (EnhancedMultivariateTrafficLSTM, MultivariateTrafficLSTM):
        try:
            model = cls(input_size, hidden_size, layers, len(FEATURES))
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception:
            continue
    raise RuntimeError(
        f"Could not load lstm_model.pth into any known architecture. "
        f"Ensure the model was trained with this version of enhanced_train.py."
    )


def _load_gb(run_dir: Path) -> dict | None:
    gb_path = run_dir / "model" / "gb_model.joblib"
    if not gb_path.exists():
        return None
    return joblib.load(gb_path)


def _load_dataset_model(run_dir: Path) -> dict | None:
    path = run_dir / "model" / "dataset_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def _forecast_with_dataset_model(run_dir: Path, data: pd.DataFrame | Path, forecast_steps: int) -> pd.DataFrame:
    bundle = _load_dataset_model(run_dir)
    if bundle is None:
        raise FileNotFoundError(
            f"No hybrid artifacts found (missing json/scaler_params.json) and no dataset model found at {run_dir / 'model' / 'dataset_model.joblib'}."
        )
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = list(bundle.get("feature_columns") or [])
    lookback = int(bundle.get("training", {}).get("lookback", 24))
    if not feature_columns:
        raise ValueError("dataset_model.joblib is missing feature_columns")

    if isinstance(data, (str, Path)):
        # Use raw CSV here (not train_model.load_dataset) to avoid double-adding
        # time-feature columns that train_kaggle_model.add_features also creates.
        df = pd.read_csv(Path(data))
    else:
        df = data.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    df = df.reset_index(drop=True)
    if "timestamp" in df.columns:
        last_timestamp = pd.to_datetime(df["timestamp"].iloc[-1])
        freq = pd.infer_freq(pd.to_datetime(df["timestamp"]))
        freq = freq or "h"
    else:
        last_timestamp = None
        freq = "h"

    preds: list[np.ndarray] = []
    for step in range(forecast_steps):
        if len(df) < lookback + 1:
            raise ValueError(f"Need at least {lookback + 1} rows for dataset-model inference. Got {len(df)}.")
        feature_frame, derived_cols = add_features(df.iloc[-(lookback + 1) :].reset_index(drop=True), lookback)
        cols = feature_columns or derived_cols
        if hasattr(scaler, "n_features_in_") and len(cols) != int(scaler.n_features_in_):
            # Fall back to derived cols if a saved bundle is inconsistent.
            cols = derived_cols
        x = feature_frame[cols].to_numpy(dtype=float)
        x_scaled = scaler.transform(x)
        y_hat = np.clip(model.predict(x_scaled)[0], 0.0, None)
        preds.append(y_hat.astype(float))

        next_row = df.iloc[-1].copy()
        for idx, feature in enumerate(FEATURES):
            next_row[feature] = float(y_hat[idx])
        if last_timestamp is not None:
            next_row["timestamp"] = last_timestamp + pd.to_timedelta(step + 1, unit="h")
        df = pd.concat([df, pd.DataFrame([next_row])], ignore_index=True)

    pred_array = np.vstack(preds)
    residual_std = np.abs(pred_array).mean(axis=0) * 0.10
    lower_95 = np.clip(pred_array - 1.96 * residual_std.reshape(1, -1), 0.0, None)
    upper_95 = pred_array + 1.96 * residual_std.reshape(1, -1)

    if last_timestamp is not None:
        future_ts = pd.date_range(start=last_timestamp + pd.to_timedelta(1, unit="h"), periods=forecast_steps, freq=freq)
    else:
        future_ts = pd.RangeIndex(forecast_steps)

    result = pd.DataFrame(pred_array, columns=FEATURES)
    result.insert(0, "timestamp", future_ts)
    for idx, feature in enumerate(FEATURES):
        result[f"{feature}_lower_95"] = lower_95[:, idx]
        result[f"{feature}_upper_95"] = upper_95[:, idx]
    return result


def _build_gb_features(
    data: pd.DataFrame,
    bundle: dict,
    lstm: torch.nn.Module | None = None,
    input_scaler: StandardScaler | None = None,
    target_scaler: StandardScaler | None = None,
    sequence_length: int | None = None,
    sequence_input_features: list[str] | None = None,
) -> np.ndarray:
    lookback = int(bundle.get("lookback", 24))
    if len(data) < lookback + 1:
        raise ValueError(
            "Not enough rows to construct GB feature lag window for inference. "
            f"Need at least {lookback + 1} rows."
        )
    feature_frame, feature_cols = add_features(data.iloc[-(lookback + 1) :].reset_index(drop=True), lookback)
    columns = bundle.get("feature_columns", feature_cols)

    embedding_columns = [col for col in columns if col.startswith("lstm_embedding_")]
    prediction_columns = [col for col in columns if col.startswith("lstm_pred_")]
    if embedding_columns or prediction_columns:
        if lstm is None or input_scaler is None or target_scaler is None or sequence_length is None or sequence_input_features is None:
            raise ValueError(
                "GB model requires LSTM-derived features, but inference cannot construct them without a loaded LSTM and sequence parameters."
            )
        if len(data) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} rows to compute LSTM-derived features for inference. Got {len(data)} rows."
            )
        sequence_input = input_scaler.transform(
            data[sequence_input_features].to_numpy(dtype=np.float32)[-sequence_length:]
        )
        with torch.no_grad():
            embedding = lstm.encode(torch.tensor(sequence_input[None], dtype=torch.float32)).cpu().numpy()[0]
            lstm_scaled = lstm(torch.tensor(sequence_input[None], dtype=torch.float32)).cpu().numpy()[0]
        for idx, col in enumerate(embedding_columns):
            feature_frame[col] = float(embedding[idx])
        lstm_pred = inverse_transform_features(target_scaler.inverse_transform(lstm_scaled.reshape(1, -1)))[0]
        for idx, col in enumerate(prediction_columns):
            feature_frame[col] = float(lstm_pred[idx])

    missing = [col for col in columns if col not in feature_frame.columns]
    if missing:
        raise ValueError(f"GB model requires missing feature columns: {missing}")
    return feature_frame[columns].to_numpy(dtype=np.float32)


def forecast(run_dir: Path, data: pd.DataFrame | Path, forecast_steps: int = 1) -> pd.DataFrame:
    try:
        scaler_params = _load_scaler_params(run_dir)
    except FileNotFoundError:
        # Auto-benchmark can select gb-only runs which do not produce scaler_params.json.
        return _forecast_with_dataset_model(run_dir, data, forecast_steps)
    feature_cols = scaler_params.get("feature_columns", INPUT_FEATURES)
    hidden_size = DEFAULT_HIDDEN_SIZE
    layers = DEFAULT_LAYERS

    metrics_path = run_dir / "json" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training = metrics.get("training", {})
        hidden_size = int(training.get("hidden_size", hidden_size))
        layers = int(training.get("layers", layers))
        sequence_length = int(training.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    else:
        sequence_length = DEFAULT_SEQUENCE_LENGTH

    input_scaler = _rebuild_scaler(scaler_params, "input_scaler")
    target_scaler = _rebuild_scaler(scaler_params, "target_scaler")
    lstm = _load_lstm(run_dir, len(feature_cols), hidden_size, layers)
    gb_bundle = _load_gb(run_dir)

    if isinstance(data, (str, Path)):
        df = load_dataset(Path(data))
    else:
        df = data.copy()
    transformed = transform_features(df)
    if "timestamp" in transformed.columns:
        transformed = add_time_features(transformed)

    available_cols = [c for c in feature_cols if c in transformed.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        raise ValueError(f"Input data is missing required columns: {', '.join(sorted(missing))}")

    raw_input = transformed[available_cols].to_numpy(dtype=np.float32)
    if len(raw_input) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} rows for inference. Got {len(raw_input)} rows."
        )

    last_timestamp = pd.to_datetime(transformed["timestamp"].iloc[-1]) if "timestamp" in transformed.columns else None
    freq = pd.infer_freq(transformed["timestamp"]) if "timestamp" in transformed.columns else None
    freq = freq or "h"
    if last_timestamp is not None:
        future_timestamps = pd.date_range(start=last_timestamp + pd.to_timedelta(1, unit="h"), periods=forecast_steps, freq=freq)
        future_time_df = pd.DataFrame({"timestamp": future_timestamps})
        future_time_df = add_time_features(future_time_df)
        future_time_features = future_time_df[TIME_FEATURES].to_numpy(dtype=np.float32)
    else:
        future_time_features = np.zeros((forecast_steps, len(TIME_FEATURES)), dtype=np.float32)

    target_indices = [available_cols.index(feat) for feat in FEATURES if feat in available_cols]
    time_indices = [available_cols.index(feat) for feat in TIME_FEATURES if feat in available_cols]

    current_window = raw_input[-sequence_length:].copy()
    all_lstm_preds: list[np.ndarray] = []

    for step in range(forecast_steps):
        scaled_window = input_scaler.transform(current_window)
        with torch.no_grad():
            lstm_scaled = lstm(torch.tensor(scaled_window[None], dtype=torch.float32)).cpu().numpy()[0]
        lstm_pred = inverse_transform_features(target_scaler.inverse_transform(lstm_scaled.reshape(1, -1)))[0]
        all_lstm_preds.append(lstm_pred)

        next_row = current_window[-1].copy()
        for idx, feature in enumerate(FEATURES):
            if feature in available_cols:
                next_row[available_cols.index(feature)] = lstm_pred[idx]
        for tf_idx, col_idx in enumerate(time_indices):
            next_row[col_idx] = future_time_features[step, tf_idx]
        current_window = np.vstack([current_window[1:], next_row])

    lstm_pred_array = np.vstack(all_lstm_preds)

    if gb_bundle is not None:
        gb_input = _build_gb_features(
            df,
            gb_bundle,
            lstm=lstm,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            sequence_length=sequence_length,
            sequence_input_features=available_cols,
        )
        gb_pred = gb_bundle["model"].predict(gb_input)
        gb_pred = np.tile(gb_pred.reshape(1, -1), (forecast_steps, 1))
        weights = gb_bundle.get("ensemble_weights", {})
        gb_weights = np.array([weights.get("gradient_boosting", {}).get(f, 1.0) for f in FEATURES], dtype=float)
        lstm_weights = np.array([weights.get("lstm", {}).get(f, 0.0) for f in FEATURES], dtype=float)
        persistence_weights = np.array([weights.get("persistence_residual", {}).get(f, 0.0) for f in FEATURES], dtype=float)

        final_pred = gb_weights.reshape(1, -1) * gb_pred + lstm_weights.reshape(1, -1) * lstm_pred_array

        if np.any(persistence_weights > 0.0):
            last_observed = raw_input[-1]
            persistence = np.zeros((forecast_steps, len(FEATURES)), dtype=float)
            for idx, feature in enumerate(FEATURES):
                if feature in available_cols:
                    persistence[:, idx] = last_observed[available_cols.index(feature)]
            final_pred = (
                persistence_weights.reshape(1, -1) * persistence
                + (1.0 - persistence_weights.reshape(1, -1)) * final_pred
            )
    else:
        final_pred = lstm_pred_array

    final_pred = np.clip(final_pred, 0.0, None)

    interval_path = run_dir / "results" / "prediction_intervals.csv"
    if interval_path.exists():
        intervals_df = pd.read_csv(interval_path)
        residual_std = np.array(
            [
                float((intervals_df[f"{feat}_upper_95"] - intervals_df[f"{feat}_lower_95"]).mean()) / 3.92
                for feat in FEATURES
            ],
            dtype=float,
        )
    else:
        residual_std = np.abs(final_pred).mean(axis=0) * 0.10

    lower_95 = np.clip(final_pred - 1.96 * residual_std.reshape(1, -1), 0.0, None)
    upper_95 = final_pred + 1.96 * residual_std.reshape(1, -1)

    if last_timestamp is not None:
        future_ts = pd.date_range(start=last_timestamp + pd.to_timedelta(1, unit="h"), periods=forecast_steps, freq=freq)
    else:
        future_ts = pd.RangeIndex(forecast_steps)

    result = pd.DataFrame(final_pred, columns=FEATURES)
    result.insert(0, "timestamp", future_ts)
    for idx, feature in enumerate(FEATURES):
        result[f"{feature}_lower_95"] = lower_95[:, idx]
        result[f"{feature}_upper_95"] = upper_95[:, idx]

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed training run (must contain model/ and json/ subdirs).",
    )
    parser.add_argument("--data", required=True, help="Telemetry CSV path for the most recent history.")
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=24,
        help="Number of future time steps to predict (default 24).",
    )
    parser.add_argument(
        "--output",
        default="forecast.csv",
        help="Output CSV path for forecast results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    result = forecast(run_dir, Path(args.data), args.forecast_steps)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Forecast saved -> {out}")


if __name__ == "__main__":
    main()

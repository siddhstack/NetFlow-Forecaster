"""Hybrid LSTM + Gradient Boosting trainer for telemetry experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from calibrate_predictions import apply_calibration, calibrate
from run_layout import artifact_path, ensure_run_layout
from train_kaggle_model import add_features
from metrics_utils import spike_thresholds_from_quantile
from train_model import FEATURES, INPUT_FEATURES, TIME_FEATURES, SpikeWeightedLoss, create_sequences, inverse_transform_features, load_dataset, transform_features


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class EnhancedMultivariateTrafficLSTM(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.norm = nn.LayerNorm(hidden_size)
        self.context_dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
                for _ in range(output_size)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(weights * out, dim=1)
        context = self.context_dropout(self.norm(context + out[:, -1]))
        return torch.cat([head(context) for head in self.heads], dim=1)


class StackedHybridLSTM(nn.Module):
    """Legacy stacked LSTM used by earlier hybrid runs before attention was added."""

    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


SimpleLSTM = EnhancedMultivariateTrafficLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ml/telemetry.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--spike-quantile", type=float, default=0.90)
    parser.add_argument("--spike-weight", type=float, default=4.0)
    parser.add_argument("--focal-gamma", type=float, default=0.3)
    parser.add_argument(
        "--feature-spike-multipliers",
        default="1.0,1.0,1.0",
        help="Comma-separated spike loss multipliers for traffic, latency, packet loss.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological training fraction.")
    parser.add_argument("--test-ratio", type=float, default=0.82, help="Chronological test start fraction; validation is between train and test.")
    parser.add_argument("--train-split", type=float, default=None, help="Deprecated alias for --test-ratio.")
    parser.add_argument("--validation-split", type=float, default=None, help="Deprecated alias for --train-ratio.")
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--gb-weight", type=float, default=0.65)
    parser.add_argument("--lstm-weight", type=float, default=0.35)
    parser.add_argument("--tune-ensemble-weights", action="store_true", default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=False, help="Export the LSTM component to ONNX.")
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True, help="Apply validation-only prediction calibration.")
    parser.add_argument("--predict-traffic-delta", action="store_true", help="Compatibility flag for candidate search; traffic remains level-predicted in this trainer.")
    parser.add_argument(
        "--spike-lift-near",
        default="",
        help="Optional comma-separated near-threshold factors for post-calibration spike lift.",
    )
    parser.add_argument(
        "--spike-lift-factors",
        default="",
        help="Optional comma-separated lift factors for post-calibration spike lift.",
    )
    parser.add_argument("--output", default="lstm_model.pth")
    parser.add_argument("--output-dir", default="runs/hybrid_best")
    return parser.parse_args()


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    scale = np.maximum(target.std(axis=0, ddof=0), 1e-9)
    return float(np.mean(((prediction - target) / scale) ** 2))


def parse_feature_spike_multipliers(value: str) -> torch.Tensor:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != len(FEATURES):
        raise ValueError(
            f"--feature-spike-multipliers expects {len(FEATURES)} values, got {len(parts)}"
        )
    return torch.tensor(parts, dtype=torch.float32)


def parse_optional_feature_values(value: str) -> list[float] | None:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} comma-separated values, got {len(parts)}")
    return parts


def apply_spike_lift(
    predictions: np.ndarray,
    thresholds: dict[str, float],
    near_factors: list[float] | None,
    lift_factors: list[float] | None,
) -> np.ndarray:
    if near_factors is None or lift_factors is None:
        return predictions
    lifted = predictions.copy()
    for idx, feature in enumerate(FEATURES):
        threshold = float(thresholds[feature])
        near = lifted[:, idx] > threshold * near_factors[idx]
        lifted[near, idx] = np.maximum(lifted[near, idx], threshold * lift_factors[idx])
    return lifted


def original_index_for_sequence(sequence_idx: int, sequence_length: int) -> int:
    return sequence_idx + sequence_length


def gb_slice_for_sequences(start_idx: int, end_idx: int, sequence_length: int, lookback: int, total_rows: int) -> slice:
    start = max(0, original_index_for_sequence(start_idx, sequence_length) - lookback)
    end = min(total_rows, original_index_for_sequence(end_idx, sequence_length) - lookback)
    return slice(start, end)


def optimize_ensemble_weights(gb_pred: np.ndarray, lstm_pred: np.ndarray, actuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gb_weights = np.zeros(actuals.shape[1], dtype=float)
    for idx in range(actuals.shape[1]):
        best_weight = 0.65
        best_score = float("inf")
        threshold = float(np.quantile(actuals[:, idx], 0.90))
        actual_spikes = actuals[:, idx] > threshold
        for weight in np.linspace(0.0, 1.0, 41):
            prediction = weight * gb_pred[:, idx] + (1.0 - weight) * lstm_pred[:, idx]
            mae = float(np.mean(np.abs(prediction - actuals[:, idx])))
            pred_spikes = prediction > threshold
            tp = float(np.sum(actual_spikes & pred_spikes))
            fp = float(np.sum(~actual_spikes & pred_spikes))
            fn = float(np.sum(actual_spikes & ~pred_spikes))
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
            spike_penalty = 0.20 * float(np.std(actuals[:, idx], ddof=0)) * (1.0 - f1)
            score = mae + spike_penalty
            if score < best_score:
                best_score = score
                best_weight = float(weight)
        gb_weights[idx] = best_weight
    return gb_weights, 1.0 - gb_weights


def oversample_spikes(x_train: np.ndarray, y_train: np.ndarray, thresholds: dict[str, float], repeats: int = 3) -> tuple[np.ndarray, np.ndarray]:
    spike_mask = np.zeros(len(y_train), dtype=bool)
    for idx, feature in enumerate(FEATURES):
        spike_mask |= y_train[:, idx] > thresholds[feature]
    if not np.any(spike_mask):
        return x_train, y_train
    return (
        np.concatenate([x_train, *([x_train[spike_mask]] * repeats)], axis=0),
        np.concatenate([y_train, *([y_train[spike_mask]] * repeats)], axis=0),
    )


def persistence_baseline(actuals: np.ndarray) -> np.ndarray:
    baseline = np.empty_like(actuals)
    baseline[0] = actuals[0]
    baseline[1:] = actuals[:-1]
    return baseline


def optimize_persistence_blend(model_pred: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    persistence = persistence_baseline(actuals)
    weights = np.zeros(actuals.shape[1], dtype=float)
    for idx in range(actuals.shape[1]):
        best_weight = 0.0
        best_mae = float("inf")
        for weight in np.linspace(0.0, 0.95, 20):
            prediction = weight * persistence[:, idx] + (1.0 - weight) * model_pred[:, idx]
            mae = float(np.mean(np.abs(prediction - actuals[:, idx])))
            if mae < best_mae:
                best_mae = mae
                best_weight = float(weight)
        weights[idx] = best_weight
    return weights


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    ensure_run_layout(output_dir)
    model_path = Path(args.output)
    if not model_path.is_absolute():
        model_path = artifact_path(output_dir, model_path.name, "model")

    df = load_dataset(data_path)
    transformed = transform_features(df)
    feature_cols = INPUT_FEATURES if all(c in transformed.columns for c in TIME_FEATURES) else FEATURES
    train_ratio = args.validation_split if args.validation_split is not None else args.train_ratio
    test_ratio = args.train_split if args.train_split is not None else args.test_ratio
    if not 0.0 < train_ratio < test_ratio < 1.0:
        raise ValueError("Expected 0 < train_ratio < test_ratio < 1.")

    x_seq, _ = create_sequences(transformed[feature_cols].to_numpy(dtype=np.float32), args.sequence_length)
    _, y_seq = create_sequences(transformed[FEATURES].to_numpy(dtype=np.float32), args.sequence_length)
    if len(x_seq) < 10:
        raise ValueError("Not enough sequences. Reduce --sequence-length or collect more rows.")

    train_end = max(1, min(len(x_seq) - 2, int(train_ratio * len(x_seq))))
    test_start = max(train_end + 1, min(len(x_seq) - 1, int(test_ratio * len(x_seq))))
    x_train_raw, y_train_raw = x_seq[:train_end], y_seq[:train_end]
    x_val_raw, y_val_raw = x_seq[train_end:test_start], y_seq[train_end:test_start]
    x_test_raw, y_test_raw = x_seq[test_start:], y_seq[test_start:]

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train = input_scaler.fit_transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)
    x_val = input_scaler.transform(x_val_raw.reshape(-1, x_val_raw.shape[-1])).reshape(x_val_raw.shape)
    x_test = input_scaler.transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)
    y_train = target_scaler.fit_transform(y_train_raw)
    y_val = target_scaler.transform(y_val_raw)
    y_test = target_scaler.transform(y_test_raw)

    raw_spike_thresholds = spike_thresholds_from_quantile(inverse_transform_features(y_train_raw.copy()), args.spike_quantile)
    transformed_spike_thresholds = spike_thresholds_from_quantile(y_train_raw, args.spike_quantile)
    scaled_spike_thresholds = target_scaler.transform(
        np.array([[transformed_spike_thresholds[feature] for feature in FEATURES]], dtype=np.float32)
    )[0]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    lstm = SimpleLSTM(len(feature_cols), args.hidden_size, args.layers, len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(lstm.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5)
    criterion = SpikeWeightedLoss(
        torch.tensor(scaled_spike_thresholds, dtype=torch.float32, device=device),
        spike_weight=args.spike_weight,
        focal_gamma=args.focal_gamma,
        per_feature_spike_multipliers=parse_feature_spike_multipliers(
            args.feature_spike_multipliers
        ).to(device),
    )
    mse_criterion = nn.MSELoss()

    print("Training Hybrid LSTM component...")
    train_rows: list[dict[str, float]] = []
    best_state = {key: value.detach().clone() for key, value in lstm.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(args.epochs):
        lstm.train()
        losses: list[float] = []
        optimizer.zero_grad()
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = lstm(batch_x)
            loss = criterion(pred, batch_y) / max(args.grad_accum_steps, 1)
            loss.backward()
            if (batch_idx + 1) % max(args.grad_accum_steps, 1) == 0 or batch_idx == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(float(loss.item() * max(args.grad_accum_steps, 1)))

        lstm.eval()
        with torch.no_grad():
            val_pred = lstm(x_val_tensor)
            val_loss = mse_criterion(val_pred, y_val_tensor).item()
        scheduler.step(val_loss)
        train_loss_mean = float(np.mean(losses))
        if val_loss < best_val - args.early_stop_delta:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in lstm.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        train_rows.append(
            {
                "epoch": epoch + 1,
                "mse_loss": train_loss_mean,
                "validation_mse_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"LSTM Epoch {epoch + 1} Loss: {train_loss_mean:.4f} | Val: {val_loss:.4f}")
        min_epochs_before_stop = min(args.epochs, 40)
        if args.early_stop_patience > 0 and epoch + 1 >= min_epochs_before_stop and stale_epochs >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}; best validation loss was epoch {best_epoch}.")
            break

    lstm.load_state_dict(best_state)
    lstm.eval()

    print("Training Gradient Boosting spike component...")
    lookback = 24
    df_feat, gb_feature_cols = add_features(df, lookback=lookback)
    x_gb = df_feat[gb_feature_cols].to_numpy(dtype=float)
    y_gb = df_feat[FEATURES].to_numpy(dtype=float)

    gb_val_slice = gb_slice_for_sequences(train_end, test_start, args.sequence_length, lookback, len(x_gb))
    gb_test_slice = gb_slice_for_sequences(test_start, len(x_seq), args.sequence_length, lookback, len(x_gb))
    gb_val_train_end = max(1, gb_val_slice.start)
    gb_test_train_end = max(1, gb_test_slice.start)

    gb_train_x, gb_train_y = oversample_spikes(
        x_gb[:gb_val_train_end],
        y_gb[:gb_val_train_end],
        spike_thresholds_from_quantile(y_gb[:gb_val_train_end], args.spike_quantile),
    )
    gb_for_weight = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=500, learning_rate=0.04, max_depth=4, subsample=0.85, random_state=42)
    )
    gb_for_weight.fit(gb_train_x, gb_train_y)

    with torch.no_grad():
        val_len = gb_val_slice.stop - gb_val_slice.start
        lstm_val_scaled = lstm(x_val_tensor[:val_len]).detach().cpu().numpy()
    lstm_val_pred = inverse_transform_features(target_scaler.inverse_transform(lstm_val_scaled))
    gb_val_pred = gb_for_weight.predict(x_gb[gb_val_slice])
    val_actuals = y_gb[gb_val_slice]
    if args.tune_ensemble_weights:
        gb_weight, lstm_weight = optimize_ensemble_weights(gb_val_pred, lstm_val_pred, val_actuals)
    else:
        gb_weight = np.full(len(FEATURES), args.gb_weight, dtype=float)
        lstm_weight = np.full(len(FEATURES), args.lstm_weight, dtype=float)
    val_ensemble_pred = gb_weight.reshape(1, -1) * gb_val_pred + lstm_weight.reshape(1, -1) * lstm_val_pred
    persistence_weight = optimize_persistence_blend(val_ensemble_pred, val_actuals)
    val_persistence = persistence_baseline(val_actuals)
    val_after_persistence = (
        persistence_weight.reshape(1, -1) * val_persistence
        + (1.0 - persistence_weight.reshape(1, -1)) * val_ensemble_pred
    )
    calibration_params = None
    if args.calibrate:
        calibration_params = calibrate(val_actuals, val_after_persistence, raw_spike_thresholds)
        val_after_persistence = apply_calibration(val_after_persistence, calibration_params, val_persistence)

    gb_test_x, gb_test_y = oversample_spikes(
        x_gb[:gb_test_train_end],
        y_gb[:gb_test_train_end],
        spike_thresholds_from_quantile(y_gb[:gb_test_train_end], args.spike_quantile),
    )
    gb = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=500, learning_rate=0.04, max_depth=4, subsample=0.85, random_state=42)
    )
    gb.fit(gb_test_x, gb_test_y)

    with torch.no_grad():
        test_len = gb_test_slice.stop - gb_test_slice.start
        lstm_scaled = lstm(x_test_tensor[:test_len]).detach().cpu().numpy()
    lstm_pred = inverse_transform_features(target_scaler.inverse_transform(lstm_scaled))
    gb_pred = gb.predict(x_gb[gb_test_slice])
    actuals = y_gb[gb_test_slice]
    final_pred = gb_weight.reshape(1, -1) * gb_pred + lstm_weight.reshape(1, -1) * lstm_pred
    traffic_guard = 0.85 * gb_pred[:, 0] + 0.15 * lstm_pred[:, 0]
    final_pred[:, 0] = np.maximum(final_pred[:, 0], traffic_guard)
    test_persistence = persistence_baseline(actuals)
    final_pred = persistence_weight.reshape(1, -1) * test_persistence + (1.0 - persistence_weight.reshape(1, -1)) * final_pred
    if calibration_params is not None:
        final_pred = apply_calibration(final_pred, calibration_params, test_persistence)
    final_pred = apply_spike_lift(
        final_pred,
        raw_spike_thresholds,
        parse_optional_feature_values(args.spike_lift_near),
        parse_optional_feature_values(args.spike_lift_factors),
    )
    final_pred = np.clip(final_pred, 0.0, None)
    residuals = actuals - final_pred
    residual_std = np.std(residuals, axis=0, ddof=0)
    lower_95 = np.clip(final_pred - 1.96 * residual_std, 0.0, None)
    upper_95 = final_pred + 1.96 * residual_std

    lstm_cpu = lstm.to("cpu")
    torch.save(lstm.state_dict(), model_path)
    joblib.dump(
        {
            "model": gb,
            "feature_columns": gb_feature_cols,
            "features": FEATURES,
            "lookback": lookback,
            "ensemble_weights": {
                "gradient_boosting": {feature: float(gb_weight[idx]) for idx, feature in enumerate(FEATURES)},
                "lstm": {feature: float(lstm_weight[idx]) for idx, feature in enumerate(FEATURES)},
                "persistence_residual": {feature: float(persistence_weight[idx]) for idx, feature in enumerate(FEATURES)},
            },
            "weight_selection": "validation_grid_search" if args.tune_ensemble_weights else "manual",
        },
        artifact_path(output_dir, "gb_model.joblib", "model"),
    )
    onnx_status = "disabled"
    if args.export_onnx:
        try:
            torch.onnx.export(
                lstm_cpu,
                torch.randn(1, args.sequence_length, len(feature_cols)),
                artifact_path(output_dir, "lstm_model.onnx", "model"),
                export_params=True,
                opset_version=18,
                input_names=["input"],
                output_names=["output"],
            )
            onnx_status = "exported"
        except Exception as exc:  # pragma: no cover - depends on optional exporter packages
            onnx_status = f"skipped: {exc.__class__.__name__}: {exc}"
            print(f"ONNX export skipped: {exc}")

    metrics = {
        "training": {
            "loss": "HybridLstmMSEPlusGradientBoosting",
            "feature_columns": feature_cols,
            "gb_feature_columns": gb_feature_cols,
            "output_features": FEATURES,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "device": str(device),
            "learning_rate": args.lr,
            "requested_epochs": args.epochs,
            "epochs": len(train_rows),
            "best_epoch": best_epoch,
            "best_validation_mse_loss": best_val,
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
            "architecture": "hybrid_attention_lstm_gradient_boosting",
            "ensemble": "lstm_gradient_boosting",
            "early_stop_patience": args.early_stop_patience,
            "early_stop_delta": args.early_stop_delta,
            "spike_quantile": args.spike_quantile,
            "spike_weight": args.spike_weight,
            "focal_gamma": args.focal_gamma,
            "spike_thresholds": raw_spike_thresholds,
            "spike_lift_near": args.spike_lift_near,
            "spike_lift_factors": args.spike_lift_factors,
            "gb_weight": {feature: float(gb_weight[idx]) for idx, feature in enumerate(FEATURES)},
            "lstm_weight": {feature: float(lstm_weight[idx]) for idx, feature in enumerate(FEATURES)},
            "persistence_residual_weight": {feature: float(persistence_weight[idx]) for idx, feature in enumerate(FEATURES)},
            "weight_selection": "validation_grid_search" if args.tune_ensemble_weights else "manual",
            "prediction_interval": "residual_normal_95",
            "onnx_status": onnx_status,
            "calibration": None
            if calibration_params is None
            else {
                "scale": calibration_params.scale,
                "bias": calibration_params.bias,
                "persistence_weight": calibration_params.persistence_weight,
                "spike_boost": calibration_params.spike_boost,
            },
        }
    }
    for idx, feature in enumerate(FEATURES):
        mae = mean_absolute_error(actuals[:, idx], final_pred[:, idx])
        rmse = float(np.sqrt(mean_squared_error(actuals[:, idx], final_pred[:, idx])))
        metrics[feature] = {"mae": float(mae), "rmse": rmse}

    scaler_params = {
        "feature_columns": feature_cols,
        "scaler_type": "StandardScaler",
        "input_scaler": {"mean": input_scaler.mean_.tolist(), "scale": input_scaler.scale_.tolist(), "var": input_scaler.var_.tolist()},
        "target_scaler": {"mean": target_scaler.mean_.tolist(), "scale": target_scaler.scale_.tolist(), "var": target_scaler.var_.tolist()},
    }
    artifact_path(output_dir, "scaler_params.json", "json").write_text(json.dumps(scaler_params, indent=2), encoding="utf-8")
    artifact_path(output_dir, "metrics.json", "json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(final_pred, columns=FEATURES).to_csv(artifact_path(output_dir, "predictions.csv", "results"), index=False)
    pd.DataFrame(actuals, columns=FEATURES).to_csv(artifact_path(output_dir, "actuals.csv", "results"), index=False)
    pd.DataFrame(val_after_persistence, columns=FEATURES).to_csv(artifact_path(output_dir, "val_predictions.csv", "results"), index=False)
    pd.DataFrame(val_actuals, columns=FEATURES).to_csv(artifact_path(output_dir, "val_actuals.csv", "results"), index=False)
    interval_df = pd.DataFrame(
        {
            **{f"{feature}_lower_95": lower_95[:, idx] for idx, feature in enumerate(FEATURES)},
            **{f"{feature}_upper_95": upper_95[:, idx] for idx, feature in enumerate(FEATURES)},
        }
    )
    interval_df.to_csv(artifact_path(output_dir, "prediction_intervals.csv", "results"), index=False)
    pd.DataFrame(train_rows).to_csv(artifact_path(output_dir, "train_losses.csv", "results"), index=False)

    raw_copy = artifact_path(output_dir, data_path.name, "raw_data")
    if data_path.resolve() != raw_copy.resolve():
        shutil.copy2(data_path, raw_copy)

    print(f"\nHYBRID ENSEMBLE COMPLETE -> {output_dir}")
    print(f"Normalized ensemble MSE: {normalized_mse(final_pred, actuals):.4f}")
    print("Learned ensemble weights:")
    for idx, feature in enumerate(FEATURES):
        print(
            f"  {feature}: GradientBoosting={gb_weight[idx]:.2f}, "
            f"LSTM={lstm_weight[idx]:.2f}, PersistenceResidual={persistence_weight[idx]:.2f}"
        )
    print("This should give stronger spike capture than the pure LSTM.")


if __name__ == "__main__":
    main()

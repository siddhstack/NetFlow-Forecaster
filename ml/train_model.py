"""Train a multivariate LSTM for network telemetry prediction."""

# NOTE: Use ml/enhanced_train.py for the hybrid attention-LSTM + Gradient Boosting model with ONNX export.

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from run_layout import artifact_path, ensure_run_layout


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]
TIME_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
INPUT_FEATURES = FEATURES + TIME_FEATURES
DEFAULT_MODEL = "lstm_model.pth"


class MultivariateTrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(32, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_size // 2), output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class SpikeWeightedLoss(nn.Module):
    """MSE with extra penalty on spike timesteps.

    Adds a ``per_feature_spike_multipliers`` parameter (default [1.0, 1.8, 2.5])
    that scales spike penalties independently per feature. Traffic spikes keep
    weight 1.0x; latency spikes get 1.8x; packet-loss spikes get 2.5x.

    Rationale: traffic carries 0.5 weight in the quality score so the optimizer
    already prioritises it. The multipliers correct for that imbalance so
    latency and loss spike recall improve without hurting traffic.
    """

    def __init__(
        self,
        thresholds: torch.Tensor,
        spike_weight: float = 4.0,
        focal_gamma: float = 0.0,
        per_feature_spike_multipliers: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("thresholds", thresholds)
        self.spike_weight = spike_weight
        self.focal_gamma = focal_gamma
        if per_feature_spike_multipliers is None:
            per_feature_spike_multipliers = torch.tensor(
                [1.0, 1.8, 2.5], dtype=torch.float32
            )
        self.register_buffer(
            "per_feature_spike_multipliers", per_feature_spike_multipliers
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        squared_error = error**2
        focal = (
            1.0
            if self.focal_gamma <= 0
            else 1.0 + error.abs().pow(self.focal_gamma)
        )
        spike_mask = (target > self.thresholds).float()
        feature_weight_matrix = self.per_feature_spike_multipliers.to(device=pred.device).view(1, -1)
        spike_weight_matrix = 1.0 + self.spike_weight * spike_mask.to(device=pred.device) * feature_weight_matrix
        composite_weight_matrix = feature_weight_matrix * spike_weight_matrix

        # Explicitly mirror the paper-ready formulation
        # $\mathcal{L}_{\text{total}} = \mathbf{e}^{T}(\mathbf{W}_{\text{feature}} \odot \mathbf{W}_{\text{spike}})\mathbf{e}$
        error_vector = squared_error * focal
        weighted_error = error_vector * composite_weight_matrix
        return weighted_error.sum(dim=-1).mean()


def create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for idx in range(len(data) - seq_len):
        x.append(data[idx : idx + seq_len])
        y.append(data[idx + seq_len])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    transformed = df.copy()
    ts = pd.to_datetime(transformed["timestamp"], errors="coerce")
    transformed["hour"] = ts.dt.hour
    transformed["day_of_week"] = ts.dt.dayofweek
    transformed["hour_sin"] = np.sin(2.0 * np.pi * transformed["hour"] / 24.0)
    transformed["hour_cos"] = np.cos(2.0 * np.pi * transformed["hour"] / 24.0)
    transformed["weekday_sin"] = np.sin(2.0 * np.pi * transformed["day_of_week"] / 7.0)
    transformed["weekday_cos"] = np.cos(2.0 * np.pi * transformed["day_of_week"] / 7.0)
    return transformed


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate_data.py or collect_telemetry.py first.")
    df = pd.read_csv(path)
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    df = df.dropna(subset=FEATURES).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = add_time_features(df)
    if len(df) < 30:
        raise ValueError(f"{path} has only {len(df)} usable rows; collect or generate more telemetry.")
    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-feature transformations before scaling.

    packet_loss_pct uses sqrt instead of log1p. This preserves the relative
    severity of loss spikes while keeping values numerically stable for scaling.
    """
    transformed = df.copy()
    transformed["packet_loss_pct"] = np.sqrt(transformed["packet_loss_pct"])
    return transformed


def inverse_transform_features(values: np.ndarray) -> np.ndarray:
    """Invert transform_features for packet_loss_pct."""
    restored = values.copy()
    loss_idx = FEATURES.index("packet_loss_pct")
    restored[:, loss_idx] = np.square(restored[:, loss_idx])
    restored[:, loss_idx] = np.clip(restored[:, loss_idx], 0.0, None)
    return restored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="telemetry.csv", help="Input telemetry CSV.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Lookback window size.")
    parser.add_argument("--epochs", type=int, default=130, help="Training epochs.")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden units.")
    parser.add_argument("--layers", type=int, default=2, help="LSTM layer count.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Chronological train split.")
    parser.add_argument("--spike-quantile", type=float, default=0.9, help="Training quantile used for spike weighting.")
    parser.add_argument("--spike-weight", type=float, default=4.0, help="Extra loss weight for spike targets.")
    parser.add_argument("--per-feature-loss-weights", default="1.0,0.7,0.5", help="Comma-separated spike multipliers for traffic, latency, packet loss.")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Error focus exponent for hard examples.")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop when validation MSE stops improving. Use 0 to always run all epochs.")
    parser.add_argument("--early-stop-delta", type=float, default=1e-5, help="Minimum validation MSE improvement.")
    parser.add_argument("--seed", type=int, default=7, help="Torch and NumPy seed.")
    parser.add_argument("--output", default=DEFAULT_MODEL, help="Model weights path.")
    parser.add_argument("--output-dir", default=".", help="Directory for all training artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    ensure_run_layout(output_dir)
    model_path = Path(args.output)
    if not model_path.is_absolute():
        model_path = artifact_path(output_dir, model_path.name, "model")

    df = load_dataset(data_path)
    transformed_df = transform_features(df)
    raw = transformed_df[FEATURES].to_numpy(dtype=np.float32)
    print(f"Loaded {len(raw)} rows from {data_path}")
    for idx, feature in enumerate(FEATURES):
        print(f"  {feature:<16} {raw[:, idx].min():8.3f} to {raw[:, idx].max():8.3f}")

    feature_columns = INPUT_FEATURES if all(name in transformed_df.columns for name in TIME_FEATURES) else FEATURES
    raw_inputs = transformed_df[feature_columns].to_numpy(dtype=np.float32)
    raw_targets = transformed_df[FEATURES].to_numpy(dtype=np.float32)
    x, _ = create_sequences(raw_inputs, args.sequence_length)
    _, y = create_sequences(raw_targets, args.sequence_length)
    if len(x) < 10:
        raise ValueError("Not enough sequences. Reduce --sequence-length or collect more rows.")

    split = max(1, min(len(x) - 1, int(len(x) * args.train_split)))
    x_train = x[:split]
    y_train = y[:split]
    x_test = x[split:]
    y_test = y[split:]
    print(f"Training samples: {len(x_train)} | Test samples: {len(x_test)}")

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train = torch.tensor(
        input_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape),
        dtype=torch.float32,
    )
    y_train = torch.tensor(target_scaler.fit_transform(y_train), dtype=torch.float32)
    x_test = torch.tensor(
        input_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape),
        dtype=torch.float32,
    )
    y_test = torch.tensor(target_scaler.transform(y_test), dtype=torch.float32)

    model = MultivariateTrafficLSTM(len(feature_columns), args.hidden_size, args.layers, len(FEATURES))
    raw_thresholds = np.quantile(raw_targets[:split], args.spike_quantile, axis=0)
    scaled_spike_thresholds = target_scaler.transform(raw_thresholds.reshape(1, -1))[0]
    spike_thresholds = torch.tensor(scaled_spike_thresholds, dtype=torch.float32)
    raw_weights = [float(x) for x in args.per_feature_loss_weights.split(",") if x.strip()]
    if len(raw_weights) != len(FEATURES):
        raise ValueError(f"--per-feature-loss-weights must have {len(FEATURES)} values")
    feature_loss_weights = torch.tensor(raw_weights, dtype=torch.float32)
    criterion = SpikeWeightedLoss(
        spike_thresholds,
        args.spike_weight,
        args.focal_gamma,
        per_feature_spike_multipliers=feature_loss_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    train_rows: list[dict[str, float]] = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    print("Training multivariate LSTM...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        train_mse = torch.mean((output - y_train) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_output = model(x_test)
            val_loss = criterion(val_output, y_test)
            val_mse = torch.mean((val_output - y_test) ** 2)
        scheduler.step(float(val_loss.item()))
        train_rows.append(
            {
                "epoch": epoch + 1,
                "mse_loss": float(train_mse.item()),
                "weighted_loss": float(loss.item()),
                "validation_mse_loss": float(val_mse.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if float(val_mse.item()) < best_val_mse - args.early_stop_delta:
            best_val_mse = float(val_mse.item())
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch + 1:3d}/{args.epochs} | "
                f"mse={train_mse.item():.6f} | weighted={loss.item():.6f} | "
                f"val_mse={val_mse.item():.6f} | lr={lr:.5f}"
            )
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(f"  early stopping at epoch {epoch + 1}; best validation MSE was epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_test).numpy()
        actual_scaled = y_test.numpy()

    predictions = inverse_transform_features(target_scaler.inverse_transform(pred_scaled))
    actuals = inverse_transform_features(target_scaler.inverse_transform(actual_scaled))

    metrics = {
        "training": {
            "loss": "SpikeWeightedLoss",
            "feature_columns": feature_columns,
            "output_features": FEATURES,
            "spike_quantile": args.spike_quantile,
            "spike_threshold_scaled": spike_thresholds.tolist(),
            "spike_weight": args.spike_weight,
            "focal_gamma": args.focal_gamma,
            "packet_loss_transform": "sqrt",
            "packet_loss_inverse_transform": "square",
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "learning_rate": args.lr,
            "requested_epochs": args.epochs,
            "epochs": len(train_rows),
            "best_epoch": best_epoch,
            "best_validation_mse_loss": best_val_mse,
            "early_stop_patience": args.early_stop_patience,
            "train_split": args.train_split,
        }
    }
    print("\nTest performance:")
    print(f"{'feature':<18} {'MAE':>10} {'RMSE':>10}")
    for idx, feature in enumerate(FEATURES):
        mae = mean_absolute_error(actuals[:, idx], predictions[:, idx])
        rmse = float(np.sqrt(mean_squared_error(actuals[:, idx], predictions[:, idx])))
        metrics[feature] = {"mae": float(mae), "rmse": rmse}
        print(f"{feature:<18} {mae:10.3f} {rmse:10.3f}")

    torch.save(model.state_dict(), model_path)
    scaler_params = {
        "feature_columns": feature_columns,
        "scaler_type": "StandardScaler",
        "input_scaler": {
            "mean": input_scaler.mean_.tolist(),
            "scale": input_scaler.scale_.tolist(),
            "var": input_scaler.var_.tolist(),
        },
        "target_scaler": {
            "mean": target_scaler.mean_.tolist(),
            "scale": target_scaler.scale_.tolist(),
            "var": target_scaler.var_.tolist(),
        },
    }
    artifact_path(output_dir, "scaler_params.json", "json").write_text(json.dumps(scaler_params, indent=2), encoding="utf-8")
    pd.DataFrame(predictions, columns=FEATURES).to_csv(artifact_path(output_dir, "predictions.csv", "results"), index=False)
    pd.DataFrame(actuals, columns=FEATURES).to_csv(artifact_path(output_dir, "actuals.csv", "results"), index=False)
    pd.DataFrame(train_rows).to_csv(
        artifact_path(output_dir, "train_losses.csv", "results"),
        index=False,
    )
    artifact_path(output_dir, "metrics.json", "json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    raw_copy = artifact_path(output_dir, data_path.name, "raw_data")
    if data_path.resolve() != raw_copy.resolve():
        shutil.copy2(data_path, raw_copy)

    print(f"\nSaved model -> {model_path}")
    print(f"Saved human-readable data artifacts -> {output_dir}")


if __name__ == "__main__":
    main()

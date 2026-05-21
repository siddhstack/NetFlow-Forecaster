"""Stable Enhanced LSTM - good balance of spike reactivity."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from run_layout import artifact_path, ensure_run_layout
from train_model import FEATURES, INPUT_FEATURES, TIME_FEATURES, create_sequences, inverse_transform_features, load_dataset, transform_features


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class EnhancedMultivariateTrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, output_size: int = 3, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(nn.Linear(hidden_size, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)
        return self.head(context)


class SpikeWeightedLoss(nn.Module):
    def __init__(self, thresholds: torch.Tensor, spike_weight: float = 4.5, focal_gamma: float = 0.8):
        super().__init__()
        self.register_buffer("thresholds", thresholds)
        self.spike_weight = spike_weight
        self.focal_gamma = focal_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (pred - target).abs()
        focal = 1.0 + error.pow(self.focal_gamma)
        spike_w = 1.0 + self.spike_weight * (target > self.thresholds).float()
        return (spike_w * focal * (pred - target) ** 2).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ml/telemetry.csv")
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--spike-quantile", type=float, default=0.88)
    parser.add_argument("--spike-weight", type=float, default=4.5)
    parser.add_argument("--focal-gamma", type=float, default=0.8)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="lstm_model.pth")
    parser.add_argument("--output-dir", default="runs/stable_enhanced")
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
    feature_columns = INPUT_FEATURES if all(c in transformed_df.columns for c in TIME_FEATURES) else FEATURES

    raw_inputs = transformed_df[feature_columns].to_numpy(dtype=np.float32)
    raw_targets = transformed_df[FEATURES].to_numpy(dtype=np.float32)

    x, _ = create_sequences(raw_inputs, args.sequence_length)
    _, y = create_sequences(raw_targets, args.sequence_length)
    if len(x) < 10:
        raise ValueError("Not enough sequences. Reduce --sequence-length or collect more rows.")

    split = max(1, min(len(x) - 1, int(args.train_split * len(x))))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train = input_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = input_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    y_train = target_scaler.fit_transform(y_train)
    y_test = target_scaler.transform(y_test)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = EnhancedMultivariateTrafficLSTM(len(feature_columns), args.hidden_size, args.layers)
    raw_th = np.quantile(raw_targets[: split + args.sequence_length], args.spike_quantile, axis=0)
    scaled_th = target_scaler.transform(raw_th.reshape(1, -1))[0]
    spike_thresholds = torch.tensor(scaled_th, dtype=torch.float32)
    criterion = SpikeWeightedLoss(spike_thresholds, args.spike_weight, args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("Training Stable Enhanced Model...")
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    train_rows: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        batch_losses: list[float] = []
        batch_mses: list[float] = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))
            batch_mses.append(float(torch.mean((pred - by) ** 2).item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_test_tensor)
            val_loss = criterion(val_pred, y_test_tensor).item()
            val_mse = torch.mean((val_pred - y_test_tensor) ** 2).item()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        train_rows.append(
            {
                "epoch": epoch + 1,
                "mse_loss": float(np.mean(batch_mses)),
                "weighted_loss": float(np.mean(batch_losses)),
                "validation_mse_loss": float(val_mse),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if (epoch + 1) % 30 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch + 1:3d} | Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_scaled = model(x_test_tensor).numpy()

    predictions = inverse_transform_features(target_scaler.inverse_transform(pred_scaled))
    actuals = inverse_transform_features(target_scaler.inverse_transform(y_test))

    metrics = {
        "training": {
            "loss": "StableEnhancedSpikeWeightedLoss",
            "feature_columns": feature_columns,
            "output_features": FEATURES,
            "spike_quantile": args.spike_quantile,
            "spike_threshold_scaled": spike_thresholds.tolist(),
            "spike_weight": args.spike_weight,
            "focal_gamma": args.focal_gamma,
            "packet_loss_transform": "log1p",
            "packet_loss_inverse_transform": "expm1",
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "requested_epochs": args.epochs,
            "epochs": len(train_rows),
            "best_epoch": best_epoch,
            "best_validation_mse_loss": best_val,
            "train_split": args.train_split,
            "architecture": "attention_lstm",
        }
    }
    for idx, feature in enumerate(FEATURES):
        mae = mean_absolute_error(actuals[:, idx], predictions[:, idx])
        rmse = float(np.sqrt(mean_squared_error(actuals[:, idx], predictions[:, idx])))
        metrics[feature] = {"mae": float(mae), "rmse": rmse}

    torch.save(model.state_dict(), model_path)
    torch.onnx.export(
        model,
        torch.randn(1, args.sequence_length, len(feature_columns)),
        artifact_path(output_dir, "lstm_model.onnx", "model"),
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )

    scaler_params = {
        "feature_columns": feature_columns,
        "scaler_type": "StandardScaler",
        "input_scaler": {"mean": input_scaler.mean_.tolist(), "scale": input_scaler.scale_.tolist(), "var": input_scaler.var_.tolist()},
        "target_scaler": {"mean": target_scaler.mean_.tolist(), "scale": target_scaler.scale_.tolist(), "var": target_scaler.var_.tolist()},
    }
    artifact_path(output_dir, "scaler_params.json", "json").write_text(json.dumps(scaler_params, indent=2), encoding="utf-8")
    artifact_path(output_dir, "metrics.json", "json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(predictions, columns=FEATURES).to_csv(artifact_path(output_dir, "predictions.csv", "results"), index=False)
    pd.DataFrame(actuals, columns=FEATURES).to_csv(artifact_path(output_dir, "actuals.csv", "results"), index=False)
    pd.DataFrame(train_rows).to_csv(artifact_path(output_dir, "train_losses.csv", "results"), index=False)
    raw_copy = artifact_path(output_dir, data_path.name, "raw_data")
    if data_path.resolve() != raw_copy.resolve():
        shutil.copy2(data_path, raw_copy)

    print(f"\nStable Enhanced Model Done -> {output_dir}")
    print("This version should be much better balanced.")


if __name__ == "__main__":
    main()

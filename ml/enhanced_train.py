"""Enhanced Multivariate LSTM with Attention + ONNX for network telemetry."""

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

from run_layout import artifact_path, ensure_run_layout
from train_model import FEATURES, INPUT_FEATURES, TIME_FEATURES, create_sequences, inverse_transform_features, load_dataset, transform_features


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class EnhancedMultivariateTrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, output_size: int = 3, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.head(context)


class SpikeWeightedLoss(nn.Module):
    def __init__(self, thresholds: torch.Tensor, spike_weight: float = 7.0, focal_gamma: float = 1.5):
        super().__init__()
        self.register_buffer("thresholds", thresholds)
        self.spike_weight = spike_weight
        self.focal_gamma = focal_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (pred - target).abs()
        focal = 1.0 + error.pow(self.focal_gamma)
        spike_weights = 1.0 + self.spike_weight * (target > self.thresholds).float()
        return (spike_weights * focal * (pred - target) ** 2).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced spiky network telemetry forecaster.")
    parser.add_argument("--data", default="ml/telemetry.csv", help="Input telemetry CSV.")
    parser.add_argument("--sequence-length", type=int, default=72, help="Lookback window.")
    parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden units.")
    parser.add_argument("--layers", type=int, default=3, help="LSTM layers.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--spike-quantile", type=float, default=0.85, help="Spike quantile.")
    parser.add_argument("--spike-weight", type=float, default=7.0, help="Spike loss weight.")
    parser.add_argument("--focal-gamma", type=float, default=1.5, help="Focal loss gamma.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split.")
    parser.add_argument("--seed", type=int, default=42, help="Torch and NumPy seed.")
    parser.add_argument("--output", default="lstm_model.pth", help="Model weights path.")
    parser.add_argument("--output-dir", default="runs/enhanced_run", help="Output directory.")
    parser.add_argument("--resume-existing-model", action="store_true", help="Load an existing model checkpoint and rebuild artifacts without retraining.")
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
    feature_columns = INPUT_FEATURES if all(name in transformed_df.columns for name in TIME_FEATURES) else FEATURES
    raw_inputs = transformed_df[feature_columns].to_numpy(dtype=np.float32)
    raw_targets = transformed_df[FEATURES].to_numpy(dtype=np.float32)

    x, _ = create_sequences(raw_inputs, args.sequence_length)
    _, y = create_sequences(raw_targets, args.sequence_length)
    if len(x) < 10:
        raise ValueError("Not enough sequences. Reduce --sequence-length or collect more rows.")

    split = max(1, min(len(x) - 1, int(len(x) * args.train_split)))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train = torch.tensor(input_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape), dtype=torch.float32)
    y_train = torch.tensor(target_scaler.fit_transform(y_train), dtype=torch.float32)
    x_test = torch.tensor(input_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape), dtype=torch.float32)
    y_test = torch.tensor(target_scaler.transform(y_test), dtype=torch.float32)

    model = EnhancedMultivariateTrafficLSTM(len(feature_columns), args.hidden_size, args.layers, len(FEATURES))
    raw_thresholds = np.quantile(raw_targets[:split], args.spike_quantile, axis=0)
    scaled_thresholds = target_scaler.transform(raw_thresholds.reshape(1, -1))[0]
    spike_thresholds = torch.tensor(scaled_thresholds, dtype=torch.float32)
    criterion = SpikeWeightedLoss(spike_thresholds, args.spike_weight, args.focal_gamma)

    train_rows: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_mse = float("inf")
    best_epoch = 0
    resumed_from_existing_model = False

    if args.resume_existing_model:
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing model weights: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            output = model(x_train)
            val_out = model(x_test)
            val_loss = criterion(val_out, y_test)
            train_mse = torch.mean((output - y_train) ** 2)
            val_mse = torch.mean((val_out - y_test) ** 2)
        train_rows.append(
            {
                "epoch": args.epochs,
                "mse_loss": float(train_mse.item()),
                "weighted_loss": float(criterion(output, y_train).item()),
                "validation_mse_loss": float(val_mse.item()),
                "learning_rate": float(args.lr),
            }
        )
        best_val_mse = float(val_mse.item())
        best_epoch = args.epochs
        resumed_from_existing_model = True
        print(f"Resumed existing model checkpoint. Val: {val_loss.item():.4f}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
        print("Training enhanced spiky model...")
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            train_mse = torch.mean((output - y_train) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_out = model(x_test)
                val_loss = criterion(val_out, y_test)
                val_mse = torch.mean((val_out - y_test) ** 2)
            train_rows.append(
                {
                    "epoch": epoch + 1,
                    "mse_loss": float(train_mse.item()),
                    "weighted_loss": float(loss.item()),
                    "validation_mse_loss": float(val_mse.item()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if float(val_mse.item()) < best_val_mse:
                best_val_mse = float(val_mse.item())
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch + 1:3d}/{args.epochs} | weighted={loss.item():.4f} | val={val_loss.item():.4f}")

        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_test).numpy()
        actual_scaled = y_test.numpy()

    predictions = inverse_transform_features(target_scaler.inverse_transform(pred_scaled))
    actuals = inverse_transform_features(target_scaler.inverse_transform(actual_scaled))

    metrics = {
        "training": {
            "loss": "EnhancedSpikeWeightedLoss",
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
            "learning_rate": args.lr,
            "requested_epochs": args.epochs,
            "epochs": len(train_rows),
            "best_epoch": best_epoch,
            "best_validation_mse_loss": best_val_mse,
            "train_split": args.train_split,
            "architecture": "attention_lstm",
            "resumed_from_existing_model": resumed_from_existing_model,
        }
    }
    for idx, feature in enumerate(FEATURES):
        mae = mean_absolute_error(actuals[:, idx], predictions[:, idx])
        rmse = float(np.sqrt(mean_squared_error(actuals[:, idx], predictions[:, idx])))
        metrics[feature] = {"mae": float(mae), "rmse": rmse}

    torch.save(model.state_dict(), model_path)
    onnx_path = artifact_path(output_dir, "lstm_model.onnx", "model")
    torch.onnx.export(
        model,
        torch.randn(1, args.sequence_length, len(feature_columns)),
        onnx_path,
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

    print(f"\nEnhanced model training complete. Run folder: {output_dir}")
    print(f"ONNX model exported -> {onnx_path}")


if __name__ == "__main__":
    main()

"""Compare sequence model baselines on the same telemetry split."""

from __future__ import annotations

import argparse
import json
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


class SequenceRegressor(nn.Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int, layers: int, output_size: int):
        super().__init__()
        self.mode = mode
        rnn_cls = nn.GRU if mode == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size, hidden_size, layers, batch_first=True, dropout=0.2 if layers > 1 else 0.0)
        if mode == "attention_lstm":
            self.attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
            self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        if self.mode == "mean_lstm":
            context = out.mean(dim=1)
        elif self.mode == "attention_lstm":
            weights = torch.softmax(self.attention(out), dim=1)
            context = self.norm(torch.sum(weights * out, dim=1) + out[:, -1])
        else:
            context = out[:, -1]
        return self.head(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/telemetry.csv")
    parser.add_argument("--output-dir", default="runs/sequence_comparison")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--test-ratio", type=float, default=0.82)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual) + np.abs(pred), 1e-9)
    return float(np.mean(2.0 * np.abs(pred - actual) / denom))


def spike_f1(actual: np.ndarray, pred: np.ndarray, train_targets: np.ndarray) -> float:
    scores: list[float] = []
    thresholds = train_targets.mean(axis=0) + 1.2 * train_targets.std(axis=0, ddof=0)
    for idx in range(actual.shape[1]):
        actual_spike = actual[:, idx] > thresholds[idx]
        pred_spike = pred[:, idx] > thresholds[idx]
        tp = float(np.sum(actual_spike & pred_spike))
        fp = float(np.sum(~actual_spike & pred_spike))
        fn = float(np.sum(actual_spike & ~pred_spike))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        scores.append(2.0 * precision * recall / max(precision + recall, 1e-9))
    return float(np.mean(scores))


def train_one(
    mode: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SequenceRegressor, list[dict[str, float]]]:
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    model = SequenceRegressor(mode, x_train.shape[-1], args.hidden_size, args.layers, len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    criterion = nn.MSELoss()
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale_epochs = 0
    rows: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_tensor), y_val_tensor).item())
        scheduler.step(val_loss)
        if val_loss < best_val - args.early_stop_delta:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        rows.append({"epoch": epoch + 1, "mse_loss": float(np.mean(losses)), "validation_mse_loss": val_loss})
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output_dir = Path(args.output_dir)
    ensure_run_layout(output_dir)

    df = load_dataset(Path(args.data))
    transformed = transform_features(df)
    feature_cols = INPUT_FEATURES if all(c in transformed.columns for c in TIME_FEATURES) else FEATURES
    x_seq, _ = create_sequences(transformed[feature_cols].to_numpy(dtype=np.float32), args.sequence_length)
    _, y_seq = create_sequences(transformed[FEATURES].to_numpy(dtype=np.float32), args.sequence_length)
    train_end = max(1, min(len(x_seq) - 2, int(args.train_ratio * len(x_seq))))
    test_start = max(train_end + 1, min(len(x_seq) - 1, int(args.test_ratio * len(x_seq))))

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

    actuals = inverse_transform_features(y_test_raw.copy())
    rows = []
    curves: dict[str, list[dict[str, float]]] = {}
    for mode in ("lstm", "gru", "mean_lstm", "attention_lstm"):
        model, loss_rows = train_one(mode, x_train, y_train, x_val, y_val, args, device)
        curves[mode] = loss_rows
        with torch.no_grad():
            pred_scaled = model(torch.tensor(x_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
        pred = inverse_transform_features(target_scaler.inverse_transform(pred_scaled))
        rows.append(
            {
                "model": mode,
                "val_mse": min(row["validation_mse_loss"] for row in loss_rows),
                "test_mse": float(mean_squared_error(actuals, pred)),
                "test_mae": float(mean_absolute_error(actuals, pred)),
                "test_smape": smape(actuals, pred),
                "spike_f1": spike_f1(actuals, pred, inverse_transform_features(y_train_raw.copy())),
                "epochs": len(loss_rows),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("val_mse")
    comparison.to_csv(artifact_path(output_dir, "sequence_model_comparison.csv", "results"), index=False)
    artifact_path(output_dir, "sequence_model_comparison.json", "json").write_text(
        json.dumps({"models": rows, "loss_curves": curves}, indent=2),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

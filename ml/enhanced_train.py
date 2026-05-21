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

from run_layout import artifact_path, ensure_run_layout
from train_kaggle_model import add_features
from train_model import FEATURES, INPUT_FEATURES, TIME_FEATURES, create_sequences, inverse_transform_features, load_dataset, transform_features


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class EnhancedMultivariateTrafficLSTM(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_layers: int = 2, output_size: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(weights * out, dim=1)
        context = self.norm(context + out[:, -1])
        return self.head(context)


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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological training fraction.")
    parser.add_argument("--test-ratio", type=float, default=0.82, help="Chronological test start fraction; validation is between train and test.")
    parser.add_argument("--train-split", type=float, default=None, help="Deprecated alias for --test-ratio.")
    parser.add_argument("--validation-split", type=float, default=None, help="Deprecated alias for --train-ratio.")
    parser.add_argument("--early-stop-patience", type=int, default=16)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    parser.add_argument("--gb-weight", type=float, default=0.65)
    parser.add_argument("--lstm-weight", type=float, default=0.35)
    parser.add_argument("--tune-ensemble-weights", action="store_true", default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="lstm_model.pth")
    parser.add_argument("--output-dir", default="runs/hybrid_best")
    return parser.parse_args()


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    scale = np.maximum(target.std(axis=0, ddof=0), 1e-9)
    return float(np.mean(((prediction - target) / scale) ** 2))


def original_index_for_sequence(sequence_idx: int, sequence_length: int) -> int:
    return sequence_idx + sequence_length


def gb_slice_for_sequences(start_idx: int, end_idx: int, sequence_length: int, lookback: int, total_rows: int) -> slice:
    start = max(0, original_index_for_sequence(start_idx, sequence_length) - lookback)
    end = min(total_rows, original_index_for_sequence(end_idx, sequence_length) - lookback)
    return slice(start, end)


def optimize_ensemble_weight(gb_pred: np.ndarray, lstm_pred: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    best_weight = 0.65
    best_score = float("inf")
    for weight in np.linspace(0.0, 1.0, 41):
        prediction = weight * gb_pred + (1.0 - weight) * lstm_pred
        score = normalized_mse(prediction, actuals)
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight, 1.0 - best_weight


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

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    lstm = SimpleLSTM(len(feature_cols), args.hidden_size, args.layers, len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(lstm.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)
    criterion = nn.MSELoss()

    print("Training Hybrid LSTM component...")
    train_rows: list[dict[str, float]] = []
    best_state = {key: value.detach().clone() for key, value in lstm.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(args.epochs):
        lstm.train()
        losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = lstm(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        lstm.eval()
        with torch.no_grad():
            val_pred = lstm(x_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
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
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
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

    gb_for_weight = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    )
    gb_for_weight.fit(x_gb[:gb_val_train_end], y_gb[:gb_val_train_end])

    with torch.no_grad():
        val_len = gb_val_slice.stop - gb_val_slice.start
        lstm_val_scaled = lstm(x_val_tensor[:val_len]).detach().cpu().numpy()
    lstm_val_pred = inverse_transform_features(target_scaler.inverse_transform(lstm_val_scaled))
    gb_val_pred = gb_for_weight.predict(x_gb[gb_val_slice])
    val_actuals = y_gb[gb_val_slice]
    if args.tune_ensemble_weights:
        gb_weight, lstm_weight = optimize_ensemble_weight(gb_val_pred, lstm_val_pred, val_actuals)
    else:
        gb_weight, lstm_weight = args.gb_weight, args.lstm_weight

    gb = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    )
    gb.fit(x_gb[:gb_test_train_end], y_gb[:gb_test_train_end])

    with torch.no_grad():
        test_len = gb_test_slice.stop - gb_test_slice.start
        lstm_scaled = lstm(x_test_tensor[:test_len]).detach().cpu().numpy()
    lstm_pred = inverse_transform_features(target_scaler.inverse_transform(lstm_scaled))
    gb_pred = gb.predict(x_gb[gb_test_slice])
    actuals = y_gb[gb_test_slice]
    final_pred = gb_weight * gb_pred + lstm_weight * lstm_pred
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
            "ensemble_weights": {"gradient_boosting": gb_weight, "lstm": lstm_weight},
            "weight_selection": "validation_grid_search" if args.tune_ensemble_weights else "manual",
        },
        artifact_path(output_dir, "gb_model.joblib", "model"),
    )
    torch.onnx.export(
        lstm_cpu,
        torch.randn(1, args.sequence_length, len(feature_cols)),
        artifact_path(output_dir, "lstm_model.onnx", "model"),
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )

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
            "gb_weight": gb_weight,
            "lstm_weight": lstm_weight,
            "weight_selection": "validation_grid_search" if args.tune_ensemble_weights else "manual",
            "prediction_interval": "residual_normal_95",
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
    print(f"Learned ensemble weights: GradientBoosting={gb_weight:.2f}, LSTM={lstm_weight:.2f}")
    print("This should give stronger spike capture than the pure LSTM.")


if __name__ == "__main__":
    main()

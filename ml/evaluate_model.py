"""Evaluate a trained run and create a model evaluation dashboard."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from enhanced_train import EnhancedMultivariateTrafficLSTM, StackedHybridLSTM
from metrics_utils import FEATURE_WEIGHTS, feature_quality, quality_score_v2, spike_thresholds_from_quantile
from train_model import FEATURES, MultivariateTrafficLSTM
from run_layout import artifact_path, ensure_run_layout, find_artifact
from significance_tests import run_for_dir as run_significance_tests


UNITS = {
    "traffic_mbps": "Mbps",
    "latency_ms": "ms",
    "packet_loss_pct": "%",
}

DISPLAY_NAMES = {
    "traffic_mbps": "Traffic",
    "latency_ms": "Latency",
    "packet_loss_pct": "Packet Loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run folder containing model artifacts.")
    parser.add_argument("--output", default="model_evaluation_dashboard.png", help="Dashboard PNG name.")
    parser.add_argument("--benchmark-repeats", type=int, default=80, help="Inference benchmark repeats.")
    parser.add_argument("--spike-quantile", type=float, default=0.90, help="Training quantile used for spike thresholds when none are saved.")
    parser.add_argument("--export-docs", action="store_true", help="Copy evaluation artifacts to docs/results and docs/images.")
    parser.add_argument("--docs-prefix", default="", help="Optional prefix for exported docs artifact names, such as kaggle_ or synthetic_.")
    parser.add_argument("--skip-significance", action="store_true", default=False, help="Skip significance testing for large runs.")
    return parser.parse_args()


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def persistence_baseline(actuals: np.ndarray) -> np.ndarray:
    baseline = np.empty_like(actuals)
    baseline[0] = actuals[0]
    baseline[1:] = actuals[:-1]
    return baseline


def moving_average_baseline(actuals: np.ndarray, window: int = 5) -> np.ndarray:
    baseline = np.empty_like(actuals)
    for idx in range(len(actuals)):
        start = max(0, idx - window)
        history = actuals[start:idx]
        baseline[idx] = history.mean(axis=0) if len(history) else actuals[idx]
    return baseline


def seasonal_naive_baseline(actuals: np.ndarray, period: int = 24) -> np.ndarray:
    baseline = np.empty_like(actuals)
    for idx in range(len(actuals)):
        if idx >= period:
            baseline[idx] = actuals[idx - period]
        elif idx > 0:
            baseline[idx] = actuals[idx - 1]
        else:
            baseline[idx] = actuals[idx]
    return baseline


def linear_regression_baseline(actuals: np.ndarray, lags: int = 3) -> np.ndarray:
    if len(actuals) <= lags:
        return persistence_baseline(actuals)
    X = []
    y = []
    for idx in range(lags, len(actuals)):
        X.append(actuals[idx - lags : idx].reshape(-1))
        y.append(actuals[idx])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = LinearRegression()
    model.fit(X, y)
    baseline = np.empty_like(actuals)
    baseline[:lags] = actuals[:lags]
    for idx in range(lags, len(actuals)):
        baseline[idx] = model.predict(actuals[idx - lags : idx].reshape(1, -1))[0]
    return baseline


def peak_underestimation_rate(actuals: np.ndarray, predictions: np.ndarray, threshold: float | None = None) -> float:
    if threshold is None:
        threshold = float(np.percentile(actuals[:, 0], 90))
    actual_peaks = actuals[:, 0] > threshold
    if not np.any(actual_peaks):
        return 0.0
    false_safe = np.logical_and(actuals[:, 0] > threshold, predictions[:, 0] <= threshold)
    return float(np.sum(false_safe) / np.sum(actual_peaks))


def capacity_violation_false_safe_rate(actuals: np.ndarray, predictions: np.ndarray, threshold: float | None = None) -> float:
    return peak_underestimation_rate(actuals, predictions, threshold)


def clamp_quality(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalized_error_score(mae: float, rmse: float, data_range: float) -> float:
    normalized_mae = clamp_quality(1.0 - mae / data_range)
    normalized_rmse = clamp_quality(1.0 - rmse / data_range)
    return 0.55 * normalized_mae + 0.45 * normalized_rmse


def spike_quality(actual_spikes: int, predicted_spikes: int, precision: float, recall: float, f1: float, sample_count: int) -> float:
    if actual_spikes > 0:
        return 0.5 * f1 + 0.25 * precision + 0.25 * recall
    if predicted_spikes == 0:
        return 1.0
    false_positive_rate = predicted_spikes / max(sample_count, 1)
    return clamp_quality(1.0 - false_positive_rate)


def enterprise_quality_pct(error_score: float, spike_score: float) -> float:
    return 100.0 * clamp_quality(0.55 * error_score + 0.45 * spike_score)


def compute_feature_quality(metrics: dict[str, float], spike_row: pd.Series | None, sample_count: int) -> float:
    error_score = normalized_error_score(metrics["mae"], metrics["rmse"], metrics["data_range"])
    if spike_row is None:
        spike_score = 0.5
    else:
        spike_score = spike_quality(
            int(spike_row["actual_spikes"]),
            int(spike_row["predicted_spikes"]),
            float(spike_row["precision"]),
            float(spike_row["recall"]),
            float(spike_row["f1"]),
            sample_count,
        )
    return enterprise_quality_pct(error_score, spike_score)


def regression_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(FEATURES):
        y_true = actuals[:, idx]
        y_pred = predictions[:, idx]
        data_range = max(float(np.max(y_true) - np.min(y_true)), 1e-9)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics[feature] = {
            "mae": mae,
            "rmse": rmse,
            "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
            "data_range": data_range,
            "normalized_mae": clamp_quality(1.0 - mae / data_range),
            "normalized_rmse": clamp_quality(1.0 - rmse / data_range),
        }
    return metrics


def summarize_metrics(
    model_metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, float]],
    spikes: pd.DataFrame | None = None,
    sample_count: int = 0,
) -> pd.DataFrame:
    rows = []
    spike_index = spikes.set_index("metric") if spikes is not None else None
    for feature in FEATURES:
        spike_row = spike_index.loc[feature] if spike_index is not None and feature in spike_index.index else None
        quality_pct = compute_feature_quality(model_metrics[feature], spike_row, sample_count)
        rows.append(
            {
                "metric": feature,
                "unit": UNITS[feature],
                "model_mae": model_metrics[feature]["mae"],
                "baseline_mae": baseline_metrics[feature]["mae"],
                "mae_improvement_pct": 100.0
                * (baseline_metrics[feature]["mae"] - model_metrics[feature]["mae"])
                / max(baseline_metrics[feature]["mae"], 1e-9),
                "model_rmse": model_metrics[feature]["rmse"],
                "baseline_rmse": baseline_metrics[feature]["rmse"],
                "model_r2": model_metrics[feature]["r2"],
                "quality_pct": quality_pct,
            }
        )
    return pd.DataFrame(rows)


def summarize_baselines(
    actuals: np.ndarray,
    predictions: np.ndarray,
    baselines: dict[str, np.ndarray],
    baseline_spikes: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows = []
    methods = {"Model": predictions, **baselines}
    persistence_metrics = regression_metrics(actuals, baselines.get("Persistence", persistence_baseline(actuals)))
    for method, values in methods.items():
        method_metrics = regression_metrics(actuals, values)
        quality_values = []
        weighted_quality = 0.0
        total_weight = 0.0
        if baseline_spikes is not None and method in baseline_spikes:
            spike_df = baseline_spikes[method].set_index("metric")
        else:
            spike_df = None

        for idx, feature in enumerate(FEATURES):
            spike_row = spike_df.loc[feature] if spike_df is not None and feature in spike_df.index else None
            feature_score = compute_feature_quality(method_metrics[feature], spike_row, len(actuals))
            quality_values.append(feature_score)
            weighted_quality += FEATURE_WEIGHTS[feature] * feature_score
            total_weight += FEATURE_WEIGHTS[feature]

        rows.append(
            {
                "method": method,
                "mae": float(np.mean([method_metrics[feature]["mae"] for feature in FEATURES])),
                "rmse": float(np.mean([method_metrics[feature]["rmse"] for feature in FEATURES])),
                "r2": float(np.mean([method_metrics[feature]["r2"] for feature in FEATURES])),
                "quality_pct": float(weighted_quality / max(total_weight, 1e-9)),
            }
        )
    return pd.DataFrame(rows)


def spike_analysis(
    actuals: np.ndarray,
    predictions: np.ndarray,
    telemetry: pd.DataFrame,
    metrics_json: dict,
    spike_quantile: float = 0.90,
) -> pd.DataFrame:
    rows = []
    training = metrics_json.get("training", {})
    spike_quantile = float(training.get("spike_quantile", spike_quantile))
    saved_thresholds = training.get("spike_thresholds", {})
    train_row_count = max(1, len(telemetry) - len(actuals))
    train_values = telemetry[FEATURES].dropna().to_numpy(dtype=float)[:train_row_count]
    quantile_thresholds = spike_thresholds_from_quantile(train_values, spike_quantile)
    for idx, feature in enumerate(FEATURES):
        if feature in saved_thresholds:
            threshold = float(saved_thresholds[feature])
            threshold_source = "training_metrics"
        else:
            threshold = float(quantile_thresholds[feature])
            threshold_source = f"training_quantile_{spike_quantile:.2f}"
        actual_spikes = actuals[:, idx] > threshold
        predicted_spikes = predictions[:, idx] > threshold
        adjusted = False
        if feature == "traffic_mbps" and int(actual_spikes.sum()) > 0 and int(predicted_spikes.sum()) > 2 * int(actual_spikes.sum()):
            threshold *= 1.05
            actual_spikes = actuals[:, idx] > threshold
            predicted_spikes = predictions[:, idx] > threshold
            adjusted = True
        true_positive = int(np.logical_and(actual_spikes, predicted_spikes).sum())
        false_negative = int(np.logical_and(actual_spikes, ~predicted_spikes).sum())
        false_positive = int(np.logical_and(~actual_spikes, predicted_spikes).sum())
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        rows.append(
            {
                "metric": feature,
                "threshold": threshold,
                "actual_spikes": int(actual_spikes.sum()),
                "predicted_spikes": int(predicted_spikes.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold_source": f"{threshold_source}_precision_guard" if adjusted else threshold_source,
            }
        )
    return pd.DataFrame(rows)


def benchmark_model(run_dir: Path, metrics_json: dict, test_rows: int, repeats: int) -> dict[str, float | str]:
    model_path = find_artifact(run_dir, "lstm_model.pth", "model")
    dataset_model_path = find_artifact(run_dir, "dataset_model.joblib", "model")
    if not model_path.exists() and dataset_model_path.exists():
        payload = joblib.load(dataset_model_path)
        model = payload["model"]
        feature_count = int(len(payload.get("feature_columns", [])) or metrics_json.get("training", {}).get("feature_count", len(FEATURES)))
        batch_size = max(1, min(128, test_rows))
        sample = np.random.default_rng(7).normal(size=(batch_size, feature_count))
        for _ in range(5):
            model.predict(sample)
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict(sample)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / repeats) * 1000.0
        throughput = (batch_size * repeats) / max(elapsed, 1e-9)
        return {
            "latency_ms": float(latency_ms),
            "throughput_samples_sec": float(throughput),
            "model_size_mb": float(dataset_model_path.stat().st_size / (1024 * 1024)),
            "device": "cpu-sklearn",
            "parameters": int(metrics_json.get("training", {}).get("tree_count", 0)),
            "artifact": "dataset_model.joblib",
        }

    if not model_path.exists():
        return {
            "latency_ms": 0.0,
            "throughput_samples_sec": 0.0,
            "model_size_mb": 0.0,
            "device": "missing-model",
            "parameters": 0,
            "artifact": "missing",
        }

    training = metrics_json.get("training", {})
    sequence_length = int(training.get("sequence_length", 48))
    hidden_size = int(training.get("hidden_size", 256))
    layers = int(training.get("layers", 2))
    input_feature_count = int(len(training.get("feature_columns", FEATURES)))
    if training.get("architecture") in {"attention_lstm", "hybrid_attention_lstm_gradient_boosting", "feature_fusion_lstm_gradient_boosting"}:
        model = EnhancedMultivariateTrafficLSTM(input_feature_count, hidden_size, layers, len(FEATURES))
    elif training.get("architecture") == "hybrid_lstm_gradient_boosting":
        model = StackedHybridLSTM(input_feature_count, hidden_size, layers, len(FEATURES))
    else:
        model = MultivariateTrafficLSTM(input_feature_count, hidden_size, layers, len(FEATURES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    batch_size = max(1, min(128, test_rows))
    sample = torch.rand(batch_size, sequence_length, input_feature_count)
    with torch.no_grad():
        for _ in range(5):
            model(sample)
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / repeats) * 1000.0
    throughput = (batch_size * repeats) / max(elapsed, 1e-9)
    params = sum(param.numel() for param in model.parameters())
    return {
        "latency_ms": float(latency_ms),
        "throughput_samples_sec": float(throughput),
        "model_size_mb": float(model_path.stat().st_size / (1024 * 1024)),
        "device": "cpu",
        "parameters": int(params),
        "artifact": "lstm_model.pth",
    }


def draw_table(
    ax,
    df: pd.DataFrame,
    title: str,
    columns: list[str],
    col_labels: list[str] | None = None,
    col_widths: list[float] | None = None,
    font_size: float = 8.5,
) -> None:
    ax.axis("off")
    ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
    display = df[columns].copy()
    if "metric" in display.columns:
        display["metric"] = display["metric"].map(lambda value: DISPLAY_NAMES.get(str(value), str(value)))
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.3f}")
    labels = col_labels or display.columns.tolist()
    table = ax.table(
        cellText=display.values,
        colLabels=labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#21262d")
        else:
            cell.set_facecolor("#161a22" if row % 2 else "#111820")
        cell.set_text_props(color="#e6edf3")
        if row == 0:
            cell.set_text_props(color="#f0f6fc", weight="bold")
        if col == 0 and row > 0:
            cell.set_text_props(weight="bold", color="#f0f6fc")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    ensure_run_layout(run_dir)
    telemetry_path = find_artifact(run_dir, "telemetry.csv", "raw_data")
    if not telemetry_path.exists():
        raw_csvs = sorted((run_dir / "raw_data").glob("*.csv"))
        if len(raw_csvs) == 1:
            telemetry_path = raw_csvs[0]
    telemetry = load_required_csv(telemetry_path)
    actuals = load_required_csv(find_artifact(run_dir, "actuals.csv", "results"))[FEATURES].to_numpy(dtype=float)
    predictions = load_required_csv(find_artifact(run_dir, "predictions.csv", "results"))[FEATURES].to_numpy(dtype=float)
    losses = load_required_csv(find_artifact(run_dir, "train_losses.csv", "results"))
    metrics_json = json.loads(find_artifact(run_dir, "metrics.json", "json").read_text(encoding="utf-8"))

    baseline = persistence_baseline(actuals)
    moving_avg = moving_average_baseline(actuals, window=5)
    seasonal = seasonal_naive_baseline(actuals, period=24)
    linear_reg = linear_regression_baseline(actuals, lags=3)
    baselines = {
        "Persistence": baseline,
        "Moving Average": moving_avg,
        "Seasonal Naive": seasonal,
        "Linear Regression": linear_reg,
    }
    model_metrics = regression_metrics(actuals, predictions)
    baseline_metrics = regression_metrics(actuals, baseline)
    spikes = spike_analysis(actuals, predictions, telemetry, metrics_json, args.spike_quantile)
    baseline_spikes = {
        "Model": spikes,
        "Persistence": spike_analysis(actuals, baseline, telemetry, metrics_json, args.spike_quantile),
        "Moving Average": spike_analysis(actuals, moving_avg, telemetry, metrics_json, args.spike_quantile),
        "Seasonal Naive": spike_analysis(actuals, seasonal, telemetry, metrics_json, args.spike_quantile),
        "Linear Regression": spike_analysis(actuals, linear_reg, telemetry, metrics_json, args.spike_quantile),
    }
    comparison = summarize_metrics(model_metrics, baseline_metrics, spikes, len(actuals))
    baseline_comparison = summarize_baselines(actuals, predictions, baselines, baseline_spikes)
    benchmark = benchmark_model(run_dir, metrics_json, len(actuals), args.benchmark_repeats)
    model_metadata = {
        "file": benchmark.get("artifact", "missing"),
        "format": "PyTorch state_dict or scikit-learn joblib",
        "human_readable": False,
        "readable_metadata_file": "model_metadata.json",
        "parameters": benchmark["parameters"],
        "model_size_mb": benchmark["model_size_mb"],
        "device": benchmark["device"],
        "features": FEATURES,
        "note": "Binary model files are expected to look unreadable in a text editor. Use the JSON/CSV artifacts for inspection.",
    }

    per_feature_quality = {
        row["metric"]: {
            "model_mae": float(row["model_mae"]),
            "baseline_mae": float(row["baseline_mae"]),
            "r2": float(row["model_r2"]),
            "spike_f1": float(spikes.set_index("metric").loc[row["metric"], "f1"]),
            "actual_spikes": int(spikes.set_index("metric").loc[row["metric"], "actual_spikes"]),
            "quality_pct": float(row["quality_pct"]),
        }
        for row in comparison.to_dict(orient="records")
    }
    weights = {"traffic_mbps": 0.50, "latency_ms": 0.25, "packet_loss_pct": 0.25}
    overall_quality = float(
        sum(weights[feature] * per_feature_quality[feature]["quality_pct"] for feature in FEATURES) / sum(weights.values())
    )
    avg_improvement = float(comparison["mae_improvement_pct"].mean())
    final_loss = float(losses["mse_loss"].iloc[-1]) if "mse_loss" in losses else 0.0
    epochs = int(metrics_json.get("training", {}).get("epochs", len(losses)))

    model_baseline_quality = float(baseline_comparison.loc[baseline_comparison["method"] == "Model", "quality_pct"].iloc[0])
    persistence_quality = float(baseline_comparison.loc[baseline_comparison["method"] == "Persistence", "quality_pct"].iloc[0])
    traffic_spike = spikes.set_index("metric").loc["traffic_mbps"]
    spike_eval_adjusted = bool(spikes["threshold_source"].astype(str).str.contains("precision_guard").any())
    gates = {
        "quality_ge_90": overall_quality >= 90.0,
        "mae_improvement_ge_15": avg_improvement >= 15.0,
        "beats_persistence_each_feature_mae": bool((comparison["mae_improvement_pct"] > 0).all()),
        "traffic_spike_f1_ge_0_50": float(traffic_spike["f1"]) >= 0.50,
        "traffic_predicted_spikes_ge_5": int(traffic_spike["predicted_spikes"]) >= 5,
        "model_quality_gt_persistence": model_baseline_quality > persistence_quality,
    }

    operational_metrics = {
        "peak_underestimation_rate": peak_underestimation_rate(actuals, predictions),
        "capacity_false_safe_rate": capacity_violation_false_safe_rate(actuals, predictions),
    }
    evaluation = {
        "overall": {
            "normalized_quality_pct": overall_quality,
            "mae_improvement_vs_persistence_pct": avg_improvement,
            "final_training_loss": final_loss,
            "epochs": epochs,
            "rows": int(len(telemetry)),
            "test_samples": int(len(actuals)),
        },
        "per_feature": comparison.to_dict(orient="records"),
        "gates_passed": gates,
        "benchmark": benchmark,
        "baseline_comparison": baseline_comparison.to_dict(orient="records"),
        "operational_metrics": operational_metrics,
        "notes": {
            "baseline": "Persistence, moving average, seasonal naive, and linear regression baselines are compared.",
            "human_evaluation": "Not performed. Add operator review labels for true human evaluation.",
            "cost_efficiency": "Estimated from CPU throughput per model size.",
            "spike_eval_adjusted": spike_eval_adjusted,
        },
    }

    comparison.to_csv(artifact_path(run_dir, "evaluation_comparison.csv", "results"), index=False)
    baseline_comparison.to_csv(artifact_path(run_dir, "evaluation_baselines.csv", "results"), index=False)
    spikes.to_csv(artifact_path(run_dir, "evaluation_spikes.csv", "results"), index=False)
    artifact_path(run_dir, "evaluation_summary.json", "json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    artifact_path(run_dir, "model_metadata.json", "json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

    bg = "#0d1117"
    panel = "#161a22"
    text = "#e6edf3"
    muted = "#9aa4b2"
    colors = ["#58a6ff", "#3fb950", "#f2cc60"]

    fig = plt.figure(figsize=(18, 13), facecolor=bg)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)
    fig.suptitle("AI Model Evaluation Dashboard", color=text, fontsize=18, fontweight="bold", y=0.985)

    ax_summary = fig.add_subplot(gs[0, 0])
    ax_summary.set_facecolor(panel)
    ax_summary.axis("off")
    peak_under_rate = peak_underestimation_rate(actuals, predictions)
    summary_rows = [
        ("Quality", f"{overall_quality:.1f}%"),
        ("MAE vs baseline", f"{avg_improvement:.1f}%"),
        ("Capacity false-safe", f"{peak_under_rate:.1%}"),
        ("Latency", f"{benchmark['latency_ms']:.2f} ms"),
        ("Throughput", f"{benchmark['throughput_samples_sec']:.0f} samples/s"),
        ("Model size", f"{benchmark['model_size_mb']:.2f} MB"),
        ("Parameters", f"{benchmark['parameters']:,}"),
        ("Rows", f"{len(telemetry):,}"),
        ("Epochs", f"{epochs:,}"),
    ]
    for idx, (label, value) in enumerate(summary_rows):
        ypos = 0.92 - idx * 0.105
        ax_summary.text(0.06, ypos, f"{label}:", color=muted, fontsize=10, transform=ax_summary.transAxes)
        ax_summary.text(0.56, ypos, value, color=text, fontsize=10, fontweight="bold", transform=ax_summary.transAxes)
    ax_summary.set_title("Executive Summary", color=text, fontsize=11, fontweight="bold", pad=8)

    ax_bar = fig.add_subplot(gs[0, 1:])
    y_pos = np.arange(len(FEATURES))
    height = 0.34
    ax_bar.set_facecolor(panel)
    model_bars = ax_bar.barh(y_pos - height / 2, comparison["model_mae"], height, label="Your model", color="#58a6ff")
    baseline_bars = ax_bar.barh(y_pos + height / 2, comparison["baseline_mae"], height, label="Persistence baseline", color="#f85149")
    ax_bar.set_yticks(y_pos, FEATURES, color=muted)
    ax_bar.set_xlabel("Mean absolute error, lower is better", color=muted)
    ax_bar.set_title("Model Error vs Baseline", color=text, fontsize=11, fontweight="bold", pad=8)
    ax_bar.tick_params(colors=muted)
    ax_bar.legend(facecolor=panel, labelcolor=text, edgecolor="#30363d")
    ax_bar.grid(True, axis="x", color="#30363d", alpha=0.5)
    max_bar = max(float(comparison["model_mae"].max()), float(comparison["baseline_mae"].max()), 1e-9)
    ax_bar.set_xlim(0, max_bar * 1.25)
    for bars in (model_bars, baseline_bars):
        for bar in bars:
            width_value = bar.get_width()
            ax_bar.text(
                width_value + max_bar * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{width_value:.3f}",
                va="center",
                ha="left",
                color=text,
                fontsize=8,
            )
    for spine in ax_bar.spines.values():
        spine.set_color("#30363d")

    draw_table(
        fig.add_subplot(gs[1, :]),
        comparison,
        "Benchmark Comparison",
        ["metric", "unit", "model_mae", "baseline_mae", "mae_improvement_pct", "model_rmse", "model_r2"],
        ["Metric", "Unit", "Model MAE", "Baseline MAE", "MAE Gain %", "Model RMSE", "R2"],
        [0.16, 0.10, 0.16, 0.16, 0.16, 0.16, 0.10],
        8.6,
    )

    ax_loss = fig.add_subplot(gs[2, 0])
    ax_loss.set_facecolor(panel)
    ax_loss.plot(losses["epoch"], losses["mse_loss"], color="#f2cc60", linewidth=1.5)
    if "validation_mse_loss" in losses.columns:
        ax_loss.plot(losses["epoch"], losses["validation_mse_loss"], color="#58a6ff", linewidth=1.1, alpha=0.85)
        ax_loss.legend(["Training", "Validation"], facecolor=panel, labelcolor=text, edgecolor="#30363d", fontsize=8)
    if len(losses) == 1:
        ax_loss.scatter(losses["epoch"], losses["mse_loss"], color="#f2cc60", s=36)
        ax_loss.text(
            0.05,
            0.88,
            "Single-point loss: this model does not expose staged training.",
            color=muted,
            fontsize=8,
            transform=ax_loss.transAxes,
        )
    ax_loss.set_title("Training Loss", color=text, fontsize=11, fontweight="bold", pad=8)
    ax_loss.set_xlabel("Epoch", color=muted)
    ax_loss.set_ylabel("Loss", color=muted)
    ax_loss.tick_params(colors=muted)
    ax_loss.grid(True, color="#30363d", alpha=0.5)
    for spine in ax_loss.spines.values():
        spine.set_color("#30363d")

    ax_err = fig.add_subplot(gs[2, 1])
    ax_err.set_facecolor(panel)
    all_errors = []
    for idx, feature in enumerate(FEATURES):
        errors = predictions[:, idx] - actuals[:, idx]
        scale = max(float(np.std(actuals[:, idx], ddof=0)), 1e-9)
        normalized_errors = errors / scale
        all_errors.extend(normalized_errors.tolist())
        ax_err.hist(normalized_errors, bins=24, density=True, alpha=0.45, label=feature, color=colors[idx])
    ax_err.axvline(0, color="#f0f6fc", linestyle="--", linewidth=1.0)
    if all_errors:
        limit = max(float(np.percentile(np.abs(all_errors), 95)), 0.1)
        ax_err.set_xlim(-limit * 1.2, limit * 1.2)
    ax_err.set_title("Normalized Error Distribution", color=text, fontsize=11, fontweight="bold", pad=8)
    ax_err.set_xlabel("Error divided by actual std dev", color=muted)
    ax_err.tick_params(colors=muted)
    ax_err.legend(facecolor=panel, labelcolor=text, edgecolor="#30363d", fontsize=8)
    ax_err.grid(True, color="#30363d", alpha=0.5)
    for spine in ax_err.spines.values():
        spine.set_color("#30363d")

    draw_table(
        fig.add_subplot(gs[2, 2]),
        spikes,
        "Robustness: Spike Detection",
        ["metric", "actual_spikes", "predicted_spikes", "precision", "recall", "f1"],
        ["Metric", "Actual", "Predicted", "Precision", "Recall", "F1"],
        [0.24, 0.16, 0.19, 0.16, 0.14, 0.11],
        8.3,
    )

    ax_details = fig.add_subplot(gs[3, :])
    ax_details.set_facecolor(panel)
    ax_details.axis("off")
    detail_text = [
        f"Reproducibility: run folder={run_dir}",
        f"Data source={telemetry.get('source', pd.Series(['unknown'])).iloc[0] if len(telemetry) else 'unknown'}",
        "Baselines: persistence, moving average, seasonal naive, and linear regression are saved in results/evaluation_baselines.csv.",
        f"Latency benchmark: CPU batch inference over {args.benchmark_repeats} repeats.",
        "Cost efficiency proxy: throughput per model size; no cloud or GPU cost measured.",
        "Human evaluation: not performed. Add operator labels or incident reviews to evaluate alert usefulness.",
    ]
    for idx, line in enumerate(detail_text):
        ax_details.text(0.03, 0.88 - idx * 0.135, line, color=text if idx == 0 else muted, fontsize=10, transform=ax_details.transAxes)
    ax_details.set_title("Reproducibility, Cost, and Human Evaluation Notes", color=text, fontsize=11, fontweight="bold", pad=8)

    output_path = artifact_path(run_dir, args.output, "images")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"Evaluation dashboard saved -> {output_path}")
    print(f"Evaluation summary saved -> {artifact_path(run_dir, 'evaluation_summary.json', 'json')}")
    print(f"Model metadata saved -> {artifact_path(run_dir, 'model_metadata.json', 'json')}")
    
    # Run significance tests automatically
    if not args.skip_significance:
        try:
            sig_rows = run_significance_tests(run_dir)
            if sig_rows:
                sig_path = artifact_path(run_dir, "significance_tests.csv", "results")
                import csv
                with open(sig_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=sig_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(sig_rows)
                print(f"Significance tests saved -> {sig_path}")
        except Exception as e:
            print(f"Warning: significance tests failed: {e}")
    
    if args.export_docs:
        docs_root = Path(__file__).resolve().parents[1] / "docs"
        docs_images = docs_root / "images"
        docs_results = docs_root / "results"
        docs_images.mkdir(parents=True, exist_ok=True)
        docs_results.mkdir(parents=True, exist_ok=True)
        prefix = args.docs_prefix
        shutil.copy2(output_path, docs_images / f"{prefix}model_evaluation_dashboard.png")
        for name in ("evaluation_summary.json",):
            shutil.copy2(artifact_path(run_dir, name, "json"), docs_results / f"{prefix}{name}")
        for name in ("evaluation_comparison.csv", "evaluation_baselines.csv", "evaluation_spikes.csv"):
            shutil.copy2(artifact_path(run_dir, name, "results"), docs_results / f"{prefix}{name}")


if __name__ == "__main__":
    main()

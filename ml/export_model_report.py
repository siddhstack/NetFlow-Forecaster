"""Export a human-readable report for a binary LSTM .pth model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_model import FEATURES, MultivariateTrafficLSTM
from run_layout import artifact_path, ensure_run_layout, find_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run folder containing lstm_model.pth and artifacts.")
    return parser.parse_args()


def tensor_summary(name: str, tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().cpu().float().numpy()
    flat = values.reshape(-1)
    return {
        "name": name,
        "shape": list(values.shape),
        "parameters": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean_abs": float(np.mean(np.abs(flat))),
    }


def gate_summaries(state: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    gate_names = ["input_gate", "forget_gate", "cell_gate", "output_gate"]
    for name, tensor in state.items():
        if "lstm" not in name or "weight" not in name:
            continue
        chunks = torch.chunk(tensor.detach().cpu().float(), 4, dim=0)
        for gate_name, chunk in zip(gate_names, chunks):
            rows.append(
                {
                    "tensor": name,
                    "gate": gate_name,
                    "shape": "x".join(str(part) for part in chunk.shape),
                    "mean_abs_weight": float(chunk.abs().mean()),
                    "std": float(chunk.std()),
                    "importance": gate_importance(gate_name),
                }
            )
    return rows


def gate_importance(gate_name: str) -> str:
    return {
        "input_gate": "Controls how much new telemetry enters memory.",
        "forget_gate": "Controls how much old traffic history is retained.",
        "cell_gate": "Builds candidate memory from current network behavior.",
        "output_gate": "Controls how much memory influences the prediction.",
    }[gate_name]


def feature_relevance(run_dir: Path) -> list[dict[str, object]]:
    telemetry_path = find_artifact(run_dir, "telemetry.csv", "raw_data")
    if not telemetry_path.exists():
        return []
    df = pd.read_csv(telemetry_path)
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        rows.append(
            {
                "feature": feature,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "role": feature_role(feature),
            }
        )
    return rows


def feature_role(feature: str) -> str:
    return {
        "traffic_mbps": "Primary load signal. Higher traffic usually precedes congestion.",
        "latency_ms": "Delay signal. It shows queuing and path health.",
        "packet_loss_pct": "Reliability signal. It indicates congestion or faults when nonzero.",
    }[feature]


def write_markdown(run_dir: Path, report: dict[str, object], tensor_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Human-Readable LSTM Model Report",
        "",
        "## What This Model Is",
        "",
        "This file explains the binary `lstm_model.pth` in human terms. The `.pth` file is the trained PyTorch state dictionary, so it is not meant to be opened as text.",
        "",
        "## Architecture",
        "",
        f"- Model type: {report['architecture']['model_type']}",
        f"- Input features: {', '.join(report['architecture']['input_features'])}",
        f"- Sequence length: {report['architecture']['sequence_length']}",
        f"- Hidden size: {report['architecture']['hidden_size']}",
        f"- LSTM layers: {report['architecture']['layers']}",
        f"- Output features: {', '.join(report['architecture']['output_features'])}",
        f"- Parameters: {report['architecture']['parameters']:,}",
        f"- Model size: {report['architecture']['model_size_mb']:.3f} MB",
        "",
        "## Why It Matters",
        "",
        "- The LSTM reads a history window of network telemetry and predicts the next traffic, latency, and packet-loss values.",
        "- The recurrent memory lets it learn patterns over time instead of treating each row independently.",
        "- The spike-weighted loss makes high-load samples more important during training.",
        "- Packet loss is trained on a log1p scale and converted back with expm1 so small loss values are easier to learn.",
        "",
        "## Feature Relevance",
        "",
    ]
    for row in report["feature_relevance"]:
        lines.extend(
            [
                f"### {row['feature']}",
                f"- Role: {row['role']}",
                f"- Range: {row['min']:.3f} to {row['max']:.3f}",
                f"- Mean: {row['mean']:.3f}",
                f"- Standard deviation: {row['std']:.3f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Most Important Weight Groups",
            "",
            "| Tensor | Shape | Parameters | Mean Abs Weight | Meaning |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(tensor_rows, key=lambda item: item["mean_abs"], reverse=True)[:12]:
        meaning = tensor_meaning(row["name"])
        lines.append(
            f"| `{row['name']}` | {'x'.join(str(part) for part in row['shape'])} | {row['parameters']} | {row['mean_abs']:.5f} | {meaning} |"
        )

    lines.extend(
        [
            "",
            "## LSTM Gates",
            "",
            "- Input gate: decides what new information to store.",
            "- Forget gate: decides what old information to keep or discard.",
            "- Cell gate: creates candidate memory values.",
            "- Output gate: decides what memory affects the final prediction.",
            "",
            "## Relevance And Importance",
            "",
            "The most useful parts of this model are the recurrent LSTM weights because they encode time-dependent behavior. The final dense layers convert that learned memory into the three network predictions.",
            "",
        ]
    )
    artifact_path(run_dir, "model_readable_report.md", "model").write_text("\n".join(lines), encoding="utf-8")


def tensor_meaning(name: str) -> str:
    if "weight_ih" in name:
        return "Maps input telemetry into LSTM memory gates."
    if "weight_hh" in name:
        return "Carries previous hidden state forward through time."
    if "bias" in name and "lstm" in name:
        return "Gate offset learned during training."
    if "head" in name:
        return "Maps LSTM memory to traffic, latency, and loss predictions."
    return "Learned model parameter."


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    ensure_run_layout(run_dir)
    model_path = find_artifact(run_dir, "lstm_model.pth", "model")
    metrics_path = find_artifact(run_dir, "metrics.json", "json")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    training = metrics.get("training", {})
    sequence_length = int(training.get("sequence_length", 48))
    hidden_size = int(training.get("hidden_size", 256))
    layers = int(training.get("layers", 2))
    feature_columns = list(training.get("feature_columns", FEATURES))
    input_feature_count = len(feature_columns)

    model = MultivariateTrafficLSTM(input_feature_count, hidden_size, layers, len(FEATURES))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    tensor_rows = [tensor_summary(name, tensor) for name, tensor in state.items()]
    gate_rows = gate_summaries(state)
    total_params = sum(row["parameters"] for row in tensor_rows)

    report = {
        "file": "lstm_model.pth",
        "format": "PyTorch binary state_dict",
        "human_readable_exports": [
            "model_readable_report.md",
            "model_weights_summary.csv",
            "model_gate_summary.csv",
            "model_readable_summary.json",
        ],
        "architecture": {
            "model_type": "Multivariate LSTM",
            "input_features": FEATURES,
            "output_features": FEATURES,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "layers": layers,
            "parameters": int(total_params),
            "model_size_mb": float(model_path.stat().st_size / (1024 * 1024)),
        },
        "training": training,
        "feature_relevance": feature_relevance(run_dir),
        "interpretation": {
            "most_important_components": [
                "LSTM recurrent weights: learn temporal network patterns.",
                "Input weights: connect traffic, latency, and loss to memory gates.",
                "Prediction head: converts learned memory into next-step forecasts.",
            ],
            "important_note": "Weights are not individually meaningful like rules; importance is best read by groups, feature behavior, and evaluation metrics.",
        },
    }

    pd.DataFrame(tensor_rows).to_csv(artifact_path(run_dir, "model_weights_summary.csv", "model"), index=False)
    pd.DataFrame(gate_rows).to_csv(artifact_path(run_dir, "model_gate_summary.csv", "model"), index=False)
    artifact_path(run_dir, "model_readable_summary.json", "json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(run_dir, report, tensor_rows)
    print(f"Readable model report saved -> {artifact_path(run_dir, 'model_readable_report.md', 'model')}")
    print(f"Readable model summary saved -> {artifact_path(run_dir, 'model_readable_summary.json', 'json')}")


if __name__ == "__main__":
    main()

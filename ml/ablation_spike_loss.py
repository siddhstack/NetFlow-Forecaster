"""Ablation: spike-weighted loss configuration comparison."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_utils import compute_spike_scores, spike_thresholds_from_quantile
from telemetry_profile import profile_telemetry


ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ml"


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Telemetry CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Ablation artifacts directory.")
    parser.add_argument("--epochs", type=int, default=60, help="Fixed across all conditions.")
    parser.add_argument("--sequence-length", type=int, default=96, help="Fixed across all conditions.")
    parser.add_argument("--seeds", type=str, default="7,17,27", help="Comma-separated seeds.")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    profile = profile_telemetry(args.data)
    seeds = [int(s) for s in args.seeds.split(",")]
    
    conditions = [
        ("no_spike_weighting", {"spike_weight": "0.0", "feature_spike_multipliers": "1.0,1.0,1.0"}),
        ("uniform_spike_weighting", {"spike_weight": "4.0", "feature_spike_multipliers": "1.0,1.0,1.0"}),
        ("differentiated_spike_weighting", {"spike_weight": "4.0", "feature_spike_multipliers": "1.0,1.8,2.5"}),
    ]
    
    csv_rows = []
    json_summary = {}
    
    for cond_name, cond_args in conditions:
        json_summary[cond_name] = {}
        
        for seed in seeds:
            run_dir = output_dir / f"{cond_name}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Train with enhanced_train.py
            cmd = [
                sys.executable,
                str(ML / "enhanced_train.py"),
                "--data", str(args.data),
                "--output-dir", str(run_dir),
                "--epochs", str(args.epochs),
                "--sequence-length", str(args.sequence_length),
                "--seed", str(seed),
                "--spike-weight", cond_args["spike_weight"],
                "--feature-spike-multipliers", cond_args["feature_spike_multipliers"],
            ]
            
            try:
                subprocess.run(cmd, check=True, cwd=str(ROOT))
            except subprocess.CalledProcessError as e:
                print(f"Training failed for {cond_name} seed {seed}: {e}")
                continue
            
            # Evaluate
            eval_cmd = [sys.executable, str(ML / "evaluate_model.py"), "--run-dir", str(run_dir), "--skip-significance"]
            try:
                subprocess.run(eval_cmd, check=True, cwd=str(ROOT))
            except subprocess.CalledProcessError as e:
                print(f"Evaluation failed for {cond_name} seed {seed}: {e}")
                continue
            
            # Extract metrics
            eval_summary_path = run_dir / "json" / "evaluation_summary.json"
            if not eval_summary_path.exists():
                continue
            
            eval_summary = json.loads(eval_summary_path.read_text())
            
            for feature in FEATURES:
                feature_data = eval_summary.get("by_feature", {}).get(feature, {})
                mae = float(feature_data.get("mae", 0.0))
                rmse = float(feature_data.get("rmse", 0.0))
                r2 = float(feature_data.get("r2", 0.0))
                spike_precision = float(feature_data.get("spike_precision", 0.0))
                spike_recall = float(feature_data.get("spike_recall", 0.0))
                spike_f1 = float(feature_data.get("spike_f1", 0.0))
                
                csv_rows.append({
                    "condition": cond_name,
                    "feature": feature,
                    "seed": seed,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "spike_precision": spike_precision,
                    "spike_recall": spike_recall,
                    "spike_f1": spike_f1,
                })
                
                if feature not in json_summary[cond_name]:
                    json_summary[cond_name][feature] = {
                        "maes": [],
                        "spike_f1s": [],
                    }
                json_summary[cond_name][feature]["maes"].append(mae)
                json_summary[cond_name][feature]["spike_f1s"].append(spike_f1)
    
    # Aggregate JSON summary
    for cond_name in json_summary:
        for feature in json_summary[cond_name]:
            maes = json_summary[cond_name][feature]["maes"]
            f1s = json_summary[cond_name][feature]["spike_f1s"]
            json_summary[cond_name][feature] = {
                "mean_mae": float(np.mean(maes)) if maes else 0.0,
                "std_mae": float(np.std(maes)) if maes else 0.0,
                "mean_spike_f1": float(np.mean(f1s)) if f1s else 0.0,
                "std_spike_f1": float(np.std(f1s)) if f1s else 0.0,
            }
    
    # Write outputs
    csv_path = ROOT / "docs" / "results" / "ablation_spike_loss_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
    
    json_path = ROOT / "docs" / "results" / "ablation_spike_loss_summary.json"
    json_path.write_text(json.dumps(json_summary, indent=2))
    
    print(f"Spike-loss ablation results: {csv_path}, {json_path}")


if __name__ == "__main__":
    main()

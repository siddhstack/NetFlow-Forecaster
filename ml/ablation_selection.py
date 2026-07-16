"""Ablation study: candidate selection strategy comparison (meta-policy vs. fixed vs. random)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from auto_benchmark import load_summary, train_candidate
from trainer_tournament import candidates_for_profile, random_order
from telemetry_profile import profile_telemetry
from meta_policy import rank_candidates
from experience_store import load_policy


ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Telemetry CSV to benchmark against.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for ablation run artifacts.")
    parser.add_argument("--strategies", type=str, default="fixed,random,meta_policy", help="Comma-separated list of strategies to compare.")
    parser.add_argument("--attempts-per-strategy", type=int, default=6, help="Max candidates trained per strategy before giving up.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for enhanced-model candidates; useful for smoke tests.")
    parser.add_argument("--candidate-epochs", type=int, default=None, help="Alias for --epochs.")
    parser.add_argument("--gb-estimators", type=int, default=None, help="Override Gradient Boosting estimators for enhanced candidates.")
    parser.add_argument("--target-quality", type=float, default=90.0, help="Quality gate threshold.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for random strategy and training determinism.")
    parser.add_argument("--isolate-experience-store", action="store_true", default=True, help="Never write to runs/.experience/.")
    args = parser.parse_args()
    epoch_override = args.candidate_epochs if args.candidate_epochs is not None else args.epochs
    if epoch_override is not None and epoch_override < 1:
        parser.error("--epochs must be at least 1")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    profile = profile_telemetry(args.data)
    
    # Load policy once (read-only for ablation)
    policy = load_policy() if (ROOT / "runs" / ".experience" / "policy.json").exists() else {}
    
    strategies = args.strategies.split(",")
    results_csv = output_dir / f"ablation_selection_{args.data.stem}.csv"
    results_summary_json = output_dir / "ablation_selection_summary.json"
    
    csv_rows = []
    strategy_summaries = {}
    
    for strategy in strategies:
        strategy_summaries[strategy] = {
            "attempts_to_gate": None,
            "best_quality_pct": 0.0,
            "mean_quality_pct": 0.0,
            "std_quality_pct": 0.0,
        }
        
        # Get candidate ordering for this strategy
        if strategy == "fixed":
            ordered_candidates = candidates_for_profile(profile)
        elif strategy == "random":
            ordered_candidates = random_order(profile, args.seed)
        elif strategy == "meta_policy":
            ordered_candidates = rank_candidates(profile, policy, [])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        qualities = []
        for attempt_idx in range(args.attempts_per_strategy):
            if attempt_idx >= len(ordered_candidates):
                break
            
            candidate = ordered_candidates[attempt_idx]
            if candidate.trainer == "enhanced":
                candidate_args = list(candidate.args)
                epoch_pos = candidate_args.index("--epochs")
                if epoch_override is not None:
                    candidate_args[epoch_pos + 1] = str(epoch_override)
                if args.gb_estimators is not None:
                    candidate_args.extend(["--gb-estimators", str(args.gb_estimators)])
                candidate_args.extend(["--seed", str(args.seed)])
                candidate = replace(candidate, args=candidate_args)
            run_dir = strategy_dir / f"attempt_{attempt_idx + 1}_{candidate.id}"
            
            start = time.perf_counter()
            train_candidate(candidate, args.data, run_dir)
            summary = load_summary(run_dir)
            elapsed = time.perf_counter() - start
            
            quality = float(summary.get("overall", {}).get("normalized_quality_pct", 0.0))
            per_feature = {row["metric"]: row for row in summary.get("per_feature", [])}
            reached = quality >= args.target_quality
            
            qualities.append(quality)
            csv_rows.append({
                "strategy": strategy,
                "attempt": attempt_idx + 1,
                "candidate_id": candidate.id,
                "quality_pct": quality,
                "traffic_spike_f1": float(per_feature.get("traffic_mbps", {}).get("spike_f1", 0.0)),
                "latency_spike_f1": float(per_feature.get("latency_ms", {}).get("spike_f1", 0.0)),
                "packet_loss_spike_f1": float(per_feature.get("packet_loss_pct", {}).get("spike_f1", 0.0)),
                "reached_gate": int(reached),
                "elapsed_seconds": elapsed,
            })
            
            if reached:
                strategy_summaries[strategy]["attempts_to_gate"] = attempt_idx + 1
                break
        
        if qualities:
            strategy_summaries[strategy]["best_quality_pct"] = float(np.max(qualities))
            strategy_summaries[strategy]["mean_quality_pct"] = float(np.mean(qualities))
            strategy_summaries[strategy]["std_quality_pct"] = float(np.std(qualities))
    
    # Write CSV
    if not csv_rows:
        raise RuntimeError("Candidate-selection ablation produced no successful runs.")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Write summary JSON
    results_summary_json.write_text(json.dumps(strategy_summaries, indent=2))
    print(f"Ablation results written to {results_csv} and {results_summary_json}")


if __name__ == "__main__":
    main()

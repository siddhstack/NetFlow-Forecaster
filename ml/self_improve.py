"""Self-learning benchmark loop with persistent experience memory."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def log_system_warning(message: str) -> None:
    print(f"[self_improve] WARNING: {message}")


def verify_and_deploy_ensemble(tournament_winner, baseline_model, validation_data):
    """Protect the deployment path from regressions by falling back to the baseline."""
    ml_metrics = tournament_winner.evaluate(validation_data)
    baseline_metrics = baseline_model.evaluate(validation_data)

    ml_loss = float(ml_metrics.get("weighted_loss", float("inf")))
    baseline_loss = float(baseline_metrics.get("weighted_loss", float("inf")))

    if ml_loss > baseline_loss:
        log_system_warning("Autonomous optimization failed to beat baseline. Triggering fallback.")
        return baseline_model
    return tournament_winner

from auto_benchmark import BenchmarkResult, gates_pass, load_summary, sync_docs, train_candidate
from experience_store import ExperienceRecord, append_record, fingerprint, update_policy_incremental, utc_now
from metrics_utils import diagnose_quality_shortfall
from stack_predictions import stack_attempts
from specialist_ensemble import apply_specialists
from telemetry_profile import profile_telemetry
from trainer_tournament import next_candidate


def traffic_f1_from_run(run_dir: Path) -> float:
    spikes_path = run_dir / "results" / "evaluation_spikes.csv"
    if not spikes_path.exists():
        return 0.0
    import csv

    for row in csv.DictReader(spikes_path.open(encoding="utf-8")):
        if row["metric"] == "traffic_mbps":
            return float(row["f1"])
    return 0.0


def append_experience(profile, candidate, summary: dict, run_dir: Path, status: str) -> None:
    spikes_path = run_dir / "results" / "evaluation_spikes.csv"
    traffic = traffic_f1_from_run(run_dir)
    record = ExperienceRecord(
        timestamp=utc_now(),
        data_fingerprint=fingerprint(profile),
        profile=asdict(profile),
        candidate_id=candidate.id,
        candidate_args=candidate.args,
        quality_pct=float(summary.get("overall", {}).get("normalized_quality_pct", 0.0)),
        gates_passed=summary.get("gates_passed", {}),
        per_feature=summary.get("per_feature", []),
        traffic_spike_f1=traffic,
        mae_improvement_pct=float(summary.get("overall", {}).get("mae_improvement_vs_persistence_pct", 0.0)),
        run_dir=str(run_dir),
        status=status,
    )
    append_record(record)
    update_policy_incremental()


def self_improve_benchmark(
    data: Path,
    output_dir: Path,
    target_quality: float = 90.0,
    max_rounds: int = 5,
    attempts_per_round: int = 4,
    sync_prefix: str = "",
) -> BenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_telemetry(data)
    (output_dir / "profile.json").write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    attempts: list[dict] = []
    best_summary: dict | None = None
    best_run: Path | None = None
    log_path = output_dir / "benchmark_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            attempts.append(record)
            summary = record.get("summary", {})
            quality = float(summary.get("overall", {}).get("normalized_quality_pct", 0.0))
            best_quality = (
                -1.0
                if best_summary is None
                else float(best_summary.get("overall", {}).get("normalized_quality_pct", 0.0))
            )
            if quality > best_quality:
                best_summary = summary
                best_run = Path(record["run_dir"])

    start_round = len(attempts) // attempts_per_round
    for round_idx in range(start_round, max_rounds):
        target_attempt_count = (round_idx + 1) * attempts_per_round
        while len(attempts) < target_attempt_count:
            candidate = next_candidate(profile, attempts)
            run_dir = output_dir / f"round_{round_idx:02d}_attempt_{len(attempts):02d}_{candidate.id}"
            try:
                train_candidate(candidate, data, run_dir)
                summary = load_summary(run_dir)
            except Exception as exc:
                summary = {"error": str(exc), "overall": {"normalized_quality_pct": 0.0}, "gates_passed": {}, "per_feature": []}
            status = "SUCCESS" if gates_pass(summary, target_quality) else "FAIL"
            append_experience(profile, candidate, summary, run_dir, status)
            record = {"candidate": asdict(candidate), "run_dir": str(run_dir), "summary": summary}
            attempts.append(record)
            with (output_dir / "benchmark_log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            quality = float(summary.get("overall", {}).get("normalized_quality_pct", 0.0))
            if best_summary is None or quality > float(best_summary.get("overall", {}).get("normalized_quality_pct", 0.0)):
                best_summary = summary
                best_run = run_dir
            if status == "SUCCESS":
                (output_dir / "best_run.txt").write_text(str(run_dir), encoding="utf-8")
                if sync_prefix:
                    sync_docs(run_dir, sync_prefix)
                return BenchmarkResult("SUCCESS", str(run_dir), quality, len(attempts))
        if best_summary and float(best_summary.get("overall", {}).get("normalized_quality_pct", 0.0)) >= 82.0:
            stacked = stack_attempts([Path(attempt["run_dir"]) for attempt in attempts], output_dir / f"round_{round_idx:02d}_stacked")
            if stacked:
                summary = load_summary(stacked)
                if best_summary is None or float(summary["overall"]["normalized_quality_pct"]) > float(best_summary["overall"]["normalized_quality_pct"]):
                    best_summary, best_run = summary, stacked
                if gates_pass(summary, target_quality):
                    (output_dir / "best_run.txt").write_text(str(stacked), encoding="utf-8")
                    if sync_prefix:
                        sync_docs(stacked, sync_prefix)
                    return BenchmarkResult("SUCCESS", str(stacked), float(summary["overall"]["normalized_quality_pct"]), len(attempts))
            if best_run is not None:
                specialist = apply_specialists(best_run, output_dir / f"round_{round_idx:02d}_specialist")
                summary = load_summary(specialist)
                if float(summary["overall"]["normalized_quality_pct"]) > float(best_summary["overall"]["normalized_quality_pct"]):
                    best_summary, best_run = summary, specialist
                if gates_pass(summary, target_quality):
                    (output_dir / "best_run.txt").write_text(str(specialist), encoding="utf-8")
                    if sync_prefix:
                        sync_docs(specialist, sync_prefix)
                    return BenchmarkResult("SUCCESS", str(specialist), float(summary["overall"]["normalized_quality_pct"]), len(attempts))
    assert best_run is not None and best_summary is not None
    (output_dir / "best_run.txt").write_text(str(best_run), encoding="utf-8")
    (best_run / "json" / "benchmark_diagnosis.json").write_text(json.dumps(diagnose_quality_shortfall(best_summary), indent=2), encoding="utf-8")
    if sync_prefix:
        sync_docs(best_run, sync_prefix)
    return BenchmarkResult("BEST_EFFORT", str(best_run), float(best_summary["overall"]["normalized_quality_pct"]), len(attempts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="runs/self_improve")
    parser.add_argument("--target-quality", type=float, default=90.0)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--attempts-per-round", type=int, default=4)
    parser.add_argument("--sync-docs", action="store_true")
    parser.add_argument("--docs-prefix", default="generic_")
    args = parser.parse_args()
    result = self_improve_benchmark(
        Path(args.data),
        Path(args.output_dir),
        args.target_quality,
        args.max_rounds,
        args.attempts_per_round,
        args.docs_prefix if args.sync_docs else "",
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()

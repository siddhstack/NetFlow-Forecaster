"""Automated train/evaluate/retry benchmark loop for telemetry CSVs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from metrics_utils import diagnose_quality_shortfall
from run_layout import find_artifact
from telemetry_profile import profile_telemetry
from trainer_tournament import Candidate, next_candidate


ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ml"


@dataclass
class BenchmarkResult:
    status: str
    best_run: str
    best_quality: float
    attempts: int


def run_cmd(args: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def train_candidate(candidate: Candidate, data: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if candidate.trainer == "gb":
        cmd = [
            sys.executable,
            str(ML / "train_dataset_model.py"),
            "--data",
            str(data),
            "--output-dir",
            str(run_dir),
            *candidate.args,
        ]
    else:
        cmd = [
            sys.executable,
            str(ML / "enhanced_train.py"),
            "--data",
            str(data),
            "--output-dir",
            str(run_dir),
            *candidate.args,
        ]
    run_cmd(cmd)
    raw = find_artifact(run_dir, data.name, "raw_data")
    run_cmd([sys.executable, str(ML / "visualize.py"), "--data", str(raw), "--output-dir", str(run_dir), "--sensitivity", "1.3"])
    run_cmd([sys.executable, str(ML / "evaluate_model.py"), "--run-dir", str(run_dir)])


def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "json" / "evaluation_summary.json").read_text(encoding="utf-8"))


def gates_pass(summary: dict, target_quality: float) -> bool:
    gates = summary.get("gates_passed", {})
    return bool(gates) and all(bool(value) for value in gates.values()) and float(summary["overall"]["normalized_quality_pct"]) >= target_quality


def sync_docs(run_dir: Path, prefix: str) -> None:
    docs_images = ROOT / "docs" / "images"
    docs_results = ROOT / "docs" / "results"
    docs_images.mkdir(parents=True, exist_ok=True)
    docs_results.mkdir(parents=True, exist_ok=True)
    for name in ("traffic_prediction_dashboard.png", "model_evaluation_dashboard.png"):
        src = run_dir / "images" / name
        if src.exists():
            shutil.copy2(src, docs_images / f"{prefix}{name}")
    for folder, names in {
        "json": ("evaluation_summary.json",),
        "results": ("evaluation_comparison.csv", "evaluation_baselines.csv", "evaluation_spikes.csv"),
    }.items():
        for name in names:
            src = run_dir / folder / name
            if src.exists():
                shutil.copy2(src, docs_results / f"{prefix}{name}")


def run_benchmark(data: Path, output_dir: Path, target_quality: float = 90.0, max_attempts: int = 24, max_minutes: int = 45, sync_prefix: str = "", learn: bool = True) -> BenchmarkResult:
    if learn:
        from self_improve import self_improve_benchmark

        attempts_per_round = max(1, min(4, max_attempts))
        max_rounds = max(1, (max_attempts + attempts_per_round - 1) // attempts_per_round)
        return self_improve_benchmark(data, output_dir, target_quality, max_rounds, attempts_per_round, sync_prefix)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_telemetry(data)
    (output_dir / "profile.json").write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    attempts: list[dict] = []
    best_summary: dict | None = None
    best_run: Path | None = None
    start = time.monotonic()
    log_path = output_dir / "benchmark_log.jsonl"
    for attempt_idx in range(max_attempts):
        if (time.monotonic() - start) / 60.0 > max_minutes:
            break
        candidate = next_candidate(profile, attempts)
        run_dir = output_dir / f"attempt_{attempt_idx:02d}_{candidate.id}"
        try:
            train_candidate(candidate, data, run_dir)
            summary = load_summary(run_dir)
        except Exception as exc:
            summary = {"error": str(exc), "overall": {"normalized_quality_pct": 0.0}, "gates_passed": {}}
        quality = float(summary.get("overall", {}).get("normalized_quality_pct", 0.0))
        record = {"attempt": attempt_idx, "candidate": asdict(candidate), "run_dir": str(run_dir), "summary": summary}
        attempts.append(record)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if best_summary is None or quality > float(best_summary.get("overall", {}).get("normalized_quality_pct", 0.0)):
            best_summary = summary
            best_run = run_dir
        if gates_pass(summary, target_quality):
            (output_dir / "best_run.txt").write_text(str(run_dir), encoding="utf-8")
            if sync_prefix:
                sync_docs(run_dir, sync_prefix)
            return BenchmarkResult("SUCCESS", str(run_dir), quality, len(attempts))
    assert best_run is not None and best_summary is not None
    (output_dir / "best_run.txt").write_text(str(best_run), encoding="utf-8")
    diagnosis = diagnose_quality_shortfall(best_summary)
    (best_run / "json" / "benchmark_diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    if sync_prefix:
        sync_docs(best_run, sync_prefix)
    return BenchmarkResult("BEST_EFFORT", str(best_run), float(best_summary["overall"]["normalized_quality_pct"]), len(attempts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="runs/auto_telemetry")
    parser.add_argument("--target-quality", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--max-minutes", type=int, default=45)
    parser.add_argument("--sync-docs", action="store_true")
    parser.add_argument("--docs-prefix", default="generic_")
    parser.add_argument("--learn", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = run_benchmark(
        Path(args.data),
        Path(args.output_dir),
        args.target_quality,
        args.max_attempts,
        args.max_minutes,
        args.docs_prefix if args.sync_docs else "",
        args.learn,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()

"""Cross-platform project runner.

This is the shared Python entrypoint used by the PowerShell and Bash wrappers.
It keeps the main ML workflows in one place while preserving OS-specific
helpers for Windows and Linux users.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODES = (
    "synthetic",
    "kaggle",
    "kaggle_opt",
    "dataset_opt",
    "public_benchmark",
    "simulate",
    "live",
    "deploy",
    "destroy",
    "train",
    "visualize",
    "benchmark",
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_DIR / "ml"


def log(message: str) -> None:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def run(command: list[str | Path], cwd: Path = PROJECT_DIR) -> None:
    """Run a command with proper error handling and logging."""
    printable = " ".join(str(part) for part in command)
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            capture_output=False,
            text=True
        )
        if completed.returncode != 0:
            raise SystemExit(f"\n✗ Command failed ({completed.returncode}):\n  {printable}")
    except FileNotFoundError as e:
        raise SystemExit(f"\n✗ Command not found: {printable}\n  Error: {e}")
    except Exception as e:
        raise SystemExit(f"\n✗ Error running command: {printable}\n  Error: {e}")


def python_command() -> str:
    return sys.executable or "python"


def install_dependencies(skip: bool) -> None:
    if skip:
        return
    log("Installing Python dependencies")
    run([python_command(), "-m", "pip", "install", "-r", PROJECT_DIR / "requirements.txt"])


def new_run_dir(mode: str) -> tuple[Path, Path]:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_DIR / "runs" / f"{run_stamp}_{mode}"
    data_file = run_dir / "raw_data" / "telemetry.csv"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    return run_dir, data_file


def auto_benchmark_args(args: argparse.Namespace, data_file: Path, run_dir: Path, sync_docs: bool = False) -> list[str | Path]:
    command: list[str | Path] = [
        python_command(),
        "ml/auto_benchmark.py",
        "--data",
        data_file,
        "--output-dir",
        run_dir,
        "--target-quality",
        str(args.target_quality),
        "--max-attempts",
        str(args.max_attempts),
    ]
    if args.max_minutes is not None:
        command += ["--max-minutes", str(args.max_minutes)]
    if sync_docs:
        command += ["--sync-docs", "--docs-prefix", "generic_"]
    command.append("--learn" if args.learn else "--no-learn")
    return command


def run_model_pipeline(args: argparse.Namespace, data_file: Path, run_dir: Path) -> None:
    if args.auto_benchmark:
        log("Running auto benchmark loop")
        run(auto_benchmark_args(args, data_file, run_dir))
        return

    log("Training hybrid model")
    run([python_command(), "ml/enhanced_train.py", "--data", data_file, "--epochs", str(args.epochs), "--output-dir", run_dir])

    log("Building dashboard")
    run([python_command(), "ml/visualize.py", "--data", run_dir / "raw_data" / "telemetry.csv", "--output-dir", run_dir])

    log("Evaluating model")
    run([python_command(), "ml/evaluate_model.py", "--run-dir", run_dir])

    log("Exporting readable model report")
    run([python_command(), "ml/export_model_report.py", "--run-dir", run_dir])

    cleanup = PROJECT_DIR / "scripts" / "cleanup_runs.py"
    if cleanup.exists():
        log("Cleaning empty run folders")
        run([python_command(), cleanup])


def show_artifacts(run_dir: Path) -> None:
    print("\nRun folder:")
    print(f"  {run_dir}")
    print("Artifacts:")
    for rel in (
        "raw_data/telemetry.csv",
        "images/traffic_prediction_dashboard.png",
        "images/model_evaluation_dashboard.png",
        "json/evaluation_summary.json",
        "json/model_metadata.json",
        "model/model_readable_report.md",
        "model/lstm_model.pth",
        "model/dataset_model.joblib",
    ):
        path = run_dir / rel
        if path.exists():
            print(f"  {path}")


def load_kaggle(data_file: Path, rows: int, l_ipn: int) -> None:
    command: list[str | Path] = [
        python_command(),
        "ml/load_kaggle_data.py",
        "--rows",
        str(rows),
        "--output",
        data_file,
        "--augment",
        "--seed",
        "42",
    ]
    if l_ipn >= 0:
        command += ["--l-ipn", str(l_ipn)]
    run(command)


def run_dataset_model(data_file: Path, run_dir: Path) -> None:
    log("Training dataset spike-aware model")
    run(
        [
            python_command(),
            "ml/train_dataset_model.py",
            "--data",
            data_file,
            "--output-dir",
            run_dir,
            "--lookback",
            "24",
            "--spike-std",
            "1.2",
            "--spike-oversample",
            "0",
        ]
    )
    log("Building dashboard")
    run([python_command(), "ml/visualize.py", "--data", run_dir / "raw_data" / "telemetry.csv", "--output-dir", run_dir])
    log("Evaluating model")
    run([python_command(), "ml/evaluate_model.py", "--run-dir", run_dir])


def latest_visualizable_run() -> Path | None:
    runs_dir = PROJECT_DIR / "runs"
    if not runs_dir.exists():
        return None
    candidates = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        required = [
            run_dir / "results" / "predictions.csv",
            run_dir / "results" / "actuals.csv",
            run_dir / "results" / "train_losses.csv",
        ]
        if all(path.exists() for path in required):
            candidates.append(run_dir)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform network telemetry runner.")
    parser.add_argument("mode", nargs="?", choices=MODES, default="synthetic")
    parser.add_argument("--samples", type=int, default=720)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--l-ipn", type=int, default=-1)
    parser.add_argument("--target-quality", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--auto-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        log(f"Mode: {args.mode} | Epochs: {args.epochs} | Samples: {args.samples}")
        
        needs_python = args.mode not in {"deploy", "destroy"}
        if needs_python:
            install_dependencies(args.skip_install)

        if args.mode == "deploy":
            log("✓ Deploying ContainerLab topology")
            run(["containerlab", "deploy", "-t", "topology.clab.yml"], cwd=PROJECT_DIR / "containerlab")
            log("✓ Deployment complete")
            return
        if args.mode == "destroy":
            log("✓ Destroying ContainerLab topology")
            run(["containerlab", "destroy", "-t", "topology.clab.yml"], cwd=PROJECT_DIR / "containerlab")
            log("✓ Destruction complete")
            return

        run_dir, data_file = new_run_dir(args.mode)
        log(f"Run directory: {run_dir}")

        if args.mode == "synthetic":
            log("✓ Generating synthetic telemetry")
            run([python_command(), "ml/generate_data.py", "--hours", str(args.samples), "--output", data_file, "--seed", "7"])
            run_model_pipeline(args, data_file, run_dir)
        elif args.mode == "kaggle":
            rows = 8000 if args.samples == 720 else args.samples
            log("✓ Loading Kaggle network telemetry")
            load_kaggle(data_file, rows, args.l_ipn)
            run_model_pipeline(args, data_file, run_dir)
        elif args.mode in {"kaggle_opt", "dataset_opt"}:
            if args.mode == "kaggle_opt":
                rows = 8000 if args.samples == 720 else args.samples
                log("✓ Loading Kaggle network telemetry")
                load_kaggle(data_file, rows, args.l_ipn)
            else:
                source = ML_DIR / "telemetry.csv"
                if not source.exists():
                    raise SystemExit(f"✗ dataset_opt needs ml/telemetry.csv (not found). Use kaggle_opt for Kaggle data.")
                shutil.copy2(source, data_file)
            run_dataset_model(data_file, run_dir)
        elif args.mode == "public_benchmark":
            log("✓ Loading CICIDS2017 public benchmark dataset")
            run([python_command(), "ml/load_public_benchmark.py", "--samples", str(args.samples), "--output", data_file])
            run_model_pipeline(args, data_file, run_dir)
        elif args.mode == "simulate":
            log("✓ Collecting simulated telemetry")
            run(
                [
                    python_command(),
                    "scripts/collect_telemetry.py",
                    "--mode",
                    "simulate",
                    "--samples",
                    str(args.samples),
                    "--interval",
                    str(args.interval),
                    "--output",
                    data_file,
                ]
            )
            run_model_pipeline(args, data_file, run_dir)
        elif args.mode == "live":
            log("✓ Collecting live telemetry")
            run(
                [
                    python_command(),
                    "scripts/collect_telemetry.py",
                    "--mode",
                    "live",
                    "--samples",
                    str(args.samples),
                    "--interval",
                    str(args.interval),
                    "--output",
                    data_file,
                ]
            )
            run_model_pipeline(args, data_file, run_dir)
        elif args.mode == "train":
            source = ML_DIR / "telemetry.csv"
            if not source.exists():
                raise SystemExit(f"✗ No ml/telemetry.csv found for train mode.")
            run_model_pipeline(args, source, run_dir)
        elif args.mode == "benchmark":
            source = ML_DIR / "telemetry.csv"
            if not source.exists():
                raise SystemExit(f"✗ No ml/telemetry.csv found for benchmark mode.")
            shutil.copy2(source, data_file)
            log("✓ Running universal benchmark (meta-policy enabled)")
            run(auto_benchmark_args(args, data_file, run_dir, sync_docs=True))
        elif args.mode == "visualize":
            latest = latest_visualizable_run()
            if latest is None:
                raise SystemExit("✗ No run folders with readable CSV artifacts found.")
            run_dir = latest
            log(f"✓ Building dashboard from {run_dir}")
            run([python_command(), "ml/visualize.py", "--data", run_dir / "raw_data" / "telemetry.csv", "--output-dir", run_dir])

        log("✓ Done")
        show_artifacts(run_dir)
        
    except KeyboardInterrupt:
        log("⚠ Interrupted by user")
        sys.exit(130)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        sys.exit(1 if "✗" in str(e) else 0)
    except Exception as e:
        log(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

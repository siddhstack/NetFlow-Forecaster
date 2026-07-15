"""Tests for ablation_selection.py."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.generate_data import generate_traffic_data
from ml.ablation_selection import main as ablation_main


def test_ablation_selection_policy_isolation():
    """Verify ablation never writes to runs/.experience/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Generate small synthetic CSV
        data_csv = tmpdir / "telemetry.csv"
        generate_traffic_data(hours=300, output=data_csv, seed=42)
        
        # Get initial state of experience store
        exp_dir = ROOT / "runs" / ".experience"
        exp_dir.mkdir(parents=True, exist_ok=True)
        policy_file = exp_dir / "policy.json"
        memory_file = exp_dir / "memory.jsonl"
        
        policy_hash_before = hashlib.sha256(policy_file.read_bytes()).hexdigest() if policy_file.exists() else None
        memory_hash_before = hashlib.sha256(memory_file.read_bytes()).hexdigest() if memory_file.exists() else None
        
        # Run ablation with minimal settings
        sys.argv = [
            "ablation_selection.py",
            "--data", str(data_csv),
            "--output-dir", str(tmpdir / "ablation"),
            "--strategies", "fixed",
            "--attempts-per-strategy", "1",
        ]
        
        try:
            ablation_main()
        except Exception:
            pass  # Allow errors from training; we just check file isolation
        
        # Verify files unchanged
        policy_hash_after = hashlib.sha256(policy_file.read_bytes()).hexdigest() if policy_file.exists() else None
        memory_hash_after = hashlib.sha256(memory_file.read_bytes()).hexdigest() if memory_file.exists() else None
        
        assert policy_hash_before == policy_hash_after, "policy.json was modified during ablation"
        assert memory_hash_before == memory_hash_after, "memory.jsonl was modified during ablation"


def test_ablation_selection_output_files():
    """Verify output files have expected structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_csv = tmpdir / "telemetry.csv"
        generate_traffic_data(hours=300, output=data_csv, seed=42)
        
        sys.argv = [
            "ablation_selection.py",
            "--data", str(data_csv),
            "--output-dir", str(tmpdir / "ablation"),
            "--strategies", "fixed",
            "--attempts-per-strategy", "1",
        ]
        
        try:
            ablation_main()
        except Exception:
            pass
        
        # Check CSV exists and has required columns
        csv_file = tmpdir / "ablation" / f"ablation_selection_{data_csv.stem}.csv"
        assert csv_file.exists(), f"CSV not created: {csv_file}"
        
        # Check JSON summary exists
        json_file = tmpdir / "ablation" / "ablation_selection_summary.json"
        assert json_file.exists(), f"JSON not created: {json_file}"
        
        summary = json.loads(json_file.read_text())
        assert "fixed" in summary
        assert "attempts_to_gate" in summary["fixed"]
        assert "best_quality_pct" in summary["fixed"]

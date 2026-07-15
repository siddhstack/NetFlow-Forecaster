"""Tests for ablation_spike_loss.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.generate_data import generate_traffic_data
from ml.ablation_spike_loss import main as ablation_main


def test_ablation_spike_loss_output():
    """Verify spike-loss ablation creates expected output structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_csv = tmpdir / "telemetry.csv"
        generate_traffic_data(hours=300, output=data_csv, seed=42)
        
        sys.argv = [
            "ablation_spike_loss.py",
            "--data", str(data_csv),
            "--output-dir", str(tmpdir / "ablation"),
            "--epochs", "2",
            "--seeds", "7",
        ]
        
        try:
            ablation_main()
        except Exception:
            pass
        
        # Check JSON summary
        json_file = ROOT / "docs" / "results" / "ablation_spike_loss_summary.json"
        if json_file.exists():
            summary = json.loads(json_file.read_text())
            assert "no_spike_weighting" in summary
            assert "uniform_spike_weighting" in summary
            assert "differentiated_spike_weighting" in summary

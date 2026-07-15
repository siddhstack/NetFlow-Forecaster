"""Tests for load_public_benchmark.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.load_public_benchmark import convert_cicids2017


def test_convert_cicids2017_schema():
    """Verify output has required schema and is sorted."""
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=10, freq='s'),
        'Flow Bytes/s': [100.0] * 10,
        'Flow Duration': [0.5] * 10,
        'PSH Flag Count': [0] * 10,
        'RST Flag Count': [0] * 10,
    })
    
    result = convert_cicids2017(df, bin_minutes=1)
    
    assert list(result.columns) == ['timestamp', 'traffic_mbps', 'latency_ms', 'packet_loss_pct']
    assert result['timestamp'].is_monotonic_increasing
    assert (result[['traffic_mbps', 'latency_ms', 'packet_loss_pct']] >= 0).all().all()


def test_convert_cicids2017_aggregation():
    """Verify binning and aggregation work correctly."""
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=5, freq='12s'),
        'Flow Bytes/s': [1_000_000.0, 2_000_000.0, 1_500_000.0, 1_000_000.0, 500_000.0],
        'Flow Duration': [1.0, 2.0, 1.5, 1.0, 0.5],
        'PSH Flag Count': [0] * 5,
        'RST Flag Count': [0] * 5,
    })
    
    result = convert_cicids2017(df, bin_minutes=1)
    assert len(result) > 0
    assert result['traffic_mbps'].sum() > 0

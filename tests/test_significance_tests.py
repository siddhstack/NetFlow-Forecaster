"""Tests for significance_tests.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.significance_tests import diebold_mariano, paired_t_test


def test_diebold_mariano_significant_difference():
    """Model errors much smaller than baseline => significant."""
    model_errors = np.random.default_rng(0).normal(0, 1, 200)
    baseline_errors = np.random.default_rng(0).normal(0, 5, 200)
    
    dm_stat, dm_p = diebold_mariano(model_errors, baseline_errors)
    assert dm_p < 0.05, f"Expected significant DM test, got p={dm_p}"


def test_paired_t_test_identical_distributions():
    """Same distribution => no significance."""
    errors = np.random.default_rng(0).normal(0, 1, 100)
    
    t_stat, t_p = paired_t_test(errors, errors)
    assert t_p >= 0.05, f"Expected non-significant t-test, got p={t_p}"


def test_paired_t_test_significant_difference():
    """Model better => significant."""
    model_errors = np.random.default_rng(0).normal(0, 1, 200)
    baseline_errors = np.random.default_rng(1).normal(0, 5, 200)
    
    t_stat, t_p = paired_t_test(model_errors, baseline_errors)
    assert t_p < 0.05, f"Expected significant t-test, got p={t_p}"

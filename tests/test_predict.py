"""Tests for the inference pipeline in ml/predict.py."""
from __future__ import annotations

import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
import torch
import joblib
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from generate_data import generate_traffic_data
from predict import forecast
from train_model import FEATURES, INPUT_FEATURES, MultivariateTrafficLSTM


def _make_minimal_run(tmp_path: Path) -> Path:
    df = generate_traffic_data(hours=140, seed=42, output=tmp_path / "input_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.weekday / 7)
    data_path = tmp_path / "input_data.csv"
    df.to_csv(data_path, index=False)

    run_dir = tmp_path / "run"
    (run_dir / "json").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    raw_inputs = df[INPUT_FEATURES].to_numpy(dtype=np.float32)
    raw_targets = df[FEATURES].to_numpy(dtype=np.float32)
    scaler.fit(raw_inputs)
    target_scaler = StandardScaler().fit(raw_targets)
    scaler_params = {
        "feature_columns": INPUT_FEATURES,
        "scaler_type": "StandardScaler",
        "input_scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        "target_scaler": {
            "mean": target_scaler.mean_.tolist(),
            "scale": target_scaler.scale_.tolist(),
            "var": target_scaler.var_.tolist(),
        },
    }
    (run_dir / "json" / "scaler_params.json").write_text(
        json.dumps(scaler_params), encoding="utf-8"
    )

    model = MultivariateTrafficLSTM(len(INPUT_FEATURES), 16, 1, len(FEATURES))
    torch.save(model.state_dict(), run_dir / "model" / "lstm_model.pth")
    metrics = {"training": {"sequence_length": 48, "hidden_size": 16, "layers": 1}}
    (run_dir / "json" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


def test_predict_returns_correct_shape(tmp_path):
    run_dir = _make_minimal_run(tmp_path)
    data_path = tmp_path / "input_data.csv"
    result = forecast(run_dir, data_path, forecast_steps=6)
    assert len(result) == 6
    assert "traffic_mbps_lower_95" in result.columns
    assert "packet_loss_pct_upper_95" in result.columns


def test_predict_values_are_non_negative(tmp_path):
    run_dir = _make_minimal_run(tmp_path)
    data_path = tmp_path / "input_data.csv"
    result = forecast(run_dir, data_path, forecast_steps=4)
    assert (result[FEATURES].to_numpy() >= 0.0).all()
    assert (result[[f"{metric}_lower_95" for metric in FEATURES]].to_numpy() >= 0.0).all()


def test_predict_raises_on_missing_columns(tmp_path):
    run_dir = _make_minimal_run(tmp_path)
    df = pd.read_csv(tmp_path / "input_data.csv")
    df = df.drop(columns=["packet_loss_pct"])
    broken_path = tmp_path / "broken.csv"
    df.to_csv(broken_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        forecast(run_dir, broken_path, forecast_steps=3)


def test_predict_raises_on_too_few_rows(tmp_path):
    run_dir = _make_minimal_run(tmp_path)
    df = pd.read_csv(tmp_path / "input_data.csv").iloc[:12]
    short_path = tmp_path / "short.csv"
    df.to_csv(short_path, index=False)
    with pytest.raises(ValueError, match="usable rows"):
        forecast(run_dir, short_path, forecast_steps=2)


def test_predict_supports_dataset_model_only(tmp_path):
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor

    df = generate_traffic_data(hours=140, seed=7, output=tmp_path / "input.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)

    # Build dataset-model payload similar to train_kaggle_model.py output.
    from train_kaggle_model import add_features

    lookback = 8
    modeled, feature_cols = add_features(df, lookback)
    x = modeled[feature_cols].to_numpy(dtype=float)
    y = modeled[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    model = MultiOutputRegressor(LinearRegression()).fit(scaler.transform(x), y)

    run_dir = tmp_path / "run_ds"
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "json").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "target_features": FEATURES,
            "training": {"lookback": lookback},
        },
        run_dir / "model" / "dataset_model.joblib",
    )

    result = forecast(run_dir, input_path, forecast_steps=6)
    assert len(result) == 6
    assert "traffic_mbps_lower_95" in result.columns
    assert "packet_loss_pct_upper_95" in result.columns

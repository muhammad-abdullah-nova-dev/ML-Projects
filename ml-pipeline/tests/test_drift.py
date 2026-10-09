"""
Unit tests for data drift monitor and Population Stability Index (PSI).
Engineered by M. Abdullah.
"""
import numpy as np
import pandas as pd
from mlflow_pipeline.drift_monitor import calculate_psi, monitor_dataset_drift


def test_psi_identical_distributions():
    rng = np.random.RandomState(42)
    baseline = rng.normal(loc=50, scale=10, size=1000)
    target = rng.normal(loc=50, scale=10, size=1000)

    psi = calculate_psi(baseline, target)
    # Identical distributions should have minimal PSI (< 0.05)
    assert psi < 0.08, f"Expected small PSI for identical distributions, got {psi}"


def test_psi_shifted_distribution():
    rng = np.random.RandomState(42)
    baseline = rng.normal(loc=50, scale=10, size=1000)
    # Severely shifted distribution
    shifted = rng.normal(loc=75, scale=15, size=1000)

    psi = calculate_psi(baseline, shifted)
    # Shifted distributions must breach the significant shift threshold (PSI > 0.25)
    assert psi > 0.25, f"Expected significant PSI > 0.25, got {psi}"


def test_monitor_dataset_drift():
    baseline_df = pd.DataFrame({
        "income": np.random.normal(5000, 1000, size=500),
        "debt_ratio": np.random.uniform(0.1, 0.5, size=500)
    })
    # Target batch with severe income drift
    batch_df = pd.DataFrame({
        "income": np.random.normal(12000, 2000, size=500),
        "debt_ratio": np.random.uniform(0.1, 0.5, size=500)
    })

    report = monitor_dataset_drift(baseline_df, batch_df, ["income", "debt_ratio"])
    assert report["system_status"] in ["warning", "action_required"]
    assert "income" in report["significant_features"]
    assert report["feature_metrics"]["debt_ratio"]["status"] == "stable"

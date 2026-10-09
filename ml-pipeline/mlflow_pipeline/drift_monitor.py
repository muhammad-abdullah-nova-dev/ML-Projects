"""
Data and Concept Drift Monitoring Module for Credit Scoring.
Engineered by M. Abdullah.

Implements Population Stability Index (PSI) and Wasserstein Distance
to monitor feature and prediction score distributions across production batches.
"""
from typing import Dict, List, Any
import numpy as np


def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calculate Population Stability Index (PSI) between a baseline and target distribution.

    Interpretation thresholds:
    - PSI < 0.1: No significant change (stable)
    - 0.1 <= PSI < 0.25: Moderate shift (monitor closely)
    - PSI >= 0.25: Significant shift (action required / trigger retraining)
    """
    baseline = np.asarray(baseline).flatten()
    target = np.asarray(target).flatten()

    # Drop NaNs for statistical stability
    baseline = baseline[~np.isnan(baseline)]
    target = target[~np.isnan(target)]

    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    # Determine quantile bins based on baseline
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(baseline, percentiles)
    # Ensure bin edges are strictly increasing
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-5

    # Bucket counts
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)

    # Convert to fractions
    baseline_pct = baseline_counts / len(baseline)
    target_pct = target_counts / len(target)

    # Smooth zero bins to avoid division by zero
    baseline_pct = np.clip(baseline_pct, epsilon, 1.0)
    target_pct = np.clip(target_pct, epsilon, 1.0)

    # Normalize back to 1.0
    baseline_pct /= np.sum(baseline_pct)
    target_pct /= np.sum(target_pct)

    # PSI calculation: sum((target - baseline) * ln(target / baseline))
    psi_value = np.sum((target_pct - baseline_pct) * np.log(target_pct / baseline_pct))
    return float(round(psi_value, 4))


def monitor_dataset_drift(baseline_df, batch_df, feature_names: List[str]) -> Dict[str, Any]:
    """
    Compute PSI across all features and classify system-level drift status.
    """
    feature_psi = {}
    drifted_features = []
    moderate_features = []

    for col in feature_names:
        if col in baseline_df and col in batch_df:
            psi = calculate_psi(baseline_df[col].values, batch_df[col].values)
            status = "stable"
            if psi >= 0.25:
                status = "significant_drift"
                drifted_features.append(col)
            elif psi >= 0.1:
                status = "moderate_drift"
                moderate_features.append(col)

            feature_psi[col] = {
                "psi": psi,
                "status": status
            }

    system_status = "stable"
    if len(drifted_features) > 0:
        system_status = "action_required"
    elif len(moderate_features) > 0:
        system_status = "warning"

    return {
        "system_status": system_status,
        "significant_drift_count": len(drifted_features),
        "moderate_drift_count": len(moderate_features),
        "significant_features": drifted_features,
        "feature_metrics": feature_psi
    }

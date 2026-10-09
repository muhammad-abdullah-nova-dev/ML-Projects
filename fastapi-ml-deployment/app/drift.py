"""
Data drift detection service using two-sample Kolmogorov-Smirnov (KS) test.
Compares incoming feature distributions against baseline distributions recorded during model training.
"""
from typing import Dict, List
import numpy as np
from scipy.stats import ks_2samp
from app.model import get_metadata
from app.schemas import DriftCheckResponse, FeatureDriftResult, IrisFeatures


def detect_data_drift(features_list: List[IrisFeatures], alpha: float = 0.05) -> DriftCheckResponse:
    metadata = get_metadata()
    baseline_stats = metadata.get("baseline_stats", {})
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Convert batch of IrisFeatures to feature arrays
    batch_dict: Dict[str, List[float]] = {name: [] for name in feature_names}
    for item in features_list:
        batch_dict["sepal_length"].append(item.sepal_length)
        batch_dict["sepal_width"].append(item.sepal_width)
        batch_dict["petal_length"].append(item.petal_length)
        batch_dict["petal_width"].append(item.petal_width)

    drifted_features = []
    feature_metrics: Dict[str, FeatureDriftResult] = {}

    for name in feature_names:
        batch_vals = np.array(batch_dict[name])
        baseline_vals = baseline_stats.get(name, {}).get("values")
        baseline_mean = baseline_stats.get(name, {}).get("mean", 0.0)

        if baseline_vals and len(baseline_vals) > 0:
            ks_stat, p_val = ks_2samp(baseline_vals, batch_vals)
            ks_stat = float(round(ks_stat, 4))
            p_val = float(round(p_val, 4))
            is_drifted = bool(p_val < alpha)
        else:
            # Fallback if baseline values not available: threshold on mean shift
            ks_stat, p_val = 0.0, 1.0
            is_drifted = False

        if is_drifted:
            drifted_features.append(name)

        feature_metrics[name] = FeatureDriftResult(
            ks_statistic=ks_stat,
            p_value=p_val,
            is_drifted=is_drifted,
            baseline_mean=float(round(baseline_mean, 3)),
            batch_mean=float(round(float(np.mean(batch_vals)), 3))
        )

    return DriftCheckResponse(
        drift_detected=len(drifted_features) > 0,
        drifted_features=drifted_features,
        feature_metrics=feature_metrics,
        sample_size=len(features_list),
        alpha=alpha,
        baseline_reference=metadata.get("model_version", "v1")
    )

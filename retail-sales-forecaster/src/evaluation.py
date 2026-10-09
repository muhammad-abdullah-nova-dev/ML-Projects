"""
Forecasting evaluation metrics and comparative benchmark tools.
Engineered by M. Abdullah.
Includes MAE, RMSE, MAPE, and WAPE (Weighted Absolute Percentage Error).
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate time-series forecasting error metrics."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    # MAPE: avoid division by zero
    non_zero = actual != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100)
    else:
        mape = 0.0

    # WAPE (Weighted Absolute Percentage Error): Sum(|Actual - Pred|) / Sum(Actual)
    sum_actual = float(np.sum(actual))
    wape = float((np.sum(np.abs(actual - predicted)) / sum_actual) * 100) if sum_actual > 0 else 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 2),
        "WAPE_pct": round(wape, 2),
    }


def compare_forecasters(actual: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model_name, preds in predictions_dict.items():
        metrics = evaluate_model(actual, preds)
        metrics["Model"] = model_name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("Model")[["MAE", "RMSE", "MAPE_pct", "WAPE_pct"]]
    return df

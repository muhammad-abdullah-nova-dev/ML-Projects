"""
Model Explainability and Feature Importance Module.
Engineered by M. Abdullah.
Calculates permutation feature importance and tree-based importances
to interpret risk drivers for tabular credit scoring models.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure ml-pipeline root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def compute_feature_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calculate permutation feature importances on validation data.
    Permutation importance measures the drop in ROC-AUC when a feature is randomly shuffled.
    """
    perm_result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    ranked_indices = np.argsort(perm_result.importances_mean)[::-1]
    importance_list = []

    for idx in ranked_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        importance_list.append({
            "feature": name,
            "importance_mean": round(float(perm_result.importances_mean[idx]), 5),
            "importance_std": round(float(perm_result.importances_std[idx]), 5),
        })

    report = {
        "scoring": "roc_auc",
        "method": "permutation_importance",
        "n_repeats": n_repeats,
        "ranked_features": importance_list,
        "top_drivers": [item["feature"] for item in importance_list[:3]]
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "feature_importance.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Calculated feature importances. Top 3 risk drivers:")
    for item in importance_list[:3]:
        print(f"  - {item['feature']}: {item['importance_mean']:.5f} (+/- {item['importance_std']:.5f})")

    return report


if __name__ == "__main__":
    from mlflow_pipeline.data_ingest import ensure_dataset
    from mlflow_pipeline.preprocess import pandas_preprocess

    data_path = ensure_dataset()
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = pandas_preprocess(df, "SeriousDlqin2yrs")

    model_path = ARTIFACTS_DIR / "best_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        compute_feature_importance(model, X_test, y_test, feature_names)
    else:
        print("Please train a model first using mlflow_pipeline/train.py")

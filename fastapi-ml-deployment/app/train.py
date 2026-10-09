"""
Training pipeline for Iris Classifier with Scikit-Learn.
Includes StandardScaler, 5-Fold Stratified Cross Validation,
metrics computation, and baseline distribution metadata serialization for drift detection.
"""
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "v1"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / f"iris_clf_{MODEL_VERSION}.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


def train_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = list(iris.target_names)

    # 1. Pipeline construction (StandardScaler + RandomForest)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # 2. Rigorous 5-fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_train_score=True)

    metrics = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "cv_precision_macro_mean": float(np.mean(cv_results["test_precision_macro"])),
        "cv_recall_macro_mean": float(np.mean(cv_results["test_recall_macro"])),
        "train_accuracy_mean": float(np.mean(cv_results["train_accuracy"])),
    }

    # 3. Fit pipeline on full dataset for serving
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    # 4. Save baseline distributions for data drift monitoring
    baseline_stats = {}
    for i, col in enumerate(feature_names):
        vals = X[:, i]
        baseline_stats[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "values": vals.tolist()  # Reference sample for KS two-sample drift test
        }

    metadata = {
        "model_name": "Iris Species Classifier",
        "model_version": MODEL_VERSION,
        "model_type": "Pipeline(StandardScaler -> RandomForestClassifier)",
        "features": feature_names,
        "classes": target_names,
        "metrics": metrics,
        "baseline_stats": baseline_stats,
        "n_samples": int(X.shape[0]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "status": "active"
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model v{MODEL_VERSION} trained successfully!")
    print(f"5-Fold CV Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
    print(f"5-Fold CV Macro F1: {metrics['cv_f1_macro_mean']:.4f}")
    print(f"Saved artifacts to {MODEL_DIR}")
    return pipeline, metadata


if __name__ == "__main__":
    train_model()

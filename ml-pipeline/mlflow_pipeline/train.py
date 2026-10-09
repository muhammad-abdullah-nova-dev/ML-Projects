"""
Training and model comparison pipeline for tabular credit risk.
Engineered by M. Abdullah.

Features:
- Rigorous baseline benchmarking (Naive Baseline, Logistic Regression, Random Forest, Gradient Boosting)
- Comprehensive evaluation metrics: ROC-AUC, PR-AUC, F1-Score, Brier Score, Precision, Recall
- MLflow experiment tracking with graceful local artifact storage fallback
- Class imbalance mitigation (balanced class weighting)
- Model artifact and benchmark JSON serialization
"""
import argparse
import json
import time
from datetime import datetime, timezone
import sys
from pathlib import Path

from typing import Dict, Any

# Ensure ml-pipeline root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    accuracy_score
)

from mlflow_pipeline.data_ingest import ensure_dataset
from mlflow_pipeline.preprocess import pandas_preprocess

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute rigorous classification metrics suitable for imbalanced datasets."""
    y_pred = (y_proba >= threshold).astype(int)

    # For constant predictions (e.g. naive baseline), handle gracefully
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Define candidate models spanning baseline to non-linear ensemble methods.
    """
    models = {
        "naive_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            class_weight="balanced",
            random_state=random_state
        )
    }

    # Attempt to add XGBoost if available
    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=5.0,
            eval_metric="logloss",
            random_state=random_state
        )
    except ImportError:
        pass

    return models


def train_and_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    mlflow_tracking_uri: str = None
) -> Dict[str, Any]:
    """
    Train all candidate models, evaluate out-of-sample performance,
    log to MLflow if accessible, and select the optimal model.
    """
    models = get_candidate_models()
    benchmark_results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_model_obj = None

    # Try setting up MLflow
    mlflow_available = False
    try:
        import mlflow
        from mlflow_pipeline.mlflow_utils import set_tracking
        set_tracking(mlflow_tracking_uri)
        mlflow.set_experiment("credit_risk_benchmark")
        mlflow_available = True
    except Exception:
        pass

    print("\n" + "=" * 75)
    print("STARTING MODEL BENCHMARKING & VALIDATION")
    print("=" * 75)

    for name, model in models.items():
        start_t = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time_ms = round((time.perf_counter() - start_t) * 1000, 2)

        # Generate probabilities for positive class (1)
        if hasattr(model, "predict_proba"):
            preds_proba = model.predict_proba(X_test)[:, 1]
        else:
            preds_proba = model.predict(X_test).astype(float)

        metrics = evaluate_predictions(y_test, preds_proba)
        metrics["fit_time_ms"] = fit_time_ms

        benchmark_results[name] = metrics
        print(f"[{name.upper()}] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Brier: {metrics['brier_score']:.4f} | Fit: {fit_time_ms}ms")

        # Track best model (excluding dummy baseline)
        if name != "naive_baseline" and metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model_name = name
            best_model_obj = model

        # Log run to MLflow if available
        if mlflow_available:
            try:
                import mlflow
                import mlflow.sklearn
                with mlflow.start_run(run_name=f"benchmark_{name}"):
                    mlflow.set_tag("model_family", name)
                    mlflow.log_params({"n_features": len(feature_names)})
                    for metric_k, metric_v in metrics.items():
                        mlflow.log_metric(metric_k, metric_v)
                    mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception:
                pass

    # Save best model to artifacts
    joblib.dump(best_model_obj, ARTIFACTS_DIR / "best_model.joblib")
    
    metadata = {
        "best_model": best_model_name,
        "best_roc_auc": best_roc_auc,
        "feature_names": feature_names,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "positive_rate_test": float(np.mean(y_test)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "maintainer": "M. Abdullah",
        "benchmark": benchmark_results
    }

    with open(ARTIFACTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 75)
    print(f"Optimal Model Selected: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    print(f"Saved artifacts to {ARTIFACTS_DIR}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Credit risk model training and benchmarking pipeline")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default="SeriousDlqin2yrs")
    parser.add_argument("--tracking-uri", type=str, default=None)
    args = parser.parse_args()

    data_path = ensure_dataset(args.data)
    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test, preprocessor, feature_names = pandas_preprocess(
        df=df,
        target_col=args.target,
        test_size=0.2,
        random_state=42
    )

    train_and_benchmark(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        mlflow_tracking_uri=args.tracking_uri
    )


if __name__ == "__main__":
    main()
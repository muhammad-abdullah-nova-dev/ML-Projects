"""
Unit tests for model training, metrics calculation, and benchmarking suite.
Engineered by M. Abdullah.
"""
import numpy as np
import pytest
from mlflow_pipeline.data_ingest import generate_synthetic_credit_data
from mlflow_pipeline.train import (
    evaluate_predictions,
    get_candidate_models,
    train_and_benchmark
)
from mlflow_pipeline.preprocess import pandas_preprocess


def test_synthetic_data_generation():
    df = generate_synthetic_credit_data(n_samples=500, random_state=42)
    assert len(df) == 500
    assert "SeriousDlqin2yrs" in df.columns
    assert "RevolvingUtilizationOfUnsecuredLines" in df.columns
    # Check that default class rate is between 3% and 20% (representative imbalance)
    pos_rate = df["SeriousDlqin2yrs"].mean()
    assert 0.03 <= pos_rate <= 0.20


def test_evaluation_metrics_computation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.85, 0.9, 0.2, 0.75])
    metrics = evaluate_predictions(y_true, y_proba)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] > 0.8
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_model_benchmarking_beats_naive_baseline():
    df = generate_synthetic_credit_data(n_samples=800, random_state=42)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = pandas_preprocess(
        df=df, target_col="SeriousDlqin2yrs", test_size=0.25, random_state=42
    )

    metadata = train_and_benchmark(X_train, X_test, y_train, y_test, feature_names)
    assert metadata["best_model"] in ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    assert metadata["best_roc_auc"] > 0.65

    benchmark = metadata["benchmark"]
    assert "naive_baseline" in benchmark
    assert "logistic_regression" in benchmark
    # Machine learning model must significantly outperform dummy baseline ROC-AUC (0.50)
    assert metadata["best_roc_auc"] > benchmark["naive_baseline"]["roc_auc"]

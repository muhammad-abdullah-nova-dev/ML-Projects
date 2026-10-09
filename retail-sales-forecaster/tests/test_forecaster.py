"""
Unit tests for Retail Sales Forecaster pipeline and models.
Engineered by M. Abdullah.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_data, add_features, train_test_split_temporal
from src.model import MachineLearningForecaster, NaiveSeasonalForecaster
from src.evaluation import evaluate_model, compare_forecasters


@pytest.fixture
def sample_sales_df():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = [2000 + (i % 7) * 50 + i * 2 for i in range(100)]
    return pd.DataFrame({"date": dates, "store": 1, "sales": sales})


def test_feature_engineering_and_temporal_split(sample_sales_df):
    fe_df = add_features(sample_sales_df)
    assert "dayofweek" in fe_df.columns
    assert "lag_7" in fe_df.columns
    assert "rolling_mean_7" in fe_df.columns
    assert not fe_df.isnull().any().any()

    train_df, test_df = train_test_split_temporal(fe_df, test_days=20)
    assert len(test_df) == 20
    assert len(train_df) == len(fe_df) - 20
    # Ensure chronological order: maximum train date is before minimum test date
    assert train_df["date"].max() < test_df["date"].min()


def test_naive_and_ml_forecasters(sample_sales_df):
    fe_df = add_features(sample_sales_df)
    train_df, test_df = train_test_split_temporal(fe_df, test_days=14)

    naive = NaiveSeasonalForecaster(lag=7)
    naive.fit(train_df)
    naive_preds = naive.predict(test_df)
    assert len(naive_preds) == 14

    ml_model = MachineLearningForecaster(model_type="random_forest")
    ml_model.fit(train_df)
    ml_preds = ml_model.predict(test_df)
    assert len(ml_preds) == 14

    metrics = evaluate_model(test_df["sales"].values, ml_preds)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert metrics["MAE"] >= 0.0
    assert metrics["WAPE_pct"] >= 0.0


def test_evaluation_metrics_exact():
    actual = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 300.0])
    metrics = evaluate_model(actual, pred)

    assert metrics["MAE"] == round((10.0 + 10.0 + 0.0) / 3, 2)
    # WAPE: (10 + 10 + 0) / 600 = 20 / 600 = 3.33%
    assert metrics["WAPE_pct"] == 3.33

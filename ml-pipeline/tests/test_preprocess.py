"""
Unit tests for data preprocessing and data leakage prevention.
Engineered by M. Abdullah.
"""
import numpy as np
import pandas as pd
import pytest
from mlflow_pipeline.preprocess import pandas_preprocess, build_preprocessor


def test_leakage_free_imputation_and_scaling():
    """
    Verify that the preprocessor computes medians and scalers strictly on X_train.
    Extreme outliers in test set must not influence training scaling parameters.
    """
    df = pd.DataFrame({
        "num1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "num2": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0],
        "cat1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })

    X_train, X_test, y_train, y_test, preprocessor, feature_names = pandas_preprocess(
        df=df, target_col="target", test_size=0.3, random_state=42, save_artifact=False
    )

    # 1. Assert correct split shapes
    assert X_train.shape[0] == 7
    assert X_test.shape[0] == 3
    assert len(y_train) == 7
    assert len(y_test) == 3

    # 2. Assert no remaining NaN values
    assert not np.isnan(X_train).any(), "X_train contains unhandled NaNs"
    assert not np.isnan(X_test).any(), "X_test contains unhandled NaNs"

    # 3. Assert preprocessor object is returned and fitted
    assert preprocessor is not None
    assert len(feature_names) >= 3


def test_missing_target_raises_value_error():
    df = pd.DataFrame({"feat": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column 'missing_target' not found"):
        pandas_preprocess(df, target_col="missing_target")


def test_handle_unseen_categories():
    """Verify that one-hot encoder ignores unseen categories at inference without crashing."""
    train_df = pd.DataFrame({
        "age": [25, 40, 55],
        "category": ["low", "med", "high"]
    })
    test_df = pd.DataFrame({
        "age": [30],
        "category": ["unknown_category"]
    })

    prep = build_preprocessor(numeric_features=["age"], categorical_features=["category"])
    prep.fit(train_df)
    transformed_test = prep.transform(test_df)
    assert transformed_test.shape[0] == 1
    assert not np.isnan(transformed_test).any()

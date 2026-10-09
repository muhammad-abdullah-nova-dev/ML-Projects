"""
Leakage-Free Preprocessing Utilities for Tabular ML Pipelines.
Engineered by M. Abdullah.

CRITICAL DESIGN NOTE:
Strict Featurization Ordering is enforced:
Train/test splitting is ALWAYS performed BEFORE any imputation, scaling, or encoding.
Preprocessors are fitted strictly on X_train, and then applied to X_test and inference streams,
eliminating data leakage.
"""
from pathlib import Path
from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(numeric_features: List[str], categorical_features: Optional[List[str]] = None) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer with median imputation and scaling
    for numerical columns, and constant imputation + one-hot encoding for categoricals.
    """
    transformers = [
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        )
    ]

    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            categorical_features
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def pandas_preprocess(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifact: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Leakage-free preprocessing function.
    Splits data first, fits preprocessor exclusively on training partition,
    and returns transformed arrays along with the fitted pipeline.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values

    # Determine numeric and categorical feature columns
    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)

    # 1. Split BEFORE fitting any transforms (Prevents Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # 2. Build and fit pipeline strictly on X_train
    preprocessor = build_preprocessor(numeric_features=num_cols, categorical_features=cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 3. Derive output feature names
    feature_names = list(num_cols)
    if cat_cols and hasattr(preprocessor.named_transformers_["cat"].named_steps["encoder"], "get_feature_names_out"):
        cat_names = list(preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_cols))
        feature_names.extend(cat_names)

    # 4. Save preprocessor artifact if requested
    if save_artifact:
        joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
        print(f"Fitted preprocessor serialized to {ARTIFACTS_DIR / 'preprocessor.joblib'}")

    return X_train_trans, X_test_trans, y_train, y_test, preprocessor, feature_names


def pyspark_preprocess(spark, input_path: str, output_path: str):
    """
    Template for distributed feature engineering using PySpark DataFrame APIs.
    """
    df = spark.read.option("header", True).csv(input_path)
    df = df.na.fill({"MonthlyIncome": 0, "NumberOfDependents": 0})
    df.write.mode("overwrite").parquet(output_path)
    print(f"Wrote distributed parquet dataset to {output_path}")
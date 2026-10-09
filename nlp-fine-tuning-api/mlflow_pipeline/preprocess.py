"""
Leakage-Free Preprocessing Utilities for Tabular and Feature Pipelines.
Engineered by M. Abdullah.
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values

    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if (len(np.unique(y)) > 1 and len(y) * test_size >= len(np.unique(y)) and len(y) * (1 - test_size) >= len(np.unique(y))) else None
    )

    preprocessor = build_preprocessor(numeric_features=num_cols, categorical_features=cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    if save_artifact:
        joblib.dump(preprocessor, ARTIFACTS_DIR / "scaler.joblib")

    return X_train_trans, X_test_trans, y_train, y_test
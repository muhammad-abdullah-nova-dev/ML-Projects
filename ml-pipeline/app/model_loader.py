"""
Model and preprocessor loader for credit risk serving API.
Engineered by M. Abdullah.
Supports local joblib artifacts with auto-train fallback, as well as MLflow registry URIs.
"""
import os
from pathlib import Path
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

_cached_model = None
_cached_preprocessor = None


def load_local_pipeline():
    """Load local preprocessor and model, training automatically if absent."""
    global _cached_model, _cached_preprocessor
    if _cached_model is not None and _cached_preprocessor is not None:
        return _cached_preprocessor, _cached_model

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        print("Model or preprocessor artifacts not found. Bootstrapping training run...")
        from mlflow_pipeline.data_ingest import ensure_dataset
        from mlflow_pipeline.preprocess import pandas_preprocess
        from mlflow_pipeline.train import train_and_benchmark
        import pandas as pd

        data_path = ensure_dataset()
        df = pd.read_csv(data_path)
        X_train, X_test, y_train, y_test, preprocessor, feature_names = pandas_preprocess(
            df=df, target_col="SeriousDlqin2yrs"
        )
        train_and_benchmark(X_train, X_test, y_train, y_test, feature_names)

    _cached_preprocessor = joblib.load(PREPROCESSOR_PATH)
    _cached_model = joblib.load(MODEL_PATH)
    return _cached_preprocessor, _cached_model


def load_model(model_uri: str = None, registered_name: str = None, stage: str = "Production"):
    """Load model from MLflow if specified, otherwise return local joblib pipeline."""
    if model_uri or registered_name:
        try:
            import mlflow
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            uri = model_uri or f"models:/{registered_name}/{stage}"
            return mlflow.pyfunc.load_model(uri)
        except Exception as e:
            print(f"Failed to load from MLflow ({e}); falling back to local artifacts.")

    _, model = load_local_pipeline()
    return model
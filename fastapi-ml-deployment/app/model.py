"""
Model loader module with dynamic artifact location and automatic fallback training.
"""
import json
from pathlib import Path
import joblib

MODEL_VERSION = "v1"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / f"iris_clf_{MODEL_VERSION}.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

_cached_model = None
_cached_metadata = None


def get_metadata() -> dict:
    global _cached_metadata
    if _cached_metadata is None:
        if not METADATA_PATH.exists():
            load_model()  # triggers training which populates metadata
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                _cached_metadata = json.load(f)
        else:
            _cached_metadata = {
                "model_name": "Iris Species Classifier",
                "model_version": MODEL_VERSION,
                "model_type": "Pipeline(StandardScaler -> RandomForestClassifier)",
                "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                "classes": ["setosa", "versicolor", "virginica"],
                "metrics": {},
                "status": "uninitialized"
            }
    return _cached_metadata


def load_model():
    """
    Load the serialized model pipeline.
    If artifact does not exist on disk, trains a fresh model automatically.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not MODEL_PATH.exists():
        print(f"Model artifact not found at {MODEL_PATH}. Initializing training...")
        from app.train import train_model
        _cached_model, _ = train_model()
    else:
        _cached_model = joblib.load(MODEL_PATH)

    return _cached_model

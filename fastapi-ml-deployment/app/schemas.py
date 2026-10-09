"""
Pydantic schemas for Iris Classifier API.
Includes input validation, bounded ranges, batch schemas, and drift detection structures.
"""
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0.0, le=20.0, description="Sepal length in cm", examples=[5.1])
    sepal_width: float = Field(..., gt=0.0, le=20.0, description="Sepal width in cm", examples=[3.5])
    petal_length: float = Field(..., gt=0.0, le=20.0, description="Petal length in cm", examples=[1.4])
    petal_width: float = Field(..., gt=0.0, le=20.0, description="Petal width in cm", examples=[0.2])


class PredictionResponse(BaseModel):
    predicted_class_id: int = Field(..., description="Predicted class index (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Human-readable species name")
    probabilities: Dict[str, float] = Field(..., description="Predicted class probability distribution")
    confidence: float = Field(..., description="Probability of the top predicted class")
    model_version: str = Field(..., description="Serving model version identifier")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchPredictionRequest(BaseModel):
    instances: List[IrisFeatures] = Field(..., min_length=1, max_length=1000, description="List of feature instances")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    total_latency_ms: float
    model_version: str


class DriftCheckRequest(BaseModel):
    features: List[IrisFeatures] = Field(..., min_length=10, description="Batch of observations (min 10) to test for feature drift")
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5, description="Significance level for two-sample Kolmogorov-Smirnov test")


class FeatureDriftResult(BaseModel):
    ks_statistic: float
    p_value: float
    is_drifted: bool
    baseline_mean: float
    batch_mean: float


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    drifted_features: List[str]
    feature_metrics: Dict[str, FeatureDriftResult]
    sample_size: int
    alpha: float
    baseline_reference: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    classes: List[str]
    feature_names: List[str]
    metrics: Dict[str, float]
    trained_at: str
    status: str

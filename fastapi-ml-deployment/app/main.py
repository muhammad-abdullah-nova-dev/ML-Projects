"""
Production FastAPI application for Iris Classifier ML Serving.
Engineered by M. Abdullah.
Features:
- Validated Pydantic v2 schemas
- Single and batch inference
- In-memory metrics & Prometheus exporter
- Statistical data drift detection (KS test)
- Model metadata & readiness healthchecks
- Bearer token authentication
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.model import load_model, get_metadata, MODEL_VERSION
from app.predict import predict_instance, predict_batch_instances
from app.schemas import (
    IrisFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    ModelInfoResponse
)
from app.drift import detect_data_drift
from app.metrics import tracker
from app.utils import is_authorized


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure model is ready and loaded on startup
    load_model()
    yield


app = FastAPI(
    title="Iris Classifier Production ML API",
    description="High-performance machine learning inference API with statistical data drift monitoring, batch predictions, and Prometheus metrics. Maintained by M. Abdullah.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_auth(request: Request):
    tracker.record_request()
    if not is_authorized(request.headers):
        tracker.record_error()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid Bearer token required in Authorization header"
        )


@app.get("/", tags=["General"])
def read_root():
    return {
        "service": "Iris Classifier ML API",
        "version": "2.0.0",
        "maintainer": "M. Abdullah",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /predict/batch",
            "drift_check": "POST /drift/check",
            "metrics": "/metrics",
            "documentation": "/docs"
        }
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    model = load_model()
    is_ready = model is not None
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "model_version": MODEL_VERSION
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Governance"])
def get_model_information():
    meta = get_metadata()
    return ModelInfoResponse(
        model_name=meta.get("model_name", "Iris Species Classifier"),
        model_version=meta.get("model_version", MODEL_VERSION),
        model_type=meta.get("model_type", "RandomForestClassifier"),
        classes=meta.get("classes", ["setosa", "versicolor", "virginica"]),
        feature_names=meta.get("features", ["sepal_length", "sepal_width", "petal_length", "petal_width"]),
        metrics=meta.get("metrics", {}),
        trained_at=meta.get("trained_at", "unknown"),
        status=meta.get("status", "active")
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_auth)], tags=["Inference"])
async def predict_single(features: IrisFeatures):
    try:
        model = load_model()
        result = predict_instance(model, features, model_version=MODEL_VERSION)
        tracker.record_prediction(result.predicted_class_name, result.latency_ms)
        return result
    except Exception as e:
        tracker.record_error()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, dependencies=[Depends(verify_auth)], tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest):
    try:
        model = load_model()
        results, total_latency = predict_batch_instances(model, request.instances, model_version=MODEL_VERSION)
        for r in results:
            tracker.record_prediction(r.predicted_class_name, r.latency_ms)
        return BatchPredictionResponse(
            predictions=results,
            total_count=len(results),
            total_latency_ms=total_latency,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        tracker.record_error()
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.post("/drift/check", response_model=DriftCheckResponse, dependencies=[Depends(verify_auth)], tags=["Monitoring"])
async def check_drift(request: DriftCheckRequest):
    try:
        report = detect_data_drift(request.features, alpha=request.alpha)
        return report
    except Exception as e:
        tracker.record_error()
        raise HTTPException(status_code=500, detail=f"Drift computation error: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
def get_metrics(format: str = "prometheus"):
    if format == "json":
        return tracker.get_summary()
    return PlainTextResponse(tracker.to_prometheus_format(), media_type="text/plain; version=0.0.4")

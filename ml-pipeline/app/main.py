"""
FastAPI application for End-to-End Tabular Credit Risk Pipeline Serving.
Engineered and maintained by M. Abdullah.
Features:
- Validated Pydantic v2 underwriting input schemas
- Risk tiering and threshold-based decisioning
- Batch assessment endpoints
- Automated artifact bootstrap
- Model governance and benchmark reporting
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from app.schemas import (
    CreditApplicantFeatures,
    CreditPredictionResponse,
    BatchCreditRequest,
    BatchCreditResponse,
    ModelGovernanceResponse
)
from app.model_loader import load_local_pipeline

app = FastAPI(
    title="Enterprise Credit Risk ML Serving API",
    description="End-to-end production underwriting inference API with class-imbalance mitigation, calibration metrics, and governance reporting. Maintained by M. Abdullah.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
BENCHMARK_PATH = ARTIFACTS_DIR / "benchmark_results.json"


def get_risk_tier(score: float) -> str:
    if score < 0.15:
        return "Low Risk"
    elif score < 0.35:
        return "Moderate Risk"
    return "High Risk"


def applicant_to_df(applicant: CreditApplicantFeatures) -> pd.DataFrame:
    data = {
        "RevolvingUtilizationOfUnsecuredLines": [applicant.RevolvingUtilizationOfUnsecuredLines],
        "age": [applicant.age],
        "NumberOfTime30-59DaysPastDueNotWorse": [applicant.NumberOfTime30_59DaysPastDueNotWorse],
        "DebtRatio": [applicant.DebtRatio],
        "MonthlyIncome": [applicant.MonthlyIncome],
        "NumberOfOpenCreditLinesAndLoans": [applicant.NumberOfOpenCreditLinesAndLoans],
        "NumberOfTimes90DaysLate": [applicant.NumberOfTimes90DaysLate],
        "NumberRealEstateLoansOrLines": [applicant.NumberRealEstateLoansOrLines],
        "NumberOfTime60-89DaysPastDueNotWorse": [applicant.NumberOfTime60_89DaysPastDueNotWorse],
        "NumberOfDependents": [applicant.NumberOfDependents],
    }
    return pd.DataFrame(data)


@app.get("/", tags=["General"])
def read_root():
    return {
        "service": "Credit Risk ML Assessment Engine",
        "version": "2.0.0",
        "maintainer": "M. Abdullah",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "governance": "/model/governance",
            "single_applicant": "POST /predict",
            "batch_underwriting": "POST /predict/batch",
            "documentation": "/docs"
        }
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    try:
        preprocessor, model = load_local_pipeline()
        return {
            "status": "healthy",
            "preprocessor_ready": preprocessor is not None,
            "model_ready": model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/model/governance", response_model=ModelGovernanceResponse, tags=["Governance"])
def get_model_governance():
    if not BENCHMARK_PATH.exists():
        load_local_pipeline()  # Trigger training to produce benchmark report

    with open(BENCHMARK_PATH, "r") as f:
        meta = json.load(f)

    return ModelGovernanceResponse(
        model_name="Consumer Credit Default Risk Scorer",
        maintainer="M. Abdullah",
        best_model_algorithm=meta.get("best_model", "gradient_boosting"),
        best_roc_auc=meta.get("best_roc_auc", 0.0),
        feature_names=meta.get("feature_names", []),
        benchmark_summary=meta.get("benchmark", {}),
        status="production"
    )


@app.post("/predict", response_model=CreditPredictionResponse, tags=["Underwriting"])
def predict_credit_risk(applicant: CreditApplicantFeatures, decision_threshold: float = 0.25):
    start_t = time.perf_counter()
    try:
        preprocessor, model = load_local_pipeline()
        df = applicant_to_df(applicant)
        X_trans = preprocessor.transform(df)

        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(X_trans)[0][1])
        else:
            score = float(model.predict(X_trans)[0])

        score = round(score, 4)
        risk_tier = get_risk_tier(score)
        approved = bool(score < decision_threshold)
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return CreditPredictionResponse(
            default_risk_score=score,
            risk_tier=risk_tier,
            approved=approved,
            threshold_used=decision_threshold,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchCreditResponse, tags=["Underwriting"])
def predict_batch_credit(request: BatchCreditRequest):
    start_t = time.perf_counter()
    try:
        preprocessor, model = load_local_pipeline()
        dfs = [applicant_to_df(a) for a in request.applicants]
        combined_df = pd.concat(dfs, ignore_index=True)
        X_trans = preprocessor.transform(combined_df)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_trans)[:, 1]
        else:
            scores = model.predict(X_trans).astype(float)

        results = []
        approved_count = 0
        total_latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        per_item_latency = round(total_latency_ms / len(request.applicants), 3)

        for s in scores:
            score_val = round(float(s), 4)
            is_app = bool(score_val < request.decision_threshold)
            if is_app:
                approved_count += 1
            results.append(
                CreditPredictionResponse(
                    default_risk_score=score_val,
                    risk_tier=get_risk_tier(score_val),
                    approved=is_app,
                    threshold_used=request.decision_threshold,
                    latency_ms=per_item_latency
                )
            )

        return BatchCreditResponse(
            results=results,
            total_processed=len(results),
            approval_rate=round(approved_count / len(results), 4),
            average_risk_score=round(float(np.mean(scores)), 4),
            total_latency_ms=total_latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")
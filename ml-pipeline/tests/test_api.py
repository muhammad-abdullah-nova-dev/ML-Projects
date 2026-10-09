"""
Integration tests for credit risk underwriting API.
Engineered by M. Abdullah.
"""
import pytest


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "Credit Risk ML Assessment Engine"
    assert data["maintainer"] == "M. Abdullah"


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["preprocessor_ready"] is True
    assert data["model_ready"] is True


def test_model_governance(client):
    res = client.get("/model/governance")
    assert res.status_code == 200
    data = res.json()
    assert data["maintainer"] == "M. Abdullah"
    assert "best_model_algorithm" in data
    assert data["best_roc_auc"] > 0.6


def test_predict_single_applicant(client):
    applicant = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 7500.0,
        "NumberOfOpenCreditLinesAndLoans": 10,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0
    }
    res = client.post("/predict?decision_threshold=0.30", json=applicant)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["default_risk_score"] <= 1.0
    assert data["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert isinstance(data["approved"], bool)
    assert data["latency_ms"] >= 0.0


def test_predict_validation_error_underage(client):
    # Age under 18 must be rejected by Pydantic schema
    invalid_applicant = {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 16,
        "DebtRatio": 0.35,
        "NumberOfOpenCreditLinesAndLoans": 2
    }
    res = client.post("/predict", json=invalid_applicant)
    assert res.status_code == 422


def test_predict_batch_applicants(client):
    payload = {
        "applicants": [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.15,
                "age": 55,
                "DebtRatio": 0.20,
                "MonthlyIncome": 9500.0,
                "NumberOfOpenCreditLinesAndLoans": 8
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.95,
                "age": 28,
                "NumberOfTimes90DaysLate": 3,
                "DebtRatio": 0.85,
                "MonthlyIncome": 2500.0,
                "NumberOfOpenCreditLinesAndLoans": 14
            }
        ],
        "decision_threshold": 0.30
    }
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 2
    assert 0.0 <= data["approval_rate"] <= 1.0
    assert len(data["results"]) == 2

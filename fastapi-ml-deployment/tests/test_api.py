"""
Unit and integration tests for FastAPI ML Serving API.
Tests cover health, authentication, input validation, single/batch prediction,
Prometheus metrics, and statistical data drift detection.
"""
import pytest
from app.utils import API_TOKEN

AUTH_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Iris Classifier ML API"
    assert data["maintainer"] == "M. Abdullah"
    assert "endpoints" in data


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Iris Species Classifier"
    assert set(data["classes"]) == {"setosa", "versicolor", "virginica"}
    assert len(data["feature_names"]) == 4
    assert data["status"] == "active"


def test_predict_unauthorized(client):
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["detail"]


def test_predict_single_success(client):
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=sample, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class_id"] in [0, 1, 2]
    assert data["predicted_class_name"] in ["setosa", "versicolor", "virginica"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3
    assert data["latency_ms"] >= 0.0


def test_predict_validation_error_negative_feature(client):
    # Sepal length cannot be <= 0 according to Pydantic Field(gt=0.0)
    invalid_sample = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=invalid_sample, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_predict_batch_success(client):
    batch = {
        "instances": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
            {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}
        ]
    }
    response = client.post("/predict/batch", json=batch, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["predicted_class_name"] == "setosa"


def test_drift_check_success(client):
    # Provide synthetic observations
    features = [
        {"sepal_length": 5.1 + i*0.05, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        for i in range(12)
    ]
    payload = {"features": features, "alpha": 0.05}
    response = client.post("/drift/check", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data
    assert "feature_metrics" in data
    assert "sepal_length" in data["feature_metrics"]
    assert data["sample_size"] == 12


def test_metrics_prometheus_and_json(client):
    # Test prometheus plain text
    res_prom = client.get("/metrics")
    assert res_prom.status_code == 200
    assert "ml_requests_total" in res_prom.text

    # Test json telemetry
    res_json = client.get("/metrics?format=json")
    assert res_json.status_code == 200
    data = res_json.json()
    assert "total_requests" in data
    assert "latency_ms" in data

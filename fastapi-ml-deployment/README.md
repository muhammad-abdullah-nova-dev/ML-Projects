# ML-as-a-Service: Iris Classifier API

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

Production-grade machine learning model serving API built with FastAPI and Scikit-Learn. Engineered with enterprise patterns including statistical data drift detection (two-sample Kolmogorov-Smirnov test), single and batch inference, Prometheus telemetry, Pydantic v2 boundary validation, Bearer token authentication, and containerized Docker runtime.

Maintained and enhanced by **M. Abdullah**.

---

## 🚀 Key Features

* **Strict Input Validation**: Pydantic v2 schemas enforcing positive physiological boundaries, preventing negative lengths or malicious inputs.
* **Probability & Confidence Scoring**: Returns full class probability distribution alongside the predicted class and top confidence level.
* **Batch Inference**: High-throughput `/predict/batch` endpoint optimizing matrix transformations for multi-item payloads.
* **Statistical Data Drift Monitoring**: `/drift/check` calculates two-sample Kolmogorov-Smirnov (KS) statistics comparing incoming observation batches against baseline training data.
* **Prometheus Metrics**: In-memory telemetry exposed at `/metrics` (request counts, prediction counters by class, p50/p95/p99 latency gauges).
* **Automated Fallback Training**: If pre-trained model artifacts are absent, the application automatically fits and serializes a cross-validated pipeline on startup.
* **Bearer Token Security**: Constant-time token verification protecting inference and telemetry endpoints.

---

## 🛠️ Architecture & Tech Stack

* **Framework**: FastAPI (ASGI) + Uvicorn
* **ML Engine**: Scikit-Learn `Pipeline(StandardScaler -> RandomForestClassifier)`
* **Validation**: Pydantic v2 with custom fields and metadata
* **Statistics**: SciPy (`scipy.stats.ks_2samp`) for distribution shift analysis
* **Testing**: PyTest + FastAPI TestClient

---

## 📋 API Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/` | Service metadata, endpoint directory, and maintainer info | No |
| `GET` | `/health` | Liveness and model readiness probe | No |
| `GET` | `/model/info` | Model metadata, cross-validation metrics, feature list | No |
| `POST` | `/predict` | Single sample inference | Yes (`Bearer <token>`) |
| `POST` | `/predict/batch` | High-throughput batch inference | Yes (`Bearer <token>`) |
| `POST` | `/drift/check` | Kolmogorov-Smirnov test against baseline distributions | Yes (`Bearer <token>`) |
| `GET` | `/metrics` | Prometheus telemetry or JSON summary (`?format=json`) | No |
| `GET` | `/docs` | Interactive Swagger OpenAPI documentation | No |

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone and enter directory
cd fastapi-ml-deployment

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template configuration:
```bash
cp .env.example .env
```
Default development token is configured in `app/utils.py` as `default-dev-token` (or configure your own via `API_TOKEN`).

### 3. Model Training
```bash
python app/train.py
```
This runs 5-Fold Stratified Cross-Validation, serializes `model/iris_clf_v1.pkl`, and writes baseline statistics to `model/metadata.json`.

### 4. Running the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are accessible at: `http://localhost:8000/docs`

---

## 🧪 Testing
Run the comprehensive automated test suite:
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment
```bash
# Build the Docker image
docker build -t iris-classifier-api:latest .

# Run the container
docker run -d -p 8000:8000 -e API_TOKEN="production-secret-token" iris-classifier-api:latest
```

---

## 📡 Example API Requests

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer default-dev-token" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```
**Response:**
```json
{
  "predicted_class_id": 0,
  "predicted_class_name": "setosa",
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.00
  },
  "confidence": 0.98,
  "model_version": "v1",
  "latency_ms": 1.42
}
```

### Data Drift Check
```bash
curl -X POST "http://localhost:8000/drift/check" \
  -H "Authorization: Bearer default-dev-token" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 5.2, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2},
      {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2},
      {"sepal_length": 4.6, "sepal_width": 3.1, "petal_length": 1.5, "petal_width": 0.2},
      {"sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 5.4, "sepal_width": 3.9, "petal_length": 1.7, "petal_width": 0.4},
      {"sepal_length": 4.6, "sepal_width": 3.4, "petal_length": 1.4, "petal_width": 0.3},
      {"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2},
      {"sepal_length": 4.4, "sepal_width": 2.9, "petal_length": 1.4, "petal_width": 0.2}
    ],
    "alpha": 0.05
  }'
```

---

## 👨‍💻 Maintainer & Attribution
- **Enhanced Implementation Maintainer**: **M. Abdullah**
- **Original Project Origin**: Derivative work based on open-source project by `torresjchristopher`.
- **Copyright**: Copyright © 2026 M. Abdullah for enhancements and additions.

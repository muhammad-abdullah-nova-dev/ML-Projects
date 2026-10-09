# NLP Model Fine-Tuning & API Deployment (Hugging Face + PyTorch + FastAPI)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

An end-to-end NLP engineering pipeline to fine-tune Transformer architectures (DistilBERT / BERT) for text classification and sentiment analysis using Hugging Face Transformers and PyTorch, track experiments and metrics in MLflow, package the inference engine with FastAPI and Docker, and serve via cloud container runtimes.

Maintained and enhanced by **M. Abdullah**.

---

## 🚀 Key Features

* **Transformer Fine-Tuning**: Config-driven training pipeline leveraging Hugging Face `Trainer` API with evaluation callbacks, mixed precision, and warmup schedules.
* **Experiment Tracking**: Automatic logging of training loss, evaluation accuracy, F1-scores, and hyperparameters to MLflow.
* **Production Serving**: Asynchronous FastAPI inference service with request validation, probability scoring, and health checks.
* **Containerization & CI/CD**: Multi-stage Docker build and GitHub Actions workflows for automated linting, testing, and container deployment.

---

## 📁 Repository Structure

```
nlp-fine-tuning-api/
├── configs/
│   ├── train_config.yaml         # Training hyperparameter configuration
│   └── train.yaml                # Model search config
├── app/
│   ├── main.py                   # FastAPI model-serving application
│   ├── model_loader.py           # Transformer pipeline loader
│   └── schemas.py                # Pydantic request/response schemas
├── mlflow_pipeline/              # Feature processing and MLflow utilities
├── tests/                        # Automated unit tests
├── Dockerfile                    # Containerization manifest
├── k8s.yaml                      # Kubernetes deployment & HPA skeleton
├── train.py                      # CLI fine-tuning script
├── serve.py                      # Uvicorn ASGI server runner
└── requirements.txt              # Pinned dependencies
```

---

## ⚡ Quickstart

### 1. Installation
```bash
cd nlp-fine-tuning-api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fine-Tuning
```bash
python train.py \
  --model_name_or_path distilbert-base-uncased \
  --data_file data/dataset.csv \
  --text_col text \
  --label_col label \
  --output_dir outputs/distilbert-sentiment \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16
```

### 3. Serving API
```bash
python serve.py --model_dir outputs/distilbert-sentiment --port 8080
```
Test prediction:
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The operational reliability of this pipeline exceeded expectations."}'
```

---

## 🧪 Testing
```bash
pytest tests/ -v
```

---

## 👨‍💻 Maintainer & Attribution
- **Enhanced Implementation Maintainer**: **M. Abdullah**
- **Original Project Origin**: Derivative work based on open-source project by `torresjchristopher`.
- **Copyright**: Copyright © 2026 M. Abdullah for enhancements and additions.

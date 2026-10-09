"""
Prediction service for Iris Classifier.
Executes inference, computes class probabilities, maps labels, and tracks latency.
"""
import time
from typing import Dict, List, Tuple
import numpy as np
from app.schemas import IrisFeatures, PredictionResponse

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def feature_to_array(features: IrisFeatures) -> List[float]:
    return [
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]


def predict_instance(model, features: IrisFeatures, model_version: str = "v1") -> PredictionResponse:
    start_time = time.perf_counter()
    input_data = [feature_to_array(features)]

    pred_idx = int(model.predict(input_data)[0])
    probabilities_raw = model.predict_proba(input_data)[0]

    probabilities: Dict[str, float] = {
        name: float(round(prob, 4))
        for name, prob in zip(CLASS_NAMES, probabilities_raw)
    }
    confidence = float(round(probabilities_raw[pred_idx], 4))
    latency_ms = float(round((time.perf_counter() - start_time) * 1000, 3))

    return PredictionResponse(
        predicted_class_id=pred_idx,
        predicted_class_name=CLASS_NAMES[pred_idx],
        probabilities=probabilities,
        confidence=confidence,
        model_version=model_version,
        latency_ms=latency_ms
    )


def predict_batch_instances(model, instances: List[IrisFeatures], model_version: str = "v1") -> Tuple[List[PredictionResponse], float]:
    start_time = time.perf_counter()
    matrix = [feature_to_array(inst) for inst in instances]

    preds = model.predict(matrix)
    probas = model.predict_proba(matrix)
    total_latency_ms = float(round((time.perf_counter() - start_time) * 1000, 3))

    results: List[PredictionResponse] = []
    for i in range(len(instances)):
        idx = int(preds[i])
        probs = {name: float(round(p, 4)) for name, p in zip(CLASS_NAMES, probas[i])}
        results.append(
            PredictionResponse(
                predicted_class_id=idx,
                predicted_class_name=CLASS_NAMES[idx],
                probabilities=probs,
                confidence=float(round(probas[i][idx], 4)),
                model_version=model_version,
                latency_ms=float(round(total_latency_ms / len(instances), 3))
            )
        )

    return results, total_latency_ms

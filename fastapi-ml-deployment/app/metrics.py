"""
Metrics and telemetry tracker for production monitoring.
Tracks total requests, error rates, latencies, and prediction distributions.
Compatible with Prometheus scraping.
"""
import time
from collections import Counter
from typing import Dict, Any
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.predictions_by_class: Counter = Counter()
        self.latencies = []
        self.start_time = time.time()

    def record_request(self):
        self.request_count += 1

    def record_error(self):
        self.error_count += 1

    def record_prediction(self, class_name: str, latency_ms: float):
        self.prediction_count += 1
        self.predictions_by_class[class_name] += 1
        self.latencies.append(latency_ms)
        # Keep sliding window of last 5000 latency measurements
        if len(self.latencies) > 5000:
            self.latencies.pop(0)

    def get_summary(self) -> Dict[str, Any]:
        uptime_seconds = time.time() - self.start_time
        lat_arr = np.array(self.latencies) if self.latencies else np.array([0.0])
        return {
            "uptime_seconds": float(round(uptime_seconds, 1)),
            "total_requests": self.request_count,
            "total_predictions": self.prediction_count,
            "total_errors": self.error_count,
            "error_rate": float(round(self.error_count / max(self.request_count, 1), 4)),
            "predictions_by_class": dict(self.predictions_by_class),
            "latency_ms": {
                "mean": float(round(float(np.mean(lat_arr)), 2)),
                "p50": float(round(float(np.percentile(lat_arr, 50)), 2)),
                "p95": float(round(float(np.percentile(lat_arr, 95)), 2)),
                "p99": float(round(float(np.percentile(lat_arr, 99)), 2)),
            }
        }

    def to_prometheus_format(self) -> str:
        summary = self.get_summary()
        lines = [
            "# HELP ml_requests_total Total HTTP requests handled",
            "# TYPE ml_requests_total counter",
            f"ml_requests_total {summary['total_requests']}",
            "",
            "# HELP ml_predictions_total Total ML predictions generated",
            "# TYPE ml_predictions_total counter",
            f"ml_predictions_total {summary['total_predictions']}",
            "",
            "# HELP ml_errors_total Total errors encountered",
            "# TYPE ml_errors_total counter",
            f"ml_errors_total {summary['total_errors']}",
            "",
            "# HELP ml_latency_p95_ms 95th percentile inference latency in milliseconds",
            "# TYPE ml_latency_p95_ms gauge",
            f"ml_latency_p95_ms {summary['latency_ms']['p95']}",
        ]
        for cls, count in summary["predictions_by_class"].items():
            lines.append(f'ml_predictions_by_class_total{{class="{cls}"}} {count}')
        return "\n".join(lines) + "\n"


tracker = MetricsTracker()

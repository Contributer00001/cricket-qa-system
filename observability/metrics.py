"""
Custom Prometheus metrics for the Cricket QA service.

Registered metrics
──────────────────
cricket_requests_total          counter   (intent, status)
cricket_inference_duration_ms   histogram (intent, llm_used)
cricket_intent_confidence       histogram (intent)
cricket_model_loaded            gauge
cricket_grounding_failures_total counter

Usage (in service/app.py)
──────────────────────────
    from observability.metrics import (
        record_inference, set_model_loaded, record_grounding_failure
    )

    record_inference(intent="total_runs", status="success",
                     duration_ms=12.4, confidence=0.95, llm_used=False)
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    _REQUEST_COUNT = Counter(
        "cricket_requests_total",
        "Total inference requests",
        ["intent", "status"],
    )

    _INFERENCE_DURATION = Histogram(
        "cricket_inference_duration_ms",
        "End-to-end inference latency in milliseconds",
        ["intent", "llm_used"],
        buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
    )

    _INTENT_CONFIDENCE = Histogram(
        "cricket_intent_confidence",
        "QueryRouter confidence score per intent",
        ["intent"],
        buckets=[0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    )

    _MODEL_LOADED = Gauge(
        "cricket_model_loaded",
        "1 when the LLM is loaded and ready, 0 otherwise",
    )

    _GROUNDING_FAILURES = Counter(
        "cricket_grounding_failures_total",
        "Times the LLM produced a number not grounded in context",
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False


def record_inference(
    intent: str,
    status: str,
    duration_ms: float,
    confidence: float,
    llm_used: bool,
) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    _REQUEST_COUNT.labels(intent=intent, status=status).inc()
    _INFERENCE_DURATION.labels(intent=intent, llm_used=str(llm_used)).observe(duration_ms)
    _INTENT_CONFIDENCE.labels(intent=intent).observe(confidence)


def set_model_loaded(loaded: bool) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    _MODEL_LOADED.set(1 if loaded else 0)


def record_grounding_failure() -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    _GROUNDING_FAILURES.inc()

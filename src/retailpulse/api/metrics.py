"""Prometheus metrics registry, collectors, and scraping endpoint.

Implements P5-09, PRD §7.7 (FR-API-04), and API §1.1 / §2.8:
- GET /metrics: Prometheus text exposition format (blocked at proxy in prod per SEC-WEB-05).
- http_requests_total: Counter partitioned by method, handler, and status.
- http_request_duration_seconds: Latency histogram partitioned by method and handler.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

router = APIRouter(tags=["Metrics"])

# HTTP request count partitioned by method, path template/handler, and HTTP status code
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP request count partitioned by method, handler, and status code.",
    ["method", "handler", "status"],
)

# Request latency histogram in seconds
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds partitioned by method and handler.",
    ["method", "handler"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0),
)


def record_http_metrics(
    method: str,
    handler: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record request count and duration in Prometheus metrics."""
    clean_method = method.upper()
    status_str = str(status_code)
    HTTP_REQUESTS_TOTAL.labels(
        method=clean_method,
        handler=handler,
        status=status_str,
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=clean_method,
        handler=handler,
    ).observe(duration_seconds)


@router.get(
    "/metrics",
    summary="Prometheus Metrics",
    response_description="Prometheus text exposition format",
    include_in_schema=False,
)
def get_metrics() -> Response:
    """Prometheus metrics scraping endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


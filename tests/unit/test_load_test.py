"""Unit tests for load testing scenarios and metrics computation (P9-02).

Validates:
- EndpointMetrics statistics (p50, p90, p95, p99, mean, error_rate).
- Locust scenario configuration (RetailPulseUser, tasks, weights, SLAs).
- k6 scenario script configuration (stages, thresholds, metrics).
- In-process load test harness execution and markdown reporting.
"""

from __future__ import annotations

import asyncio
import pathlib

from scripts.run_load_test import (
    TRD_CACHED_KPI_P95_MS,
    TRD_INDEXED_READ_P95_MS,
    EndpointMetrics,
    run_load_test,
    update_performance_report_with_load_test,
)


def test_endpoint_metrics_percentiles_and_status() -> None:
    """Test latency percentile calculation, error rates, and SLA status reporting."""
    metrics = EndpointMetrics(
        name="GET /api/v1/forecasts",
        endpoint_type="Indexed Read",
        target_p95_ms=TRD_INDEXED_READ_P95_MS,
    )
    assert metrics.total == 0
    assert metrics.p95 == 0.0
    assert metrics.mean == 0.0
    assert metrics.error_rate == 0.0

    # Add 100 sample latencies: 1.0ms to 100.0ms
    metrics.latencies = [float(i) for i in range(1, 101)]
    assert metrics.total == 100
    assert metrics.p50 == 50.0
    assert metrics.p90 == 90.0
    assert metrics.p95 == 95.0
    assert metrics.p99 == 99.0
    assert metrics.mean == 50.5
    assert metrics.error_rate == 0.0
    assert metrics.status == "MET"

    # Test SLA breach
    metrics.latencies = [350.0] * 100
    assert metrics.p95 == 350.0
    assert metrics.status == "BREACHED"

    # Test error threshold breach
    metrics.latencies = [50.0] * 90
    metrics.errors = 10
    assert metrics.total == 100
    assert metrics.error_rate == 10.0
    assert metrics.status == "BREACHED"


def test_locustfile_structure() -> None:
    """Test that locustfile defines RetailPulseUser with proper tasks and headers."""
    locustfile_path = pathlib.Path("tests/load/locustfile.py")
    assert locustfile_path.exists(), "tests/load/locustfile.py must exist"

    content = locustfile_path.read_text(encoding="utf-8")
    assert "class RetailPulseUser(HttpUser):" in content
    assert "/api/v1/kpis/overview" in content
    assert "/api/v1/forecasts" in content
    assert "/api/v1/customers" in content
    assert "/api/v1/inventory/recommendations" in content
    assert "/api/v1/products" in content
    assert "SLAVerifier" in content or "TRD_INDEXED_READ_P95_MS" in content


def test_k6_scenario_structure() -> None:
    """Test that k6_scenario.js defines stages and thresholds matching TRD §7."""
    k6_path = pathlib.Path("tests/load/k6_scenario.js")
    assert k6_path.exists(), "tests/load/k6_scenario.js must exist"

    content = k6_path.read_text(encoding="utf-8")
    assert "target: 50" in content
    assert "p(95)<300" in content
    assert "p(95)<1000" in content
    assert "rate<0.01" in content
    assert "/api/v1/kpis/overview" in content
    assert "/api/v1/forecasts" in content


def test_run_load_test_smoke() -> None:
    """Smoke test running in-process load test with small concurrency and duration."""
    metrics_map = asyncio.run(run_load_test(concurrency=2, duration_seconds=3.0))

    assert len(metrics_map) == 5
    total_reqs = sum(m.total for m in metrics_map.values())
    assert total_reqs > 0, "Expected at least one request during smoke test"
    for name, metric in metrics_map.items():
        assert metric.error_rate == 0.0, f"Expected 0 errors for {name}, got {metric.errors}"
        if metric.total > 0:
            assert metric.p95 > 0.0, f"Expected valid p95 for {name}"


def test_update_performance_report_with_load_test(tmp_path: pathlib.Path) -> None:
    """Test markdown generation for performance report Section 5."""
    metrics_map: dict[str, EndpointMetrics] = {
        "GET /api/v1/forecasts": EndpointMetrics(
            name="GET /api/v1/forecasts",
            endpoint_type="Indexed Read",
            target_p95_ms=TRD_INDEXED_READ_P95_MS,
            latencies=[25.0, 30.0, 45.0, 50.0],
            errors=0,
        ),
        "GET /api/v1/kpis/overview": EndpointMetrics(
            name="GET /api/v1/kpis/overview",
            endpoint_type="Cached KPI Aggregates",
            target_p95_ms=TRD_CACHED_KPI_P95_MS,
            latencies=[12.0, 15.0, 18.0, 20.0],
            errors=0,
        ),
    }

    report_file = tmp_path / "performance.md"
    report_file.write_text(
        "# Performance Report\n\n## 5. Migration Artifacts\n\nContent.\n", encoding="utf-8"
    )

    summary_md = update_performance_report_with_load_test(
        metrics_map=metrics_map,
        concurrency=50,
        duration_s=12.0,
        report_path=report_file,
    )

    assert "## 5. Concurrency & Load Testing Verification (P9-02)" in summary_md
    assert "GET /api/v1/forecasts" in summary_md
    assert "GET /api/v1/kpis/overview" in summary_md
    assert "50 concurrent users" in summary_md

    updated_content = report_file.read_text(encoding="utf-8")
    assert "## 5. Concurrency & Load Testing Verification (P9-02)" in updated_content
    assert "## 6. Migration Artifacts" in updated_content

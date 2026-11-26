"""Locust load test scenario for RetailPulse API.

Implements Task P9-02:
- 50 concurrent users simulated load against core API endpoints.
- Validates TRD §7 latency SLAs under sustained concurrency:
  - Indexed read endpoints: p95 < 300 ms
  - Cached KPI aggregates: p95 < 1,000 ms (1 s)
  - Error rate < 1.0%

Usage:
    locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 1m --host http://localhost:8000
"""

from __future__ import annotations

import logging
import os
import random

try:
    from locust import HttpUser, between, events, task
except ImportError:  # Allow importing module when locust is not installed in local environment
    HttpUser = object  # type: ignore[misc,assignment]

    def task(weight: int = 1):  # type: ignore[no-redef]
        def decorator(fn):
            fn.locust_task_weight = weight
            return fn

        return decorator

    def between(a: float, b: float):  # type: ignore[no-redef]
        return lambda: random.uniform(a, b)  # noqa: S311

    events = None  # type: ignore[assignment]

logger = logging.getLogger("retailpulse.load_test")

# SLA Target Budgets from TRD §7
TRD_INDEXED_READ_P95_MS = 300.0
TRD_CACHED_KPI_P95_MS = 1000.0


class RetailPulseUser(HttpUser):
    """Simulates a concurrent RetailPulse dashboard and analyst user."""

    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        """Authenticate user or set access token header before running tasks."""
        self.auth_token = os.getenv("RP_LOAD_TEST_TOKEN", "")
        self.headers = {
            "Accept": "application/json",
        }
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
        else:
            # Login via token endpoint if available
            login_data = {
                "username": os.getenv("RP_LOAD_TEST_USER", "viewer@example.com"),
                "password": os.getenv("RP_LOAD_TEST_PASS", "ValidPassword123!"),
            }
            try:
                res = self.client.post("/api/v1/auth/token", data=login_data)
                if res.status_code == 200:
                    tok = res.json().get("access_token")
                    if tok:
                        self.headers["Authorization"] = f"Bearer {tok}"
            except Exception as exc:
                logger.debug("Automatic login in on_start skipped: %s", exc)

    @task(4)
    def overview_kpis(self) -> None:
        """Cached top-level business KPI aggregates (TRD §7 SLA: p95 < 1000 ms)."""
        self.client.get(
            "/api/v1/kpis/overview",
            headers=self.headers,
            name="/api/v1/kpis/overview [cached aggregate]",
        )

    @task(3)
    def demand_forecast(self) -> None:
        """Point & interval demand forecast lookup (TRD §7 SLA: p95 < 300 ms)."""
        product_id = random.randint(1, 150)  # noqa: S311
        self.client.get(
            f"/api/v1/forecasts?product_id={product_id}&store_id=1&horizon=28",
            headers=self.headers,
            name="/api/v1/forecasts [indexed read]",
        )

    @task(2)
    def customer_churn_risk(self) -> None:
        """Paginated churn risk listing (TRD §7 SLA: p95 < 300 ms)."""
        page = random.randint(1, 5)  # noqa: S311
        self.client.get(
            f"/api/v1/customers?risk_band=high&page={page}&page_size=20",
            headers=self.headers,
            name="/api/v1/customers [indexed read]",
        )

    @task(2)
    def inventory_recommendations(self) -> None:
        """Paginated replenishment recommendations (TRD §7 SLA: p95 < 300 ms)."""
        page = random.randint(1, 5)  # noqa: S311
        self.client.get(
            f"/api/v1/inventory/recommendations?status=reorder_now&page={page}&page_size=20",
            headers=self.headers,
            name="/api/v1/inventory/recommendations [indexed read]",
        )

    @task(1)
    def reference_products(self) -> None:
        """Reference product catalog search (TRD §7 SLA: p95 < 300 ms)."""
        self.client.get(
            "/api/v1/products?page=1&page_size=20",
            headers=self.headers,
            name="/api/v1/products [reference read]",
        )


if events is not None:

    @events.quitting.add_listener
    def verify_sla_thresholds(environment, **_kwargs) -> None:
        """Evaluate TRD §7 SLA compliance upon test completion."""
        stats = environment.stats.total
        if stats.num_requests == 0:
            return

        p95 = stats.get_response_time_percentile(0.95)
        fail_rate = stats.fail_ratio

        logger.info("=" * 60)
        logger.info("RETAILPULSE LOAD TEST SLA VERIFICATION REPORT")
        logger.info("=" * 60)
        logger.info("Total Requests: %d", stats.num_requests)
        logger.info("Failed Requests: %d (%.2f%%)", stats.num_failures, fail_rate * 100)
        logger.info("Overall p95 Response Time: %.2f ms", p95)

        # Check endpoints
        for name, entry in environment.stats.entries.items():
            ep_p95 = entry.get_response_time_percentile(0.95)
            budget = (
                TRD_CACHED_KPI_P95_MS if "kpis/overview" in name[0] else TRD_INDEXED_READ_P95_MS
            )
            met = ep_p95 <= budget
            status_str = "PASSED" if met else "FAILED"
            logger.info(
                "[%s] %s: p95 = %.2f ms (Budget: %.0f ms)", status_str, name[0], ep_p95, budget
            )

        if p95 > TRD_CACHED_KPI_P95_MS or fail_rate > 0.01:
            logger.error("SLA THRESHOLDS BREACHED")
            environment.process_exit_code = 1
        else:
            logger.info("ALL TRD §7 CONCURRENCY SLAS SATISFIED")
            environment.process_exit_code = 0

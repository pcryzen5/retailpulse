"""RBAC Matrix and Route Inventory integration test suite.

Implements P5-10, PRD §7.7 (FR-SEC-01), API §2, and SECURITY §4 (SEC-AUTHZ-01, SEC-AUTHZ-02):
- Route inventory test: introspects app.routes to guarantee zero unmapped routes in the catalogue.
- Parametrized RBAC test matrix over every endpoint x {anonymous, viewer, analyst, admin}.
- Enforces 401 for unauthenticated calls to protected routes.
- Enforces 403 for authenticated calls with insufficient role.
- Confirms authorized execution proceeds beyond auth layer (not 401, not 403) for sufficient role.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi.routing import APIRoute

# Complete RBAC Catalogue from docs/API.md §2
# Catalogue entry schema: (method, path_template, minimum_role, request_kwargs_builder)
# minimum_role: "public", "viewer", "analyst", "admin"
RBAC_CATALOGUE = [
    # 2.8 Operations (root)
    ("GET", "/health/live", "public", lambda s: {}),
    ("GET", "/health/ready", "public", lambda s: {}),
    ("GET", "/health/startup", "public", lambda s: {}),
    ("GET", "/metrics", "public", lambda s: {}),
    # 2.1 Auth and users
    (
        "POST",
        "/api/v1/auth/login",
        "public",
        lambda s: {
            "json": {
                "username": "viewer@example.com",
                "password": "ValidPassword123!",
            }
        },
    ),
    ("GET", "/api/v1/auth/me", "viewer", lambda s: {}),
    (
        "POST",
        "/api/v1/auth/change-password",
        "viewer",
        lambda s: {
            "json": {"current_password": "ValidPassword123!", "new_password": "ValidPassword123!"}
        },
    ),
    ("GET", "/api/v1/users", "admin", lambda s: {}),
    (
        "POST",
        "/api/v1/users",
        "admin",
        lambda s: {
            "json": {
                "email": "new@bar.com",
                "full_name": "New User",
                "role": "viewer",
                "password": "newpassword123!",
            }
        },
    ),
    (
        "PATCH",
        "/api/v1/users/{user_id}",
        "admin",
        lambda s: {
            "path_params": {"user_id": s["viewer_id"]},
            "json": {"full_name": "Updated"},
        },
    ),
    # 2.2 Reference data and sales
    ("GET", "/api/v1/stores", "viewer", lambda s: {}),
    ("GET", "/api/v1/products", "viewer", lambda s: {}),
    (
        "GET",
        "/api/v1/products/{product_id}",
        "viewer",
        lambda s: {"path_params": {"product_id": s["product_id"]}},
    ),
    ("GET", "/api/v1/sales/daily", "viewer", lambda s: {"params": {"store_id": s["store_id"]}}),
    # 2.3 KPIs and forecasts
    ("GET", "/api/v1/kpis/overview", "viewer", lambda s: {}),
    (
        "GET",
        "/api/v1/forecasts",
        "viewer",
        lambda s: {"params": {"product_id": s["product_id"], "store_id": s["store_id"]}},
    ),
    ("GET", "/api/v1/forecasts/accuracy", "viewer", lambda s: {}),
    # 2.4 Customers
    ("GET", "/api/v1/segments", "viewer", lambda s: {}),
    ("GET", "/api/v1/customers", "viewer", lambda s: {}),
    (
        "GET",
        "/api/v1/customers/{customer_id}",
        "viewer",
        lambda s: {"path_params": {"customer_id": s["customer_id"]}},
    ),
    ("GET", "/api/v1/customers/export", "analyst", lambda s: {}),
    ("GET", "/api/v1/churn/summary", "viewer", lambda s: {}),
    # 2.5 Inventory
    ("GET", "/api/v1/inventory/summary", "viewer", lambda s: {}),
    ("GET", "/api/v1/inventory/recommendations", "viewer", lambda s: {}),
    ("GET", "/api/v1/inventory/recommendations/export", "analyst", lambda s: {}),
    (
        "GET",
        "/api/v1/inventory/params/{product_id}",
        "viewer",
        lambda s: {"path_params": {"product_id": s["product_id"]}},
    ),
    (
        "PUT",
        "/api/v1/inventory/params/{product_id}",
        "analyst",
        lambda s: {
            "path_params": {"product_id": s["product_id"]},
            "json": {"lead_time_days": 10.0, "service_level": 0.95},
        },
    ),
    (
        "POST",
        "/api/v1/inventory/what-if",
        "analyst",
        lambda s: {
            "json": {
                "product_id": s["product_id"],
                "store_id": s["store_id"],
                "overrides": {"lead_time_days": 10.0},
            }
        },
    ),
    (
        "POST",
        "/api/v1/inventory/simulate",
        "analyst",
        lambda s: {
            "json": {"scope": {"abc_class": "A", "limit": 10}, "policy": {"service_level": 0.95}}
        },
    ),
    ("GET", "/api/v1/inventory/simulations/latest", "viewer", lambda s: {}),
    # 2.6 Models, monitoring, alerts
    ("GET", "/api/v1/models", "viewer", lambda s: {}),
    (
        "GET",
        "/api/v1/models/{run_id}",
        "viewer",
        lambda s: {"path_params": {"run_id": s["model_run_id"]}},
    ),
    (
        "POST",
        "/api/v1/models/{task}/train",
        "admin",
        lambda s: {"path_params": {"task": "forecast"}, "json": {"algorithms": ["lightgbm"]}},
    ),
    (
        "POST",
        "/api/v1/models/{run_id}/promote",
        "admin",
        lambda s: {"path_params": {"run_id": s["model_run_id"]}},
    ),
    ("GET", "/api/v1/monitoring/drift", "viewer", lambda s: {}),
    (
        "GET",
        "/api/v1/monitoring/drift/{report_id}",
        "viewer",
        lambda s: {"path_params": {"report_id": s["drift_report_id"]}},
    ),
    (
        "GET",
        "/api/v1/monitoring/drift/{report_id}/html",
        "viewer",
        lambda s: {"path_params": {"report_id": s["drift_report_id"]}},
    ),
    ("GET", "/api/v1/monitoring/performance", "viewer", lambda s: {}),
    ("GET", "/api/v1/alerts", "viewer", lambda s: {}),
    (
        "POST",
        "/api/v1/alerts/{alert_id}/ack",
        "analyst",
        lambda s: {"path_params": {"alert_id": s["alert_id"]}},
    ),
    # 2.7 Data and pipelines
    (
        "POST",
        "/api/v1/uploads/sales",
        "analyst",
        lambda s: {
            "files": {
                "file": (
                    "s.csv",
                    io.BytesIO(
                        b"invoice_no,invoice_date,sku,quantity,unit_price\n1,2026-01-01,A,1,10\n"
                    ),
                    "text/csv",
                )
            },
            "data": {"source_name": "generic_csv"},
        },
    ),
    ("GET", "/api/v1/uploads", "analyst", lambda s: {}),
    (
        "GET",
        "/api/v1/uploads/{batch_id}",
        "analyst",
        lambda s: {"path_params": {"batch_id": s["batch_id"]}},
    ),
    ("GET", "/api/v1/pipelines/runs", "viewer", lambda s: {}),
    (
        "GET",
        "/api/v1/pipelines/runs/{run_id}",
        "viewer",
        lambda s: {"path_params": {"run_id": s["pipeline_run_id"]}},
    ),
    (
        "POST",
        "/api/v1/pipelines/{flow_name}/trigger",
        "analyst",
        lambda s: {
            "path_params": {"flow_name": "score_all"},
            "json": {"params": {}},
        },
    ),
]


def test_route_inventory_against_rbac_catalogue(api_client: dict[str, Any]):
    """Verify every route in FastAPI is explicitly listed in catalogue (SEC-AUTHZ-01)."""
    app = api_client["app"]

    catalogue_endpoints = {(method.upper(), path) for method, path, _, _ in RBAC_CATALOGUE}

    app_endpoints = set()
    for route in app.routes:
        if isinstance(route, APIRoute):
            # Skip documentation endpoints
            if route.path in ("/docs", "/redoc", "/openapi.json"):
                continue
            for m in route.methods:
                if m != "HEAD":
                    app_endpoints.add((m.upper(), route.path))

    # Assert that all app routes are covered by the catalogue
    unmapped_routes = app_endpoints - catalogue_endpoints
    assert not unmapped_routes, f"Found unmapped routes in app: {unmapped_routes}"


@pytest.mark.parametrize(
    ("method", "path_template", "min_role", "kwargs_builder"),
    RBAC_CATALOGUE,
)
@pytest.mark.parametrize("client_role", ["anonymous", "viewer", "analyst", "admin"])
def test_rbac_matrix_enforcement(  # noqa: PLR0917
    api_client: dict[str, Any],
    method: str,
    path_template: str,
    min_role: str,
    kwargs_builder: Any,
    client_role: str,
):
    """Parametrized RBAC test verifying authorization bounds for each route and role."""
    client = api_client["client"]
    headers = api_client["headers"][client_role]
    seeds = api_client["seeds"]

    # Build kwargs and substitute path params
    req_kwargs = kwargs_builder(seeds)
    path_params = req_kwargs.pop("path_params", {})
    resolved_path = path_template
    for k, v in path_params.items():
        resolved_path = resolved_path.replace(f"{{{k}}}", str(v))

    # Dispatch HTTP call
    call_kwargs = dict(req_kwargs)
    call_kwargs["headers"] = headers

    func = getattr(client, method.lower())
    response = func(resolved_path, **call_kwargs)

    role_levels = {"anonymous": 0, "viewer": 1, "analyst": 2, "admin": 3}
    min_role_level = 0 if min_role == "public" else role_levels[min_role]
    client_level = role_levels[client_role]

    if client_role == "anonymous":
        if min_role == "public":
            # Public endpoint allows anonymous access (not 401)
            assert response.status_code != 401
        else:
            # Protected endpoint requires authentication
            assert response.status_code == 401
            assert response.json()["type"] == "urn:retailpulse:problem:unauthorized"
    elif client_level < min_role_level:
        # Authenticated user has insufficient permissions
        assert response.status_code == 403
        assert response.json()["type"] == "urn:retailpulse:problem:forbidden"
    else:
        # User has sufficient role: must be authorized (never 401 or 403)
        assert response.status_code not in (401, 403)

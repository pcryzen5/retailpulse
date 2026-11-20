"""Unit tests for FastAPI app factory, middleware, RFC 9457 error handlers, and health endpoints.

Implements P5-01 verification:
- Settings and lifespan initialization.
- OpenAPI docs toggle based on environment (disabled in prod).
- Security headers (nosniff, DENY, no-referrer, Permissions-Policy, Cache-Control: no-store).
- Request ID generation, echoing, and sanitization.
- Timing header (X-Process-Time).
- Health probes (/health/live, /health/ready, /health/startup).
- RFC 9457 error response shape across domain exceptions and unhandled errors.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.testclient import TestClient

from retailpulse.api.app import create_app
from retailpulse.config import Settings
from retailpulse.errors import (
    AuthenticationError,
    ConflictError,
    DatabaseError,
    DomainValidationError,
    NotFoundError,
    PermissionDeniedError,
)

# Error router to trigger domain and unhandled exceptions
error_router = APIRouter(prefix="/test-errors")


class DummyInput(BaseModel):
    name: str = Field(min_length=3)
    age: int = Field(ge=18)


@error_router.post("/validation")
def trigger_validation(data: DummyInput) -> dict[str, str]:
    return {"status": "ok", "name": data.name}


@error_router.get("/not-found")
def trigger_not_found() -> None:
    raise NotFoundError("Entity SKU-999 not found")


@error_router.get("/conflict")
def trigger_conflict() -> None:
    raise ConflictError("Pipeline run already in progress")


@error_router.get("/domain-validation")
def trigger_domain_validation() -> None:
    raise DomainValidationError("Invalid horizon range: must be between 1 and 28")


@error_router.get("/auth-error")
def trigger_auth_error() -> None:
    raise AuthenticationError("Raw internal credential leak attempt")


@error_router.get("/permission-denied")
def trigger_permission_denied() -> None:
    raise PermissionDeniedError("Admin role required for this action")


@error_router.get("/database-error")
def trigger_database_error() -> None:
    raise DatabaseError("FATAL: connection to server at '127.0.0.1', port 5432 failed")


@error_router.get("/unhandled-error")
def trigger_unhandled() -> None:
    raise RuntimeError("Unexpected zero division or crash with /secret/path/to/key")


@pytest.fixture
def test_app():
    """Create test FastAPI application with error triggering routes."""
    app = create_app()
    app.include_router(error_router)
    return app


@pytest.fixture
def client(test_app):
    """Synchronous test client."""
    return TestClient(test_app, raise_server_exceptions=False)


# --- 1. App Factory & Environment Config Tests ---


def test_app_factory_creates_app() -> None:
    """Verify application initializes with expected metadata and state."""
    app = create_app()
    assert app.title == "RetailPulse API"
    assert app.version == "0.1.0"
    assert isinstance(app.state.settings, Settings)


def test_docs_enabled_in_dev() -> None:
    """Verify Swagger and OpenAPI endpoints are active in dev environment."""
    dev_settings = Settings(env="dev", enable_docs=None)
    app = create_app(settings=dev_settings)
    assert app.docs_url == "/docs"
    assert app.openapi_url == "/openapi.json"
    assert app.redoc_url == "/redoc"


def test_docs_disabled_in_prod() -> None:
    """Verify Swagger and OpenAPI endpoints are disabled in prod environment."""
    prod_settings = Settings(
        env="prod",
        jwt_secret="prod-super-secure-jwt-secret-min-32-chars-long!",
        pseudonym_salt="prod-super-secure-salt-min-32-chars-long!",
        enable_docs=None,
    )
    app = create_app(settings=prod_settings)
    assert app.docs_url is None
    assert app.openapi_url is None
    assert app.redoc_url is None


def test_cors_configuration() -> None:
    """Verify CORS origins are set when configured."""
    cors_settings = Settings(cors_origins="https://dashboard.example.com,https://api.example.com")
    app = create_app(settings=cors_settings)
    client = TestClient(app)
    response = client.get(
        "/health/live",
        headers={"Origin": "https://dashboard.example.com"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://dashboard.example.com"


# --- 2. Middleware Tests (Security Headers, Timing, Request ID) ---


def test_security_headers_present(client: TestClient) -> None:
    """Verify SEC-WEB-01 mandatory hardening headers on responses."""
    resp = client.get("/health/live")
    assert resp.status_code == 200

    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert "camera=()" in resp.headers["Permissions-Policy"]
    assert resp.headers["Cache-Control"] == "no-store"


def test_timing_header_present(client: TestClient) -> None:
    """Verify X-Process-Time is calculated and returned."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert "X-Process-Time" in resp.headers
    # E.g. "0.45ms"
    assert re.match(r"^\d+(\.\d+)?ms$", resp.headers["X-Process-Time"])


def test_request_id_generated_when_missing(client: TestClient) -> None:
    """Verify X-Request-ID is generated and returned when not provided."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    req_id = resp.headers.get("X-Request-ID")
    assert req_id is not None
    assert len(req_id) >= 16


def test_request_id_echoes_safe_client_header(client: TestClient) -> None:
    """Verify safe client X-Request-ID is echoed back."""
    custom_id = "custom-test-req-12345"
    resp = client.get("/health/live", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == custom_id


def test_request_id_replaces_unsafe_client_header(client: TestClient) -> None:
    """Verify unsafe X-Request-ID is rejected and replaced with clean UUID."""
    unsafe_id = "<script>alert(1)</script>"
    resp = client.get("/health/live", headers={"X-Request-ID": unsafe_id})
    assert resp.status_code == 200
    req_id = resp.headers["X-Request-ID"]
    assert req_id != unsafe_id
    assert "<script>" not in req_id


# --- 3. Health Probes Tests (/health/*) ---


def test_health_live(client: TestClient) -> None:
    """Verify /health/live returns 200 OK."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_startup(client: TestClient) -> None:
    """Verify /health/startup returns 200 OK."""
    resp = client.get("/health/startup")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_root_alias(client: TestClient) -> None:
    """Verify /health root returns 200 OK."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("retailpulse.api.routers.health.check_database")
@patch("retailpulse.api.routers.health._check_service_reachability")
def test_health_ready_success(
    mock_service_check: MagicMock,
    mock_check_db: MagicMock,
    client: TestClient,
) -> None:
    """Verify /health/ready returns 200 when DB is connected and migrations current."""
    mock_check_db.return_value = {
        "connected": True,
        "is_up_to_date": True,
        "current_revision": "0006_rev",
        "head_revision": "0006_rev",
        "error": None,
    }
    mock_service_check.return_value = "ready"

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["database"]["connected"] is True
    assert data["database"]["migrations"] == "current"
    assert data["dependencies"]["mlflow"] == "ready"
    assert data["dependencies"]["prefect"] == "ready"


@patch("retailpulse.api.routers.health.check_database")
@patch("retailpulse.api.routers.health._check_service_reachability")
def test_health_ready_degraded_dependencies_still_200(
    mock_service_check: MagicMock,
    mock_check_db: MagicMock,
    client: TestClient,
) -> None:
    """Verify MLflow and Prefect degradation does not fail readiness probe (API §2.8)."""
    mock_check_db.return_value = {
        "connected": True,
        "is_up_to_date": True,
        "current_revision": "0006_rev",
        "head_revision": "0006_rev",
        "error": None,
    }
    mock_service_check.return_value = "degraded"

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["dependencies"]["mlflow"] == "degraded"
    assert data["dependencies"]["prefect"] == "degraded"


@patch("retailpulse.api.routers.health.check_database")
@patch("retailpulse.api.routers.health._check_service_reachability")
def test_health_ready_db_down_returns_503_problem(
    mock_service_check: MagicMock,
    mock_check_db: MagicMock,
    client: TestClient,
) -> None:
    """Verify /health/ready returns RFC 9457 503 when DB is unreachable."""
    mock_service_check.return_value = "degraded"
    mock_check_db.return_value = {
        "connected": False,
        "is_up_to_date": False,
        "current_revision": None,
        "head_revision": "0006_rev",
        "error": "Connection refused",
    }

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.headers["Content-Type"] == "application/problem+json"

    problem = resp.json()
    assert problem["type"] == "urn:retailpulse:problem:model_unavailable"
    assert problem["status"] == 503
    assert problem["title"] == "Service Not Ready"
    assert "request_id" in problem
    assert any(err["field"] == "database.connected" for err in problem["errors"])


@patch("retailpulse.api.routers.health.check_database")
@patch("retailpulse.api.routers.health._check_service_reachability")
def test_health_ready_migrations_outdated_returns_503_problem(
    mock_service_check: MagicMock,
    mock_check_db: MagicMock,
    client: TestClient,
) -> None:
    """Verify /health/ready returns RFC 9457 503 when migrations are not at head."""
    mock_service_check.return_value = "degraded"
    mock_check_db.return_value = {
        "connected": True,
        "is_up_to_date": False,
        "current_revision": "0004_rev",
        "head_revision": "0006_rev",
        "error": None,
    }

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.headers["Content-Type"] == "application/problem+json"

    problem = resp.json()
    assert problem["type"] == "urn:retailpulse:problem:model_unavailable"
    assert problem["status"] == 503
    assert any(err["field"] == "database.migrations" for err in problem["errors"])


# --- 4. RFC 9457 Error Response Formats ---


def test_error_not_found(client: TestClient) -> None:
    """Verify NotFoundError maps to 404 Problem Details."""
    resp = client.get("/test-errors/not-found")
    assert resp.status_code == 404
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:not_found"
    assert body["title"] == "Not Found"
    assert body["status"] == 404
    assert body["detail"] == "Entity SKU-999 not found"
    assert body["instance"] == "/test-errors/not-found"
    assert "request_id" in body


def test_error_conflict(client: TestClient) -> None:
    """Verify ConflictError maps to 409 Problem Details."""
    resp = client.get("/test-errors/conflict")
    assert resp.status_code == 409
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:conflict"
    assert body["title"] == "Conflict"
    assert body["status"] == 409
    assert body["detail"] == "Pipeline run already in progress"


def test_error_domain_validation(client: TestClient) -> None:
    """Verify DomainValidationError maps to 422 Problem Details."""
    resp = client.get("/test-errors/domain-validation")
    assert resp.status_code == 422
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:validation_error"
    assert body["title"] == "Validation Error"
    assert body["status"] == 422
    assert "horizon" in body["detail"]


def test_error_authentication_sanitized(client: TestClient) -> None:
    """Verify AuthenticationError returns generic message without leaking details."""
    resp = client.get("/test-errors/auth-error")
    assert resp.status_code == 401
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:unauthorized"
    assert body["title"] == "Unauthorized"
    assert body["status"] == 401
    # Check generic detail per API §1.1
    assert body["detail"] == "Invalid credentials or authentication token"
    assert "credential leak" not in body["detail"]


def test_error_permission_denied(client: TestClient) -> None:
    """Verify PermissionDeniedError maps to 403 Problem Details."""
    resp = client.get("/test-errors/permission-denied")
    assert resp.status_code == 403
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:forbidden"
    assert body["title"] == "Forbidden"
    assert body["status"] == 403


def test_error_database_sanitized(client: TestClient) -> None:
    """Verify DatabaseError returns 503 and hides raw SQL/connection strings (SEC-WEB-04)."""
    resp = client.get("/test-errors/database-error")
    assert resp.status_code == 503
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:model_unavailable"
    assert body["title"] == "Service Unavailable"
    assert body["status"] == 503
    assert body["detail"] == "Database service is temporarily unavailable"
    # Never leak raw IP, port, or SQL syntax
    assert "127.0.0.1" not in body["detail"]
    assert "5432" not in body["detail"]


def test_error_unhandled_sanitized(client: TestClient) -> None:
    """Verify unhandled Exception returns 500 without leaking stack traces (SEC-WEB-04)."""
    resp = client.get("/test-errors/unhandled-error")
    assert resp.status_code == 500
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:internal_error"
    assert body["title"] == "Internal Server Error"
    assert body["status"] == 500
    assert body["detail"] == "An unexpected internal server error occurred."
    # Never leak stack trace, variable names, or paths
    assert "secret" not in str(body)
    assert "zero division" not in str(body)


def test_error_http_not_found_on_route(client: TestClient) -> None:
    """Verify 404 on nonexistent endpoint returns RFC 9457 format."""
    resp = client.get("/nonexistent/endpoint/path")
    assert resp.status_code == 404
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:not_found"
    assert body["status"] == 404
    assert body["instance"] == "/nonexistent/endpoint/path"
    assert "request_id" in body


def test_error_request_validation_field_errors(client: TestClient) -> None:
    """Verify Pydantic input validation failure returns 422 with granular field errors."""
    resp = client.post("/test-errors/validation", json={"name": "a", "age": 15})
    assert resp.status_code == 422
    assert resp.headers["Content-Type"] == "application/problem+json"

    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:validation_error"
    assert body["title"] == "Validation Error"
    assert body["status"] == 422
    assert "errors" in body
    fields = [err["field"] for err in body["errors"]]
    assert "name" in fields
    assert "age" in fields

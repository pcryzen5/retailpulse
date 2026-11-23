"""Unit tests for rate limiting, Prometheus metrics, OpenAPI tags, and docs toggle.

Implements P5-09 verification:
- 429 response on rate-limited endpoints with RFC 9457 problem details and Retry-After header.
- GET /metrics returns Prometheus format exposition containing HTTP request counters.
- OpenAPI schema contains configured tags metadata and API descriptions.
- RP_ENABLE_DOCS and environment behavior:
  - In prod with enable_docs=None: /docs and /openapi.json disabled (404).
  - In prod with enable_docs=True: /docs and /openapi.json enabled (200).
  - In dev: /docs and /openapi.json enabled (200).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import sqlalchemy as sa
from pydantic import SecretStr
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from retailpulse.api.app import create_app
from retailpulse.config import Settings
from retailpulse.db.base import Base
from retailpulse.db.models.app import AppUser
from retailpulse.db.session import get_db
from retailpulse.security.password import hash_password


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def ops_db_engine():
    """In-memory SQLite engine with translated schemas."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    engine = engine.execution_options(
        schema_translate_map={"app": None, "core": None, "ops": None, "ml": None}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def ops_db_session(ops_db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=ops_db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_user(ops_db_session: Session) -> dict[str, Any]:
    """Seed user for authentication checks."""
    user = AppUser(
        user_id=uuid.uuid4(),
        email="ratelimit_user@example.com",
        full_name="Rate Limit User",
        role="viewer",
        password_hash=hash_password("ValidPassword123!"),
        is_active=True,
    )
    ops_db_session.add(user)
    ops_db_session.commit()
    return {"user_id": user.user_id, "email": user.email}


# --- Rate Limiting Tests ---


def test_rate_limit_login_exceeded_429(ops_db_session: Session, seeded_user: dict[str, Any]):
    """Verify that exceeding login rate limit (5/min per IP) returns 429 with Retry-After header."""
    settings = Settings(env="test")
    app = create_app(settings=settings)
    app.dependency_overrides[get_db] = lambda: ops_db_session
    if hasattr(app.state, "limiter"):
        app.state.limiter.enabled = True
        app.state.limiter.reset()
    client = TestClient(app)

    # Login rate limit is 5 per minute per IP
    responses = []
    for _ in range(6):
        res = client.post(
            "/api/v1/auth/login",
            json={"username": seeded_user["email"], "password": "WrongPassword123!"},
        )
        responses.append(res)

    # First 5 attempts fail authentication (401)
    for r in responses[:5]:
        assert r.status_code == 401

    # 6th attempt hits rate limit (429)
    rate_limited_res = responses[5]
    assert rate_limited_res.status_code == 429
    assert "Retry-After" in rate_limited_res.headers
    body = rate_limited_res.json()
    assert body["type"] == "urn:retailpulse:problem:rate_limited"
    assert body["title"] == "Rate Limited"
    assert "Rate limit exceeded" in body["detail"]


# --- Prometheus Metrics Tests ---


def test_metrics_endpoint_returns_prometheus_data(ops_db_session: Session):
    """Verify GET /metrics exposes Prometheus text format and counts requests."""
    settings = Settings(env="test")
    app = create_app(settings=settings)
    app.dependency_overrides[get_db] = lambda: ops_db_session
    client = TestClient(app)

    # Trigger health request to generate traffic
    client.get("/health/live")

    # Fetch metrics
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert "text/plain" in metrics_res.headers["content-type"]
    metrics_text = metrics_res.text

    # Verify standard metrics and labels are present
    assert "http_requests_total" in metrics_text
    assert "http_request_duration_seconds" in metrics_text
    assert 'handler="/health/live"' in metrics_text


# --- OpenAPI Metadata and Tags Tests ---


def test_openapi_schema_tags_and_metadata():
    """Verify OpenAPI schema includes required tags and API descriptions."""
    settings = Settings(env="test")
    app = create_app(settings=settings)
    client = TestClient(app)

    res = client.get("/openapi.json")
    assert res.status_code == 200
    schema = res.json()

    assert schema["info"]["title"] == "RetailPulse API"
    assert "tags" in schema
    tag_names = {t["name"] for t in schema["tags"]}
    expected_tags = {
        "Auth",
        "Users",
        "Reference",
        "KPIs",
        "Forecasts",
        "Customers",
        "Inventory",
        "Models",
        "Monitoring",
        "Alerts",
        "Uploads",
        "Pipelines",
        "Health",
    }
    assert expected_tags.issubset(tag_names)


# --- Docs Toggle and Production Environment Tests ---


def test_docs_disabled_in_prod_config():
    """Verify /docs and /openapi.json are disabled (404) in prod by default (SEC-WEB-05)."""
    prod_secret = SecretStr("a" * 48)
    settings = Settings(
        env="prod",
        jwt_secret=prod_secret,
        pseudonym_salt=prod_secret,
        enable_docs=None,  # default in prod
    )
    assert not settings.docs_enabled

    app = create_app(settings=settings)
    client = TestClient(app)

    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_explicitly_enabled_in_prod():
    """Verify RP_ENABLE_DOCS=true overrides default prod behavior."""
    prod_secret = SecretStr("a" * 48)
    settings = Settings(
        env="prod",
        jwt_secret=prod_secret,
        pseudonym_salt=prod_secret,
        enable_docs=True,  # explicitly enabled
    )
    assert settings.docs_enabled

    app = create_app(settings=settings)
    client = TestClient(app)

    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_docs_enabled_in_dev_by_default():
    """Verify /docs and /openapi.json are enabled (200) in dev by default."""
    settings = Settings(env="dev")
    assert settings.docs_enabled

    app = create_app(settings=settings)
    client = TestClient(app)

    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200

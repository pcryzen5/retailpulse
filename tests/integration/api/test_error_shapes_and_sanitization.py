"""Integration tests for RFC 9457 Problem Details formatting and error sanitization.

Implements P5-10, API §1.1, and SECURITY §7 (SEC-WEB-04):
- application/problem+json content-type on all error responses.
- Structure: type, title, status, detail, instance, request_id.
- X-Request-ID header echoes the request_id in the JSON body.
- Sanitization guarantees: zero stack traces, raw SQL queries, internal filesystem paths,
  or secrets leaked to client responses.
"""

from __future__ import annotations

import io
import re
from typing import Any

from starlette.testclient import TestClient

from retailpulse.api.app import create_app
from retailpulse.config import Settings

FORBIDDEN_LEAK_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r'File ".*\.py", line \d+', re.IGNORECASE),
    re.compile(r"SELECT .* FROM", re.IGNORECASE),
    re.compile(r"INSERT INTO .*", re.IGNORECASE),
    re.compile(r"postgresql(\+\w+)?://", re.IGNORECASE),
    re.compile(r"\$argon2id\$", re.IGNORECASE),
    re.compile(r"password_hash", re.IGNORECASE),
]


def _assert_valid_problem_details(
    response, expected_status: int, expected_type_suffix: str
) -> dict[str, Any]:
    """Assert response conforms to RFC 9457 Problem Details and sanitization requirements."""
    assert response.status_code == expected_status
    assert "application/problem+json" in response.headers.get("content-type", "")

    body = response.json()
    assert body["status"] == expected_status
    assert body["type"] == f"urn:retailpulse:problem:{expected_type_suffix}"
    assert "title" in body
    assert "detail" in body
    assert "instance" in body
    assert "request_id" in body

    # Header X-Request-ID must match body request_id
    assert response.headers.get("X-Request-ID") == body["request_id"]

    # SEC-WEB-04: Verify zero secret/internal leaks in raw response text
    raw_text = response.text
    for pattern in FORBIDDEN_LEAK_PATTERNS:
        assert not pattern.search(raw_text), (
            f"Found leaked pattern {pattern.pattern} in response: {raw_text}"
        )

    return body


def test_error_shape_401_unauthorized(api_client: dict[str, Any]):
    """Verify 401 Unauthorized returns valid problem details without credentials leakage."""
    client = api_client["client"]
    res = client.get("/api/v1/auth/me", headers={})
    _assert_valid_problem_details(res, 401, "unauthorized")


def test_error_shape_403_forbidden(api_client: dict[str, Any]):
    """Verify 403 Forbidden returns valid problem details."""
    client = api_client["client"]
    headers = api_client["headers"]["viewer"]
    res = client.get("/api/v1/users", headers=headers)
    _assert_valid_problem_details(res, 403, "forbidden")


def test_error_shape_404_not_found(api_client: dict[str, Any]):
    """Verify 404 Not Found returns valid problem details."""
    client = api_client["client"]
    headers = api_client["headers"]["viewer"]
    res = client.get("/api/v1/products/999999", headers=headers)
    _assert_valid_problem_details(res, 404, "not_found")


def test_error_shape_415_unsupported_media_type(api_client: dict[str, Any]):
    """Verify 415 Unsupported Media Type on invalid upload extension."""
    client = api_client["client"]
    headers = api_client["headers"]["analyst"]
    files = {"file": ("malicious.exe", io.BytesIO(b"MZfakebinary"), "application/x-dosexec")}
    res = client.post(
        "/api/v1/uploads/sales",
        headers=headers,
        files=files,
        data={"source_name": "generic_csv"},
    )
    _assert_valid_problem_details(res, 415, "unsupported_media_type")


def test_error_shape_422_validation_error(api_client: dict[str, Any]):
    """Verify 422 Validation Error returns field errors if present."""
    client = api_client["client"]
    headers = api_client["headers"]["viewer"]
    res = client.get("/api/v1/products?page_size=500", headers=headers)
    body = _assert_valid_problem_details(res, 422, "validation_error")
    assert "page_size" in body["detail"] or "errors" in body


def test_error_shape_500_sanitized_internal_error():
    """Verify unhandled 500 error sanitizes tracebacks and internal details completely."""
    app = create_app(settings=Settings(env="test"))

    @app.get("/trigger-crash", include_in_schema=False)
    def trigger_crash():
        raise RuntimeError(
            "Internal DB connection failed: postgresql://admin:secretpass@10.0.0.1:5432/db"
        )

    client = TestClient(app, raise_server_exceptions=False)
    res = client.get("/trigger-crash")
    body = _assert_valid_problem_details(res, 500, "internal_error")

    # Verify secret error message is NOT in detail
    assert "secretpass" not in res.text
    assert "postgresql://" not in res.text
    assert body["detail"] == "An unexpected internal server error occurred."

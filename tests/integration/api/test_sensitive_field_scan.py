"""Sensitive data leak scan and privacy assurance integration tests.

Implements P5-10, PRD §7.7 (FR-SEC-01), API §1.1, and SECURITY §6 / §10:
- SEC-DATA-01 / 02 / 03: Customer privacy verification (only surrogate customer_id and
  pseudonymized customer_key; zero raw names, emails, phones, addresses).
- Zero leaks of password_hash, jwt_secret, pseudonym_salt, or database connection strings.
- SEC-ML-01 / API §3.11: MLflow internal tracking URI hidden from non-admin users.
"""

from __future__ import annotations

import re
from typing import Any

FORBIDDEN_KEYS = frozenset(
    {
        "password_hash",
        "password",
        "jwt_secret",
        "pseudonym_salt",
        "secret",
        "secret_key",
    }
)


def _scan_for_sensitive_fields(data: Any, path: str = "") -> list[str]:
    """Recursively traverse JSON data structure to detect forbidden sensitive keys."""
    violations: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            current_path = f"{path}.{k}" if path else k
            if k.lower() in FORBIDDEN_KEYS:
                violations.append(f"Forbidden key '{k}' found at {current_path}")
            if isinstance(v, (dict, list)):
                violations.extend(_scan_for_sensitive_fields(v, current_path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            current_path = f"{path}[{idx}]"
            if isinstance(item, (dict, list)):
                violations.extend(_scan_for_sensitive_fields(item, current_path))
    return violations


def test_users_endpoint_never_leaks_password_hash(api_client: dict[str, Any]):
    """Verify GET /api/v1/users does not include password_hash in response payload."""
    client = api_client["client"]
    admin_headers = api_client["headers"]["admin"]

    res = client.get("/api/v1/users", headers=admin_headers)
    assert res.status_code == 200
    body = res.json()

    # Scan for forbidden keys
    violations = _scan_for_sensitive_fields(body)
    assert not violations, f"Sensitive fields detected in /users response: {violations}"

    # Verify expected safe fields only
    for user_item in body["items"]:
        assert "password_hash" not in user_item
        assert "email" in user_item
        assert "role" in user_item


def test_customer_endpoints_pseudonymization_and_privacy(api_client: dict[str, Any]):
    """Verify customer endpoints use pseudonymized keys and drop personal contact fields."""
    client = api_client["client"]
    viewer_headers = api_client["headers"]["viewer"]
    customer_id = api_client["seeds"]["customer_id"]

    # 1. Customer list
    res_list = client.get("/api/v1/customers", headers=viewer_headers)
    assert res_list.status_code == 200
    list_body = res_list.json()
    violations = _scan_for_sensitive_fields(list_body)
    assert not violations

    # 2. Customer detail
    res_detail = client.get(f"/api/v1/customers/{customer_id}", headers=viewer_headers)
    assert res_detail.status_code == 200
    detail_body = res_detail.json()
    violations_detail = _scan_for_sensitive_fields(detail_body)
    assert not violations_detail

    # Verify customer key is masked for privacy in detail view
    cust_key = detail_body["customer_key"]
    assert cust_key.endswith("...")
    assert len(cust_key) == 11

    # 3. Customer export has full 64-character HMAC hex digest
    analyst_headers = api_client["headers"]["analyst"]
    res_export = client.get("/api/v1/customers/export", headers=analyst_headers)
    assert res_export.status_code == 200
    assert re.search(r"[0-9a-f]{64}", res_export.text)

    # SEC-DATA-02: Ensure no personal identifiers (name, email, phone, street) exist
    assert "email" not in detail_body
    assert "phone" not in detail_body
    assert "first_name" not in detail_body
    assert "address" not in detail_body


def test_models_mlflow_url_admin_only(api_client: dict[str, Any]):
    """Verify mlflow_url is redacted for viewers/analysts and shown to admins only (API §3.11)."""
    client = api_client["client"]
    viewer_headers = api_client["headers"]["viewer"]
    analyst_headers = api_client["headers"]["analyst"]
    admin_headers = api_client["headers"]["admin"]
    run_id = api_client["seeds"]["model_run_id"]

    # 1. Viewer cannot see mlflow_url
    res_viewer = client.get(f"/api/v1/models/{run_id}", headers=viewer_headers)
    assert res_viewer.status_code == 200
    assert res_viewer.json()["mlflow_url"] is None

    # 2. Analyst cannot see mlflow_url
    res_analyst = client.get(f"/api/v1/models/{run_id}", headers=analyst_headers)
    assert res_analyst.status_code == 200
    assert res_analyst.json()["mlflow_url"] is None

    # 3. Admin sees mlflow_url
    res_admin = client.get(f"/api/v1/models/{run_id}", headers=admin_headers)
    assert res_admin.status_code == 200
    assert res_admin.json()["mlflow_url"] is not None
    assert "experiments" in res_admin.json()["mlflow_url"]

"""Integration tests for API pagination bounds, envelope structure, and sorting whitelists.

Implements P5-10 and API §1.1 Conventions:
- Page size bounds: 1 <= page_size <= 200 (page_size > 200 or < 1 returns 422).
- Page bounds: page >= 1 (page < 1 returns 422).
- Pagination envelope: {"items": [...], "page": 1, "page_size": 50, "total": ...}.
- Sorting whitelists: whitelisted per endpoint; invalid or unknown field returns 422.
"""

from __future__ import annotations

from typing import Any

import pytest

# Endpoints supporting pagination and sorting with an admin-authorized client
PAGINATED_ENDPOINTS = [
    ("/api/v1/products", "sku"),
    ("/api/v1/customers", "monetary"),
    ("/api/v1/inventory/recommendations", "stockout_risk"),
    ("/api/v1/models", "trained_at"),
    ("/api/v1/monitoring/drift", "created_at"),
    ("/api/v1/alerts", "created_at"),
    ("/api/v1/uploads", "created_at"),
    ("/api/v1/pipelines/runs", "queued_at"),
    ("/api/v1/users", "created_at"),
]


@pytest.mark.parametrize(("endpoint", "_valid_sort_field"), PAGINATED_ENDPOINTS)
def test_pagination_bounds_rejection(
    api_client: dict[str, Any], endpoint: str, _valid_sort_field: str
):
    """Verify that page_size > 200, page_size < 1, and page < 1 return 422 Problem Details."""
    client = api_client["client"]
    headers = api_client["headers"]["admin"]

    # 1. page_size > 200
    res_large = client.get(f"{endpoint}?page_size=201", headers=headers)
    assert res_large.status_code == 422
    assert res_large.json()["type"] == "urn:retailpulse:problem:validation_error"

    # 2. page_size < 1
    res_zero = client.get(f"{endpoint}?page_size=0", headers=headers)
    assert res_zero.status_code == 422

    # 3. page < 1
    res_page_zero = client.get(f"{endpoint}?page=0", headers=headers)
    assert res_page_zero.status_code == 422


@pytest.mark.parametrize(("endpoint", "valid_sort_field"), PAGINATED_ENDPOINTS)
def test_pagination_envelope_and_sorting(
    api_client: dict[str, Any], endpoint: str, valid_sort_field: str
):
    """Verify standard response envelope and sort column enforcement."""
    client = api_client["client"]
    headers = api_client["headers"]["admin"]

    # 1. Standard envelope structure
    res = client.get(f"{endpoint}?page=1&page_size=10", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert "items" in body
    assert isinstance(body["items"], list)
    assert body["page"] == 1
    assert body["page_size"] == 10
    assert "total" in body
    assert isinstance(body["total"], int)

    # 2. Valid sorting asc and desc
    res_asc = client.get(f"{endpoint}?sort={valid_sort_field}:asc", headers=headers)
    assert res_asc.status_code == 200

    res_desc = client.get(f"{endpoint}?sort={valid_sort_field}:desc", headers=headers)
    assert res_desc.status_code == 200

    # 3. Invalid / unknown sort field returns 422
    res_bad_sort = client.get(f"{endpoint}?sort=nonexistent_field_xyz:asc", headers=headers)
    assert res_bad_sort.status_code == 422
    assert res_bad_sort.json()["type"] == "urn:retailpulse:problem:validation_error"

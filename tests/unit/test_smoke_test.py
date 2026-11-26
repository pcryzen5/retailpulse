"""Unit tests for the deployment smoke test script (scripts/smoke_test.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.smoke_test import run_smoke_test


def test_smoke_test_success_flow() -> None:
    """Verify that smoke test succeeds when readiness, login, and read endpoints respond 200."""
    with patch("scripts.smoke_test.requests.get") as mock_get, patch(
        "scripts.smoke_test.requests.post"
    ) as mock_post:
        # Mock /health/ready
        ready_resp = MagicMock(status_code=200)
        ready_resp.json.return_value = {"status": "ok", "db": "connected"}

        # Mock /api/v1/auth/me
        me_resp = MagicMock(status_code=200)
        me_resp.json.return_value = {"email": "admin@retailpulse.local", "role": "admin"}

        # Mock /api/v1/kpis/overview
        overview_resp = MagicMock(status_code=200)
        overview_resp.json.return_value = {"total_sales": 10000}

        mock_get.side_effect = [ready_resp, me_resp, overview_resp]

        # Mock /api/v1/auth/login
        login_resp = MagicMock(status_code=200)
        login_resp.json.return_value = {
            "access_token": "fake_jwt_token_abc123",
            "token_type": "bearer",
        }
        mock_post.return_value = login_resp

        result = run_smoke_test(
            base_url="http://localhost:8000",
            email="admin@retailpulse.local",
            password="test_password",
            max_retries=2,
            retry_delay_seconds=0.01,
        )

        assert result is True
        assert mock_get.call_count == 3
        mock_post.assert_called_once()


def test_smoke_test_fails_when_ready_probe_fails() -> None:
    """Verify that smoke test fails if /health/ready does not return 200 within retries."""
    with patch("scripts.smoke_test.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=503)

        result = run_smoke_test(
            base_url="http://localhost:8000",
            email="admin@retailpulse.local",
            password="test_password",
            max_retries=2,
            retry_delay_seconds=0.01,
        )

        assert result is False


def test_smoke_test_fails_when_login_rejected() -> None:
    """Verify that smoke test fails if credentials are rejected."""
    with patch("scripts.smoke_test.requests.get") as mock_get, patch(
        "scripts.smoke_test.requests.post"
    ) as mock_post:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_post.return_value = MagicMock(status_code=401, text="Unauthorized")

        result = run_smoke_test(
            base_url="http://localhost:8000",
            email="admin@retailpulse.local",
            password="wrong_password",
            max_retries=1,
            retry_delay_seconds=0.01,
        )

        assert result is False

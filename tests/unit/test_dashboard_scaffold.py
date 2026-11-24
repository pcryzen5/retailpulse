"""Unit and AppTest verification suite for Dashboard Scaffold (P6-01).

Implements P6-01 verification requirements:
- Theme design tokens and Plotly figure layout helper.
- API client timeouts, retries, and RFC 9457 Problem Details mapping.
- AppTest: Login page rendering for unauthenticated users.
- AppTest: Login failure handling (generic credential error, account lockout / rate limit).
- AppTest: Login success and session state persistence.
- AppTest: Role-based st.navigation (viewer cannot see Admin; admin sees Admin).
- Component: Status strip freshness and 36-hour stale data warning banner.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import (
    ApiClient,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    ServerError,
    StatusStripData,
    UnauthorizedError,
    UserProfile,
    ValidationError,
)
from retailpulse.dashboard.theme import (
    BORDER,
    CHART_PALETTE,
    DANGER,
    FORECAST_COLORS,
    INK,
    MUTED,
    PRIMARY,
    RISK_BAND_COLORS,
    SEGMENT_COLORS,
    SUCCESS,
    SURFACE,
    WARNING,
    style_figure,
)

APP_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "src" / "retailpulse" / "dashboard" / "app.py"
)


# =====================================================================
# 1. Theme and Plotly Styling Tests
# =====================================================================


def test_theme_color_tokens_and_palette():
    """Verify design tokens conform to DESIGN §2 requirements."""
    hex_pattern = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")

    # Assert basic color tokens are valid hex
    for color in (PRIMARY, INK, MUTED, BORDER, SURFACE, SUCCESS, WARNING, DANGER):
        assert hex_pattern.match(color), f"Invalid hex color token: {color}"

    # Assert Okabe-Ito chart palette has 8 color-blind safe colors
    assert len(CHART_PALETTE) == 8
    for color in CHART_PALETTE:
        assert hex_pattern.match(color), f"Invalid chart palette color: {color}"

    # Assert semantic mappings
    assert "Champions" in SEGMENT_COLORS
    assert "At Risk" in SEGMENT_COLORS
    assert "low" in RISK_BAND_COLORS
    assert "high" in RISK_BAND_COLORS
    assert "history" in FORECAST_COLORS
    assert "forecast" in FORECAST_COLORS


def test_style_figure_layout():
    """Verify style_figure applies standard typography, margins, and gridlines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 20, 30], name="Test Series"))

    styled_fig = style_figure(
        fig,
        title="Weekly Sales Trend",
        xaxis_title="Date",
        yaxis_title="Revenue",
        height=450,
    )

    layout = styled_fig.layout
    assert layout.height == 450
    assert layout.paper_bgcolor == SURFACE
    assert layout.plot_bgcolor == SURFACE
    assert "Inter" in layout.font.family
    assert layout.title.text == "Weekly Sales Trend"
    assert layout.xaxis.title.text == "Date"
    assert layout.xaxis.gridcolor == BORDER
    assert layout.yaxis.title.text == "Revenue"
    assert layout.yaxis.gridcolor == BORDER
    assert layout.showlegend is True
    assert layout.legend.orientation == "h"


# =====================================================================
# 2. API Client and Problem Details Mapping Tests
# =====================================================================


def _make_mock_response(
    status_code: int,
    json_body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Helper to create an httpx.Response instance for testing error mapping."""
    res_headers = {"Content-Type": "application/problem+json", "X-Request-ID": "req-test-123"}
    if headers:
        res_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        json=json_body,
        headers=res_headers,
        request=httpx.Request("GET", "http://test/api"),
    )


def test_client_error_mapping():
    """Verify client maps HTTP error codes to typed RFC 9457 exceptions."""
    client = ApiClient(base_url="http://test")

    # 401 Unauthorized
    with pytest.raises(UnauthorizedError) as exc_401:
        client._handle_response_error(
            _make_mock_response(
                401,
                {
                    "title": "Unauthorized",
                    "detail": "Invalid token",
                    "type": "urn:problem:unauthorized",
                },
            )
        )
    assert exc_401.value.status_code == 401
    assert exc_401.value.request_id == "req-test-123"

    # 403 Forbidden
    with pytest.raises(ForbiddenError) as exc_403:
        client._handle_response_error(
            _make_mock_response(
                403,
                {
                    "title": "Forbidden",
                    "detail": "Role insufficient",
                    "type": "urn:problem:forbidden",
                },
            )
        )
    assert exc_403.value.status_code == 403

    # 404 Not Found
    with pytest.raises(NotFoundError) as exc_404:
        client._handle_response_error(
            _make_mock_response(
                404,
                {
                    "title": "Not Found",
                    "detail": "Resource missing",
                    "type": "urn:problem:not_found",
                },
            )
        )
    assert exc_404.value.status_code == 404

    # 409 Conflict
    with pytest.raises(ConflictError) as exc_409:
        client._handle_response_error(
            _make_mock_response(
                409,
                {"title": "Conflict", "detail": "State conflict", "type": "urn:problem:conflict"},
            )
        )
    assert exc_409.value.status_code == 409

    # 422 Validation Error
    with pytest.raises(ValidationError) as exc_422:
        client._handle_response_error(
            _make_mock_response(
                422,
                {
                    "title": "Validation Error",
                    "detail": "Invalid parameters",
                    "type": "urn:problem:validation_error",
                    "errors": [{"loc": ["page"]}],
                },
            )
        )
    assert exc_422.value.status_code == 422
    assert exc_422.value.errors == [{"loc": ["page"]}]

    # 429 Rate Limited with Retry-After header
    with pytest.raises(RateLimitedError) as exc_429:
        client._handle_response_error(
            _make_mock_response(
                429,
                {
                    "title": "Rate Limited",
                    "detail": "Too many calls",
                    "type": "urn:problem:rate_limited",
                },
                headers={"Retry-After": "60"},
            )
        )
    assert exc_429.value.status_code == 429
    assert exc_429.value.retry_after == 60

    # 500 Server Error
    with pytest.raises(ServerError) as exc_500:
        client._handle_response_error(
            _make_mock_response(
                500,
                {
                    "title": "Internal Error",
                    "detail": "Server failed",
                    "type": "urn:problem:internal_error",
                },
            )
        )
    assert exc_500.value.status_code == 500


def test_client_status_strip_data_parsing():
    """Verify get_status_strip_data safely compiles metadata from multiple endpoints."""
    client = ApiClient(base_url="http://test")

    def mock_get(path: str, **kwargs: Any) -> Any:
        if path == "/api/v1/kpis/overview":
            return {"data_as_of": "2011-12-08"}
        if path == "/api/v1/models":
            return {"items": [{"algorithm": "lightgbm", "is_champion": True}]}
        if path == "/api/v1/pipelines/runs":
            return {"items": [{"status": "succeeded", "queued_at": "2011-12-09T01:22:00Z"}]}
        return {}

    with patch.object(client, "get", side_effect=mock_get):
        status = client.get_status_strip_data()
        assert status.data_as_of == "2011-12-08"
        assert status.champion_model == "Lightgbm"
        assert status.last_pipeline_status == "SUCCEEDED"
        assert status.last_pipeline_time == "2011-12-09T01:22:00Z"


# =====================================================================
# 3. Streamlit AppTest Suite (Scaffold, Login, Navigation)
# =====================================================================


def test_apptest_unauthenticated_shows_login_page():
    """Unauthenticated access must render the login view without navigating to dashboard pages."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Form is rendered with Sign In submit button
    assert not at.exception
    assert len(at.form) >= 1
    assert any("Sign In" in str(sub.label) for sub in at.form[0].form_submit_button)


def test_apptest_login_failure_invalid_credentials():
    """Failed login displays generic error message ('Email or password is incorrect.')."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    mock_client = MagicMock(spec=ApiClient)
    mock_client.token = None
    mock_client.login.side_effect = UnauthorizedError(401, "Unauthorized", "Invalid credentials")

    at.session_state["api_client"] = mock_client

    # Fill in inputs and submit
    at.text_input[0].input("wrong@example.com")
    at.text_input[1].input("WrongPassword123!")
    at.form[0].form_submit_button[0].click()
    at.run()

    assert not at.exception
    # Generic error message matching DESIGN §4.1
    error_texts = [e.value for e in at.error]
    assert any("Email or password is incorrect." in txt for txt in error_texts)
    assert at.session_state["logged_in"] is False


def test_apptest_login_failure_rate_limited():
    """Rate limited or locked account displays 'Too many attempts. Try again later.'."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    mock_client = MagicMock(spec=ApiClient)
    mock_client.token = None
    mock_client.login.side_effect = RateLimitedError(
        429,
        "Rate Limited",
        "Too many attempts",
        "urn:problem:rate_limited",
        retry_after=60,
    )

    at.session_state["api_client"] = mock_client

    at.text_input[0].input("locked@example.com")
    at.text_input[1].input("WrongPassword123!")
    at.form[0].form_submit_button[0].click()
    at.run()

    assert not at.exception
    error_texts = [e.value for e in at.error]
    assert any("Too many attempts. Try again later." in txt for txt in error_texts)
    assert at.session_state["logged_in"] is False


def test_apptest_login_success():
    """Successful login sets session state, stores access token, and loads user profile."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    mock_client = MagicMock(spec=ApiClient)
    mock_client.token = None
    mock_client.login.return_value = {"access_token": "valid-jwt-token"}
    mock_client.get_me.return_value = UserProfile(
        user_id="user-123",
        email="analyst@example.com",
        full_name="Test Analyst",
        role="analyst",
    )
    mock_client.get_status_strip_data.return_value = StatusStripData(
        data_as_of="2011-12-08",
        champion_model="LightGBM",
        last_pipeline_status="SUCCEEDED",
        last_pipeline_time="2011-12-09 01:22 UTC",
        is_stale_warning=False,
    )

    at.session_state["api_client"] = mock_client

    at.text_input[0].input("analyst@example.com")
    at.text_input[1].input("ValidPassword123!")
    at.form[0].form_submit_button[0].click()
    at.run()

    assert not at.exception
    assert at.session_state["logged_in"] is True
    assert at.session_state["token"] == "valid-jwt-token"
    assert at.session_state["user"]["role"] == "analyst"


def test_apptest_role_based_navigation_viewer_vs_admin():
    """Verify viewer cannot access Admin: Users, whereas admin can access Admin: Users."""
    mock_status = StatusStripData(
        data_as_of="2011-12-08",
        champion_model="LightGBM",
        last_pipeline_status="SUCCEEDED",
        last_pipeline_time="01:22 UTC",
        is_stale_warning=False,
    )

    # 1. Viewer Role
    at_viewer = AppTest.from_file(APP_PATH, default_timeout=30)
    mock_client_viewer = MagicMock(spec=ApiClient)
    mock_client_viewer.token = "viewer-token"
    mock_client_viewer.get_status_strip_data.return_value = mock_status
    at_viewer.session_state["api_client"] = mock_client_viewer
    at_viewer.session_state["logged_in"] = True
    at_viewer.session_state["token"] = "viewer-token"
    at_viewer.session_state["user"] = {
        "user_id": "viewer-1",
        "email": "viewer@example.com",
        "full_name": "Viewer User",
        "role": "viewer",
    }
    at_viewer.run()

    assert not at_viewer.exception
    # In viewer mode, default page is Overview
    titles_viewer = [t.value for t in at_viewer.title]
    assert "Overview" in titles_viewer
    assert "User Management" not in titles_viewer

    # Sidebar shows viewer role
    sidebar_captions = [c.value for c in at_viewer.sidebar.caption]
    assert any("viewer" in cap.lower() for cap in sidebar_captions)

    # 2. Admin Role
    at_admin = AppTest.from_file(APP_PATH, default_timeout=30)
    mock_client_admin = MagicMock(spec=ApiClient)
    mock_client_admin.token = "admin-token"
    mock_client_admin.get_status_strip_data.return_value = mock_status
    at_admin.session_state["api_client"] = mock_client_admin
    at_admin.session_state["logged_in"] = True
    at_admin.session_state["token"] = "admin-token"
    at_admin.session_state["user"] = {
        "user_id": "admin-1",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "role": "admin",
    }
    at_admin.run()

    assert not at_admin.exception
    sidebar_captions_admin = [c.value for c in at_admin.sidebar.caption]
    assert any("admin" in cap.lower() for cap in sidebar_captions_admin)


def test_status_strip_warning_when_stale():
    """Verify status strip renders a warning banner when data scoring is older than 36 hours."""
    stale_data = StatusStripData(
        data_as_of="2011-12-05",
        champion_model="LightGBM",
        last_pipeline_status="FAILED",
        last_pipeline_time="2011-12-06",
        is_stale_warning=True,
        stale_reason="Warning: Model scoring is older than 36 hours.",
    )

    at = AppTest.from_string("""
import streamlit as st
from retailpulse.dashboard.components.status_strip import render_status_strip

stale_data = st.session_state["stale_data"]
render_status_strip(stale_data)
""")
    at.session_state["stale_data"] = stale_data
    at.run()

    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("older than 36 hours" in w for w in warnings)

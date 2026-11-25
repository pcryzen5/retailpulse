"""AppTest and unit test suite for the Model Monitoring dashboard view.

Verifies P6-08:
- 4 Tabs rendering: Models, Performance, Drift, Alerts.
- Models tab: Model runs table, candidate comparison (up to 4 models).
- Admin-only actions: 'Train New Model' and 'Promote to Champion'.
- Performance tab: Realized vs backtest vs baseline WAPE timeline chart.
- Drift tab: Evidently drift reports table, feature PSI chart, and sandboxed HTML viewer.
- Alerts tab: Operational alerts with role-gated 'Acknowledge' button (Analyst/Admin only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import ApiClient, UserProfile

MONITORING_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "retailpulse"
    / "dashboard"
    / "views"
    / "monitoring.py"
)


def _create_mock_client(**custom_responses: Any) -> MagicMock:
    """Create a fully-stubbed ApiClient instance for Model Monitoring testing."""
    client = MagicMock(spec=ApiClient)
    client.token = "fake-jwt-token"

    default_models = {
        "items": [
            {
                "run_id": "00000000-0000-0000-0000-000000000001",
                "task": "forecast",
                "algorithm": "lightgbm",
                "is_champion": True,
                "trained_at": "2011-12-08T04:00:00Z",
                "metrics": {
                    "wape": 0.312,
                    "mase": 0.845,
                    "coverage": 0.812,
                },
                "gate_passed": True,
                "gate_reasons": ["WAPE 0.312 beats baseline 0.510"],
            },
            {
                "run_id": "00000000-0000-0000-0000-000000000002",
                "task": "forecast",
                "algorithm": "prophet",
                "is_champion": False,
                "trained_at": "2011-12-07T12:00:00Z",
                "metrics": {
                    "wape": 0.385,
                    "mase": 0.920,
                    "coverage": 0.765,
                },
                "gate_passed": True,
                "gate_reasons": ["WAPE 0.385 beats baseline 0.510"],
            },
        ],
        "total": 2,
    }
    client.get_models.return_value = custom_responses.get("models", default_models)

    default_perf = {
        "items": [
            {
                "window_start": "2011-11-15",
                "value": 0.34,
                "baseline_value": 0.51,
            },
            {
                "window_start": "2011-11-22",
                "value": 0.36,
                "baseline_value": 0.50,
            },
        ],
        "total": 2,
    }
    client.get_model_performance.return_value = custom_responses.get("performance", default_perf)

    default_drift = {
        "items": [
            {
                "report_id": "00000000-0000-0000-0000-000000000001",
                "task": "forecast",
                "kind": "data_drift",
                "share_drifted_features": 0.15,
                "dataset_drift": False,
                "created_at": "2011-12-08T06:00:00Z",
            }
        ],
        "total": 1,
    }
    client.get_drift_reports.return_value = custom_responses.get("drift", default_drift)
    client.get_drift_report_html.return_value = "<html><body>Drift Report</body></html>"

    default_alerts = {
        "items": [
            {
                "alert_id": 101,
                "severity": "warning",
                "title": "Realized WAPE Degraded",
                "message": "Realized WAPE 0.63 exceeded warning threshold 0.50.",
                "created_at": "2011-12-08T10:00:00Z",
                "status": "open",
            }
        ],
        "total": 1,
    }
    client.get_alerts.return_value = custom_responses.get("alerts", default_alerts)

    return client


def _setup_app_test(mock_client: MagicMock, role: str = "admin") -> AppTest:
    """Initialize AppTest with authenticated session state."""
    at = AppTest.from_file(MONITORING_PATH, default_timeout=30)
    at.session_state["api_client"] = mock_client
    at.session_state["logged_in"] = True
    at.session_state["user"] = UserProfile(
        user_id="u-mon-1",
        email=f"{role}@retailpulse.com",
        role=role,
        full_name="Test User",
    )
    return at


def test_monitoring_renders_full_successfully_for_admin() -> None:
    """Verify that all 4 tabs and admin-only actions render for admin."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="admin")
    at.run()

    assert len(at.exception) == 0, f"Unexpected exception: {at.exception}"

    # Verify Title
    titles = [t.value for t in at.title]
    assert "Model Monitoring" in titles

    # Verify 4 tabs exist
    tabs = [t.label for t in at.tabs]
    assert "Models" in tabs
    assert "Performance" in tabs
    assert "Drift" in tabs
    assert "Alerts" in tabs

    # Verify Admin HAS 'Train New Model' button
    train_buttons = [b for b in at.button if "Train New Model" in b.label]
    assert len(train_buttons) == 1

    # Verify Admin HAS 'Promote to Champion' button for challenger
    promote_buttons = [b for b in at.button if "Promote to Champion" in b.label]
    assert len(promote_buttons) >= 1

    # Verify Models & Drift Dataframes rendered
    assert len(at.dataframe) >= 2

    # Verify Plotly charts rendered (Performance timeline + Drift PSI)
    plotly_charts = at.get("plotly_chart")
    assert len(plotly_charts) >= 2

    # Verify Admin HAS 'Acknowledge' alert button
    ack_buttons = [b for b in at.button if "Acknowledge" in b.label]
    assert len(ack_buttons) >= 1


def test_monitoring_role_gating_viewer() -> None:
    """CRITICAL VERIFICATION: Viewers must NOT have train, promote, or acknowledge buttons."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="viewer")
    at.run()

    assert len(at.exception) == 0

    # 1. Viewer role must NOT see 'Train New Model' button
    train_buttons = [b for b in at.button if "Train New Model" in b.label]
    assert len(train_buttons) == 0

    # 2. Viewer role must NOT see 'Promote to Champion' button
    promote_buttons = [b for b in at.button if "Promote to Champion" in b.label]
    assert len(promote_buttons) == 0

    # 3. Viewer sees admin-restriction caption
    captions = [c.value for c in at.caption]
    assert any("restricted to Admins" in c for c in captions)

    # 4. Viewer role must NOT see 'Acknowledge' alert button
    ack_buttons = [b for b in at.button if "Acknowledge" in b.label]
    assert len(ack_buttons) == 0


def test_monitoring_role_gating_analyst() -> None:
    """Verify Analysts CAN acknowledge alerts but CANNOT train or promote models."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0

    # Analyst cannot train or promote
    train_buttons = [b for b in at.button if "Train New Model" in b.label]
    assert len(train_buttons) == 0
    promote_buttons = [b for b in at.button if "Promote to Champion" in b.label]
    assert len(promote_buttons) == 0

    # Analyst CAN acknowledge alerts
    ack_buttons = [b for b in at.button if "Acknowledge" in b.label]
    assert len(ack_buttons) >= 1


def test_monitoring_empty_models_registry() -> None:
    """Verify graceful handling when no model runs exist."""
    client = _create_mock_client(models={"items": [], "total": 0})
    at = _setup_app_test(client, role="admin")
    at.run()

    assert len(at.exception) == 0
    info_texts = [i.value for i in at.info]
    assert any("No model runs found" in t for t in info_texts)

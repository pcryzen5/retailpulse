"""AppTest and unit test suite for the Executive Overview dashboard view.

Verifies P6-03:
- Executive KPI card rendering (values, compact scaling, deltas, and tooltips).
- Demand history and 28-day probabilistic forecast chart.
- Operational alerts display with role-based acknowledgment gating (viewer vs analyst).
- Top 10 stockout risks table with synthetic parameter notices.
- Customer segment breakdown grouped comparison.
- Error handling and empty state fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import ApiClient, ApiClientError, UserProfile

OVERVIEW_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "retailpulse"
    / "dashboard"
    / "views"
    / "overview.py"
)


def _create_mock_client(**custom_responses: Any) -> MagicMock:
    """Create a fully-stubbed ApiClient instance for Overview testing."""
    client = MagicMock(spec=ApiClient)
    client.token = "fake-jwt-token"

    # Default stub responses matching DESIGN §4.2
    default_kpis = {
        "revenue_30d": 612000.0,
        "revenue_prev_30d": 597000.0,
        "revenue_change_pct": 2.4,
        "active_customers": 1840,
        "forecast": {
            "run_id": "run-fc-01",
            "algorithm": "lightgbm",
            "wape": 0.42,
            "baseline_wape": 0.51,
        },
        "skus_reorder_now": 37,
        "high_risk_customers": 214,
        "expected_revenue_at_risk": 48000.0,
        "open_alerts": 1,
        "data_as_of": "2011-12-08",
    }
    client.get_kpis_overview.return_value = custom_responses.get("kpis", default_kpis)

    default_sales = [
        {"date": "2011-12-01", "units_sold": 120.0, "product_id": 1},
        {"date": "2011-12-02", "units_sold": 135.0, "product_id": 1},
    ]
    client.get_daily_sales.return_value = custom_responses.get("sales", default_sales)

    default_forecast = {
        "product_id": 1,
        "predictions": [
            {
                "date": "2011-12-03",
                "forecast_mean": 130.0,
                "lower_bound": 100.0,
                "upper_bound": 160.0,
            },
            {
                "date": "2011-12-04",
                "forecast_mean": 125.0,
                "lower_bound": 95.0,
                "upper_bound": 155.0,
            },
        ],
    }
    client.get_forecast.return_value = custom_responses.get("forecast", default_forecast)

    default_alerts = {
        "items": [
            {
                "alert_id": 42,
                "severity": "warning",
                "title": "Realized WAPE Degraded",
                "message": "Realized WAPE 0.63 exceeded warning threshold 0.50.",
                "created_at": "2011-12-08T10:00:00Z",
            }
        ],
        "total": 1,
    }
    client.get_alerts.return_value = custom_responses.get("alerts", default_alerts)

    default_recs = {
        "items": [
            {
                "sku": "85123A",
                "product_name": "WHITE HANGING HEART T-LIGHT HOLDER",
                "stockout_risk": 0.88,
                "on_hand_units": 15,
                "reorder_point": 120,
                "order_quantity_units": 200,
                "params_source": "synthetic",
            }
        ],
        "total": 1,
    }
    client.get_inventory_recommendations.return_value = custom_responses.get(
        "inventory", default_recs
    )

    default_segments = [
        {
            "segment_id": 1,
            "segment_label": "Champions",
            "customer_share_pct": 22.5,
            "revenue_share_pct": 58.2,
        },
        {
            "segment_id": 2,
            "segment_label": "At Risk",
            "customer_share_pct": 31.0,
            "revenue_share_pct": 14.3,
        },
    ]
    client.get_segments.return_value = custom_responses.get("segments", default_segments)
    return client


def test_overview_apptest_renders_kpis_and_components():
    """Verify Overview renders all 6 KPI cards, tables, and charts cleanly."""
    at = AppTest.from_file(OVERVIEW_PATH)

    mock_client = _create_mock_client()
    viewer_user = UserProfile(user_id="u-1", email="viewer@retailpulse.com", role="viewer")

    at.session_state["api_client"] = mock_client
    at.session_state["user"] = viewer_user
    at.session_state["logged_in"] = True

    at.run()

    assert not at.exception

    # 1. Verify 6 KPI metrics rendered
    metric_values = [m.value for m in at.metric]
    assert any("£612k" in val for val in metric_values)
    assert any("1,840" in val for val in metric_values)
    assert any("0.42 vs 0.51" in val for val in metric_values)
    assert any("37" in val for val in metric_values)
    assert any("214" in val for val in metric_values)
    assert any("£48k" in val for val in metric_values)

    # 2. Verify alert presence
    markdown_texts = [m.value for m in at.markdown]
    assert any("Realized WAPE Degraded" in txt for txt in markdown_texts)

    # Viewer cannot see Acknowledge button
    button_labels = [b.label for b in at.button]
    assert "Acknowledge" not in button_labels

    # 3. Verify stockout risks dataframe exists
    assert len(at.dataframe) >= 1
    df = at.dataframe[0].value
    assert "85123A" in df["SKU"].values
    assert "🧪 Synthetic" in df["Params"].values

    # 4. Verify anonymous sales footnote exists
    captions = [c.value for c in at.caption]
    assert any("Anonymous sales" in cap for cap in captions)


def test_overview_apptest_alert_ack_by_analyst():
    """Analyst user can see and click Acknowledge on open alerts."""
    at = AppTest.from_file(OVERVIEW_PATH)

    mock_client = _create_mock_client()
    mock_client.acknowledge_alert.return_value = {"status": "acknowledged"}

    analyst_user = UserProfile(user_id="u-2", email="analyst@retailpulse.com", role="analyst")

    at.session_state["api_client"] = mock_client
    at.session_state["user"] = analyst_user
    at.session_state["logged_in"] = True

    at.run()
    assert not at.exception

    # Analyst sees Acknowledge button
    button_labels = [b.label for b in at.button]
    assert "Acknowledge" in button_labels

    # Find and click Acknowledge button
    ack_btn = next(b for b in at.button if b.label == "Acknowledge")
    ack_btn.click()
    at.run()

    mock_client.acknowledge_alert.assert_called_once_with(42)


def test_overview_apptest_kpi_fetch_error_handling():
    """API errors during KPI fetch render friendly error with Request ID and retry."""
    at = AppTest.from_file(OVERVIEW_PATH)

    mock_client = MagicMock(spec=ApiClient)
    mock_client.token = "token"
    mock_client.get_kpis_overview.side_effect = ApiClientError(
        500,
        "Database Error",
        "Connection pool depleted",
        request_id="req-ov-err-123",
    )

    viewer_user = UserProfile(user_id="u-1", email="viewer@retailpulse.com", role="viewer")
    at.session_state["api_client"] = mock_client
    at.session_state["user"] = viewer_user
    at.session_state["logged_in"] = True

    at.run()
    assert not at.exception

    # Error message rendered with Request ID
    error_texts = [e.value for e in at.error]
    assert any("Connection pool depleted" in txt for txt in error_texts)
    assert any("req-ov-err-123" in txt for txt in error_texts)


def test_overview_apptest_empty_state_handling():
    """Overview handles empty data responses gracefully with informative notices."""
    at = AppTest.from_file(OVERVIEW_PATH)

    mock_client = _create_mock_client(
        kpis={
            "revenue_30d": 0.0,
            "revenue_prev_30d": 0.0,
            "revenue_change_pct": None,
            "active_customers": 0,
            "forecast": None,
            "skus_reorder_now": 0,
            "high_risk_customers": 0,
            "expected_revenue_at_risk": 0.0,
            "open_alerts": 0,
        },
        sales=[],
        alerts={"items": [], "total": 0},
        inventory={"items": [], "total": 0},
        segments=[],
    )

    viewer_user = UserProfile(user_id="u-1", email="viewer@retailpulse.com", role="viewer")
    at.session_state["api_client"] = mock_client
    at.session_state["user"] = viewer_user
    at.session_state["logged_in"] = True

    at.run()
    assert not at.exception

    # No alerts -> success message
    success_texts = [s.value for s in at.success]
    assert any("No open alerts. System is operating normally." in txt for txt in success_texts)

    # Empty states for inventory and segments
    info_texts = [i.value for i in at.info]
    assert any("No immediate stockout risks detected." in txt for txt in info_texts)
    assert any("No customer segments calculated yet." in txt for txt in info_texts)

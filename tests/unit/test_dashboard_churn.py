"""AppTest and unit test suite for the Churn Risk dashboard view.

Verifies P6-06:
- Filter bar rendering (Risk Band radio, Segment dropdown, Customer search).
- Summary KPI metric cards (High, Medium, Low counts, and Expected Revenue at Risk).
- Churn probability distribution histogram with risk band threshold shading.
- Selected customer detail panel with plain-language SHAP drivers reasons chart and RFM metrics.
- Customer list dataframe table.
- Role-gated CSV export: Analysts/Admins have export button; Viewers have NO export button.
- Segment pre-selection via session state cross-navigation.
- Empty state and error state handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import ApiClient, ApiClientError, UserProfile

CHURN_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "retailpulse"
    / "dashboard"
    / "views"
    / "churn.py"
)


def _create_mock_client(**custom_responses: Any) -> MagicMock:
    """Create a fully-stubbed ApiClient instance for Churn Risk testing."""
    client = MagicMock(spec=ApiClient)
    client.token = "fake-jwt-token"

    default_summary = {
        "as_of_date": "2011-12-08",
        "total_scored": 1840,
        "band_counts": {
            "high": 214,
            "medium": 531,
            "low": 1095,
        },
        "expected_revenue_at_risk": 48000.0,
        "mean_churn_probability": 0.38,
        "probability_histogram": [
            {
                "bin_start": round(i * 0.1, 1),
                "bin_end": round((i + 1) * 0.1, 1),
                "count": 150 + i * 5,
            }
            for i in range(10)
        ],
    }
    client.get_churn_summary.return_value = custom_responses.get("summary", default_summary)

    default_segments = [
        {"segment_id": 1, "segment_label": "Champions"},
        {"segment_id": 2, "segment_label": "Loyal"},
        {"segment_id": 3, "segment_label": "At Risk"},
    ]
    client.get_segments.return_value = custom_responses.get("segments", default_segments)

    default_customers = {
        "items": [
            {
                "customer_id": 101,
                "customer_key": "CUST-1001",
                "segment_id": 3,
                "segment_label": "At Risk",
                "churn_probability": 0.82,
                "risk_band": "high",
                "recency_days": 110,
                "frequency": 3,
                "monetary": 420.50,
                "top_drivers": [
                    {
                        "feature": "recency_days",
                        "impact": 0.35,
                        "direction": "increases_risk",
                    },
                    {
                        "feature": "order_trend",
                        "impact": 0.22,
                        "direction": "increases_risk",
                    },
                ],
            },
            {
                "customer_id": 102,
                "customer_key": "CUST-1002",
                "segment_id": 1,
                "segment_label": "Champions",
                "churn_probability": 0.15,
                "risk_band": "low",
                "recency_days": 14,
                "frequency": 8,
                "monetary": 1250.00,
                "top_drivers": [
                    {
                        "feature": "orders_last_90d",
                        "impact": 0.28,
                        "direction": "decreases_risk",
                    }
                ],
            },
        ],
        "total": 2,
    }
    client.get_customers.return_value = custom_responses.get("customers", default_customers)

    default_customer_detail = {
        "customer_id": 101,
        "customer_key": "CUST-1001",
        "segment": {"segment_id": 3, "segment_label": "At Risk"},
        "rfm": {
            "recency_days": 110,
            "frequency": 3,
            "monetary": 420.50,
            "r_score": 1,
            "f_score": 3,
            "m_score": 3,
            "rfm_score": "133",
        },
        "churn": {
            "probability": 0.82,
            "risk_band": "high",
            "as_of_date": "2011-12-08",
            "top_reasons": [
                {
                    "feature": "recency_days",
                    "impact": 0.35,
                    "direction": "increases_risk",
                },
                {
                    "feature": "order_trend",
                    "impact": 0.22,
                    "direction": "increases_risk",
                },
            ],
        },
    }
    client.get_customer_detail.return_value = custom_responses.get(
        "customer_detail", default_customer_detail
    )

    client.export_customers_csv.return_value = (
        b"customer_key,risk_band,churn_probability\nCUST-1001,high,0.82\n"
    )

    return client


def _setup_app_test(mock_client: MagicMock, role: str = "analyst") -> AppTest:
    """Initialize AppTest with authenticated session state."""
    at = AppTest.from_file(CHURN_PATH, default_timeout=30)
    at.session_state["api_client"] = mock_client
    at.session_state["logged_in"] = True
    at.session_state["user"] = UserProfile(
        user_id="u-test-1",
        email=f"{role}@retailpulse.com",
        role=role,
        full_name="Test User",
    )
    return at


def test_churn_renders_full_successfully_for_analyst() -> None:
    """Verify that filters, KPI cards, charts, customer detail, and export render for analyst."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0, f"Unexpected exception: {at.exception}"

    # Verify Title and Caption
    titles = [t.value for t in at.title]
    assert "Churn Risk" in titles

    # Verify Filters: Risk band radio, segment selectbox, search input
    assert len(at.radio) >= 1
    assert len(at.selectbox) >= 2  # segment filter + customer detail picker
    assert len(at.text_input) >= 1

    # Verify Top KPI Metric Cards
    metric_labels = [m.label for m in at.metric]
    assert "High Risk Customers" in metric_labels
    assert "Medium Risk Customers" in metric_labels
    assert "Low Risk Customers" in metric_labels
    assert "Expected Revenue at Risk" in metric_labels

    # Verify Charts rendered (histogram + reasons horizontal bar chart)
    plotly_charts = at.get("plotly_chart")
    assert len(plotly_charts) >= 2

    # Verify Customer List Table
    assert len(at.dataframe) >= 1

    # Verify Analyst HAS export button
    export_buttons = [b for b in at.download_button if "Export" in b.label]
    assert len(export_buttons) == 1
    assert export_buttons[0].label == "Export Customers CSV"


def test_churn_viewer_has_no_export_button() -> None:
    """CRITICAL VERIFICATION: Viewers must NOT have an export button rendered."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="viewer")
    at.run()

    assert len(at.exception) == 0

    # Viewers see tables and charts, but NO export button
    export_buttons = [b for b in at.download_button if "Export" in b.label]
    assert len(export_buttons) == 0, "Viewer role must not see any Export CSV download button"


def test_churn_preselects_segment_from_cross_navigation() -> None:
    """Verify preselected segment from Customer Segments cross-navigation is honored."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="analyst")
    # Simulate cross-navigation state set by Segments view
    at.session_state["churn_filter_segment"] = "At Risk"
    at.session_state["churn_filter_segment_id"] = 3
    at.run()

    assert len(at.exception) == 0
    # Segment selectbox value should be 'At Risk'
    assert at.selectbox[0].value == "At Risk"


def test_churn_empty_state_when_no_scores() -> None:
    """Verify friendly empty state when churn summary has 0 scored customers."""
    client = _create_mock_client(summary={"total_scored": 0, "band_counts": {}})
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert len(at.info) >= 1 or len(at.warning) >= 1
    info_texts = [i.value for i in at.info]
    assert any("No active churn scores" in t for t in info_texts)


def test_churn_error_state_on_api_failure() -> None:
    """Verify error state handled cleanly on ApiClientError."""
    client = _create_mock_client()
    client.get_churn_summary.side_effect = ApiClientError(
        status_code=500,
        title="Internal Server Error",
        detail="Scoring table unavailable",
    )
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) >= 1
    error_texts = [e.value for e in at.error]
    assert any(
        "Internal Server Error" in t or "Scoring table unavailable" in t for t in error_texts
    )

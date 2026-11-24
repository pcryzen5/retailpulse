"""AppTest and unit test suite for the Customer Segments dashboard view.

Verifies P6-05:
- Segment cards rendering (label, customer count, customer share %,
  revenue share %, RFM averages, and persona description).
- Top KPI summary metrics (total segments, scored customers, top revenue contributor).
- Grouped bar chart comparing customer share vs revenue share across segments.
- Segment comparison table and CSV export button.
- Cross-navigation to Churn Risk setting session state segment filter.
- Graceful empty state when no segments exist.
- Error state handling on API failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import ApiClient, ApiClientError, UserProfile

SEGMENTS_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "retailpulse"
    / "dashboard"
    / "views"
    / "segments.py"
)


def _create_mock_client(**custom_responses: Any) -> MagicMock:
    """Create a fully-stubbed ApiClient instance for Customer Segments testing."""
    client = MagicMock(spec=ApiClient)
    client.token = "fake-jwt-token"

    default_segments = [
        {
            "segment_id": 1,
            "segment_label": "Champions",
            "n_customers": 950,
            "pct_customers": 0.22,
            "revenue_share": 0.58,
            "avg_recency_days": 12.4,
            "avg_frequency": 9.1,
            "avg_monetary": 1420.50,
            "description": "High-value shoppers who buy often and made recent purchases.",
        },
        {
            "segment_id": 2,
            "segment_label": "Loyal",
            "n_customers": 1200,
            "pct_customers": 0.28,
            "revenue_share": 0.22,
            "avg_recency_days": 28.5,
            "avg_frequency": 4.5,
            "avg_monetary": 450.00,
            "description": "Consistent, dependable customers with frequent engagement.",
        },
        {
            "segment_id": 3,
            "segment_label": "At Risk",
            "n_customers": 850,
            "pct_customers": 0.20,
            "revenue_share": 0.12,
            "avg_recency_days": 115.0,
            "avg_frequency": 3.2,
            "avg_monetary": 380.00,
            "description": (
                "Previously frequent or high-spending customers who have not ordered recently."
            ),
        },
        {
            "segment_id": 4,
            "segment_label": "Hibernating",
            "n_customers": 1300,
            "pct_customers": 0.30,
            "revenue_share": 0.08,
            "avg_recency_days": 210.0,
            "avg_frequency": 1.2,
            "avg_monetary": 110.00,
            "description": "Customers with low recency and engagement who have largely lapsed.",
        },
    ]
    client.get_segments.return_value = custom_responses.get("segments", default_segments)
    return client


def _setup_app_test(mock_client: MagicMock, role: str = "analyst") -> AppTest:
    """Initialize AppTest with authenticated session state."""
    at = AppTest.from_file(SEGMENTS_PATH, default_timeout=30)
    at.session_state["api_client"] = mock_client
    at.session_state["user"] = UserProfile(
        user_id="u-analyst-1",
        email="analyst@retailpulse.com",
        role=role,
        full_name="Test User",
    )
    return at


def test_segments_renders_full_successfully() -> None:
    """Verify that cards, metrics, grouped bar chart, table, and download button render."""
    client = _create_mock_client()
    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0, f"Unexpected exception: {at.exception}"

    # Verify Title and Caption
    titles = [t.value for t in at.title]
    assert "Customer Segments" in titles

    # Verify Top KPI Cards
    metric_labels = [m.label for m in at.metric]
    assert "Total Segments" in metric_labels
    assert "Total Scored Customers" in metric_labels
    assert "Top Revenue Contributor" in metric_labels

    # Verify Segment Cards metrics (Customers, Revenue Share, Avg Spend)
    assert any("Revenue Share" in m for m in metric_labels)
    assert any("Avg Spend" in m for m in metric_labels)

    # Verify Grouped Bar Chart rendered
    plotly_charts = at.get("plotly_chart")
    assert len(plotly_charts) >= 1

    # Verify Dataframe table and CSV download button
    assert len(at.dataframe) >= 1
    assert len(at.download_button) >= 1
    assert at.download_button[0].label == "Download Segments CSV"


def test_segments_cross_navigation_sets_filter() -> None:
    """Verify clicking 'View Churn Risk' sets the target segment filter in session state."""
    client = _create_mock_client()
    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0

    # Locate navigation buttons (one per segment card)
    buttons = at.button
    churn_buttons = [b for b in buttons if "View Churn Risk" in b.label]
    assert len(churn_buttons) >= 4

    # Click first button (Champions, segment_id = 1)
    churn_buttons[0].click().run()

    assert len(at.exception) == 0
    assert at.session_state.get("churn_filter_segment") == "Champions"
    assert at.session_state.get("churn_filter_segment_id") == 1


def test_segments_empty_state() -> None:
    """Verify friendly empty state when no customer segments exist."""
    client = _create_mock_client(segments=[])
    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert len(at.info) >= 1 or len(at.warning) >= 1
    info_texts = [i.value for i in at.info]
    assert any("No customer segments found" in t for t in info_texts)


def test_segments_error_state() -> None:
    """Verify error state handled gracefully when API call fails."""
    client = _create_mock_client()
    client.get_segments.side_effect = ApiClientError(
        status_code=500,
        title="Internal Server Error",
        detail="Database query timed out.",
    )

    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) >= 1
    error_texts = [e.value for e in at.error]
    assert any("Internal Server Error" in t or "Database query timed out" in t for t in error_texts)

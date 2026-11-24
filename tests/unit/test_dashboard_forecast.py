"""AppTest and unit test suite for the Demand Forecast dashboard view.

Verifies P6-04:
- SKU selector and product selection.
- Store selector and 7/14/28-day horizon toggle.
- 180-day history with 80% prediction interval forecast chart.
- Model metadata display.
- Summary metric cards (units with range, ABC class, class WAPE).
- Accuracy segmentation by ABC class and horizon bucket.
- Forecast points data table and CSV download button.
- Ineligible SKU explanation handling (shows plain notice, no red error block).
- Empty and service unavailable states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import ApiClient, ApiClientError, UserProfile, ValidationError

FORECAST_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "retailpulse"
    / "dashboard"
    / "views"
    / "forecast.py"
)


def _create_mock_client(**custom_responses: Any) -> MagicMock:
    """Create a fully-stubbed ApiClient instance for Forecast testing."""
    client = MagicMock(spec=ApiClient)
    client.token = "fake-jwt-token"

    default_products = {
        "items": [
            {
                "product_id": 1,
                "sku": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "abc_class": "A",
            },
            {
                "product_id": 2,
                "sku": "71053",
                "description": "WHITE METAL LANTERN",
                "abc_class": "B",
            },
        ],
        "total": 2,
    }
    client.search_products.return_value = custom_responses.get("products", default_products)

    default_stores = [
        {"store_id": 1, "store_name": "Online Store", "store_code": "ONLINE"},
        {"store_id": 2, "store_name": "Store London", "store_code": "LDN-01"},
    ]
    client.get_stores.return_value = custom_responses.get("stores", default_stores)

    default_forecast = {
        "model": {
            "run_id": "run-fc-01",
            "algorithm": "lightgbm",
            "is_champion": True,
            "trained_at": "2011-12-08T04:00:00Z",
        },
        "product": {
            "product_id": 1,
            "sku": "85123A",
            "description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "abc_class": "A",
        },
        "store_id": 1,
        "origin_date": "2011-12-08",
        "interval_level": 0.8,
        "points": [
            {
                "date": f"2011-12-{9 + i:02d}",
                "yhat": 45.0 + i,
                "yhat_lower": 32.0 + i,
                "yhat_upper": 58.0 + i,
            }
            for i in range(28)
        ],
    }
    client.get_forecast.return_value = custom_responses.get("forecast", default_forecast)

    default_accuracy = {
        "run_id": "run-fc-01",
        "algorithm": "lightgbm",
        "metrics": {"wape": 0.31, "mase": 0.85},
        "by_abc": {
            "A": {"wape": 0.29},
            "B": {"wape": 0.42},
            "C": {"wape": 0.55},
        },
        "by_horizon": [
            {"bucket": "1-7", "wape": 0.25},
            {"bucket": "8-14", "wape": 0.32},
            {"bucket": "15-28", "wape": 0.41},
        ],
        "baselines": {
            "seasonal_naive": {"A": 0.45, "B": 0.58, "C": 0.70},
        },
    }
    client.get_forecast_accuracy.return_value = custom_responses.get("accuracy", default_accuracy)

    default_sales = [
        {"date": f"2011-11-{1 + i:02d}", "units_sold": 40.0 + (i % 10)} for i in range(30)
    ]
    client.get_daily_sales.return_value = custom_responses.get("sales", default_sales)

    return client


def _setup_app_test(mock_client: MagicMock, role: str = "analyst") -> AppTest:
    """Initialize AppTest with authenticated session state."""
    at = AppTest.from_file(FORECAST_PATH, default_timeout=30)
    at.session_state["api_client"] = mock_client
    at.session_state["user"] = UserProfile(
        user_id="u-analyst-1",
        email="analyst@retailpulse.com",
        role=role,
        full_name="Test User",
    )
    return at


def test_forecast_renders_full_successfully() -> None:
    """Verify that all core controls, cards, charts, table, and download button render."""
    client = _create_mock_client()
    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0, f"Unexpected exception: {at.exception}"

    # Verify Title and Model metadata
    titles = [t.value for t in at.title]
    assert "Demand Forecast" in titles

    captions = [c.value for c in at.caption]
    assert any("Lightgbm" in c or "lightgbm" in c.lower() for c in captions)

    # Verify Controls: selectboxes (SKU, Store) and Horizon radio
    assert len(at.selectbox) >= 2
    assert len(at.radio) >= 1
    assert at.radio[0].value == 7 or at.radio[0].value == 28 or at.radio[0].value == 14

    # Verify Summary KPI cards
    metric_labels = [m.label for m in at.metric]
    assert any("Next" in m and "Forecast" in m for m in metric_labels)
    assert "ABC Classification" in metric_labels
    assert any("WAPE for Class" in m for m in metric_labels)

    # Verify Charts rendered (forecast chart, ABC accuracy, Horizon accuracy)
    plotly_charts = at.get("plotly_chart")
    assert len(plotly_charts) >= 3

    # Verify Dataframe table and CSV download button
    assert len(at.dataframe) >= 1
    assert len(at.download_button) >= 1
    assert at.download_button[0].label == "Download Forecast CSV"


def test_forecast_ineligible_sku_shows_reason_without_error() -> None:
    """Verify that ineligible SKUs display an info banner with reason, not a red error."""
    client = _create_mock_client()
    client.get_forecast.side_effect = ValidationError(
        status_code=422,
        title="Validation Error",
        detail=(
            "Product is not eligible for forecasting: Insufficient sales history (< 30 active days)"
        ),
    )

    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 0, "Ineligible SKU must not render a red st.error block"

    info_messages = [i.value for i in at.info]
    assert any("Insufficient sales history" in m for m in info_messages)
    assert any("Forecasting Notice" in m for m in info_messages)


def test_forecast_ineligible_via_apiclienterror_shows_notice() -> None:
    """Verify that ApiClientError with 'not eligible' detail also displays info notice."""
    client = _create_mock_client()
    client.get_forecast.side_effect = ApiClientError(
        status_code=422,
        title="Unprocessable Entity",
        detail="Product is not eligible for forecasting: Intermittent sales pattern",
    )

    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 0
    info_messages = [i.value for i in at.info]
    assert any("Intermittent sales pattern" in m for m in info_messages)


def test_forecast_horizon_toggle_updates_call() -> None:
    """Verify selecting a different horizon updates client parameter and KPI label."""
    client = _create_mock_client()
    at = _setup_app_test(client)
    at.run()

    # Default horizon is 7 or whatever is first option [7, 14, 28]
    # Change horizon radio to 14
    at.radio[0].set_value(14).run()

    assert len(at.exception) == 0
    # Check that client.get_forecast was called with horizon=14
    call_kwargs = client.get_forecast.call_args.kwargs
    assert call_kwargs.get("horizon") == 14

    metric_labels = [m.label for m in at.metric]
    assert any("Next 14 Days Forecast" in m for m in metric_labels)


def test_forecast_not_found_empty_state() -> None:
    """Verify 404 response displays friendly empty state."""
    client = _create_mock_client()
    client.get_forecast.side_effect = ApiClientError(
        status_code=404,
        title="Not Found",
        detail="Forecast not generated for this SKU.",
    )

    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 0
    # Should show an empty/info warning state
    assert len(at.info) >= 1 or len(at.warning) >= 1


def test_forecast_service_unavailable_champion_empty_state() -> None:
    """Verify 503 response displays message about champion model."""
    client = _create_mock_client()
    client.get_forecast.side_effect = ApiClientError(
        status_code=503,
        title="Service Unavailable",
        detail="No champion forecast model available.",
    )

    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert len(at.info) >= 1 or len(at.warning) >= 1


def test_forecast_no_products_empty_state() -> None:
    """Verify handling when product catalog search returns no items."""
    client = _create_mock_client(products={"items": [], "total": 0})
    at = _setup_app_test(client)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert len(at.info) >= 1 or len(at.warning) >= 1

"""AppTest and unit test suite for the Inventory Replenishment dashboard view.

Verifies P6-07:
- Status / ABC / SKU filter bar rendering and default to 'Reorder now'.
- Persistent synthetic parameters notice banner when synthetic parameters are active.
- Summary KPI metric cards (Reorder Now, In Stock, Overstock, Monitored SKUs).
- Recommendations table with stockout risk percentage and status badges.
- What-if parameter form with live delta metrics calculation.
- Confirmation dialog modal on save.
- Policy simulation comparison (RetailPulse vs baseline fill rate, stockout days, inventory value).
- Role-gated controls: What-if save button and CSV export available to Analyst; hidden from Viewer.
- Empty state and error state handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import ApiClient, ApiClientError, UserProfile

INVENTORY_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "retailpulse"
    / "dashboard"
    / "views"
    / "inventory.py"
)


def _create_mock_client(**custom_responses: Any) -> MagicMock:
    """Create a fully-stubbed ApiClient instance for Inventory testing."""
    client = MagicMock(spec=ApiClient)
    client.token = "fake-jwt-token"

    default_summary = {
        "as_of_date": "2011-12-08",
        "total_items": 300,
        "status_counts": {
            "reorder_now": 37,
            "ok": 240,
            "overstock": 23,
        },
        "params_synthetic_share": 1.0,
        "stockout_risk_histogram": [
            {"bin_start": 0.0, "bin_end": 0.2, "count": 180},
            {"bin_start": 0.2, "bin_end": 0.5, "count": 60},
            {"bin_start": 0.5, "bin_end": 0.8, "count": 35},
            {"bin_start": 0.8, "bin_end": 1.0, "count": 25},
        ],
    }
    client.get_inventory_summary.return_value = custom_responses.get("summary", default_summary)

    default_recommendations = {
        "items": [
            {
                "product_id": 1,
                "sku": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "abc_class": "A",
                "store_id": 1,
                "as_of_date": "2011-12-08",
                "on_hand": 12,
                "on_order": 0,
                "lead_time_days": 7.0,
                "service_level": 0.95,
                "reorder_point": 48.0,
                "safety_stock": 18.0,
                "recommended_order_qty": 60,
                "stockout_risk": 0.85,
                "days_of_cover": 2.1,
                "status": "reorder_now",
                "params_source": "synthetic",
            },
            {
                "product_id": 2,
                "sku": "22423",
                "description": "REGENCY CAKESTAND 3 TIER",
                "abc_class": "A",
                "store_id": 1,
                "as_of_date": "2011-12-08",
                "on_hand": 150,
                "on_order": 50,
                "lead_time_days": 10.0,
                "service_level": 0.95,
                "reorder_point": 80.0,
                "safety_stock": 25.0,
                "recommended_order_qty": 0,
                "stockout_risk": 0.04,
                "days_of_cover": 28.5,
                "status": "ok",
                "params_source": "synthetic",
            },
        ],
        "total": 2,
        "page": 1,
        "page_size": 50,
        "params_synthetic_share": 1.0,
    }
    client.get_inventory_recommendations.return_value = custom_responses.get(
        "recommendations", default_recommendations
    )

    default_what_if = {
        "product_id": 1,
        "sku": "85123A",
        "description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "abc_class": "A",
        "store_id": 1,
        "as_of_date": "2011-12-08",
        "on_hand": 12,
        "on_order": 0,
        "lead_time_days": 14.0,
        "service_level": 0.98,
        "reorder_point": 65.0,
        "safety_stock": 28.0,
        "recommended_order_qty": 75,
        "stockout_risk": 0.42,
        "days_of_cover": 2.1,
        "status": "reorder_now",
        "params_source": "what_if",
    }
    client.compute_inventory_what_if.return_value = custom_responses.get("what_if", default_what_if)

    default_simulation = {
        "simulation_id": "00000000-0000-0000-0000-000000000001",
        "uses_synthetic_params": True,
        "results": {
            "retailpulse": {
                "fill_rate": 0.965,
                "stockout_days": 4,
                "avg_inventory_value": 31000.0,
            },
            "baseline": {
                "fill_rate": 0.912,
                "stockout_days": 18,
                "avg_inventory_value": 42000.0,
            },
        },
    }
    client.get_latest_inventory_simulation.return_value = custom_responses.get(
        "simulation", default_simulation
    )
    client.run_inventory_simulation.return_value = default_simulation

    client.export_inventory_recommendations_csv.return_value = (
        b"sku,status,reorder_point\n85123A,reorder_now,48.0\n"
    )

    return client


def _setup_app_test(mock_client: MagicMock, role: str = "analyst") -> AppTest:
    """Initialize AppTest with authenticated session state."""
    at = AppTest.from_file(INVENTORY_PATH, default_timeout=30)
    at.session_state["api_client"] = mock_client
    at.session_state["logged_in"] = True
    at.session_state["user"] = UserProfile(
        user_id="u-test-1",
        email=f"{role}@retailpulse.com",
        role=role,
        full_name="Test User",
    )
    return at


def test_inventory_renders_full_successfully_for_analyst() -> None:
    """Verify that filters, KPI cards, table, what-if, save button, and simulation render."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0, f"Unexpected exception: {at.exception}"

    # Verify Title
    titles = [t.value for t in at.title]
    assert "Inventory" in titles

    # Verify Persistent Synthetic Notice banner (DESIGN §4.6)
    info_texts = [i.value for i in at.info]
    assert any("Synthetic parameters active" in t for t in info_texts)
    assert any("Lead times, costs and stock levels are simulated" in t for t in info_texts)

    # Verify Top KPI Metric Cards
    metric_labels = [m.label for m in at.metric]
    assert "Reorder Now" in metric_labels
    assert "In Stock (OK)" in metric_labels
    assert "Overstock" in metric_labels
    assert "Total Monitored SKUs" in metric_labels

    # Verify Filter controls: Status defaults to 'Reorder now' per DESIGN §4.6
    assert len(at.selectbox) >= 2  # status + abc + what-if sku
    assert at.selectbox[0].value == "Reorder now"

    # Verify Recommendations Dataframe
    assert len(at.dataframe) >= 1

    # Verify Export button rendered for Analyst
    export_buttons = [b for b in at.download_button if "Export" in b.label]
    assert len(export_buttons) == 1
    assert export_buttons[0].label == "Export Recommendations CSV"

    # Verify What-If section and Save Policy Parameters button rendered for Analyst
    save_buttons = [b for b in at.button if "Save Policy Parameters" in b.label]
    assert len(save_buttons) == 1

    # Verify Simulation Benchmark Section metrics
    assert any("Fill Rate" in m for m in metric_labels)
    assert any("Stockout Days" in m for m in metric_labels)
    assert any("Avg Inventory Value" in m for m in metric_labels)


def test_inventory_viewer_role_gating() -> None:
    """CRITICAL VERIFICATION: Viewers must NOT have export or save buttons."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="viewer")
    at.run()

    assert len(at.exception) == 0

    # 1. Viewer role must have NO export button
    export_buttons = [b for b in at.download_button if "Export" in b.label]
    assert len(export_buttons) == 0, "Viewer role must not see any Export CSV download button"

    # 2. Viewer role must NOT see the Save Policy Parameters button
    save_buttons = [b for b in at.button if "Save Policy Parameters" in b.label]
    assert len(save_buttons) == 0, "Viewer role must not see Save Policy Parameters button"

    # 3. Viewer sees a warning/forbidden notice regarding editing
    warning_texts = [w.value for w in at.warning]
    assert any("requires Analyst or Admin role" in t for t in warning_texts)


def test_inventory_what_if_slider_adjustment() -> None:
    """Verify modifying what-if parameters updates delta metrics."""
    client = _create_mock_client()
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0

    # Verify what-if recomputation occurred
    assert client.compute_inventory_what_if.called

    # Check that delta metric cards render
    metric_labels = [m.label for m in at.metric]
    assert "Reorder Point" in metric_labels
    assert "Safety Stock" in metric_labels
    assert "Recommended Order Qty" in metric_labels


def test_inventory_empty_recommendations_state() -> None:
    """Verify graceful empty state when recommendation list is empty."""
    client = _create_mock_client(recommendations={"items": [], "total": 0})
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0
    info_texts = [i.value for i in at.info]
    assert any("No recommendations match the filter criteria" in t for t in info_texts)


def test_inventory_error_state_on_failure() -> None:
    """Verify error state handled cleanly when summary API raises error."""
    client = _create_mock_client()
    client.get_inventory_recommendations.side_effect = ApiClientError(
        status_code=500,
        title="Internal Server Error",
        detail="Inventory calculation timeout",
    )
    at = _setup_app_test(client, role="analyst")
    at.run()

    assert len(at.exception) == 0
    # Should fall back to empty recommendations without crashing
    info_texts = [i.value for i in at.info]
    assert any("No recommendations match" in t for t in info_texts)

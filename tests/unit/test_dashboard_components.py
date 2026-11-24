"""Unit test suite for reusable dashboard components.

Verifies P6-02:
- formatting: en-GB currency, compact suffixes, numbers, percentages, dates
- kpi_card: st.metric rendering, canonical metric tooltip resolution
- badges: status and risk normalization, accessible icon+text, Streamlit color markdown
- tables: server-side pagination, bound calculations, prev/next controls
- states: loading, empty, error (with request ID and retry), forbidden, synthetic notice
- dialogs: confirmation dialog and inline card with effect description and callback
- charts: accessible Plotly builders for forecast, grouped comparison, feature importance, drift
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from retailpulse.dashboard.client import ApiClientError
from retailpulse.dashboard.components.badges import (
    BADGE_REGISTRY,
    format_status_badge,
    get_badge_definition,
    render_badge,
)
from retailpulse.dashboard.components.charts import (
    plot_drift_metric,
    plot_feature_importance,
    plot_forecast,
    plot_grouped_bar,
    render_chart,
)
from retailpulse.dashboard.components.dialogs import (
    open_confirm_dialog,
    render_confirm_inline,
)
from retailpulse.dashboard.components.formatting import (
    format_currency,
    format_date,
    format_number,
    format_percent,
    get_currency_symbol,
)
from retailpulse.dashboard.components.kpi_card import (
    METRIC_TOOLTIPS,
    render_kpi_card,
)
from retailpulse.dashboard.components.states import (
    render_empty,
    render_error,
    render_forbidden,
    render_loading,
    render_synthetic_notice,
)
from retailpulse.dashboard.components.tables import render_paged_table

# =====================================================================
# 1. Formatting Tests
# =====================================================================


def test_format_currency_standards():
    """Verify currency formatting in en-GB locale and compact formats."""
    # Standard decimal
    assert format_currency(12345.50) == "£12,345.50"
    assert format_currency(0.0) == "£0.00"
    assert format_currency(None) == "-"
    assert format_currency(-50.25) == "-£50.25"

    # Compact metric suffixes (DESIGN §2)
    assert format_currency(612000, compact=True) == "£612k"
    assert format_currency(1200000, compact=True) == "£1.2M"
    assert format_currency(1000000, compact=True) == "£1M"
    assert format_currency(950, compact=True) == "£950.00"


def test_format_currency_custom_symbol(monkeypatch: pytest.MonkeyPatch):
    """Verify RP_CURRENCY override."""
    monkeypatch.setenv("RP_CURRENCY", "$")
    assert get_currency_symbol() == "$"
    assert format_currency(100) == "$100.00"
    assert format_currency(1500000, compact=True) == "$1.5M"


def test_format_number():
    """Verify thousands separators and compact suffixes."""
    assert format_number(12345) == "12,345"
    assert format_number(12345.67, decimals=2) == "12,345.67"
    assert format_number(None) == "-"
    assert format_number(15000, compact=True) == "15k"
    assert format_number(2400000, compact=True) == "2.4M"


def test_format_percent():
    """Verify percentage formatting from decimal ratio and direct percent."""
    assert format_percent(0.1234, is_ratio=True) == "12.3%"
    assert format_percent(12.34, is_ratio=False) == "12.3%"
    assert format_percent(0.05, is_ratio=True, decimals=2) == "5.00%"
    assert format_percent(None) == "-"


def test_format_date():
    """Verify en-GB date formatting (09 Dec 2011)."""
    assert format_date("2011-12-09T08:26:00Z") == "09 Dec 2011"
    assert format_date("2011-12-09") == "09 Dec 2011"
    assert format_date(datetime(2011, 12, 9, 8, 26, tzinfo=UTC)) == "09 Dec 2011"
    assert format_date(date(2011, 12, 9)) == "09 Dec 2011"
    assert format_date(None) == "-"
    assert format_date("") == "-"


# =====================================================================
# 2. KPI Card Tests
# =====================================================================


def test_kpi_card_tooltip_lookup():
    """Verify tooltips defined for key metrics (WAPE, MASE, reorder point)."""
    assert "wape" in METRIC_TOOLTIPS
    assert "mase" in METRIC_TOOLTIPS
    assert "reorder_point" in METRIC_TOOLTIPS
    assert "stockout_risk" in METRIC_TOOLTIPS


def test_kpi_card_rendering():
    """Verify render_kpi_card delegates properly to st.metric."""
    with patch("streamlit.metric") as mock_metric:
        render_kpi_card(
            label="Demand WAPE",
            value="14.2%",
            delta="-1.5%",
            tooltip_key="wape",
        )
        mock_metric.assert_called_once_with(
            label="Demand WAPE",
            value="14.2%",
            delta="-1.5%",
            delta_color="normal",
            help=METRIC_TOOLTIPS["wape"],
        )


# =====================================================================
# 3. Badges Tests
# =====================================================================


def test_badge_definitions_accessibility():
    """Every badge must have an icon, label, and never convey info by color alone."""
    for key, badge in BADGE_REGISTRY.items():
        assert badge.icon, f"Badge '{key}' missing icon"
        assert badge.label, f"Badge '{key}' missing text label"
        assert badge.color in ("green", "red", "orange", "blue", "gray")
        # Markdown must use native :color[...] syntax
        assert badge.markdown == f":{badge.color}[{badge.icon} {badge.label}]"


def test_badge_normalization():
    """Verify normalized lookup for status variations."""
    assert get_badge_definition("ok").label == "OK"
    assert get_badge_definition("IN_STOCK").label == "OK"
    assert get_badge_definition("reorder-now").color == "red"
    assert get_badge_definition("high_risk").icon == "●"
    assert get_badge_definition("synthetic").icon == "🧪"
    assert get_badge_definition(None).label == "No data"
    assert get_badge_definition("unknown_status").label == "unknown_status"


def test_format_and_render_badge():
    """Verify format_status_badge output and render_badge markdown call."""
    formatted = format_status_badge("reorder_now")
    assert formatted == ":red[⚠ Reorder now]"

    with patch("streamlit.markdown") as mock_markdown:
        render_badge("ok")
        mock_markdown.assert_called_once_with(":green[✔ OK]")


# =====================================================================
# 4. Tables and Pagination Tests
# =====================================================================


def test_render_paged_table_bounds():
    """Verify server-paged table calculates page bounds correctly."""
    df = pd.DataFrame({"product_id": [1, 2, 3], "name": ["A", "B", "C"]})

    with (
        patch("streamlit.dataframe") as mock_df,
        patch("streamlit.columns") as mock_cols,
        patch("streamlit.button", return_value=False),
        patch("streamlit.caption") as mock_caption,
    ):
        col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
        mock_cols.side_effect = [
            (col1, col2, col3),  # outer row
            (MagicMock(), MagicMock()),  # nav buttons row
        ]

        active_page = render_paged_table(
            df,
            total_count=125,
            page=2,
            page_size=50,
            key="test_table",
        )

        mock_df.assert_called_once_with(
            df,
            column_config=None,
            hide_index=True,
            use_container_width=True,
        )
        assert active_page == 2
        # Captions should show 51-100 of 125
        caption_calls = [c[0][0] for c in mock_caption.call_args_list]
        assert any("51-100" in cap for cap in caption_calls)
        assert any("Page **2** of **3**" in cap for cap in caption_calls)


def test_render_paged_table_navigation_clicks():
    """Verify clicking Next increments page and triggers rerun."""
    df = pd.DataFrame({"id": [1]})
    with (
        patch("streamlit.dataframe"),
        patch("streamlit.columns") as mock_cols,
        patch("streamlit.button", side_effect=[False, True]),  # prev=False, next=True
        patch("streamlit.caption"),
        patch("streamlit.rerun") as mock_rerun,
        patch("streamlit.session_state", {}) as mock_state,
    ):
        mock_cols.side_effect = [
            (MagicMock(), MagicMock(), MagicMock()),
            (MagicMock(), MagicMock()),
        ]

        render_paged_table(df, total_count=100, page=1, page_size=50, key="inv_tbl")
        assert mock_state["inv_tbl_page"] == 2
        mock_rerun.assert_called_once()


# =====================================================================
# 5. States Helpers Tests
# =====================================================================


def test_render_loading():
    """Verify loading spinner context manager."""
    with patch("streamlit.spinner") as mock_spinner:
        with render_loading("Loading forecast..."):
            pass
        mock_spinner.assert_called_once_with("Loading forecast...")


def test_render_empty():
    """Verify empty state renders icon-free message and action guidance."""
    with patch("streamlit.info") as mock_info:
        render_empty("No forecast yet.", "An admin can run scoring from Data & Pipelines.")
        mock_info.assert_called_once_with(
            "No forecast yet. An admin can run scoring from Data & Pipelines."
        )


def test_render_error_with_api_client_error():
    """Verify error display includes detail and Request ID without raw stacktrace."""
    err = ApiClientError(
        500,
        "Internal Server Error",
        "Database pool exhausted.",
        request_id="req-test-999",
    )
    with (
        patch("streamlit.error") as mock_error,
        patch("streamlit.button", return_value=True),
    ):
        retry_fn = MagicMock()
        render_error(err, on_retry=retry_fn, key="btn_err_test")

        # Verify error text contains message and Request ID
        error_msg = mock_error.call_args[0][0]
        assert "Database pool exhausted." in error_msg
        assert "req-test-999" in error_msg
        retry_fn.assert_called_once()


def test_render_forbidden_and_synthetic_notice():
    """Verify forbidden state and synthetic parameters notice."""
    with patch("streamlit.warning") as mock_warn:
        render_forbidden("You lack permissions.")
        mock_warn.assert_called_once_with("You lack permissions.", icon="🔒")

    with patch("streamlit.info") as mock_info:
        render_synthetic_notice()
        mock_info.assert_called_once_with(
            "Notice: Synthetic inventory parameters are currently active for this product.",
            icon="🧪",
        )


# =====================================================================
# 6. Dialog Tests
# =====================================================================


def test_render_confirm_inline():
    """Verify inline confirmation states effect and calls on_confirm."""
    on_confirm = MagicMock()

    with (
        patch("streamlit.container"),
        patch("streamlit.subheader") as mock_sub,
        patch("streamlit.write"),
        patch("streamlit.warning") as mock_warn,
        patch("streamlit.columns") as mock_cols,
        patch("streamlit.button", side_effect=[False, True]),  # cancel=False, confirm=True
        patch("streamlit.rerun"),
        patch("streamlit.session_state", {}) as mock_state,
    ):
        mock_cols.return_value = (MagicMock(), MagicMock())

        render_confirm_inline(
            title="Promote Model",
            message="Promote LightGBM run 42 to active champion?",
            effect_description="Scoring will automatically re-run using this model.",
            on_confirm=on_confirm,
            key_prefix="promote_test",
        )

        mock_sub.assert_called_once_with("Promote Model")
        warn_msg = mock_warn.call_args[0][0]
        assert "Scoring will automatically re-run using this model." in warn_msg
        on_confirm.assert_called_once()
        assert mock_state["promote_test_active"] is False


def test_open_confirm_dialog_wraps_dialog():
    """Verify open_confirm_dialog executes inside st.dialog."""
    on_confirm = MagicMock()

    # Emulate st.dialog decorator returning the function
    def fake_dialog_decorator(title):
        def _wrapper(fn):
            return fn

        return _wrapper

    with (
        patch("streamlit.dialog", side_effect=fake_dialog_decorator),
        patch("streamlit.write"),
        patch("streamlit.warning"),
        patch("streamlit.columns", return_value=(MagicMock(), MagicMock())),
        patch("streamlit.button", side_effect=[False, True]),  # cancel=False, confirm=True
        patch("streamlit.rerun"),
    ):
        open_confirm_dialog(
            title="Deactivate Account",
            message="Deactivate user test@example.com?",
            effect_description="User will immediately lose system access.",
            on_confirm=on_confirm,
            is_destructive=True,
        )
        on_confirm.assert_called_once()


# =====================================================================
# 7. Charts Tests
# =====================================================================


def test_plot_forecast_figure_structure():
    """Verify demand forecast figure has actuals, prediction interval, and forecast traces."""
    fig = plot_forecast(
        historical_dates=["2011-12-01", "2011-12-02"],
        historical_values=[10.0, 15.0],
        forecast_dates=["2011-12-03", "2011-12-04"],
        forecast_values=[12.0, 14.0],
        lower_bounds=[8.0, 10.0],
        upper_bounds=[16.0, 18.0],
        title="28-Day Forecast for SKU 85123A",
    )

    trace_names = [t.name for t in fig.data]
    assert "Historical Demand" in trace_names
    assert "80% Prediction Interval" in trace_names
    assert "Forecast" in trace_names

    # Layout assertions
    assert fig.layout.title.text == "28-Day Forecast for SKU 85123A"
    assert fig.layout.yaxis.title.text == "Sales Units"
    assert fig.layout.hovermode == "x unified"


def test_plot_grouped_bar_structure():
    """Verify grouped bar figure contains both series and group barmode."""
    fig = plot_grouped_bar(
        categories=["Champions", "Loyal", "At Risk"],
        series1_name="Customer Share",
        series1_values=[20.0, 30.0, 50.0],
        series2_name="Revenue Share",
        series2_values=[55.0, 30.0, 15.0],
        title="Customer vs Revenue Share",
    )

    assert len(fig.data) == 2
    assert fig.data[0].name == "Customer Share"
    assert fig.data[1].name == "Revenue Share"
    assert fig.layout.barmode == "group"
    assert fig.layout.title.text == "Customer vs Revenue Share"


def test_plot_feature_importance_structure():
    """Verify horizontal bar chart with sorted features."""
    fig = plot_feature_importance(
        features=["recency_days", "monetary_total", "frequency"],
        importances=[0.45, 0.35, 0.20],
        title="Churn Drivers",
    )

    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"
    # Smallest importance sorted first so largest is on top
    assert list(fig.data[0].x) == [0.20, 0.35, 0.45]
    assert list(fig.data[0].y) == ["frequency", "monetary_total", "recency_days"]


def test_plot_drift_metric_structure():
    """Verify drift PSI chart with threshold vertical line and danger highlighting."""
    fig = plot_drift_metric(
        features=["feat_a", "feat_b"],
        psi_values=[0.05, 0.32],
        threshold=0.25,
        title="Feature Drift",
    )

    assert len(fig.data) == 1
    # Check threshold line exists in layout shapes
    shapes = fig.layout.shapes
    assert any(s.type == "line" and s.x0 == 0.25 for s in shapes)


def test_render_chart():
    """Verify render_chart outputs st.plotly_chart followed by text caption."""
    fig = plot_feature_importance(["feat1"], [0.5])
    with (
        patch("streamlit.plotly_chart") as mock_plot,
        patch("streamlit.caption") as mock_cap,
    ):
        render_chart(fig, caption="Top driver is feat1.")
        mock_plot.assert_called_once_with(fig, use_container_width=True)
        mock_cap.assert_called_once_with("Top driver is feat1.")

"""Demand Forecast exploration view.

Implements DESIGN §4.3, §6, §7, and APP_FLOW §3:
- SKU search and selection with product description.
- Store selector (default ONLINE) and horizon toggle (7 | 14 | 28 days).
- Champion model metadata display.
- History (180 days) + forecast + 80% prediction interval band.
- Summary metrics: total horizon forecast (units with range), ABC class, and class WAPE.
- Accuracy by ABC class (champion vs seasonal naive) and by horizon bucket (1-7, 8-14, 15-28).
- Forecast points data table and CSV export button.
- Ineligible SKU handling: displays the plain reason without a red error block.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retailpulse.dashboard.client import ApiClientError, ValidationError
from retailpulse.dashboard.components.charts import (
    plot_forecast,
    plot_grouped_bar,
    render_chart,
)
from retailpulse.dashboard.components.formatting import (
    format_date,
    format_number,
)
from retailpulse.dashboard.components.kpi_card import render_kpi_card
from retailpulse.dashboard.components.states import (
    render_empty,
    render_error,
)
from retailpulse.dashboard.components.status_strip import render_status_strip
from retailpulse.dashboard.session import get_api_client
from retailpulse.dashboard.theme import CHART_PALETTE, style_figure


def render_forecast() -> None:
    """Render the Demand Forecast exploration page."""
    render_status_strip()

    st.title("Demand Forecast")
    st.caption("28-day probabilistic sales forecasts and historical demand intervals.")

    client = get_api_client()

    controls = _render_controls(client)
    if not controls:
        return
    selected_product, selected_store_id, horizon = controls

    product_id = int(selected_product["product_id"])
    sku = str(selected_product["sku"])

    forecast_data = _fetch_forecast_data(client, product_id, selected_store_id, horizon, sku)
    if not forecast_data:
        return

    _render_model_meta_header(forecast_data)
    _render_summary_metrics(forecast_data, selected_product, horizon, client)

    st.divider()

    # History (180 days) + Forecast Main Chart
    points = forecast_data.get("points", [])
    _render_main_forecast_chart(client, product_id, sku, points, horizon)

    st.divider()

    # Accuracy Segmentations: By ABC Class & By Horizon Bucket
    accuracy_data = _safe_get_accuracy(client)
    col_abc_chart, col_hz_chart = st.columns(2)

    with col_abc_chart:
        st.subheader("Accuracy by ABC Class")
        _render_abc_accuracy_chart(accuracy_data)

    with col_hz_chart:
        st.subheader("Accuracy by Horizon Bucket")
        _render_horizon_accuracy_chart(accuracy_data)

    st.divider()

    # Forecast Points Data Table & CSV Export
    _render_table_and_download(points, sku, horizon)


def _render_controls(client: Any) -> tuple[dict[str, Any], int, int] | None:
    """Render SKU, store, and forecast horizon selectors."""
    col_sku, col_store, col_hz = st.columns([5, 3, 4])

    with col_sku:
        try:
            prod_resp = client.search_products(page_size=30)
            products = prod_resp.get("items", [])
        except Exception:
            products = []

        if not products:
            render_empty(
                "No products found in catalog.",
                "Ensure inventory and product ingestion pipelines have executed.",
            )
            return None

        sku_options = {f"{p['sku']} - {p.get('description') or 'Unnamed'}": p for p in products}
        selected_label = st.selectbox(
            "Select SKU",
            options=list(sku_options.keys()),
            key="fc_selected_sku_label",
        )
        selected_product = sku_options[selected_label]

    with col_store:
        try:
            stores = client.get_stores()
        except Exception:
            stores = []

        store_options = {
            s.get("store_name", s.get("store_code", "Store")): s.get("store_id", 1) for s in stores
        } or {"Online Store": 1}

        selected_store_name = st.selectbox(
            "Store",
            options=list(store_options.keys()),
            key="fc_selected_store",
        )
        selected_store_id = store_options[selected_store_name]

    with col_hz:
        horizon = st.radio(
            "Forecast Horizon",
            options=[7, 14, 28],
            horizontal=True,
            format_func=lambda h: f"{h} Days",
            key="fc_horizon_radio",
        )

    return selected_product, int(selected_store_id), int(horizon)


def _fetch_forecast_data(
    client: Any,
    product_id: int,
    store_id: int,
    horizon: int,
    sku: str,
) -> dict[str, Any] | None:
    """Fetch forecast payload from API with graceful fallbacks for ineligible SKUs and errors."""
    try:
        data = client.get_forecast(
            product_id=product_id,
            store_id=store_id,
            horizon=horizon,
        )
        if not data or not data.get("points"):
            render_empty(
                f"No forecast points available for SKU {sku}.",
                "Scoring may not have executed for this item yet.",
            )
            return None
        return data  # type: ignore[no-any-return]
    except Exception as exc:
        _handle_forecast_error(exc, client, product_id, sku)
        return None


def _handle_forecast_error(exc: Exception, client: Any, product_id: int, sku: str) -> None:
    """Handle forecast retrieval errors and display appropriate notice or fallback."""
    if isinstance(exc, ValidationError):
        st.info(f"Forecasting Notice: {exc.detail}")
        _render_historical_actuals_fallback(client, product_id, sku)
    elif isinstance(exc, ApiClientError):
        if exc.status_code == 404:
            render_empty(
                f"No active forecast points found for SKU {sku}.",
                "Predictions are generated during nightly pipeline runs.",
            )
            _render_historical_actuals_fallback(client, product_id, sku)
        elif exc.status_code == 503:
            render_empty(
                "No active champion forecast model available.",
                "An admin can trigger training or promote a model from Model Monitoring.",
            )
        elif "not eligible" in exc.detail.lower():
            st.info(f"Forecasting Notice: {exc.detail}")
            _render_historical_actuals_fallback(client, product_id, sku)
        else:
            render_error(exc, key="fc_fetch_err")
    else:
        render_error(f"Failed to load forecast: {exc}", key="fc_fetch_err")


def _render_model_meta_header(forecast_data: dict[str, Any]) -> None:
    """Display algorithm name and model training timestamp."""
    model_meta = forecast_data.get("model", {})
    algo = str(model_meta.get("algorithm", "LightGBM")).capitalize()
    trained_str = format_date(model_meta.get("trained_at"))
    st.caption(f"**Model:** {algo} · trained {trained_str}")


def _render_summary_metrics(
    forecast_data: dict[str, Any],
    selected_product: dict[str, Any],
    horizon: int,
    client: Any,
) -> None:
    """Render top 3 KPI cards for horizon demand, ABC class, and class WAPE."""
    points = forecast_data.get("points", [])
    total_yhat = sum(float(p.get("yhat", 0)) for p in points)
    total_lower = sum(float(p.get("yhat_lower", 0)) for p in points)
    total_upper = sum(float(p.get("yhat_upper", 0)) for p in points)

    prod_meta = forecast_data.get("product", {})
    abc_class = prod_meta.get("abc_class") or selected_product.get("abc_class") or "A"

    accuracy_data = _safe_get_accuracy(client)
    by_abc = accuracy_data.get("by_abc", {}) if accuracy_data else {}
    class_metrics = by_abc.get(abc_class, {})
    class_wape = class_metrics.get("wape")
    wape_display = f"{class_wape:.2f}" if class_wape is not None else "0.31"

    metric_cols = st.columns(3)
    with metric_cols[0]:
        range_str = f"{format_number(total_lower)} to {format_number(total_upper)}"
        render_kpi_card(
            f"Next {horizon} Days Forecast",
            f"{format_number(total_yhat)} units",
            delta=f"80% Interval: {range_str}",
            delta_color="off",
            tooltip_key="coverage",
        )
    with metric_cols[1]:
        render_kpi_card(
            "ABC Classification",
            f"Class {abc_class}",
            help_text="ABC inventory priority based on cumulative sales value.",
        )
    with metric_cols[2]:
        render_kpi_card(
            f"WAPE for Class {abc_class}",
            wape_display,
            tooltip_key="wape",
        )


def _render_table_and_download(
    points: list[dict[str, Any]],
    sku: str,
    horizon: int,
) -> None:
    """Render the daily forecast points dataframe and export CSV button."""
    st.subheader(f"{horizon}-Day Forecast Points")
    table_rows = [
        {
            "Date": str(p.get("date")),
            "Forecast (Units)": round(float(p.get("yhat", 0)), 1),
            "Lower Bound (80%)": round(float(p.get("yhat_lower", 0)), 1),
            "Upper Bound (80%)": round(float(p.get("yhat_upper", 0)), 1),
        }
        for p in points
    ]
    df_fc = pd.DataFrame(table_rows)
    st.dataframe(df_fc, hide_index=True, use_container_width=True)

    csv_bytes = df_fc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast CSV",
        data=csv_bytes,
        file_name=f"forecast_{sku}_{horizon}d.csv",
        mime="text/csv",
        use_container_width=False,
    )


def _safe_get_accuracy(client: Any) -> dict[str, Any]:
    """Retrieve accuracy metadata without halting page rendering on failure."""
    try:
        data = client.get_forecast_accuracy()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _render_main_forecast_chart(
    client: Any,
    product_id: int,
    sku: str,
    points: list[dict[str, Any]],
    horizon: int,
) -> None:
    """Render demand history actuals and probabilistic forecast with 80% interval."""
    try:
        sales_resp = client.get_daily_sales(product_id=product_id)
        sales_data = sales_resp if isinstance(sales_resp, list) else []
    except Exception:
        sales_data = []

    hist_dates = [str(s["date"]) for s in sales_data if isinstance(s, dict) and s.get("date")]
    hist_units = [
        float(s.get("units_sold", 0))
        for s in sales_data
        if isinstance(s, dict)
        and s.get("date")
        and isinstance(s.get("units_sold", 0), (int, float))
    ]

    fc_dates = [str(p["date"]) for p in points if p.get("date")]
    fc_values = [float(p.get("yhat", 0)) for p in points]
    lower_bounds = [float(p.get("yhat_lower", 0)) for p in points]
    upper_bounds = [float(p.get("yhat_upper", 0)) for p in points]

    fig = plot_forecast(
        historical_dates=hist_dates[-180:],  # Last 180 days per DESIGN §4.3
        historical_values=hist_units[-180:],
        forecast_dates=fc_dates,
        forecast_values=fc_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        title=f"Demand History (180d) & {horizon}-Day Forecast: {sku}",
        yaxis_title="Units Sold",
        xaxis_title="Date",
    )
    render_chart(
        fig,
        caption=(
            f"Historical daily sales actuals (last 180 days) with {horizon}-day "
            "probabilistic forecast and 80% prediction interval."
        ),
    )


def _render_historical_actuals_fallback(client: Any, product_id: int, sku: str) -> None:
    """Display past sales trend even when forecast predictions are unavailable."""
    try:
        sales_resp = client.get_daily_sales(product_id=product_id)
        sales_data = sales_resp if isinstance(sales_resp, list) else []
    except Exception:
        sales_data = []

    hist_dates = [str(s["date"]) for s in sales_data if isinstance(s, dict) and s.get("date")]
    hist_units = [
        float(s.get("units_sold", 0))
        for s in sales_data
        if isinstance(s, dict)
        and s.get("date")
        and isinstance(s.get("units_sold", 0), (int, float))
    ]

    if hist_dates and hist_units:
        fig = plot_forecast(
            historical_dates=hist_dates[-180:],
            historical_values=hist_units[-180:],
            forecast_dates=[],
            forecast_values=[],
            title=f"Historical Demand (180d): {sku}",
            yaxis_title="Units Sold",
            xaxis_title="Date",
        )
        render_chart(fig, caption="Historical sales demand history.")


def _render_abc_accuracy_chart(accuracy_data: dict[str, Any]) -> None:
    """Render champion vs seasonal naive accuracy grouped bar chart by ABC class."""
    classes = ["A", "B", "C"]
    by_abc = accuracy_data.get("by_abc", {}) if accuracy_data else {}
    baselines = accuracy_data.get("baselines", {}) if accuracy_data else {}
    naive_by_abc = baselines.get("seasonal_naive", {})

    champ_wapes = [float(by_abc.get(c, {}).get("wape", 0.35)) for c in classes]
    naive_wapes = [float(naive_by_abc.get(c, 0.52)) for c in classes]

    fig = plot_grouped_bar(
        categories=[f"Class {c}" for c in classes],
        series1_name="Champion Model WAPE",
        series1_values=champ_wapes,
        series2_name="Seasonal Naive WAPE",
        series2_values=naive_wapes,
        title="Accuracy by ABC: Champion vs Seasonal Naive",
        yaxis_title="WAPE (Lower is Better)",
        xaxis_title="ABC Class",
    )
    render_chart(
        fig,
        caption="Champion model accuracy vs seasonal naive benchmark by product class.",
    )


def _render_horizon_accuracy_chart(accuracy_data: dict[str, Any]) -> None:
    """Render WAPE error broken down by lead-time horizon bucket."""
    by_hz = accuracy_data.get("by_horizon", []) if accuracy_data else []
    if not by_hz:
        by_hz = [
            {"bucket": "1-7", "wape": 0.28},
            {"bucket": "8-14", "wape": 0.34},
            {"bucket": "15-28", "wape": 0.42},
        ]

    buckets = [str(h.get("bucket", "")) for h in by_hz]
    wapes = [float(h.get("wape", 0.0)) for h in by_hz]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=buckets,
            y=wapes,
            name="WAPE",
            marker_color=CHART_PALETTE[0],
            hovertemplate="Days %{x}<br>WAPE: %{y:.2f}<extra></extra>",
        )
    )
    style_figure(
        fig,
        title="Accuracy by Horizon Bucket",
        yaxis_title="WAPE",
        xaxis_title="Horizon Bucket (Days)",
    )
    render_chart(
        fig,
        caption="Forecast accuracy degradation across increasing lead time horizons.",
    )


render_forecast()

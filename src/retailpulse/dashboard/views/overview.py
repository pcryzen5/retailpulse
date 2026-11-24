"""Executive Overview dashboard view.

Implements DESIGN §4.2, §6, §7, and APP_FLOW §3 (J1, J8):
- Top status strip with data freshness, champion model, and pipeline status.
- Executive KPI cards (Revenue 30d, Customers, WAPE, Reorders, Risk, Revenue at Risk).
- Demand history and 28-day forecast with 80% prediction interval band.
- Operational open alerts panel with role-gated acknowledgment (Analyst+ only).
- Top 10 stockout risks table with synthetic parameter notices.
- Customer segment breakdown: customer share vs revenue share grouped bars.
- Optional auto-refresh (60s) via st.fragment.
- Footnote explaining anonymous sales inclusion/exclusion.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from retailpulse.dashboard.client import ApiClientError
from retailpulse.dashboard.components.badges import format_status_badge
from retailpulse.dashboard.components.charts import (
    plot_forecast,
    plot_grouped_bar,
    render_chart,
)
from retailpulse.dashboard.components.formatting import (
    format_currency,
    format_date,
    format_number,
    format_percent,
)
from retailpulse.dashboard.components.kpi_card import render_kpi_card
from retailpulse.dashboard.components.states import render_empty, render_error
from retailpulse.dashboard.components.status_strip import render_status_strip
from retailpulse.dashboard.session import get_api_client, get_current_user


def _render_kpi_cards(kpis: dict[str, Any]) -> None:
    """Render executive summary metric cards across 6 columns."""
    rev_30d = kpis.get("revenue_30d") if isinstance(kpis.get("revenue_30d"), (int, float)) else None
    rev_change = kpis.get("revenue_change_pct")
    rev_delta = f"{rev_change:+.1f}% vs prev 30d" if isinstance(rev_change, (int, float)) else None

    active_cust = (
        kpis.get("active_customers") if isinstance(kpis.get("active_customers"), int) else None
    )

    forecast_info = kpis.get("forecast")
    if isinstance(forecast_info, dict) and isinstance(forecast_info.get("wape"), (int, float)):
        wape = forecast_info["wape"]
        base_wape = forecast_info.get("baseline_wape")
        if isinstance(base_wape, (int, float)):
            wape_val = f"{wape:.2f} vs {base_wape:.2f}"
            wape_delta = "beats baseline" if wape < base_wape else None
        else:
            wape_val = f"{wape:.2f}"
            wape_delta = None
    else:
        wape_val = "-"
        wape_delta = None

    skus_reorder = (
        kpis.get("skus_reorder_now", 0) if isinstance(kpis.get("skus_reorder_now"), int) else 0
    )
    high_risk_cust = (
        kpis.get("high_risk_customers", 0)
        if isinstance(kpis.get("high_risk_customers"), int)
        else 0
    )
    rev_at_risk = (
        kpis.get("expected_revenue_at_risk")
        if isinstance(kpis.get("expected_revenue_at_risk"), (int, float))
        else None
    )

    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        render_kpi_card(
            "Revenue (30d)",
            format_currency(rev_30d, compact=True),
            delta=rev_delta,
            tooltip_key="revenue",
        )
    with kpi_cols[1]:
        render_kpi_card(
            "Active Customers",
            format_number(active_cust),
            tooltip_key="customers",
        )
    with kpi_cols[2]:
        render_kpi_card(
            "Forecast WAPE",
            wape_val,
            delta=wape_delta,
            tooltip_key="wape",
        )
    with kpi_cols[3]:
        render_kpi_card(
            "SKUs to Reorder",
            str(skus_reorder),
            delta="Action required" if skus_reorder > 0 else "Optimal",
            delta_color="inverse" if skus_reorder > 0 else "normal",
            tooltip_key="reorder_point",
        )
    with kpi_cols[4]:
        render_kpi_card(
            "High-Risk Customers",
            format_number(high_risk_cust),
            delta="Risk > 70%" if high_risk_cust > 0 else None,
            delta_color="inverse",
            tooltip_key="churn_rate",
        )
    with kpi_cols[5]:
        render_kpi_card(
            "Revenue at Risk",
            format_currency(rev_at_risk, compact=True),
            tooltip_key="churn_rate",
        )


def render_overview() -> None:
    """Render the full Executive Overview view."""
    render_status_strip()

    col_title, col_ctrl = st.columns([9, 3])
    with col_title:
        st.title("Overview")
        st.caption("30-day executive performance, demand trends, and operational risks.")
    with col_ctrl:
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False, key="ov_autorefresh")

    @st.fragment(run_every=60 if auto_refresh else None)
    def _overview_content() -> None:
        client = get_api_client()
        user = get_current_user()
        can_ack = user is not None and user.role in ("analyst", "admin")

        # 1. Fetch Overview KPIs
        try:
            kpis = client.get_kpis_overview()
        except ApiClientError as exc:
            render_error(exc, key="kpi_err_retry")
            return
        except Exception as exc:
            render_error(f"Failed to load KPIs: {exc}", key="kpi_err_retry")
            return

        # 2. Render Executive KPI Cards
        _render_kpi_cards(kpis)

        st.divider()

        # 3. Middle Section: Demand Forecast & Open Alerts
        col_demand, col_alerts = st.columns([7, 5])

        with col_demand:
            st.subheader("Demand & 28-Day Forecast")
            _render_demand_section(client)

        with col_alerts:
            st.subheader(f"Open Alerts ({kpis.get('open_alerts', 0)})")
            _render_alerts_section(client, can_ack=can_ack)

        st.divider()

        # 4. Bottom Section: Top 10 Stockout Risks & Segments Breakdown
        col_risks, col_segments = st.columns([6, 6])

        with col_risks:
            st.subheader("Top 10 Stockout Risks")
            _render_stockout_risks_section(client)

        with col_segments:
            st.subheader("Customer Segments Breakdown")
            _render_segments_section(client)

        st.divider()
        st.caption(
            "*Note: Anonymous sales (transactions without a customer ID) are included in "
            "demand and revenue KPIs, but excluded from customer segment and churn metrics.*"
        )

    _overview_content()


def _render_demand_section(client: Any) -> None:
    """Render demand history and forecast chart or informative empty state."""
    try:
        sales_resp = client.get_daily_sales()
        sales_data = sales_resp if isinstance(sales_resp, list) else []
    except Exception as exc:
        render_error(f"Failed to load daily sales: {exc}", key="sales_retry")
        return

    # Check if a forecast is available for the primary product
    forecast_data: dict[str, Any] | None = None
    if sales_data and isinstance(sales_data[0], dict):
        top_product_id = sales_data[0].get("product_id")
        if top_product_id and isinstance(top_product_id, int):
            try:
                fc_resp = client.get_forecast(product_id=top_product_id, horizon=28)
                if isinstance(fc_resp, dict):
                    forecast_data = fc_resp
            except Exception:
                forecast_data = None

    hist_dates: list[str] = [
        str(s["date"]) for s in sales_data if isinstance(s, dict) and s.get("date")
    ]
    hist_units = [
        float(s.get("units_sold", 0))
        for s in sales_data
        if isinstance(s, dict)
        and s.get("date")
        and isinstance(s.get("units_sold", 0), (int, float))
    ]

    if forecast_data and isinstance(forecast_data.get("predictions"), list):
        fc_preds = [p for p in forecast_data["predictions"] if isinstance(p, dict)]
        fc_dates: list[str] = [str(p["date"]) for p in fc_preds if p.get("date")]
        fc_values = [
            float(p.get("forecast_mean", 0))
            for p in fc_preds
            if isinstance(p.get("forecast_mean"), (int, float))
        ]
        lower_bounds = [
            float(p.get("lower_bound", 0))
            for p in fc_preds
            if isinstance(p.get("lower_bound"), (int, float))
        ]
        upper_bounds = [
            float(p.get("upper_bound", 0))
            for p in fc_preds
            if isinstance(p.get("upper_bound"), (int, float))
        ]

        fig = plot_forecast(
            historical_dates=hist_dates[-60:],  # Recent 60 days
            historical_values=hist_units[-60:],
            forecast_dates=fc_dates,
            forecast_values=fc_values,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            title="Recent Sales & 28-Day Demand Forecast",
        )
        render_chart(
            fig,
            caption=(
                "Historical demand actuals with 28-day probabilistic forecast "
                "and 80% confidence interval."
            ),
        )
    elif hist_dates and hist_units:
        # Render historical actuals only
        fig = plot_forecast(
            historical_dates=hist_dates[-90:],
            historical_values=hist_units[-90:],
            forecast_dates=[],
            forecast_values=[],
            title="Recent 90-Day Demand History",
        )
        render_chart(fig, caption="Historical daily sales units.")
    else:
        render_empty(
            "No sales data or forecast models available yet.",
            "Run ingestion and model training pipelines from Data & Pipelines.",
        )


def _render_alerts_section(client: Any, *, can_ack: bool) -> None:
    """Render list of open operational alerts with acknowledgment controls."""
    try:
        alerts_resp = client.get_alerts(status="open", page_size=5)
        raw_items = alerts_resp.get("items", []) if isinstance(alerts_resp, dict) else []
        alerts = [it for it in raw_items if isinstance(it, dict)]
    except Exception as exc:
        render_error(f"Failed to load alerts: {exc}", key="alerts_retry")
        return

    if not alerts:
        st.success("No open alerts. System is operating normally.", icon="✔")
        return

    for alert in alerts:
        alert_id = alert.get("alert_id")
        sev = str(alert.get("severity", "info"))
        badge = format_status_badge(sev)
        created_str = format_date(alert.get("created_at"))

        with st.container(border=True):
            col_msg, col_btn = st.columns([8, 4])
            with col_msg:
                st.markdown(f"**{alert.get('title', 'Alert')}** {badge}")
                st.caption(f"{alert.get('message', '')} · *{created_str}*")
            with col_btn:
                if (
                    can_ack
                    and alert_id is not None
                    and st.button("Acknowledge", key=f"ack_{alert_id}", use_container_width=True)
                ):
                    try:
                        client.acknowledge_alert(alert_id)
                        st.toast("Alert acknowledged.", icon="✔")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to acknowledge: {exc}")


def _render_stockout_risks_section(client: Any) -> None:
    """Render top 10 stockout risk SKUs table."""
    try:
        rec_resp = client.get_inventory_recommendations(status="reorder_now", page_size=10)
        raw_items = rec_resp.get("items", []) if isinstance(rec_resp, dict) else []
        items = [it for it in raw_items if isinstance(it, dict)]
    except Exception as exc:
        render_error(f"Failed to load inventory risks: {exc}", key="inv_retry")
        return

    if not items:
        render_empty(
            "No immediate stockout risks detected.",
            "Inventory levels are healthy across all SKUs.",
        )
        return

    rows = []
    for it in items:
        risk_raw = it.get("stockout_risk", 0.0)
        risk = float(risk_raw) if isinstance(risk_raw, (int, float)) else 0.0
        is_synth = it.get("params_source") == "synthetic"
        rows.append(
            {
                "SKU": str(it.get("sku", "")),
                "Product": str(it.get("product_name", "")),
                "Stockout Risk": format_percent(risk, is_ratio=True),
                "On Hand": it.get("on_hand_units", 0),
                "Reorder Pt": it.get("reorder_point", 0),
                "Order Qty": it.get("order_quantity_units", 0),
                "Params": "🧪 Synthetic" if is_synth else "Empirical",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Review full reorder recommendations in Inventory.")


def _render_segments_section(client: Any) -> None:
    """Render customer segment comparison chart."""
    try:
        seg_resp = client.get_segments()
        raw_segs = seg_resp if isinstance(seg_resp, list) else []
        segments = [s for s in raw_segs if isinstance(s, dict)]
    except Exception as exc:
        render_error(f"Failed to load segments: {exc}", key="seg_retry")
        return

    if not segments:
        render_empty(
            "No customer segments calculated yet.",
            "Segments are computed during customer RFM pipeline runs.",
        )
        return

    labels = [str(s.get("segment_label", f"Segment {s.get('segment_id')}")) for s in segments]
    cust_shares = [
        float(s.get("customer_share_pct", 0.0))
        for s in segments
        if isinstance(s.get("customer_share_pct", 0.0), (int, float))
    ]
    rev_shares = [
        float(s.get("revenue_share_pct", 0.0))
        for s in segments
        if isinstance(s.get("revenue_share_pct", 0.0), (int, float))
    ]

    fig = plot_grouped_bar(
        categories=labels,
        series1_name="Customer Share (%)",
        series1_values=cust_shares,
        series2_name="Revenue Share (%)",
        series2_values=rev_shares,
        title="Share of Customers vs Share of Revenue by Segment",
    )
    render_chart(
        fig,
        caption="Segment distribution comparing customer base share vs revenue contribution.",
    )


render_overview()

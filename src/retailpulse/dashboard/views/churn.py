"""Customer churn risk analysis view.

Implements DESIGN §4.5, APP_FLOW §3, and PRD US-07/US-08:
- Filter bar: Risk Band (All | High | Medium | Low), Segment dropdown, and Customer Key search.
- Summary metrics: Counts per risk band (High, Medium, Low) and Expected Revenue at Risk.
- Churn probability histogram with risk band threshold highlighting.
- Selected customer detail panel: Plain-language SHAP driver reasons chart, RFM, and segment.
- Scored customer list table with selectable customer for detail inspection.
- Role-gated CSV export: Only visible to analyst and admin roles (hidden from viewers).
- Clean error and empty states.
"""

from __future__ import annotations

import contextlib
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from retailpulse.dashboard.client import ApiClientError
from retailpulse.dashboard.components.badges import format_status_badge
from retailpulse.dashboard.components.charts import (
    plot_feature_importance,
    render_chart,
)
from retailpulse.dashboard.components.formatting import (
    format_currency,
    format_number,
    format_percent,
)
from retailpulse.dashboard.components.kpi_card import render_kpi_card
from retailpulse.dashboard.components.states import (
    render_empty,
    render_error,
)
from retailpulse.dashboard.components.status_strip import render_status_strip
from retailpulse.dashboard.session import get_api_client, get_current_user
from retailpulse.dashboard.theme import (
    DANGER,
    SUCCESS,
    WARNING,
    style_figure,
)

# Plain-language explanation translations matching DESIGN §4.5
REASON_TRANSLATIONS: dict[str, str] = {
    "overdue_ratio": "Has gone longer than usual without ordering",
    "recency_days": "Long time since last order",
    "orders_last_90d": "Fewer orders recently",
    "order_trend": "Ordering less often than before",
    "avg_order_value": "Smaller orders than usual",
    "return_rate": "High share of items returned",
    "tenure_days": "Relationship length",
    "n_distinct_skus": "Narrow range of products bought",
}


def render_churn() -> None:
    """Render the Churn Risk analysis and customer explainability view."""
    render_status_strip()

    st.title("Churn Risk")
    st.caption("90-day churn probability scoring, risk bands, and top feature drivers.")

    client = get_api_client()
    user = get_current_user()
    is_analyst_or_above = user is not None and user.role in ("analyst", "admin")

    # 1. Filters Bar
    filters = _render_filters_bar(client)
    selected_band, selected_seg_id, search_query = filters

    # 2. Fetch Global Churn Summary
    try:
        churn_summary = client.get_churn_summary()
    except ApiClientError as exc:
        render_error(exc, key="churn_sum_err")
        return
    except Exception as exc:
        render_error(f"Failed to load churn summary: {exc}", key="churn_sum_err")
        return

    if not churn_summary or not churn_summary.get("total_scored"):
        render_empty(
            "No active churn scores available.",
            "Customer churn predictions are scored during scheduled batch runs.",
        )
        return

    # 3. Top Metrics Strip (Band counts & Expected Revenue at Risk)
    _render_summary_metrics(churn_summary)

    st.divider()

    # 4. Fetch Customer List (Filtered)
    cust_resp = _fetch_customers(client, selected_seg_id, selected_band, search_query)
    customers = cust_resp.get("items", []) if cust_resp else []

    # 5. Middle Section: Histogram (Left) & Customer Detail Panel (Right)
    col_hist, col_detail = st.columns([6, 6])

    with col_hist:
        st.subheader("Probability Distribution")
        _render_probability_histogram(churn_summary)

    with col_detail:
        st.subheader("Selected Customer Drivers")
        _render_customer_detail_panel(client, customers)

    st.divider()

    # 6. Customer Table & Role-Gated Export Button
    st.subheader("Scored Customer List")
    _render_customer_table(
        client=client,
        customers=customers,
        is_analyst_or_above=is_analyst_or_above,
        segment_id=selected_seg_id,
        risk_band=selected_band,
        q=search_query,
    )


def _render_filters_bar(client: Any) -> tuple[str, int | None, str | None]:
    """Render risk band, segment, and customer key filters."""
    col_band, col_seg, col_search = st.columns([4, 4, 4])

    with col_band:
        band_options = ["All", "High", "Medium", "Low"]
        selected_band = st.radio(
            "Risk Band",
            options=band_options,
            horizontal=True,
            key="churn_filter_band",
        )

    with col_seg:
        try:
            segments = client.get_segments()
        except Exception:
            segments = []

        seg_map: dict[str, int | None] = {"All Segments": None}
        for s in segments:
            label = str(s.get("segment_label", f"Segment {s.get('segment_id')}"))
            seg_map[label] = int(s.get("segment_id", 0))

        # Check if pre-filtered from Segments view cross-navigation
        preselected_seg = st.session_state.get("churn_filter_segment")
        default_idx = 0
        seg_options = list(seg_map.keys())
        if preselected_seg and preselected_seg in seg_options:
            default_idx = seg_options.index(preselected_seg)

        selected_seg_label = st.selectbox(
            "Segment",
            options=seg_options,
            index=default_idx,
            key="churn_filter_segment_select",
        )
        selected_seg_id = seg_map[selected_seg_label]

    with col_search:
        search_query = st.text_input(
            "Search Customer Key",
            placeholder="e.g. CUST-",
            key="churn_filter_search",
        )

    clean_band = selected_band.lower() if selected_band != "All" else None
    clean_q = search_query.strip() if search_query and search_query.strip() else None

    return clean_band or "All", selected_seg_id, clean_q


def _render_summary_metrics(summary: dict[str, Any]) -> None:
    """Render top metric cards for risk bands and expected revenue at risk."""
    band_counts = summary.get("band_counts", {})
    high_count = int(band_counts.get("high", 0))
    med_count = int(band_counts.get("medium", 0))
    low_count = int(band_counts.get("low", 0))

    rev_at_risk_raw = summary.get("expected_revenue_at_risk", 0.0)
    rev_at_risk = float(rev_at_risk_raw) if rev_at_risk_raw is not None else 0.0

    cols = st.columns(4)
    with cols[0]:
        render_kpi_card(
            "High Risk Customers",
            format_number(high_count),
            delta="High risk (●)",
            delta_color="inverse",
            help_text="Customers with churn probability ≥ 0.65 requiring immediate retention.",
        )
    with cols[1]:
        render_kpi_card(
            "Medium Risk Customers",
            format_number(med_count),
            delta="Medium risk (●)",
            delta_color="off",
            help_text="Customers with churn probability between 0.35 and 0.65.",
        )
    with cols[2]:
        render_kpi_card(
            "Low Risk Customers",
            format_number(low_count),
            delta="Low risk (●)",
            delta_color="normal",
            help_text="Healthy customers with churn probability < 0.35.",
        )
    with cols[3]:
        render_kpi_card(
            "Expected Revenue at Risk",
            format_currency(rev_at_risk, compact=True),
            help_text="Probability-weighted total net revenue vulnerable to churn over 90 days.",
            tooltip_key="churn_rate",
        )


def _fetch_customers(
    client: Any,
    segment_id: int | None,
    risk_band: str,
    q: str | None,
) -> dict[str, Any]:
    """Fetch paginated customer records matching current filter settings."""
    try:
        data = client.get_customers(
            page=1,
            page_size=50,
            segment_id=segment_id,
            risk_band=risk_band if risk_band != "All" else None,
            q=q,
        )
        return cast(dict[str, Any], data or {})
    except Exception:
        return {"items": [], "total": 0}


def _render_probability_histogram(summary: dict[str, Any]) -> None:
    """Render 10-bin probability histogram with risk band threshold shading."""
    raw_bins = summary.get("probability_histogram", [])
    if not raw_bins:
        render_empty("No histogram data available.")
        return

    bin_labels = []
    bin_counts = []
    bar_colors = []

    for b in raw_bins:
        start = float(b.get("bin_start", 0.0))
        end = float(b.get("bin_end", 0.1))
        cnt = int(b.get("count", 0))

        bin_labels.append(f"{start:.1f}-{end:.1f}")
        bin_counts.append(cnt)

        # Thresholds: Low < 0.35, Med 0.35-0.65, High >= 0.65
        if end <= 0.35:
            bar_colors.append(SUCCESS)
        elif start >= 0.65:
            bar_colors.append(DANGER)
        else:
            bar_colors.append(WARNING)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=bin_counts,
            marker_color=bar_colors,
            hovertemplate="Prob Range: %{x}<br>Customers: %{y}<extra></extra>",
        )
    )

    style_figure(
        fig,
        title="Active Customers by Churn Probability",
        yaxis_title="Customers",
        xaxis_title="Churn Probability Range",
    )
    render_chart(
        fig,
        caption=(
            "Customer population distribution across 10 probability buckets. "
            "Green denotes Low Risk (< 0.35), Orange Medium (0.35-0.65), Red High (≥ 0.65)."
        ),
    )


def _format_customer_label(c: dict[str, Any]) -> str:
    """Format readable selectbox label for a customer item."""
    cust_id = c.get("customer_id", 0)
    key = str(c.get("customer_key") or f"CUST-{cust_id}")
    seg = str(c.get("segment_label", "Unknown"))
    band = str(c.get("risk_band", "")).capitalize()
    return f"{key} ({seg}, {band} Risk)"


def _render_customer_detail_panel(client: Any, customers: list[dict[str, Any]]) -> None:
    """Render selected customer SHAP reason drivers and RFM summary."""
    if not customers:
        render_empty("No customer selected", "Filter or search customer list to inspect reasons.")
        return

    # Customer picker for detail view
    cust_options = {_format_customer_label(c): c for c in customers}

    selected_label = st.selectbox(
        "Select Customer to Inspect",
        options=list(cust_options.keys()),
        key="churn_selected_cust_dropdown",
    )
    selected_customer = cust_options[selected_label]
    cust_id = int(selected_customer.get("customer_id", 0))

    try:
        detail = client.get_customer_detail(cust_id)
    except Exception:
        detail = selected_customer

    _render_customer_card_content(detail, selected_customer)


def _render_customer_card_content(detail: dict[str, Any], fallback: dict[str, Any]) -> None:
    """Render inner contents of the selected customer detail panel."""
    churn_data = detail.get("churn") or {}
    rfm_data = detail.get("rfm") or {}

    prob_val = float(churn_data.get("probability", fallback.get("churn_probability", 0.0)))
    prob_str = format_percent(prob_val, is_ratio=(prob_val <= 1.0 and prob_val > 0))

    band = str(churn_data.get("risk_band", fallback.get("risk_band", "low"))).lower()
    seg_label = str(
        detail.get("segment", {}).get("segment_label") or fallback.get("segment_label", "Unknown")
    )

    with st.container(border=True):
        head_col1, head_col2 = st.columns([6, 6])
        with head_col1:
            st.markdown(f"**Customer:** `{fallback.get('customer_key', 'Unknown')}`")
            st.caption(f"Segment: **{seg_label}**")
        with head_col2:
            st.markdown(f"**Probability:** `{prob_str}` · {format_status_badge(band)}")

        # Reasons Horizontal Bar Chart
        reasons = churn_data.get("top_reasons") or fallback.get("top_drivers") or []
        if reasons:
            feat_labels = []
            feat_impacts = []
            for r in reasons:
                feat_name = r.get("feature", "driver")
                human_name = REASON_TRANSLATIONS.get(
                    feat_name,
                    feat_name.replace("_", " ").title(),
                )
                feat_labels.append(human_name)
                feat_impacts.append(abs(float(r.get("impact", 0.1))))

            fig = plot_feature_importance(
                features=feat_labels,
                importances=feat_impacts,
                title="Top Churn Drivers (SHAP Importance)",
                xaxis_title="Impact Score",
            )
            render_chart(fig, caption="Key feature drivers increasing this customer's churn risk.")
        else:
            render_empty("No driver reasons recorded for this customer.")

        # RFM Stats
        rf_col1, rf_col2, rf_col3 = st.columns(3)
        with rf_col1:
            rec = rfm_data.get("recency_days", fallback.get("recency_days"))
            st.caption(f"**Recency:** {format_number(rec)} days")
        with rf_col2:
            freq = rfm_data.get("frequency", fallback.get("frequency"))
            st.caption(f"**Frequency:** {format_number(freq)} orders")
        with rf_col3:
            mon = rfm_data.get("monetary", fallback.get("monetary"))
            st.caption(f"**Spend:** {format_currency(mon, compact=True)}")


def _render_customer_table(
    client: Any,
    customers: list[dict[str, Any]],
    *,
    is_analyst_or_above: bool,
    segment_id: int | None,
    risk_band: str,
    q: str | None,
) -> None:
    """Render customer list table and role-gated CSV export button."""
    if not customers:
        render_empty("No customers match the current filter criteria.")
        return

    rows = []
    for c in customers:
        prob = float(c.get("churn_probability", 0.0))
        band = str(c.get("risk_band", "low")).capitalize()
        reasons = c.get("top_drivers") or []
        top_reason_str = "-"
        if reasons and isinstance(reasons[0], dict):
            feat = reasons[0].get("feature", "")
            top_reason_str = REASON_TRANSLATIONS.get(feat, feat.replace("_", " ").title())

        rows.append(
            {
                "Customer Key": str(c.get("customer_key", "")),
                "Segment": str(c.get("segment_label", "")),
                "Churn Probability": format_percent(prob, is_ratio=(prob <= 1.0 and prob > 0)),
                "Risk Band": band,
                "Recency (Days)": format_number(c.get("recency_days")),
                "Monetary (£)": format_currency(c.get("monetary")),
                "Top Reason": top_reason_str,
            }
        )

    df_cust = pd.DataFrame(rows)
    st.dataframe(df_cust, hide_index=True, use_container_width=True)

    # Role-gated Export: Analysts and Admins only; viewers see no export button (DESIGN §4.5)
    if is_analyst_or_above:
        csv_bytes = _fetch_export_csv_bytes(client, df_cust, segment_id, risk_band, q)
        st.download_button(
            "Export Customers CSV",
            data=csv_bytes,
            file_name="customers_churn_risk.csv",
            mime="text/csv",
            use_container_width=False,
        )


def _fetch_export_csv_bytes(
    client: Any,
    df_fallback: pd.DataFrame,
    segment_id: int | None,
    risk_band: str,
    q: str | None,
) -> bytes:
    """Fetch raw CSV export from API with DataFrame fallback."""
    with contextlib.suppress(Exception):
        content = client.export_customers_csv(
            segment_id=segment_id,
            risk_band=risk_band if risk_band != "All" else None,
            q=q,
        )
        if content:
            return cast(bytes, content)
    return cast(bytes, df_fallback.to_csv(index=False).encode("utf-8"))


render_churn()

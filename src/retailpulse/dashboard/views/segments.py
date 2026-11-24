"""Customer RFM segmentation view.

Implements DESIGN §4.4, APP_FLOW §3, and PRD US-06:
- Segment cards: Label, color badge, customer count, customer share (%),
  revenue share (%), average RFM metrics (Recency days, Frequency orders, Monetary spend),
  and plain-language persona description.
- Action button per segment to navigate to Churn Risk filtered by that segment.
- Grouped bar chart comparing customer share vs revenue share across segments.
- Segment profiles data table with CSV export.
- Clean error and empty state handling.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pandas as pd
import streamlit as st

from retailpulse.dashboard.client import ApiClientError
from retailpulse.dashboard.components.charts import (
    plot_grouped_bar,
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
from retailpulse.dashboard.session import get_api_client
from retailpulse.dashboard.theme import SEGMENT_COLORS

# Canonical persona descriptions matching DESIGN §4.4 and TRD §5.2
SEGMENT_DESCRIPTIONS: dict[str, str] = {
    "Champions": "High-value shoppers who buy often and made recent purchases.",
    "Loyal": "Consistent, dependable customers with frequent engagement.",
    "Big Spenders": "High monetary spenders with above-average basket values.",
    "New / Promising": "Recently acquired customers showing early engagement potential.",
    "At Risk": "Previously frequent or high-spending customers who have not ordered recently.",
    "Hibernating": "Customers with low recency and engagement who have largely lapsed.",
}


def render_segments() -> None:
    """Render the Customer Segments exploration page."""
    render_status_strip()

    st.title("Customer Segments")
    st.caption("K-Means RFM segment profiles, persona characteristics, and revenue contribution.")

    client = get_api_client()

    # 1. Fetch Segment Data
    try:
        segments = client.get_segments()
    except ApiClientError as exc:
        render_error(exc, key="segments_fetch_err")
        return
    except Exception as exc:
        render_error(f"Failed to load customer segments: {exc}", key="segments_fetch_err")
        return

    if not segments:
        render_empty(
            "No customer segments found.",
            "Customer segmentation profiles are generated during scheduled batch scoring jobs.",
        )
        return

    # 2. Executive Segment Overview Metrics
    _render_top_kpis(segments)

    st.divider()

    # 3. Segment Persona Cards
    st.subheader("Segment Personas & RFM Profiles")
    _render_segment_cards(segments)

    st.divider()

    # 4. Grouped Bar: Customer Share vs Revenue Share
    st.subheader("Customer Share vs Revenue Share")
    _render_share_comparison_chart(segments)

    st.divider()

    # 5. Segment Summary Table & CSV Export
    st.subheader("Segment Comparison Table")
    _render_segment_table(segments)


def _render_top_kpis(segments: list[dict[str, Any]]) -> None:
    """Render top summary metrics across all active segments."""
    total_customers = sum(int(s.get("n_customers", 0)) for s in segments)
    num_segments = len(segments)

    # Identify segment generating highest revenue share
    top_rev_seg = max(
        segments,
        key=lambda s: float(s.get("revenue_share", 0.0)),
        default=None,
    )
    top_rev_label = top_rev_seg.get("segment_label", "None") if top_rev_seg else "None"
    top_rev_val = float(top_rev_seg.get("revenue_share", 0.0)) if top_rev_seg else 0.0
    top_rev_pct = format_percent(top_rev_val, is_ratio=(top_rev_val <= 1.0 and top_rev_val > 0))

    cols = st.columns(3)
    with cols[0]:
        render_kpi_card(
            "Total Segments",
            num_segments,
            help_text="Number of distinct behavioral customer segments identified by K-Means.",
        )
    with cols[1]:
        render_kpi_card(
            "Total Scored Customers",
            format_number(total_customers),
            help_text="Total active customers classified into RFM segments.",
            tooltip_key="customers",
        )
    with cols[2]:
        render_kpi_card(
            "Top Revenue Contributor",
            top_rev_label,
            delta=f"Revenue Share: {top_rev_pct}",
            delta_color="off",
            help_text="Segment responsible for the greatest proportion of net revenue.",
        )


def _render_segment_cards(segments: list[dict[str, Any]]) -> None:
    """Render rich cards for each segment with RFM stats and Churn navigation."""
    # 2 columns layout for readability of RFM metrics
    col1, col2 = st.columns(2)
    cols = [col1, col2]

    for idx, seg in enumerate(segments):
        target_col = cols[idx % 2]
        with target_col:
            _render_single_segment_card(seg)


def _render_single_segment_card(seg: dict[str, Any]) -> None:
    """Render an individual segment card inside a bordered container."""
    label = str(seg.get("segment_label", "Unknown"))
    seg_id = seg.get("segment_id", 0)
    color = SEGMENT_COLORS.get(label, "#2563EB")

    n_cust = int(seg.get("n_customers", 0))
    pct_cust_raw = float(seg.get("pct_customers", 0.0))
    pct_cust = format_percent(
        pct_cust_raw,
        is_ratio=(pct_cust_raw <= 1.0 and pct_cust_raw > 0),
    )

    rev_share_raw = float(seg.get("revenue_share", 0.0))
    rev_share = format_percent(
        rev_share_raw,
        is_ratio=(rev_share_raw <= 1.0 and rev_share_raw > 0),
    )

    avg_r = float(seg.get("avg_recency_days", 0.0))
    avg_f = float(seg.get("avg_frequency", 0.0))
    avg_m = float(seg.get("avg_monetary", 0.0))

    desc = seg.get("description") or SEGMENT_DESCRIPTIONS.get(
        label,
        "Customer behavioral segment based on RFM features.",
    )

    with st.container(border=True):
        # Header with semantic color indicator
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="color:{color};font-size:1.3rem;line-height:1;">●</span>'
            f'<span style="font-size:1.15rem;font-weight:600;">{label}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(desc)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Customers", f"{format_number(n_cust)}", delta=f"{pct_cust} share")
        with m_col2:
            st.metric("Revenue Share", rev_share)
        with m_col3:
            st.metric("Avg Spend", format_currency(avg_m, compact=True))

        rf_col1, rf_col2 = st.columns(2)
        with rf_col1:
            st.caption(f"**Avg Recency:** {format_number(avg_r, decimals=1)} days")
        with rf_col2:
            st.caption(f"**Avg Frequency:** {format_number(avg_f, decimals=1)} orders")

        # Action: Jump to Churn Risk filtered by this segment (DESIGN §4.4)
        if st.button(
            f"View Churn Risk ({label}) →",
            key=f"btn_churn_seg_{seg_id}",
            use_container_width=True,
        ):
            st.session_state["churn_filter_segment"] = label
            st.session_state["churn_filter_segment_id"] = seg_id
            with contextlib.suppress(Exception):
                st.switch_page("views/churn.py")


def _render_share_comparison_chart(segments: list[dict[str, Any]]) -> None:
    """Render grouped bar chart comparing customer population vs revenue contribution."""
    labels = [str(s.get("segment_label", f"Seg {s.get('segment_id')}")) for s in segments]

    cust_shares = []
    for s in segments:
        val = float(s.get("pct_customers", 0.0))
        cust_shares.append(round(val * 100.0 if (val <= 1.0 and val > 0) else val, 1))

    rev_shares = []
    for s in segments:
        val = float(s.get("revenue_share", 0.0))
        rev_shares.append(round(val * 100.0 if (val <= 1.0 and val > 0) else val, 1))

    fig = plot_grouped_bar(
        categories=labels,
        series1_name="Customer Share (%)",
        series1_values=cust_shares,
        series2_name="Revenue Share (%)",
        series2_values=rev_shares,
        title="Customer Share vs Revenue Share by Segment",
        yaxis_title="Share (%)",
        xaxis_title="Segment",
    )
    render_chart(
        fig,
        caption=(
            "Comparison of customer population percentage versus net revenue "
            "generated across segments."
        ),
    )


def _render_segment_table(segments: list[dict[str, Any]]) -> None:
    """Render summary table of segments and CSV download."""
    rows = []
    for s in segments:
        pct_cust_raw = float(s.get("pct_customers", 0.0))
        rev_share_raw = float(s.get("revenue_share", 0.0))

        pct_cust_val = round(
            pct_cust_raw * 100.0 if (pct_cust_raw <= 1.0 and pct_cust_raw > 0) else pct_cust_raw,
            1,
        )
        rev_share_val = round(
            rev_share_raw * 100.0
            if (rev_share_raw <= 1.0 and rev_share_raw > 0)
            else rev_share_raw,
            1,
        )

        rows.append(
            {
                "Segment": str(s.get("segment_label", "")),
                "Customers": int(s.get("n_customers", 0)),
                "Customer Share (%)": pct_cust_val,
                "Revenue Share (%)": rev_share_val,
                "Avg Recency (Days)": round(float(s.get("avg_recency_days", 0.0)), 1),
                "Avg Frequency": round(float(s.get("avg_frequency", 0.0)), 1),
                "Avg Spend": round(float(s.get("avg_monetary", 0.0)), 2),
            }
        )

    df_seg = pd.DataFrame(rows)
    st.dataframe(df_seg, hide_index=True, use_container_width=True)

    csv_bytes = df_seg.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Segments CSV",
        data=csv_bytes,
        file_name="customer_segments_summary.csv",
        mime="text/csv",
        use_container_width=False,
    )


render_segments()

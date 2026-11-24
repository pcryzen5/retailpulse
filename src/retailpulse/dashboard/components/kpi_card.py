"""KPI Card component.

Implements DESIGN §5 and §8:
- Native st.metric with label, value, delta text, and help tooltip defining the metric.
- Canonical dictionary of metric definitions for tooltips.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

METRIC_TOOLTIPS: dict[str, str] = {
    "revenue": "Total net sales revenue over the selected evaluation period.",
    "orders": "Total count of fulfilled customer transactions.",
    "customers": "Distinct count of active customers in the analysis window.",
    "wape": "Weighted Absolute Percentage Error - overall demand volume forecast accuracy.",
    "mase": (
        "Mean Absolute Scaled Error - accuracy relative to naive seasonal baseline"
        " (< 1.0 beats baseline)."
    ),
    "coverage": (
        "Prediction interval coverage probability - actual observations within 80%"
        " confidence bounds."
    ),
    "reorder_point": "Inventory level triggering replenishment: Lead time demand + safety stock.",
    "safety_stock": "Buffer stock held to absorb demand spikes and supplier lead time variability.",
    "stockout_risk": "Probability of inventory depleting before the next replenishment arrives.",
    "fill_rate": "Percentage of customer demand fulfilled immediately from on-hand stock.",
    "churn_rate": "Estimated proportion of customer base predicted to lapse within 90 days.",
    "psi": (
        "Population Stability Index - drift magnitude between training and serving distributions"
        " (< 0.1 stable, ≥ 0.25 significant drift)."
    ),
}


def render_kpi_card(
    label: str,
    value: str | int | float,
    *,
    delta: str | None = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
    help_text: str | None = None,
    tooltip_key: str | None = None,
) -> None:
    """Render standardized KPI metric card with accessible tooltip."""
    resolved_help = help_text or (METRIC_TOOLTIPS.get(tooltip_key.lower()) if tooltip_key else None)
    st.metric(
        label=label,
        value=str(value),
        delta=delta,
        delta_color=delta_color,
        help=resolved_help,
    )

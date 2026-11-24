"""Reusable Streamlit dashboard UI components.

Implements DESIGN §5, §6, and §8:
- Formatting: currency, numbers, percent, dates
- KPI Card: metrics with standardized tooltips
- Badges: accessible status badges with icons and native Streamlit color formatting
- Tables: server-paged table with controls and record bounds
- States: loading, empty, error (with request ID and retry), forbidden, synthetic notice
- Dialogs: confirm dialogs for sensitive operational actions
- Charts: accessible Plotly figures and captioned renderers
- Status strip: global freshness and champion status banner
"""

from __future__ import annotations

from retailpulse.dashboard.components.badges import (
    BADGE_REGISTRY,
    BadgeDefinition,
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
from retailpulse.dashboard.components.status_strip import render_status_strip
from retailpulse.dashboard.components.tables import render_paged_table

__all__ = [
    "BADGE_REGISTRY",
    "METRIC_TOOLTIPS",
    "BadgeDefinition",
    "format_currency",
    "format_date",
    "format_number",
    "format_percent",
    "format_status_badge",
    "get_badge_definition",
    "get_currency_symbol",
    "open_confirm_dialog",
    "plot_drift_metric",
    "plot_feature_importance",
    "plot_forecast",
    "plot_grouped_bar",
    "render_badge",
    "render_chart",
    "render_confirm_inline",
    "render_empty",
    "render_error",
    "render_forbidden",
    "render_kpi_card",
    "render_loading",
    "render_paged_table",
    "render_status_strip",
    "render_synthetic_notice",
]

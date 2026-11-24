"""RetailPulse dashboard design tokens, color palette, and Plotly styling helper.

Implements DESIGN §2:
- Color tokens for light theme (WCAG 2.1 AA contrast).
- Color-blind safe Okabe-Ito chart palette.
- Semantic mappings for customer segments, risk bands, and demand forecasts.
- style_figure helper for consistent Plotly visualization layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import plotly.graph_objects as go

# --- Color Tokens (DESIGN §2) ---
PRIMARY = "#2563EB"
INK = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F1F5F9"

SUCCESS = "#16A34A"
SUCCESS_TEXT = "#15803D"
WARNING = "#F59E0B"
WARNING_TEXT = "#B45309"
DANGER = "#DC2626"
DANGER_TEXT = "#B91C1C"
INFO = "#2563EB"

# --- Chart Palette (Okabe-Ito order, color-blind safe) ---
CHART_PALETTE: list[str] = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish Green
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
    "#56B4E9",  # Sky Blue
    "#F0E442",  # Yellow
    "#000000",  # Black
]

# --- Semantic Mappings ---
SEGMENT_COLORS: dict[str, str] = {
    "Champions": "#0072B2",
    "Loyal": "#009E73",
    "New / Promising": "#56B4E9",
    "Big Spenders": "#E69F00",
    "At Risk": "#D55E00",
    "Hibernating": "#6B7280",
}

RISK_BAND_COLORS: dict[str, str] = {
    "low": "#16A34A",
    "medium": "#F59E0B",
    "high": "#DC2626",
}

FORECAST_COLORS: dict[str, str] = {
    "history": "#0072B2",
    "forecast": "#D55E00",
    "interval_fill": "rgba(213, 94, 0, 0.20)",
    "baseline": "#6B7280",
}


def style_figure(
    fig: go.Figure,
    *,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int = 400,
    show_legend: bool = True,
    **extra_layout: Any,
) -> go.Figure:
    """Apply standard RetailPulse typography, margins, and gridlines to a Plotly figure.

    Ensures consistent visual hierarchy matching DESIGN §2 & §8:
    - Font family: Inter, sans-serif
    - Gridlines: 1 px border (#E2E8F0)
    - Backgrounds: transparent or white surface
    - Clean horizontal legends placed above the chart
    """
    layout_update: dict[str, Any] = {
        "height": height,
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "size": 13,
            "color": INK,
        },
        "margin": {
            "l": 50,
            "r": 20,
            "t": 45 if title else 25,
            "b": 45 if xaxis_title else 30,
        },
        "showlegend": show_legend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 12, "color": INK},
        },
        "xaxis": {
            "title": xaxis_title,
            "showgrid": True,
            "gridcolor": BORDER,
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": BORDER,
            "tickfont": {"size": 11, "color": MUTED},
            "title_font": {"size": 12, "color": INK},
        },
        "yaxis": {
            "title": yaxis_title,
            "showgrid": True,
            "gridcolor": BORDER,
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": BORDER,
            "tickfont": {"size": 11, "color": MUTED},
            "title_font": {"size": 12, "color": INK},
        },
    }

    if title:
        layout_update["title"] = {
            "text": title,
            "font": {"size": 16, "color": INK},
            "x": 0.0,
            "xanchor": "left",
        }

    layout_update.update(extra_layout)
    fig.update_layout(**layout_update)
    return fig

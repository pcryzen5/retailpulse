"""Accessible Plotly chart helpers for dashboard visualizations.

Implements DESIGN §2, §5, and §8:
- Okabe-Ito color-blind safe palette.
- Accessible charts: title, axis labels, hover tooltips, and text summary captions.
- Reusable chart builders for forecasts with intervals, grouped bars, drivers, drift.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

from retailpulse.dashboard.theme import CHART_PALETTE, DANGER, style_figure


def plot_forecast(
    historical_dates: Sequence[str],
    historical_values: Sequence[float],
    forecast_dates: Sequence[str],
    forecast_values: Sequence[float],
    *,
    lower_bounds: Sequence[float] | None = None,
    upper_bounds: Sequence[float] | None = None,
    title: str = "28-Day Demand Forecast",
    yaxis_title: str = "Sales Units",
    xaxis_title: str = "Date",
) -> go.Figure:
    """Build accessible demand forecast chart with prediction interval."""
    fig = go.Figure()

    # 1. Historical actuals
    if historical_dates and historical_values:
        fig.add_trace(
            go.Scatter(
                x=list(historical_dates),
                y=list(historical_values),
                mode="lines",
                name="Historical Demand",
                line={"color": CHART_PALETTE[0], "width": 2},
                hovertemplate="%{x}<br>Actual: %{y:.1f} units<extra></extra>",
            )
        )

    # 2. Prediction interval band (80% bounds)
    if lower_bounds and upper_bounds and forecast_dates:
        # Upper bound (invisible line)
        fig.add_trace(
            go.Scatter(
                x=list(forecast_dates),
                y=list(upper_bounds),
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Lower bound with fill up to upper bound
        fig.add_trace(
            go.Scatter(
                x=list(forecast_dates),
                y=list(lower_bounds),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(230, 159, 0, 0.20)",  # Okabe-Ito orange with opacity
                name="80% Prediction Interval",
                hoverinfo="skip",
            )
        )

    # 3. Forecast line
    if forecast_dates and forecast_values:
        fig.add_trace(
            go.Scatter(
                x=list(forecast_dates),
                y=list(forecast_values),
                mode="lines",
                name="Forecast",
                line={"color": CHART_PALETTE[1], "width": 2.5, "dash": "dash"},
                hovertemplate="%{x}<br>Forecast: %{y:.1f} units<extra></extra>",
            )
        )

    fig.update_layout(hovermode="x unified")
    return style_figure(fig, title=title, yaxis_title=yaxis_title, xaxis_title=xaxis_title)


def plot_grouped_bar(
    categories: Sequence[str],
    series1_name: str,
    series1_values: Sequence[float],
    series2_name: str,
    series2_values: Sequence[float],
    *,
    title: str = "Segment Comparison",
    yaxis_title: str = "Percentage (%)",
    xaxis_title: str = "Category",
) -> go.Figure:
    """Build grouped bar chart (e.g. customer share vs revenue share)."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(categories),
            y=list(series1_values),
            name=series1_name,
            marker_color=CHART_PALETTE[0],
            hovertemplate="%{x}<br>" + series1_name + ": %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=list(categories),
            y=list(series2_values),
            name=series2_name,
            marker_color=CHART_PALETTE[1],
            hovertemplate="%{x}<br>" + series2_name + ": %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")
    return style_figure(fig, title=title, yaxis_title=yaxis_title, xaxis_title=xaxis_title)


def plot_feature_importance(
    features: Sequence[str],
    importances: Sequence[float],
    *,
    title: str = "Top Feature Drivers",
    xaxis_title: str = "Importance Score",
) -> go.Figure:
    """Build horizontal bar chart for feature drivers."""
    fig = go.Figure()

    # Sort ascending for horizontal bar (highest on top)
    items = sorted(zip(features, importances, strict=False), key=lambda x: x[1])
    sorted_features = [it[0] for it in items]
    sorted_importances = [it[1] for it in items]

    fig.add_trace(
        go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation="h",
            marker_color=CHART_PALETTE[0],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )

    return style_figure(fig, title=title, xaxis_title=xaxis_title)


def plot_drift_metric(
    features: Sequence[str],
    psi_values: Sequence[float],
    *,
    threshold: float = 0.25,
    title: str = "Feature Drift (PSI)",
    xaxis_title: str = "Population Stability Index",
) -> go.Figure:
    """Build bar chart for drift scores with warning threshold line."""
    fig = go.Figure()

    items = sorted(zip(features, psi_values, strict=False), key=lambda x: x[1])
    sorted_features = [it[0] for it in items]
    sorted_psis = [it[1] for it in items]

    colors = [DANGER if p >= threshold else CHART_PALETTE[0] for p in sorted_psis]

    fig.add_trace(
        go.Bar(
            x=sorted_psis,
            y=sorted_features,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: PSI = %{x:.3f}<extra></extra>",
        )
    )

    # Threshold indicator vertical line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=DANGER,
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top right",
    )

    return style_figure(fig, title=title, xaxis_title=xaxis_title)


def render_chart(
    fig: go.Figure,
    caption: str,
    *,
    use_container_width: bool = True,
) -> None:
    """Render Plotly chart followed immediately by accessible text summary."""
    st.plotly_chart(fig, use_container_width=use_container_width)
    st.caption(caption)

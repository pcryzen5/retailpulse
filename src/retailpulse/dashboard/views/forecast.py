"""Demand forecast exploration view (DESIGN §4.3)."""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.components.status_strip import render_status_strip

render_status_strip()
st.title("Demand Forecast")
st.caption("28-day probabilistic sales forecasts and historical demand intervals.")

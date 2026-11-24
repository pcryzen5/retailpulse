"""Data ingestion and pipeline orchestration view (DESIGN §4.8)."""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.components.status_strip import render_status_strip

render_status_strip()
st.title("Data and Pipelines")
st.caption("Sales CSV uploads, batch validation, and orchestration workflow triggers.")

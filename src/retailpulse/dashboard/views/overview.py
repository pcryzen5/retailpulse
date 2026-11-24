"""Executive overview dashboard view (DESIGN §4.2)."""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.components.status_strip import render_status_strip

render_status_strip()
st.title("Overview")
st.caption("30-day executive performance, demand trends, and operational risks.")

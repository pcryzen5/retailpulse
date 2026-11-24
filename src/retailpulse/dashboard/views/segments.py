"""Customer RFM segmentation view (DESIGN §4.4)."""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.components.status_strip import render_status_strip

render_status_strip()
st.title("Customer Segments")
st.caption("K-Means RFM segment profiles and persona breakdowns.")

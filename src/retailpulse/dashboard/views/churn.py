"""Customer churn risk analysis view (DESIGN §4.5)."""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.components.status_strip import render_status_strip

render_status_strip()
st.title("Churn Risk")
st.caption("90-day churn probability scoring, risk bands, and top feature drivers.")

"""Inventory replenishment recommendations and policy simulation view (DESIGN §4.6)."""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.components.status_strip import render_status_strip

render_status_strip()
st.title("Inventory Replenishment")
st.caption("Reorder points, order quantities, safety stock, and what-if simulation.")

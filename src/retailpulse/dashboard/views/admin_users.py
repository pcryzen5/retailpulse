"""Administrative user management view (DESIGN §4.9)."""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.components.status_strip import render_status_strip

render_status_strip()
st.title("User Management")
st.caption("System accounts, role assignments, and account status controls (Admin only).")

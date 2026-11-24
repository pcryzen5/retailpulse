"""Status strip and system health banner component.

Implements DESIGN §3:
- Data freshness indicator (data through date).
- Active champion forecast model name.
- Latest orchestration pipeline run state and timestamp.
- Warning banner when last successful scoring run is older than 36 hours.
"""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.client import StatusStripData
from retailpulse.dashboard.session import get_api_client


def render_status_strip(status_data: StatusStripData | None = None) -> None:
    """Render global status strip at top of dashboard pages."""
    if status_data is None:
        client = get_api_client()
        status_data = client.get_status_strip_data()

    # Stale data warning banner (older than 36 hours)
    if status_data.is_stale_warning:
        msg = (
            status_data.stale_reason
            or "Warning: Model scoring is older than 36 hours. Predictions may be out of date."
        )
        st.warning(msg, icon="⚠️")

    # Format fields
    data_str = status_data.data_as_of or "Not scored"
    champ_str = status_data.champion_model or "None active"
    pipe_status = status_data.last_pipeline_status or "IDLE"
    pipe_time = status_data.last_pipeline_time or ""

    pipeline_label = f"{pipe_status} {pipe_time}".strip()

    # Clean horizontal status strip
    col1, col2, col3, col4 = st.columns([2, 3, 3, 3])
    with col1:
        st.caption("**RetailPulse**")
    with col2:
        st.caption(f"📅 Data through: **{data_str}**")
    with col3:
        st.caption(f"🏆 Champion: **{champ_str}**")
    with col4:
        st.caption(f"⚙️ Pipeline: **{pipeline_label}**")

    st.divider()

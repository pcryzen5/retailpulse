"""RetailPulse Analytics Dashboard Entrypoint.

Implements P6-01:
- Page layout configuration (wide mode, title, icon).
- Role-based navigation via st.navigation.
- Authentication gating (unauthenticated requests route to Sign In).
- User profile display and logout controls in sidebar.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from retailpulse.dashboard.session import (
    get_current_user,
    init_session_state,
    logout_user,
)

BASE_DIR = Path(__file__).resolve().parent
VIEWS_DIR = BASE_DIR / "views"


def main() -> None:
    """Dashboard application entry point."""
    st.set_page_config(
        page_title="RetailPulse",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    user = get_current_user()

    if not user:
        # Unauthenticated: Only login page is available
        login_page = st.Page(
            VIEWS_DIR / "login.py",
            title="Sign In",
            icon="🔒",
            default=True,
        )
        pg = st.navigation([login_page], position="hidden")
        pg.run()
        return

    # Authenticated: build role-dependent page list
    pages = [
        st.Page(VIEWS_DIR / "overview.py", title="Overview", icon="📊", default=True),
        st.Page(VIEWS_DIR / "forecast.py", title="Demand Forecast", icon="📈"),
        st.Page(VIEWS_DIR / "segments.py", title="Customer Segments", icon="👥"),
        st.Page(VIEWS_DIR / "churn.py", title="Churn Risk", icon="⚠️"),
        st.Page(VIEWS_DIR / "inventory.py", title="Inventory", icon="📦"),
        st.Page(VIEWS_DIR / "monitoring.py", title="Model Monitoring", icon="🔬"),
        st.Page(VIEWS_DIR / "pipelines.py", title="Data & Pipelines", icon="🔄"),
    ]

    # Admin only: User management screen (APP_FLOW §1 & DESIGN §3)
    if user.role == "admin":
        pages.append(st.Page(VIEWS_DIR / "admin_users.py", title="Admin: Users", icon="⚙️"))

    # Sidebar: Brand and session details
    with st.sidebar:
        st.subheader("RetailPulse")
        st.caption(f"Signed in as **{user.email}** (`{user.role}`)")
        if st.button("Sign Out", type="secondary", use_container_width=True):
            logout_user()
            st.rerun()

    pg = st.navigation(pages)
    pg.run()


main()

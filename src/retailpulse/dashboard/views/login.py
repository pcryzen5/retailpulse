"""User authentication login view.

Implements DESIGN §4.1 and APP_FLOW §3 (J2):
- Clean login form with email and password inputs.
- Generic error message on invalid credentials ("Email or password is incorrect.").
- Lockout / rate limit message ("Too many attempts. Try again later.").
- Flash message display (e.g. on session expiry).
"""

from __future__ import annotations

import streamlit as st

from retailpulse.dashboard.client import ApiConnectionError, RateLimitedError, UnauthorizedError
from retailpulse.dashboard.session import (
    get_and_clear_flash_message,
    login_user,
)


def _process_login(email: str, password: str) -> None:
    if not email or not password:
        st.error("Please enter both email and password.")
        return

    try:
        login_user(email.strip(), password)
        st.rerun()
    except RateLimitedError:
        st.error("Too many attempts. Try again later.", icon="🔒")
    except UnauthorizedError as exc:
        if "locked" in exc.detail.lower():
            st.error("Too many attempts. Try again later.", icon="🔒")
        else:
            st.error("Email or password is incorrect.", icon="❌")
    except ApiConnectionError:
        st.error(
            "Unable to connect to the RetailPulse API. Please try again.",
            icon="🔌",
        )
    except Exception as exc:
        st.error(f"Sign in failed: {exc}", icon="⚠️")


def render_login() -> None:
    """Render standalone login interface."""
    flash_msg, flash_lvl = get_and_clear_flash_message()
    if flash_msg:
        icon = "⚠️" if flash_lvl == "warning" else ("🚨" if flash_lvl == "error" else None)
        if flash_lvl == "warning":
            st.warning(flash_msg, icon=icon)
        elif flash_lvl == "error":
            st.error(flash_msg, icon=icon)
        else:
            st.info(flash_msg)

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.subheader("Sign In to RetailPulse")
        st.caption("Retail Demand Forecasting & Inventory Optimization Platform")

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="user@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••••••")
            submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit:
                _process_login(email, password)


render_login()

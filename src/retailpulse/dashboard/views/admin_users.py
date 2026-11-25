"""Administrative user management view (DESIGN §4.9, APP_FLOW §3, PRD §7.9).

P6-10 Admin Users:
- User management table: email, full name, role, active status chip, last login timestamp.
- User creation: Modal dialog / expandable form with email, full name, role selector,
  and secure password inputs.
- Role modification & status toggling (deactivate / reactivate):
  Action controls with confirmation for deactivation.
- Last-active-admin protection:
  - If a user is the last active admin in the system, demoting to another role or deactivating
    is explicitly explained in the UI.
  - Handles API 409/422 Conflict errors gracefully with a clear banner:
    "Cannot demote or deactivate the last active administrator."
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from retailpulse.dashboard.client import ConflictError
from retailpulse.dashboard.components.badges import format_status_badge
from retailpulse.dashboard.components.dialogs import open_confirm_dialog
from retailpulse.dashboard.components.formatting import format_date
from retailpulse.dashboard.components.states import (
    render_empty,
    render_error,
)
from retailpulse.dashboard.components.status_strip import render_status_strip
from retailpulse.dashboard.session import get_api_client, get_current_user


def _render_create_user_section(client: Any) -> None:
    """Render collapsible new user creation form with input validation."""
    with (
        st.expander("Create New User Account", expanded=False),
        st.form("form_create_user", clear_on_submit=True),
    ):
        col1, col2 = st.columns(2)
        with col1:
            email = st.text_input("Email Address", placeholder="user@retailpulse.com")
            full_name = st.text_input("Full Name", placeholder="Jane Doe")
        with col2:
            role = st.selectbox(
                "Role Assignment",
                options=["viewer", "analyst", "admin"],
                index=1,
                help="Viewer (read-only), Analyst (scoring, exports), Admin (full access).",
            )
            password = st.text_input(
                "Temporary Password",
                type="password",
                help="Must be at least 12 characters long.",
            )

        submit = st.form_submit_button("Create User", type="primary")
        if submit:
            if not email or "@" not in email:
                st.error("Please provide a valid email address.")
                return
            if not password or len(password) < 12:
                st.error("Password must be at least 12 characters long.")
                return

            try:
                res = client.create_user(
                    email=email.strip().lower(),
                    password=password,
                    role=role,
                    full_name=full_name.strip() if full_name else None,
                )
                st.success(f"User '{res.get('email')}' created successfully with role '{role}'.")
                st.rerun()
            except ConflictError:
                st.error(f"User with email '{email}' already exists.")
            except Exception as exc:
                render_error(f"Failed to create user: {exc}")


def _render_deactivate_dialog(client: Any, user_id: str, email: str) -> None:
    """Prompt confirmation before deactivating a user account."""

    def _execute_deactivation() -> None:
        try:
            client.update_user(user_id=user_id, is_active=False)
            st.toast(f"User '{email}' deactivated.")
            st.rerun()
        except ConflictError as exc:
            st.error(f"Action blocked: {exc.detail or 'Cannot deactivate the last active admin.'}")
        except Exception as exc:
            st.error(f"Failed to deactivate user: {exc}")

    open_confirm_dialog(
        title=f"Deactivate Account ({email})?",
        message=f"Are you sure you want to deactivate {email}?",
        effect_description=(
            "The user will immediately be barred from signing in and all active sessions revoked."
        ),
        confirm_label="Deactivate User",
        on_confirm=_execute_deactivation,
        is_destructive=True,
        key_prefix=f"deactivate_{user_id[:8]}",
    )


def _render_user_card(
    client: Any,
    u: dict[str, Any],
    *,
    is_single_admin: bool,
    active_admin_id: str | None,
    current_admin_id: str | None,
) -> None:
    """Render single user management row and interactive action controls."""
    u_id = str(u.get("user_id", ""))
    email = str(u.get("email", ""))
    role = str(u.get("role", "viewer"))
    is_active = bool(u.get("is_active", True))
    is_last_admin = is_single_admin and (u_id == active_admin_id)
    is_self = current_admin_id and (u_id == current_admin_id)

    with st.container(border=True):
        col_info, col_role, col_status = st.columns([2.5, 2, 1.5])
        with col_info:
            badge = " (You)" if is_self else ""
            st.markdown(f"**{email}**{badge}")
            st.caption(f"Name: {u.get('full_name') or '—'} | ID: `{u_id[:8]}`")
            if is_last_admin:
                st.caption("🔒 Protected: Last active administrator")

        with col_role:
            role_options = ["viewer", "analyst", "admin"]
            curr_idx = role_options.index(role) if role in role_options else 0
            new_role = st.selectbox(
                "Role",
                options=role_options,
                index=curr_idx,
                key=f"select_role_{u_id[:8]}",
                disabled=is_last_admin,
                label_visibility="collapsed",
            )
            if new_role != role and st.button(
                "Apply Role Change",
                key=f"btn_apply_role_{u_id[:8]}",
                use_container_width=True,
            ):
                try:
                    client.update_user(user_id=u_id, role=new_role)
                    st.toast(f"Role updated to {new_role} for {email}")
                    st.rerun()
                except ConflictError as exc:
                    st.error(f"Cannot change role: {exc.detail}")
                except Exception as exc:
                    st.error(f"Failed to change role: {exc}")

        with col_status:
            if is_active:
                if st.button(
                    "Deactivate",
                    key=f"btn_deact_{u_id[:8]}",
                    disabled=is_last_admin,
                    use_container_width=True,
                ):
                    _render_deactivate_dialog(client, u_id, email)
            elif st.button(
                "Reactivate",
                key=f"btn_react_{u_id[:8]}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    client.update_user(user_id=u_id, is_active=True)
                    st.toast(f"User {email} reactivated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to reactivate: {exc}")


def _render_user_management_table(
    client: Any,
    users: list[dict[str, Any]],
    current_admin_id: str | None,
) -> None:
    """Render table of users and role modification/deactivation actions."""
    st.subheader("System Accounts")

    # Count active admins for last-admin indicator
    active_admins = [u for u in users if u.get("role") == "admin" and u.get("is_active")]
    is_single_admin = len(active_admins) == 1
    active_admin_id = str(active_admins[0].get("user_id")) if is_single_admin else None

    if is_single_admin:
        last_admin = active_admins[0]
        st.warning(
            f"⚠️ Last Active Administrator: `{last_admin.get('email')}`. "
            "To prevent lockout, this administrator cannot be demoted or deactivated.",
            icon="⚠️",
        )

    # User Summary Overview
    summary_rows = [
        {
            "Email": u.get("email"),
            "Full Name": u.get("full_name") or "—",
            "Role": str(u.get("role", "")).upper(),
            "Status": format_status_badge("ok" if u.get("is_active") else "reorder now"),
            "Active": "Yes" if u.get("is_active") else "No (Inactive)",
            "Last Login": format_date(u.get("last_login_at"))
            if u.get("last_login_at")
            else "Never",
            "Created": format_date(u.get("created_at")),
        }
        for u in users
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Account Actions & Role Management")

    for u in users:
        _render_user_card(
            client,
            u,
            is_single_admin=is_single_admin,
            active_admin_id=active_admin_id,
            current_admin_id=current_admin_id,
        )


def render_admin_users_view() -> None:
    """Entry point for the Administrative User Management view."""
    render_status_strip()
    st.title("User Management")
    st.caption("System accounts, role assignments, and account status controls (Admin only).")

    user = get_current_user()
    if not user or user.role != "admin":
        st.error(
            "Access Denied: The User Management dashboard is strictly restricted to Administrators."
        )
        return

    client = get_api_client()

    # 1. Create User Section
    _render_create_user_section(client)
    st.divider()

    # 2. Users Table & Actions
    try:
        users_resp = client.get_users(page=1, page_size=100)
        users = users_resp.get("items", []) if users_resp else []
    except Exception as exc:
        render_error(f"Failed to load user accounts: {exc}")
        return

    if not users:
        render_empty("No system user accounts found.")
        return

    _render_user_management_table(
        client,
        users,
        current_admin_id=user.user_id if user else None,
    )


if __name__ == "__main__" or "streamlit" in globals():
    render_admin_users_view()

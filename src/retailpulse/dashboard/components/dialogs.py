"""Confirmation dialog components for sensitive operations.

Implements DESIGN §5 and APP_FLOW §3 (J5, J6):
- Explicit effect description before triggering destructive or critical actions:
    Promote model, Train model, Save inventory parameters, Deactivate user.
- Cancel action cleanly aborts without mutation.
- Uses native st.dialog when available, with inline fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import streamlit as st


def open_confirm_dialog(
    title: str,
    message: str,
    effect_description: str,
    on_confirm: Callable[[], None],
    *,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    is_destructive: bool = False,
    key_prefix: str = "confirm",
) -> None:
    """Display modal confirmation dialog stating exact operational effects."""

    @st.dialog(title)
    def _dialog() -> None:
        st.write(message)
        st.warning(f"**Effect:** {effect_description}", icon="⚠️")

        col_cancel, col_confirm = st.columns([1, 1])
        with col_cancel:
            if st.button(cancel_label, key=f"{key_prefix}_cancel", use_container_width=True):
                st.rerun()
        with col_confirm:
            btn_type: Literal["primary", "secondary"] = "secondary" if is_destructive else "primary"
            if st.button(
                confirm_label,
                key=f"{key_prefix}_confirm",
                type=btn_type,
                use_container_width=True,
            ):
                on_confirm()
                st.rerun()

    _dialog()


def render_confirm_inline(
    title: str,
    message: str,
    effect_description: str,
    on_confirm: Callable[[], None],
    *,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    is_destructive: bool = False,
    key_prefix: str = "inline_confirm",
) -> None:
    """Render inline confirmation card (for embedded workflows or testing)."""
    with st.container(border=True):
        st.subheader(title)
        st.write(message)
        st.warning(f"**Effect:** {effect_description}", icon="⚠️")

        col_cancel, col_confirm = st.columns([1, 1])
        with col_cancel:
            if st.button(cancel_label, key=f"{key_prefix}_cancel", use_container_width=True):
                st.session_state[f"{key_prefix}_active"] = False
                st.rerun()
        with col_confirm:
            btn_type: Literal["primary", "secondary"] = "secondary" if is_destructive else "primary"
            if st.button(
                confirm_label,
                key=f"{key_prefix}_confirm",
                type=btn_type,
                use_container_width=True,
            ):
                st.session_state[f"{key_prefix}_active"] = False
                on_confirm()
                st.rerun()

"""Loading, empty, error, and permission state presentation helpers.

Implements DESIGN §6:
- Loading: st.spinner with specific operational text.
- Empty: Icon-free plain message + next-step guidance.
- Error: Inline message with Request ID and Retry action (no raw tracebacks).
- Forbidden: Role-gated clear message.
- Synthetic notice: Visible badge/banner for synthetic inventory parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import streamlit as st

from retailpulse.dashboard.client import ApiClientError


@contextmanager
def render_loading(message: str = "Loading...") -> Iterator[None]:
    """Context manager for localized non-blocking loading spinner."""
    with st.spinner(message):
        yield


def render_empty(message: str, action_guidance: str | None = None) -> None:
    """Render clean, icon-free empty state with optional next-step guidance."""
    text = f"{message} {action_guidance or ''}".strip()
    st.info(text)


def render_error(
    error: Exception | ApiClientError | str,
    *,
    on_retry: Callable[[], None] | None = None,
    key: str = "btn_retry",
) -> None:
    """Render inline error with plain-language message, Request ID, and Retry button."""
    if isinstance(error, ApiClientError):
        detail_msg = error.detail or error.title or "An unexpected API error occurred."
        request_id = error.request_id
    elif isinstance(error, Exception):
        detail_msg = str(error) or "An unexpected system error occurred."
        request_id = getattr(error, "request_id", None)
    else:
        detail_msg = str(error)
        request_id = None

    display_text = f"**Error:** {detail_msg}"
    if request_id:
        display_text += f"\n\n*Request ID:* `{request_id}`"

    st.error(display_text, icon="⚠️")

    if st.button("Retry", key=key, type="secondary"):
        if on_retry is not None:
            on_retry()
        else:
            st.rerun()


def render_forbidden(message: str = "You don't have access to this action.") -> None:
    """Render 403 Forbidden state for insufficient role access."""
    st.warning(message, icon="🔒")


def render_synthetic_notice() -> None:
    """Render visible notice that synthetic inventory parameters are active."""
    st.info(
        "Notice: Synthetic inventory parameters are currently active for this product.",
        icon="🧪",
    )

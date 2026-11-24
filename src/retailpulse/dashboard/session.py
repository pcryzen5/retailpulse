"""Streamlit session state and authentication helpers.

Implements P6-01:
- Initialization of session state variables.
- Token storage and authenticated API client retrieval.
- Sign-in, sign-out, and session expiry handlers.
"""

from __future__ import annotations

import logging

import streamlit as st

from retailpulse.dashboard.client import ApiClient, UnauthorizedError, UserProfile

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize required session state fields if not already populated."""
    if "token" not in st.session_state:
        st.session_state["token"] = None
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "flash_message" not in st.session_state:
        st.session_state["flash_message"] = None
    if "flash_level" not in st.session_state:
        st.session_state["flash_level"] = "info"
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = ApiClient()


def get_api_client() -> ApiClient:
    """Retrieve the session-scoped API client, syncing authentication token."""
    init_session_state()
    client: ApiClient = st.session_state["api_client"]
    token = st.session_state.get("token")
    if client.token != token:
        client.set_token(token)
    return client


def set_api_client(client: ApiClient) -> None:
    """Explicitly assign an API client instance (useful for test stubs)."""
    st.session_state["api_client"] = client
    if client.token:
        st.session_state["token"] = client.token


def get_current_user() -> UserProfile | None:
    """Return currently authenticated UserProfile, or None if unauthenticated."""
    init_session_state()
    user_val = st.session_state.get("user")
    if not user_val or not st.session_state.get("logged_in"):
        return None
    if isinstance(user_val, UserProfile):
        return user_val
    return UserProfile(
        user_id=str(user_val["user_id"]),
        email=str(user_val["email"]),
        role=str(user_val["role"]),
        full_name=user_val.get("full_name"),
    )


def login_user(email: str, password: str) -> UserProfile:
    """Authenticate user credentials against API, updating session state.

    Raises:
        UnauthorizedError: On invalid credentials or locked account.
        ApiClientError: On other API failures.
    """
    client = get_api_client()
    login_data = client.login(username=email, password=password)
    token = login_data["access_token"]

    st.session_state["token"] = token
    client.set_token(token)

    # Fetch fresh user profile
    profile = client.get_me()
    st.session_state["user"] = {
        "user_id": profile.user_id,
        "email": profile.email,
        "full_name": profile.full_name,
        "role": profile.role,
    }
    st.session_state["logged_in"] = True
    return profile


def logout_user() -> None:
    """Clear session authentication state and token."""
    st.session_state["token"] = None
    st.session_state["user"] = None
    st.session_state["logged_in"] = False
    client = get_api_client()
    client.set_token(None)


def handle_unauthorized(error: UnauthorizedError) -> None:
    """Handle 401 session expiry: clear credentials and notify user."""
    logger.warning("Session unauthorized (%s), logging out user.", error)
    logout_user()
    st.session_state["flash_message"] = "Your session expired. Please sign in again."
    st.session_state["flash_level"] = "warning"


def set_flash_message(message: str, level: str = "info") -> None:
    """Set transient user notification message."""
    st.session_state["flash_message"] = message
    st.session_state["flash_level"] = level


def get_and_clear_flash_message() -> tuple[str | None, str]:
    """Retrieve and clear transient notification message."""
    msg = st.session_state.get("flash_message")
    lvl = st.session_state.get("flash_level", "info")
    st.session_state["flash_message"] = None
    st.session_state["flash_level"] = "info"
    return msg, lvl

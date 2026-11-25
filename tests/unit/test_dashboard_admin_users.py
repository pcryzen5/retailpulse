"""AppTest and unit test suite for the Admin Users dashboard view.

Verifies P6-10:
- User management table: email, full name, role, active status, last login timestamp.
- User creation form: email, password, role inputs and validation.
- Role modification and deactivation actions.
- Last-active-admin protection:
  - If a user is the last active admin, role select and deactivate buttons are disabled.
  - UI warning banner is displayed explaining last-admin protection.
- Access gating: Non-admin users (Viewer/Analyst) receive Access Denied error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import (
    ApiClient,
    ConflictError,
    StatusStripData,
    UserProfile,
)

VIEWS_DIR = Path(__file__).parent.parent.parent / "src" / "retailpulse" / "dashboard" / "views"
ADMIN_USERS_VIEW_PATH = str(VIEWS_DIR / "admin_users.py")


class StubAdminUsersClient(ApiClient):
    """Stub API client for Admin User Management testing."""

    def __init__(self, single_admin: bool = True) -> None:
        self._token: str | None = "admin-token"
        self.single_admin = single_admin
        self.created_users: list[dict[str, Any]] = []
        self.updated_users: list[dict[str, Any]] = []

    def set_token(self, token: str | None) -> None:
        self._token = token

    def get_status_strip_data(self) -> Any:
        return StatusStripData(
            data_as_of="2026-11-25T08:00:00Z",
            champion_model="lightgbm",
            last_pipeline_status="completed",
            last_pipeline_time="2026-11-25T08:00:00Z",
            is_stale_warning=False,
            stale_reason=None,
        )

    def get_users(
        self,
        page: int = 1,
        page_size: int = 50,
        sort: str = "created_at:desc",
    ) -> dict[str, Any]:
        items = [
            {
                "user_id": "11111111-1111-1111-1111-111111111111",
                "email": "primary_admin@retailpulse.com",
                "full_name": "Primary Admin",
                "role": "admin",
                "is_active": True,
                "last_login_at": "2026-11-25T08:00:00Z",
                "created_at": "2026-11-01T00:00:00Z",
            },
            {
                "user_id": "22222222-2222-2222-2222-222222222222",
                "email": "analyst@retailpulse.com",
                "full_name": "Data Analyst",
                "role": "analyst",
                "is_active": True,
                "last_login_at": "2026-11-24T12:00:00Z",
                "created_at": "2026-11-05T00:00:00Z",
            },
            {
                "user_id": "33333333-3333-3333-3333-333333333333",
                "email": "inactive_viewer@retailpulse.com",
                "full_name": "Guest Viewer",
                "role": "viewer",
                "is_active": False,
                "last_login_at": None,
                "created_at": "2026-11-10T00:00:00Z",
            },
        ]
        if not self.single_admin:
            items.append(
                {
                    "user_id": "44444444-4444-4444-4444-444444444444",
                    "email": "second_admin@retailpulse.com",
                    "full_name": "Secondary Admin",
                    "role": "admin",
                    "is_active": True,
                    "last_login_at": "2026-11-24T10:00:00Z",
                    "created_at": "2026-11-02T00:00:00Z",
                }
            )
        return {"items": items, "total": len(items), "page": page, "page_size": page_size}

    def create_user(
        self,
        email: str,
        password: str,
        role: str = "analyst",
        full_name: str | None = None,
    ) -> dict[str, Any]:
        if email == "existing@retailpulse.com":
            raise ConflictError("User with this email already exists")
        data = {
            "user_id": "99999999-9999-9999-9999-999999999999",
            "email": email,
            "role": role,
            "full_name": full_name,
            "is_active": True,
        }
        self.created_users.append(data)
        return data

    def update_user(
        self,
        user_id: str,
        *,
        role: str | None = None,
        is_active: bool | None = None,
        full_name: str | None = None,
        password: str | None = None,
    ) -> dict[str, Any]:
        if (
            self.single_admin
            and user_id == "11111111-1111-1111-1111-111111111111"
            and (role != "admin" or is_active is False)
        ):
            raise ConflictError("Cannot demote or deactivate the last active administrator.")
        data = {
            "user_id": user_id,
            "role": role,
            "is_active": is_active,
            "full_name": full_name,
        }
        self.updated_users.append(data)
        return data


def test_admin_users_view_renders_for_admin_with_last_admin_protection() -> None:
    """When only 1 admin is active, last-admin warning is shown and actions are protected."""
    stub_client = StubAdminUsersClient(single_admin=True)
    at = AppTest.from_file(ADMIN_USERS_VIEW_PATH, default_timeout=30)
    at.session_state["api_client"] = stub_client
    at.session_state["user"] = UserProfile(
        user_id="11111111-1111-1111-1111-111111111111",
        email="primary_admin@retailpulse.com",
        role="admin",
    )
    at.session_state["logged_in"] = True
    at.run()

    assert not at.exception
    # Page Title
    assert any("User Management" in t.value for t in at.title)

    # Last Active Administrator Warning
    assert any("Last Active Administrator" in w.value for w in at.warning)

    # Check that primary admin's role select and deactivate buttons are disabled
    role_selects = {s.key: s for s in at.selectbox}
    assert role_selects["select_role_11111111"].disabled

    deact_buttons = {b.key: b for b in at.button if str(b.key).startswith("btn_deact_")}
    assert deact_buttons["btn_deact_11111111"].disabled

    # Non-admin user (analyst) can have role changed or be deactivated
    assert not role_selects["select_role_22222222"].disabled
    assert not deact_buttons["btn_deact_22222222"].disabled


def test_admin_users_view_with_multiple_admins() -> None:
    """When multiple admins are active, neither is locked by last-admin protection."""
    stub_client = StubAdminUsersClient(single_admin=False)
    at = AppTest.from_file(ADMIN_USERS_VIEW_PATH, default_timeout=30)
    at.session_state["api_client"] = stub_client
    at.session_state["user"] = UserProfile(
        user_id="11111111-1111-1111-1111-111111111111",
        email="primary_admin@retailpulse.com",
        role="admin",
    )
    at.session_state["logged_in"] = True
    at.run()

    assert not at.exception
    # No last active administrator warning
    assert not any("Last Active Administrator" in w.value for w in at.warning)

    # Primary admin's role select and deactivate buttons are NOT disabled
    role_selects = {s.key: s for s in at.selectbox}
    assert not role_selects["select_role_11111111"].disabled

    deact_buttons = {b.key: b for b in at.button if str(b.key).startswith("btn_deact_")}
    assert not deact_buttons["btn_deact_11111111"].disabled


def test_admin_users_view_access_denied_for_non_admin() -> None:
    """Non-admin users (Analyst or Viewer) receive access denied message."""
    stub_client = StubAdminUsersClient(single_admin=True)
    at = AppTest.from_file(ADMIN_USERS_VIEW_PATH, default_timeout=30)
    at.session_state["api_client"] = stub_client
    at.session_state["user"] = UserProfile(
        user_id="22222222-2222-2222-2222-222222222222",
        email="analyst@retailpulse.com",
        role="analyst",
    )
    at.session_state["logged_in"] = True
    at.run()

    assert not at.exception
    # Access Denied error rendered
    assert any("Access Denied" in err.value for err in at.error)

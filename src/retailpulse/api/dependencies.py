"""FastAPI authentication and authorization dependencies.

Implements SEC-AUTH-05, SEC-AUTHZ-01, and API §1.1:
- OAuth2 Bearer token extraction via HTTP Authorization header.
- Token decoding and user resolution from database on every request.
- Role resolution directly from database so role changes and deactivations apply immediately.
- require_role(min_role) dependency factory supporting viewer < analyst < admin hierarchy.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Callable
from typing import Annotated

import sqlalchemy as sa
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from retailpulse.db.models.app import AppUser
from retailpulse.db.session import get_db
from retailpulse.errors import AuthenticationError, PermissionDeniedError
from retailpulse.security.jwt import decode_access_token
from retailpulse.timeutil import utc_now

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

ROLE_HIERARCHY: dict[str, int] = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
}


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[Session, Depends(get_db)],
) -> AppUser:
    """Resolve and authenticate current user from Bearer token and database.

    Enforces SEC-AUTH-05: The user is queried from DB on every request to ensure
    deactivations, lockouts, and role adjustments take effect instantaneously.
    """
    payload = decode_access_token(token)
    sub = payload.get("sub")
    if not sub:
        raise AuthenticationError("Invalid authentication token: missing subject claim")

    try:
        user_uuid = uuid.UUID(str(sub))
    except (ValueError, TypeError) as exc:
        raise AuthenticationError("Invalid authentication token: malformed subject") from exc

    user = session.execute(
        sa.select(AppUser).where(AppUser.user_id == user_uuid)
    ).scalar_one_or_none()

    if user is None:
        raise AuthenticationError("User account not found or invalid token")

    if not user.is_active:
        raise AuthenticationError("User account is inactive")

    if user.locked_until is not None:
        locked_dt = (
            user.locked_until.replace(tzinfo=datetime.UTC)
            if user.locked_until.tzinfo is None
            else user.locked_until
        )
        if locked_dt > utc_now():
            raise AuthenticationError("User account is currently locked out")

    return user


def require_role(min_role: str) -> Callable[[AppUser], AppUser]:
    """Dependency factory enforcing minimum required role in the hierarchy.

    Hierarchy: viewer (1) < analyst (2) < admin (3).

    Args:
        min_role: Minimum role required ('viewer', 'analyst', 'admin').

    Returns:
        FastAPI dependency callable returning the authenticated AppUser.
    """
    req_level = ROLE_HIERARCHY.get(min_role)
    if req_level is None:
        raise ValueError(f"Unknown role requirement: {min_role}")

    def role_dependency(
        current_user: Annotated[AppUser, Depends(get_current_user)],
    ) -> AppUser:
        user_level = ROLE_HIERARCHY.get(current_user.role, 0)
        if user_level < req_level:
            raise PermissionDeniedError(
                f"Operation requires '{min_role}' role or higher "
                f"(current role: '{current_user.role}')"
            )
        return current_user

    return role_dependency

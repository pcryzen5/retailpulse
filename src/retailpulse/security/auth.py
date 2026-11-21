"""User authentication, account lockout lifecycle, and security controls.

Implements SEC-AUTH-01, SEC-AUTH-03, SEC-AUTH-05, and SEC-AUTHZ-03:
- Authentication with Argon2id and timing attack protection.
- Account lockout after 5 consecutive failed attempts for 15 minutes.
- Uniform generic failure messages across all authentication failure modes.
- Last active administrator protection against deactivation or demotion.
"""

from __future__ import annotations

import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.app import AppUser
from retailpulse.errors import AuthenticationError, ConflictError
from retailpulse.security.password import (
    hash_password,
    validate_password,
    verify_dummy_password,
    verify_password,
)
from retailpulse.timeutil import utc_now

MAX_FAILED_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15
GENERIC_AUTH_ERROR_MSG = "Invalid credentials or authentication token"


def authenticate_user(
    session: Session,
    email: str,
    password: str,
) -> AppUser:
    """Authenticate a user by email and password enforcing lockout and timing mitigation.

    Args:
        session: Active database session.
        email: Raw email input.
        password: Raw candidate password.

    Returns:
        The authenticated AppUser entity.

    Raises:
        AuthenticationError: If credentials fail, user is inactive, or account is locked.
    """
    clean_email = email.strip().lower()
    now = utc_now()

    user = session.execute(
        sa.select(AppUser).where(AppUser.email == clean_email)
    ).scalar_one_or_none()

    # 1. Unknown user: run dummy hash to neutralize timing disparity (SEC-AUTH-03)
    if user is None:
        verify_dummy_password(password)
        raise AuthenticationError(GENERIC_AUTH_ERROR_MSG)

    # 2. Check active lockout state
    if user.locked_until is not None:
        locked_dt = (
            user.locked_until.replace(tzinfo=datetime.UTC)
            if user.locked_until.tzinfo is None
            else user.locked_until
        )
        if locked_dt > now:
            # Still locked out: equalize timing and return uniform generic message
            verify_dummy_password(password)
            raise AuthenticationError(GENERIC_AUTH_ERROR_MSG)
        # Lockout period has elapsed: reset counters
        user.failed_login_count = 0
        user.locked_until = None

    # 3. Check inactive user
    if not user.is_active:
        verify_dummy_password(password)
        raise AuthenticationError(GENERIC_AUTH_ERROR_MSG)

    # 4. Verify password
    is_valid, needs_rehash = verify_password(user.password_hash, password)

    if not is_valid:
        user.failed_login_count += 1
        if user.failed_login_count >= MAX_FAILED_LOGIN_ATTEMPTS:
            user.locked_until = now + datetime.timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        session.commit()
        raise AuthenticationError(GENERIC_AUTH_ERROR_MSG)

    # 5. Successful login: reset failures, record timestamp, rehash if needed
    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = now
    if needs_rehash:
        user.password_hash = hash_password(password)

    session.commit()
    return user


def ensure_not_last_admin(
    session: Session,
    target_user: AppUser,
    *,
    new_role: str | None = None,
    new_is_active: bool | None = None,
) -> None:
    """Enforce SEC-AUTHZ-03: last active admin cannot be demoted or deactivated."""
    is_demoting = new_role is not None and new_role != "admin"
    is_deactivating = new_is_active is False

    if target_user.role == "admin" and target_user.is_active and (is_demoting or is_deactivating):
        other_active_admins = session.scalar(
            sa.select(sa.func.count(AppUser.user_id)).where(
                AppUser.role == "admin",
                AppUser.is_active.is_(True),
                AppUser.user_id != target_user.user_id,
            )
        )
        if (other_active_admins or 0) == 0:
            raise ConflictError("Cannot demote or deactivate the last active administrator")


def change_user_password(
    session: Session,
    user: AppUser,
    current_password: str,
    new_password: str,
) -> None:
    """Validate current credentials and update password with policy enforcement."""
    is_valid, _ = verify_password(user.password_hash, current_password)
    if not is_valid:
        raise AuthenticationError(GENERIC_AUTH_ERROR_MSG)

    validate_password(new_password)
    user.password_hash = hash_password(new_password)
    session.commit()

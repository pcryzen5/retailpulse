"""JSON Web Token (JWT) creation, signing, and verification.

Implements SEC-AUTH-02:
- Fixed algorithm HS256 enforced on encode and decode.
- Required claims: sub, iat, exp, jti, role.
- Enforces strict secret length and validation.
- Uniform AuthenticationError on expiration, tampering, or invalid algorithms.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

import jwt
from jwt.exceptions import (
    DecodeError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAlgorithmError,
    InvalidSignatureError,
    MissingRequiredClaimError,
    PyJWTError,
)

from retailpulse.config import get_settings
from retailpulse.errors import AuthenticationError
from retailpulse.timeutil import utc_now

JWT_ALGORITHM = "HS256"


def create_access_token(
    user_id: uuid.UUID | str,
    role: str,
    *,
    expires_delta: datetime.timedelta | None = None,
    secret_key: str | None = None,
) -> str:
    """Create a signed JWT access token with standard claims (SEC-AUTH-02).

    Args:
        user_id: Subject identifier (UUID or UUID string).
        role: User role name ('admin', 'analyst', 'viewer').
        expires_delta: Optional custom lifetime duration; defaults to settings.jwt_expire_minutes.
        secret_key: Optional override for signing secret.

    Returns:
        Encoded JWT string.
    """
    settings = get_settings()
    secret = secret_key or settings.jwt_secret.get_secret_value()

    now = utc_now()
    if expires_delta is not None:
        expire_dt = now + expires_delta
    else:
        expire_dt = now + datetime.timedelta(minutes=settings.jwt_expire_minutes)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire_dt.timestamp()),
        "jti": str(uuid.uuid4()),
    }

    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(
    token: str,
    *,
    secret_key: str | None = None,
) -> dict[str, Any]:
    """Decode and verify an access token enforcing algorithm allowlist and claims.

    Args:
        token: Raw encoded JWT string.
        secret_key: Optional override for signing secret.

    Returns:
        Decoded payload dictionary with verified claims.

    Raises:
        AuthenticationError: If token is expired, tampered, has wrong algorithm, or missing claims.
    """
    settings = get_settings()
    secret = secret_key or settings.jwt_secret.get_secret_value()

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[JWT_ALGORITHM],
            options={
                "require": ["sub", "iat", "exp", "jti"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
            },
        )
        return payload
    except ExpiredSignatureError as exc:
        raise AuthenticationError("Authentication token has expired") from exc
    except (
        InvalidSignatureError,
        DecodeError,
        InvalidAlgorithmError,
        ImmatureSignatureError,
        MissingRequiredClaimError,
        PyJWTError,
    ) as exc:
        raise AuthenticationError("Invalid authentication token") from exc

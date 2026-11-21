"""Argon2id password hashing, verification, and policy enforcement.

Implements SEC-AUTH-01 and SEC-AUTH-03:
- Argon2id password hashing via argon2-cffi.
- Password policy: length 12-128 characters, deny-list check.
- Timing attack mitigation: dummy hash verification for unknown users.
- Rehash detection for parameter upgrades.
"""

from __future__ import annotations

import contextlib

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type

from retailpulse.errors import DomainValidationError

# Argon2id password hasher tuned for OWASP recommendations (~100-300ms)
_HASHER = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128

# Common and predictable passwords deny-list per SEC-AUTH-01
PASSWORD_DENYLIST: frozenset[str] = frozenset(
    {
        "123456789012",
        "1234567890123",
        "12345678901234",
        "123456789012345",
        "password1234",
        "password12345",
        "password123456",
        "administrator",
        "administrator1",
        "retailpulse123",
        "retailpulse1234",
        "letmein123456",
        "welcome123456",
        "changeme12345",
        "qwertyuiop12",
        "qwertyuiop123",
        "supersecret12",
        "supersecret123",
        "adminpassword",
        "adminpassword1",
        "correcthorsebatterystaple",
        "iloveyou12345",
        "trustnoone123",
        "dragon1234567",
        "monkey1234567",
        "football1234",
    }
)

# Pre-computed dummy Argon2id hash for constant-time evaluation on unknown users
DUMMY_ARGON2ID_HASH: str = _HASHER.hash("RetailPulse-Dummy-Hash-Secret-123!")


def validate_password(password: str) -> None:
    """Validate password against length constraints and deny-list policy (SEC-AUTH-01).

    Args:
        password: Raw candidate password string.

    Raises:
        DomainValidationError: If password is too short, too long, or matches deny-list.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise DomainValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters in length"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise DomainValidationError(
            f"Password must not exceed {MAX_PASSWORD_LENGTH} characters in length"
        )

    norm_pwd = password.strip().lower()
    if norm_pwd in PASSWORD_DENYLIST:
        raise DomainValidationError(
            "Password is too common or easily guessable. Please choose a stronger passphrase."
        )


def hash_password(password: str, *, validate: bool = True) -> str:
    """Hash password using Argon2id with per-user salt.

    Args:
        password: Raw password string.
        validate: Whether to validate password policy before hashing (default True).

    Returns:
        Argon2id encoded hash string.
    """
    if validate:
        validate_password(password)
    return _HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> tuple[bool, bool]:
    """Verify password against an Argon2id hash and check if rehashing is needed.

    Args:
        password_hash: Stored Argon2id hash string.
        password: Plaintext password candidate.

    Returns:
        Tuple of (is_valid, needs_rehash).
    """
    try:
        is_valid = _HASHER.verify(password_hash, password)
        needs_rehash = _HASHER.check_needs_rehash(password_hash)
        return is_valid, needs_rehash
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False, False


def verify_dummy_password(password: str) -> None:
    """Execute dummy verification to equalize timing for unknown users (SEC-AUTH-03)."""
    with contextlib.suppress(VerifyMismatchError, VerificationError, InvalidHashError):
        _HASHER.verify(DUMMY_ARGON2ID_HASH, password)

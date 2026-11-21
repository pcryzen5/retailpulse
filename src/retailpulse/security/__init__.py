"""Security, authentication, and authorization utilities."""

from retailpulse.security.auth import (
    authenticate_user,
    change_user_password,
    ensure_not_last_admin,
)
from retailpulse.security.jwt import create_access_token, decode_access_token
from retailpulse.security.password import (
    hash_password,
    validate_password,
    verify_password,
)
from retailpulse.security.pseudonymize import (
    mask_customer_key,
    pseudonymize_customer_id,
    pseudonymize_series,
    verify_customer_key,
)

__all__ = [
    "authenticate_user",
    "change_user_password",
    "create_access_token",
    "decode_access_token",
    "ensure_not_last_admin",
    "hash_password",
    "mask_customer_key",
    "pseudonymize_customer_id",
    "pseudonymize_series",
    "validate_password",
    "verify_customer_key",
    "verify_password",
]

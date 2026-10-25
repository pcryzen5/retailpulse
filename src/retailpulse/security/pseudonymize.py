"""Customer pseudonymization using HMAC-SHA256.

Per SEC-DATA-01 and TRD §4.3 Rule 6:
- customer_key = HMAC_SHA256(RP_PSEUDONYM_SALT, canonical_integer_text)
- Raw customer IDs must never reach the database, logs, or downstream files
- Missing/null customer IDs retain customer_key = None
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pandas as pd
from pydantic import SecretStr

from retailpulse.config import get_settings


def _get_salt_str(salt: str | SecretStr | None = None) -> str:
    """Resolve salt to raw string from argument or application settings."""
    if salt is not None:
        if isinstance(salt, SecretStr):
            return salt.get_secret_value()
        return str(salt)
    return get_settings().pseudonym_salt.get_secret_value()


def pseudonymize_customer_id(
    customer_id: Any,
    salt: str | SecretStr | None = None,
) -> str | None:
    """Pseudonymize a single canonical customer ID using HMAC-SHA256.

    Args:
        customer_id: Normalized integer string (e.g. '13085') or None.
        salt: Optional salt string or SecretStr. If None, loaded from Settings.

    Returns:
        Hex-encoded 64-character SHA-256 HMAC string, or None if customer_id is empty/null.
    """
    if customer_id is None or pd.isna(customer_id):
        return None

    raw_str = str(customer_id).strip()
    if not raw_str or raw_str.lower() in ("nan", "none", "null"):
        return None

    salt_str = _get_salt_str(salt)
    key_bytes = salt_str.encode("utf-8")
    msg_bytes = raw_str.encode("utf-8")

    return hmac.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()


def pseudonymize_series(
    series: pd.Series,
    salt: str | SecretStr | None = None,
) -> pd.Series:
    """Vectorized pseudonymization of a pandas Series of customer IDs.

    Args:
        series: Series containing canonical customer ID strings or nulls.
        salt: Optional salt string or SecretStr. If None, loaded from Settings.

    Returns:
        Series of 64-character HMAC hex strings and None/null values.
    """
    if series.empty:
        return pd.Series([], dtype="object", index=series.index)

    salt_str = _get_salt_str(salt)
    key_bytes = salt_str.encode("utf-8")

    # Cache unique non-null values to speed up hashing across large datasets
    unique_ids = series.dropna().unique()
    id_map: dict[str, str | None] = {}
    for uid in unique_ids:
        s = str(uid).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            id_map[uid] = None
        else:
            id_map[uid] = hmac.new(key_bytes, s.encode("utf-8"), hashlib.sha256).hexdigest()

    return series.map(id_map).astype("object")


def mask_customer_key(customer_key: str | None, prefix_len: int = 8) -> str:
    """Format customer_key for display per SEC-DATA-03 (at most first 8 characters).

    Args:
        customer_key: 64-character hex key or None.
        prefix_len: Number of initial characters to expose (default 8).

    Returns:
        Truncated key (e.g., 'a1b2c3d4...') or empty string if None.
    """
    if not customer_key:
        return ""
    key_str = str(customer_key).strip()
    if len(key_str) <= prefix_len:
        return key_str
    return f"{key_str[:prefix_len]}..."


def verify_customer_key(
    customer_id: Any,
    expected_key: str,
    salt: str | SecretStr | None = None,
) -> bool:
    """Securely verify whether a raw customer ID matches a known customer_key.

    Uses hmac.compare_digest for constant-time comparison against timing attacks.

    Args:
        customer_id: Raw or canonical customer ID.
        expected_key: Hex customer_key to check against.
        salt: Optional salt override.

    Returns:
        True if the customer ID hashes to expected_key, False otherwise.
    """
    computed = pseudonymize_customer_id(customer_id, salt=salt)
    if computed is None or not expected_key:
        return False
    return hmac.compare_digest(computed.lower(), expected_key.strip().lower())

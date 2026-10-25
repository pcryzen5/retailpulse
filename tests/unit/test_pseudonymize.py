"""Unit tests for customer pseudonymization per SEC-DATA-01 and TRD §4.3 Rule 6."""

from __future__ import annotations

import pandas as pd
from pydantic import SecretStr

from retailpulse.security.pseudonymize import (
    mask_customer_key,
    pseudonymize_customer_id,
    pseudonymize_series,
    verify_customer_key,
)

TEST_SALT = "test-salt-secret-at-least-32-bytes-long!"


def test_pseudonymize_customer_id_deterministic() -> None:
    """Verify HMAC-SHA256 pseudonymization is deterministic and produces 64-character hex."""
    key1 = pseudonymize_customer_id("13085", salt=TEST_SALT)
    key2 = pseudonymize_customer_id("13085", salt=TEST_SALT)

    assert key1 is not None
    assert key2 is not None
    assert key1 == key2
    assert len(key1) == 64
    assert all(c in "0123456789abcdef" for c in key1)


def test_pseudonymize_different_salts_produce_different_keys() -> None:
    """Verify different salts yield distinct hashes for the same customer ID."""
    key_a = pseudonymize_customer_id("17850", salt="salt-one-32-bytes-long-padded!!")
    key_b = pseudonymize_customer_id("17850", salt="salt-two-32-bytes-long-padded!!")

    assert key_a != key_b


def test_pseudonymize_null_and_empty_inputs_return_none() -> None:
    """Verify missing, empty, or nan values return None without error."""
    assert pseudonymize_customer_id(None, salt=TEST_SALT) is None
    assert pseudonymize_customer_id("", salt=TEST_SALT) is None
    assert pseudonymize_customer_id("   ", salt=TEST_SALT) is None
    assert pseudonymize_customer_id("nan", salt=TEST_SALT) is None
    assert pseudonymize_customer_id("None", salt=TEST_SALT) is None
    assert pseudonymize_customer_id("null", salt=TEST_SALT) is None
    assert pseudonymize_customer_id(float("nan"), salt=TEST_SALT) is None


def test_pseudonymize_with_secret_str() -> None:
    """Verify pseudonymization accepts Pydantic SecretStr as salt."""
    secret = SecretStr("secret-salt-at-least-32-bytes-long!!")
    key = pseudonymize_customer_id("14000", salt=secret)
    assert key is not None
    assert len(key) == 64


def test_pseudonymize_series() -> None:
    """Verify vectorized pseudonymization preserves nulls and maps unique values."""
    series = pd.Series(["13085", None, "17850", "", "nan", "13085"])
    hashed_series = pseudonymize_series(series, salt=TEST_SALT)

    assert len(hashed_series) == 6
    assert hashed_series.iloc[0] is not None
    assert len(str(hashed_series.iloc[0])) == 64

    # Row 0 and row 5 have identical customer ID -> identical hash
    assert hashed_series.iloc[0] == hashed_series.iloc[5]

    # Nulls / empty values remain None
    assert pd.isna(hashed_series.iloc[1]) or hashed_series.iloc[1] is None
    assert pd.isna(hashed_series.iloc[3]) or hashed_series.iloc[3] is None
    assert pd.isna(hashed_series.iloc[4]) or hashed_series.iloc[4] is None

    # Different IDs produce different hashes
    assert hashed_series.iloc[0] != hashed_series.iloc[2]


def test_mask_customer_key() -> None:
    """Verify customer key masking for display (SEC-DATA-03)."""
    full_key = "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0"
    masked = mask_customer_key(full_key, prefix_len=8)
    assert masked == "a1b2c3d4..."
    assert len(masked) == 11

    # Empty or None returns empty string
    assert mask_customer_key(None) == ""
    assert mask_customer_key("") == ""


def test_verify_customer_key() -> None:
    """Verify constant-time customer key verification."""
    cid = "17850"
    expected_key = pseudonymize_customer_id(cid, salt=TEST_SALT)
    assert expected_key is not None

    # Correct match
    assert verify_customer_key(cid, expected_key, salt=TEST_SALT) is True

    # Wrong customer ID
    assert verify_customer_key("17851", expected_key, salt=TEST_SALT) is False

    # Wrong salt
    assert verify_customer_key(cid, expected_key, salt="other-salt-32-bytes-long-here!!") is False

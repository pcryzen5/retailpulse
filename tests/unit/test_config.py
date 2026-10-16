"""Unit tests for configuration, errors, logging, and time utility."""

from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import SecretStr

from retailpulse.config import Settings, get_settings
from retailpulse.errors import ConfigError, RetailPulseError
from retailpulse.logging import (
    add_request_id,
    bind_request_id,
    configure_logging,
    get_logger,
    reset_request_id,
)
from retailpulse.timeutil import utc_now

pytestmark = pytest.mark.unit


def test_default_settings() -> None:
    """Verify default settings align with TRD §3 specifications."""
    settings = Settings()
    assert settings.env == "dev"
    assert settings.currency == "GBP"
    assert settings.seed == 42
    assert settings.forecast_top_n == 300
    assert settings.forecast_horizon_days == 28
    assert settings.backtest_folds == 5
    assert settings.interval_level == 0.80
    assert settings.churn_window_days == 90
    assert settings.churn_band_medium == 0.30
    assert settings.churn_band_high == 0.60
    assert settings.rfm_window_days == 730
    assert settings.dedupe_exact_rows is True
    assert settings.default_service_level == 0.95
    assert settings.overstock_days == 90
    assert settings.auto_promote is False
    assert settings.docs_enabled is True  # True in dev


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variables with RP_ prefix override defaults."""
    monkeypatch.setenv("RP_ENV", "test")
    monkeypatch.setenv("RP_SEED", "99")
    monkeypatch.setenv("RP_CURRENCY", "EUR")
    monkeypatch.setenv("RP_FORECAST_TOP_N", "30")

    settings = Settings()
    assert settings.env == "test"
    assert settings.seed == 99
    assert settings.currency == "EUR"
    assert settings.forecast_top_n == 30


def test_prod_fails_on_short_secrets() -> None:
    """Verify production startup fails when secrets are shorter than 32 bytes."""
    with pytest.raises(ConfigError, match="must be at least 32 bytes"):
        Settings(
            env="prod",
            jwt_secret=SecretStr("short-secret"),
            pseudonym_salt=SecretStr("a" * 32),
        )

    with pytest.raises(ConfigError, match="must be at least 32 bytes"):
        Settings(
            env="prod",
            jwt_secret=SecretStr("a" * 32),
            pseudonym_salt=SecretStr("short-salt"),
        )


def test_prod_fails_on_example_secrets() -> None:
    """Verify production startup fails when secrets use known placeholders."""
    with pytest.raises(ConfigError, match="cannot be a known example"):
        Settings(
            env="prod",
            jwt_secret=SecretStr("insecure-dev-jwt-secret-min-32-bytes-long!"),
            pseudonym_salt=SecretStr("a" * 32),
        )


def test_prod_succeeds_with_valid_secrets() -> None:
    """Verify production mode succeeds with adequate, non-example secrets."""
    settings = Settings(
        env="prod",
        jwt_secret=SecretStr("k8Jd!92_xmLPq#71nZ0vR-4tWbYxM2sQ"),
        pseudonym_salt=SecretStr("u3Np@84_vLqZs#19mK2vX-9tRbCxN5wT"),
    )
    assert settings.env == "prod"
    assert settings.docs_enabled is False  # disabled by default in prod


def test_churn_band_threshold_validation() -> None:
    """Verify churn medium band must be strictly less than high band."""
    with pytest.raises(ConfigError, match="must be strictly less than"):
        Settings(churn_band_medium=0.60, churn_band_high=0.50)


def test_get_settings_cache() -> None:
    """Verify get_settings returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_errors_hierarchy() -> None:
    """Verify domain error inheritance."""
    err = ConfigError("Invalid config", details={"key": "val"})
    assert isinstance(err, RetailPulseError)
    assert err.message == "Invalid config"
    assert err.details == {"key": "val"}


def test_timeutil_utc_now() -> None:
    """Verify timeutil.utc_now provides timezone-aware UTC datetimes."""
    now = utc_now()
    assert now.tzinfo is not None
    assert now.tzinfo == UTC


def test_logging_and_request_id() -> None:
    """Verify logging configuration and request_id contextvar injection."""
    configure_logging(level="DEBUG", json_format=True)
    logger = get_logger("test_logger")
    assert logger is not None

    token = bind_request_id("test-req-123")
    try:
        event: dict[str, object] = {"event": "test"}
        processed = add_request_id(None, "info", event)  # type: ignore[arg-type]
        assert processed["request_id"] == "test-req-123"
    finally:
        reset_request_id(token)

    event2: dict[str, object] = {"event": "test2"}
    processed2 = add_request_id(None, "info", event2)  # type: ignore[arg-type]
    assert "request_id" not in processed2

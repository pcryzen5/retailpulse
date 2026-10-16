"""Application settings and environment configuration for RetailPulse."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from retailpulse.errors import ConfigError

_EXAMPLE_SECRETS = {
    "insecure-dev-jwt-secret-min-32-bytes-long!",
    "insecure-dev-salt-min-32-bytes-long!",
    "change-me",
    "placeholder",
    "example",
}


class Settings(BaseSettings):
    """Runtime configuration loaded strictly from environment variables (prefix RP_)."""

    model_config = SettingsConfigDict(
        env_prefix="RP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = Field(
        default="dev",
        description="Deployment environment name",
    )
    database_url: str = Field(
        default="postgresql+psycopg://rp_app:devpassword@localhost:5432/retailpulse",
        description="SQLAlchemy database connection URL",
    )
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000",
        description="MLflow tracking server URI",
    )
    prefect_api_url: str = Field(
        default="http://prefect:4200/api",
        description="Prefect API endpoint",
    )
    jwt_secret: SecretStr = Field(
        default=SecretStr("insecure-dev-jwt-secret-min-32-bytes-long!"),
        description="Secret key for signing access tokens",
    )
    jwt_expire_minutes: int = Field(
        default=30,
        ge=1,
        description="Access token validity duration in minutes",
    )
    pseudonym_salt: SecretStr = Field(
        default=SecretStr("insecure-dev-salt-min-32-bytes-long!"),
        description="Salt for HMAC-SHA256 customer pseudonymization",
    )
    api_base_url: str = Field(
        default="http://api:8000",
        description="Base URL of the API for internal service calls",
    )
    cors_origins: list[str] | str = Field(
        default="",
        description="Allowed CORS origins (comma-separated or list)",
    )
    max_upload_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum upload size in megabytes",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for local raw, interim, and report files",
    )
    seed: int = Field(
        default=42,
        description="Global deterministic random seed",
    )
    forecast_top_n: int = Field(
        default=300,
        ge=1,
        description="Number of top SKUs to forecast",
    )
    forecast_horizon_days: int = Field(
        default=28,
        ge=1,
        le=90,
        description="Forecast horizon in days",
    )
    backtest_folds: int = Field(
        default=5,
        ge=1,
        description="Number of rolling-origin backtest folds",
    )
    interval_level: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Nominal prediction interval coverage",
    )
    churn_window_days: int = Field(
        default=90,
        ge=1,
        description="Window in days defining churn outcome",
    )
    churn_band_medium: float = Field(
        default=0.30,
        gt=0.0,
        lt=1.0,
        description="Threshold probability for medium churn risk",
    )
    churn_band_high: float = Field(
        default=0.60,
        gt=0.0,
        lt=1.0,
        description="Threshold probability for high churn risk",
    )
    rfm_window_days: int = Field(
        default=730,
        ge=30,
        description="Historical lookback window for RFM features",
    )
    dedupe_exact_rows: bool = Field(
        default=True,
        description="Whether to drop exact duplicate input rows during cleaning",
    )
    default_service_level: float = Field(
        default=0.95,
        gt=0.5,
        lt=1.0,
        description="Default inventory cycle service level",
    )
    overstock_days: int = Field(
        default=90,
        ge=1,
        description="Days of cover threshold for overstock status",
    )
    auto_promote: bool = Field(
        default=False,
        description="Whether candidate models can auto-promote without manual approval",
    )
    enable_docs: bool | None = Field(
        default=None,
        description="Whether Swagger/OpenAPI documentation endpoints are active",
    )
    alert_webhook_url: SecretStr | None = Field(
        default=None,
        description="Optional operator-configured alert webhook URL",
    )
    currency: str = Field(
        default="GBP",
        description="Primary display currency code",
    )
    log_level: str = Field(
        default="INFO",
        description="Root logging severity level",
    )

    @property
    def docs_enabled(self) -> bool:
        """Resolve whether docs are enabled based on environment default."""
        if self.enable_docs is not None:
            return self.enable_docs
        return self.env != "prod"

    @model_validator(mode="after")
    def validate_production_security(self) -> Settings:
        """Enforce strict secret length and non-default constraints in production."""
        if self.env == "prod":
            jwt_val = self.jwt_secret.get_secret_value()
            if len(jwt_val.encode("utf-8")) < 32:
                raise ConfigError(
                    "RP_JWT_SECRET must be at least 32 bytes in production environments"
                )
            if any(bad in jwt_val.lower() for bad in _EXAMPLE_SECRETS):
                raise ConfigError(
                    "RP_JWT_SECRET cannot be a known example or placeholder secret in production"
                )

            salt_val = self.pseudonym_salt.get_secret_value()
            if len(salt_val.encode("utf-8")) < 32:
                raise ConfigError(
                    "RP_PSEUDONYM_SALT must be at least 32 bytes in production environments"
                )
            if any(bad in salt_val.lower() for bad in _EXAMPLE_SECRETS):
                raise ConfigError(
                    "RP_PSEUDONYM_SALT cannot be a known example or placeholder "
                    "secret in production"
                )

        if self.churn_band_medium >= self.churn_band_high:
            raise ConfigError("RP_CHURN_BAND_MEDIUM must be strictly less than RP_CHURN_BAND_HIGH")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve the cached global Settings instance.

    Returns:
        The singleton Settings instance initialized from environment variables.
    """
    return Settings()

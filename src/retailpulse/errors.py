"""Domain exceptions for RetailPulse."""

from __future__ import annotations


class RetailPulseError(Exception):
    """Base exception for all RetailPulse domain errors."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(RetailPulseError):
    """Raised when configuration validation or loading fails."""


class DomainValidationError(RetailPulseError):
    """Raised when domain entity or input validation fails."""


class IngestionError(RetailPulseError):
    """Raised when file reading, parsing, or adapter transformation fails."""


class DataQualityError(RetailPulseError):
    """Raised when data quality checks fail or threshold gates are violated."""


class NotFoundError(RetailPulseError):
    """Raised when an expected entity or resource is not found."""


class ConflictError(RetailPulseError):
    """Raised when a concurrent action or state conflict occurs."""


class AuthenticationError(RetailPulseError):
    """Raised when authentication fails."""


class PermissionDeniedError(RetailPulseError):
    """Raised when an authenticated user lacks required permissions."""


class PipelineError(RetailPulseError):
    """Raised when an orchestration flow or execution step fails."""


class ModelGateError(RetailPulseError):
    """Raised when a candidate model fails the champion gate."""


class ServiceUnavailableError(RetailPulseError):
    """Raised when an active model, forecast, or required service output is unavailable."""


class DatabaseError(RetailPulseError):
    """Raised when database operations, connectivity, or migrations fail."""

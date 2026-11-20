"""RFC 9457 Problem Details error formatting and global FastAPI exception handlers.

Implements P5-01, API §1.1, and SECURITY §7 (SEC-WEB-04):
- application/problem+json media type responses.
- Maps domain exceptions and HTTP errors to RFC 9457 URN types.
- Sanitizes error output: zero stack traces, SQL, file paths, or secrets leaked to client.
- Always echoes request_id in response body and X-Request-ID header.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from retailpulse.errors import (
    AuthenticationError,
    ConfigError,
    ConflictError,
    DatabaseError,
    DataQualityError,
    DomainValidationError,
    IngestionError,
    ModelGateError,
    NotFoundError,
    PermissionDeniedError,
    PipelineError,
    RetailPulseError,
)
from retailpulse.logging import get_logger, request_id_ctx

logger = get_logger("retailpulse.api.errors")

PROBLEM_MEDIA_TYPE = "application/problem+json"

# Status code to (type_suffix, title) mapping per API §1.1
STATUS_TO_PROBLEM_TYPE: dict[int, tuple[str, str]] = {
    400: ("bad_request", "Bad Request"),
    401: ("unauthorized", "Unauthorized"),
    403: ("forbidden", "Forbidden"),
    404: ("not_found", "Not Found"),
    405: ("bad_request", "Method Not Allowed"),
    409: ("conflict", "Conflict"),
    413: ("payload_too_large", "Payload Too Large"),
    415: ("unsupported_media_type", "Unsupported Media Type"),
    422: ("validation_error", "Validation Error"),
    429: ("rate_limited", "Rate Limited"),
    500: ("internal_error", "Internal Server Error"),
    503: ("model_unavailable", "Service Unavailable"),
}


def _resolve_request_id(request: Request) -> str:
    """Extract request_id from request state, contextvar, or generate fallback."""
    return getattr(request.state, "request_id", None) or request_id_ctx.get() or str(uuid.uuid4())


def create_problem_response(
    status_code: int,
    *,
    title: str | None = None,
    detail: str | None = None,
    instance: str | None = None,
    request_id: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    type_suffix: str | None = None,
) -> JSONResponse:
    """Construct an RFC 9457 compliant Problem Details JSONResponse.

    Args:
        status_code: HTTP status code.
        title: Human-readable summary of problem type.
        detail: Human-readable explanation specific to this occurrence.
        instance: URI reference identifying the specific occurrence of the problem.
        request_id: Tracing UUID associated with the request.
        errors: Optional granular field-level validation errors.
        type_suffix: Suffix for urn:retailpulse:problem:<type_suffix>.

    Returns:
        JSONResponse with media_type application/problem+json and X-Request-ID header.
    """
    default_suffix, default_title = STATUS_TO_PROBLEM_TYPE.get(
        status_code,
        ("bad_request" if status_code < 500 else "internal_error", "Error"),
    )

    eff_suffix = type_suffix or default_suffix
    eff_title = title or default_title
    eff_req_id = request_id or str(uuid.uuid4())
    eff_detail = detail or eff_title

    payload: dict[str, Any] = {
        "type": f"urn:retailpulse:problem:{eff_suffix}",
        "title": eff_title,
        "status": status_code,
        "detail": eff_detail,
        "instance": instance or "",
        "request_id": eff_req_id,
    }

    if errors is not None:
        payload["errors"] = errors

    headers = {
        "Content-Type": PROBLEM_MEDIA_TYPE,
        "X-Request-ID": eff_req_id,
    }

    return JSONResponse(
        content=payload,
        status_code=status_code,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


async def starlette_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle Starlette and FastAPI HTTPExceptions."""
    req_id = _resolve_request_id(request)
    default_suffix, default_title = STATUS_TO_PROBLEM_TYPE.get(
        exc.status_code,
        ("bad_request" if exc.status_code < 500 else "internal_error", "HTTP Error"),
    )

    detail_str = str(exc.detail) if exc.detail else default_title

    # Log non-client errors
    if exc.status_code >= 500:
        logger.error(
            "HTTP server error occurred",
            status_code=exc.status_code,
            detail=detail_str,
            path=request.url.path,
            request_id=req_id,
        )

    return create_problem_response(
        status_code=exc.status_code,
        title=default_title,
        detail=detail_str,
        instance=request.url.path,
        request_id=req_id,
        type_suffix=default_suffix,
    )


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Format Pydantic / FastAPI input validation errors into RFC 9457 format."""
    req_id = _resolve_request_id(request)

    field_errors: list[dict[str, Any]] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        # Extract meaningful field identifier (e.g. skip "body", "query")
        clean_parts = [str(x) for x in loc if str(x) not in ("body", "query", "path")]
        field_name = ".".join(clean_parts) if clean_parts else ".".join(str(x) for x in loc)
        field_errors.append(
            {
                "field": field_name,
                "message": err.get("msg", "Validation failed"),
            }
        )

    detail = (
        f"{field_errors[0]['field']}: {field_errors[0]['message']}"
        if len(field_errors) == 1
        else f"{len(field_errors)} validation errors detected"
    )

    return create_problem_response(
        status_code=422,
        title="Validation Error",
        detail=detail,
        instance=request.url.path,
        request_id=req_id,
        errors=field_errors,
        type_suffix="validation_error",
    )


async def domain_exception_handler(  # noqa: PLR0915
    request: Request,
    exc: RetailPulseError,
) -> JSONResponse:
    """Map domain exceptions from retailpulse.errors to RFC 9457 Problem Details."""
    req_id = _resolve_request_id(request)

    if isinstance(exc, DomainValidationError):
        status_code = 422
        suffix = "validation_error"
        title = "Validation Error"
        detail = exc.message
    elif isinstance(exc, NotFoundError):
        status_code = 404
        suffix = "not_found"
        title = "Not Found"
        detail = exc.message
    elif isinstance(exc, ConflictError):
        status_code = 409
        suffix = "conflict"
        title = "Conflict"
        detail = exc.message
    elif isinstance(exc, AuthenticationError):
        status_code = 401
        suffix = "unauthorized"
        title = "Unauthorized"
        # Always return generic message to prevent username enumeration per API §1.1
        detail = "Invalid credentials or authentication token"
    elif isinstance(exc, PermissionDeniedError):
        status_code = 403
        suffix = "forbidden"
        title = "Forbidden"
        detail = exc.message
    elif isinstance(exc, (IngestionError, DataQualityError)):
        status_code = 422
        suffix = "validation_error"
        title = "Data Ingestion / Quality Error"
        detail = exc.message
    elif isinstance(exc, ModelGateError):
        status_code = 422
        suffix = "validation_error"
        title = "Model Gate Validation Failed"
        detail = exc.message
    elif isinstance(exc, DatabaseError):
        status_code = 503
        suffix = "model_unavailable"
        title = "Service Unavailable"
        # SEC-WEB-04: Never leak raw database queries, connection strings, or internal errors
        detail = "Database service is temporarily unavailable"
        logger.error(
            "Database domain error encountered",
            error=exc.message,
            path=request.url.path,
            request_id=req_id,
        )
    elif isinstance(exc, ConfigError):
        status_code = 500
        suffix = "internal_error"
        title = "Configuration Error"
        detail = "Internal service configuration error"
        logger.error(
            "Configuration domain error encountered",
            error=exc.message,
            path=request.url.path,
            request_id=req_id,
        )
    elif isinstance(exc, PipelineError):
        status_code = 500
        suffix = "internal_error"
        title = "Pipeline Execution Error"
        detail = "Pipeline execution failed"
        logger.error(
            "Pipeline domain error encountered",
            error=exc.message,
            path=request.url.path,
            request_id=req_id,
        )
    else:
        status_code = 400
        suffix = "bad_request"
        title = "Bad Request"
        detail = exc.message

    return create_problem_response(
        status_code=status_code,
        title=title,
        detail=detail,
        instance=request.url.path,
        request_id=req_id,
        type_suffix=suffix,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all for unhandled exceptions, enforcing SEC-WEB-04 (zero detail leakage)."""
    req_id = _resolve_request_id(request)

    logger.error(
        "Unhandled internal exception",
        exc_type=type(exc).__name__,
        error=str(exc),
        path=request.url.path,
        request_id=req_id,
        exc_info=exc,
    )

    return create_problem_response(
        status_code=500,
        title="Internal Server Error",
        detail="An unexpected internal server error occurred.",
        instance=request.url.path,
        request_id=req_id,
        type_suffix="internal_error",
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all RFC 9457 exception handlers on the FastAPI application."""
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RetailPulseError, domain_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

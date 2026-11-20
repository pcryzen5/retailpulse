"""HTTP Middleware for RetailPulse API service.

Implements P5-01, TRD §7, API §1.1, and SECURITY §7 (SEC-WEB-01, SEC-WEB-04):
- RequestIDMiddleware: Generates or sanitizes X-Request-ID, binds structlog context,
  and echoes in responses and error bodies.
- TimingMiddleware: Measures execution latency, logs structured HTTP metrics,
  and attaches X-Process-Time header.
- SecurityHeadersMiddleware: Enforces strict HTTP security headers on all responses
  (nosniff, DENY frame options, no-referrer, minimal Permissions-Policy, Cache-Control: no-store).
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from retailpulse.logging import bind_request_id, get_logger, reset_request_id

logger = get_logger("retailpulse.api.http")

# Safe request ID pattern: alphanumeric, hyphen, underscore, 1-64 chars
SAFE_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generates, sanitizes, and binds request ID for logging and response tracking."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process incoming request and echo validated request ID."""
        client_req_id = request.headers.get("X-Request-ID")
        if client_req_id and SAFE_REQUEST_ID_REGEX.match(client_req_id):
            request_id = client_req_id
        else:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        token = bind_request_id(request_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            reset_request_id(token)


class TimingMiddleware(BaseHTTPMiddleware):
    """Measures request duration and attaches X-Process-Time diagnostic header."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Track processing latency and log request details."""
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"

        client_host = request.client.host if request.client else None
        logger.info(
            "http_request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(process_time_ms, 2),
            client_ip=client_host,
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects mandatory HTTP security headers per SEC-WEB-01 on all responses."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add hardening headers to HTTP responses."""
        response = await call_next(request)

        # SEC-WEB-01 mandatory baseline security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )

        # Default Cache-Control: no-store for API responses if not explicitly specified
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"

        return response

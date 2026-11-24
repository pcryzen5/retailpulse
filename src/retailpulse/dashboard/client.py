"""RetailPulse Dashboard HTTP API Client.

Implements P6-01 and ARCHITECTURE ADR-006:
- Pure API client for all dashboard communication.
- Configurable timeouts and retries for transient errors.
- Typed mapping of RFC 9457 Problem Details responses to specialized exceptions.
- Session token management and Bearer authentication injection.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)


# --- Exception Hierarchy (mapped to RFC 9457 Problem Details) ---


class ApiClientError(Exception):
    """Base exception for all RetailPulse API communication failures."""

    def __init__(
        self,
        status_code: int,
        title: str,
        detail: str,
        type_uri: str = "urn:retailpulse:problem:error",
        *,
        request_id: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(f"[{status_code}] {title}: {detail}")
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.type_uri = type_uri
        self.request_id = request_id
        self.errors = errors or []


class UnauthorizedError(ApiClientError):
    """401 Unauthorized: Session missing, expired, or invalid credentials."""


class ForbiddenError(ApiClientError):
    """403 Forbidden: Authenticated user lacks role permissions."""


class NotFoundError(ApiClientError):
    """404 Not Found: Requested resource does not exist."""


class ConflictError(ApiClientError):
    """409 Conflict: Resource conflict or invalid state transition."""


class ValidationError(ApiClientError):
    """422 Validation Error: Malformed parameters or schema validation failure."""


class RateLimitedError(ApiClientError):
    """429 Rate Limited: Request threshold exceeded."""

    def __init__(
        self,
        status_code: int,
        title: str,
        detail: str,
        type_uri: str,
        *,
        request_id: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(status_code, title, detail, type_uri, request_id=request_id)
        self.retry_after = retry_after


class ServerError(ApiClientError):
    """500 / 502 / 503 / 504: Internal server or upstream service error."""


class ApiConnectionError(ApiClientError):
    """Network connection failure or timeout when communicating with the API."""

    def __init__(self, message: str) -> None:
        super().__init__(
            status_code=503,
            title="Service Unavailable",
            detail=message,
            type_uri="urn:retailpulse:problem:service_unavailable",
        )


# --- Typed Data Models ---


@dataclass(frozen=True)
class UserProfile:
    """Current authenticated user profile."""

    user_id: str
    email: str
    full_name: str | None
    role: str


@dataclass(frozen=True)
class StatusStripData:
    """System freshness and metadata for global status strip (DESIGN §3)."""

    data_as_of: str | None
    champion_model: str | None
    last_pipeline_status: str | None
    last_pipeline_time: str | None
    is_stale_warning: bool
    stale_reason: str | None = None


# --- HTTP Client Implementation ---


class ApiClient:
    """Thread-safe synchronous HTTP client for communicating with the RetailPulse API."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        raw_url = base_url or os.environ.get("RP_API_BASE_URL") or "http://127.0.0.1:8000"
        self.base_url = raw_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._token: str | None = token
        self._http_client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=3.0),
            follow_redirects=True,
        )

    def set_token(self, token: str | None) -> None:
        """Update or clear the active Bearer authentication token."""
        self._token = token

    @property
    def token(self) -> str | None:
        """Current Bearer authentication token."""
        return self._token

    def _build_headers(self, custom_headers: dict[str, str] | None = None) -> dict[str, str]:
        """Construct request headers including Bearer authorization if authenticated."""
        headers = {
            "Accept": "application/json, application/problem+json",
            "User-Agent": "RetailPulse-Dashboard/1.0",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if custom_headers:
            headers.update(custom_headers)
        return headers

    def _handle_response_error(self, response: httpx.Response) -> None:
        """Parse RFC 9457 Problem Details or generic error payload into a typed exception."""
        status_code = response.status_code
        request_id = response.headers.get("X-Request-ID")

        try:
            body = response.json()
        except Exception:
            body = {}

        type_uri = body.get("type", "urn:retailpulse:problem:error")
        title = body.get("title", response.reason_phrase or "API Error")
        detail = body.get("detail", response.text or "An unexpected error occurred.")
        errors = body.get("errors")

        if status_code == 401:
            raise UnauthorizedError(
                status_code, title, detail, type_uri, request_id=request_id, errors=errors
            )
        if status_code == 403:
            raise ForbiddenError(
                status_code, title, detail, type_uri, request_id=request_id, errors=errors
            )
        if status_code == 404:
            raise NotFoundError(
                status_code, title, detail, type_uri, request_id=request_id, errors=errors
            )
        if status_code == 409:
            raise ConflictError(
                status_code, title, detail, type_uri, request_id=request_id, errors=errors
            )
        if status_code == 422:
            raise ValidationError(
                status_code, title, detail, type_uri, request_id=request_id, errors=errors
            )
        if status_code == 429:
            retry_header = response.headers.get("Retry-After")
            retry_after = int(retry_header) if retry_header and retry_header.isdigit() else None
            raise RateLimitedError(
                status_code, title, detail, type_uri, request_id=request_id, retry_after=retry_after
            )
        if status_code >= 500:
            raise ServerError(
                status_code, title, detail, type_uri, request_id=request_id, errors=errors
            )

        raise ApiClientError(
            status_code, title, detail, type_uri, request_id=request_id, errors=errors
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Execute HTTP request with retry logic for transient network/server failures."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        req_headers = self._build_headers(headers)
        req_timeout = timeout if timeout is not None else self.timeout

        is_idempotent = method.upper() in ("GET", "HEAD", "OPTIONS")
        attempts = 1 + (self.max_retries if is_idempotent else 0)

        for attempt in range(attempts):
            try:
                response = self._http_client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                    headers=req_headers,
                    timeout=req_timeout,
                )
                if response.is_success:
                    return response

                # Retry on transient upstream errors
                if (
                    is_idempotent
                    and response.status_code in (502, 503, 504)
                    and attempt < attempts - 1
                ):
                    backoff = 0.2 * (2**attempt)
                    time.sleep(backoff)
                    continue

                self._handle_response_error(response)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                if attempt < attempts - 1:
                    time.sleep(0.2 * (2**attempt))
                    continue
                err_msg = f"Failed to connect to RetailPulse API at {self.base_url}: {exc}"
                raise ApiConnectionError(err_msg) from exc

        raise ApiConnectionError(f"Request to {url} failed after {attempts} attempts.")

    # --- Convenience HTTP Verbs ---

    def get(self, path: str, **kwargs: Any) -> Any:
        """Perform GET request and parse JSON response."""
        res = self.request("GET", path, **kwargs)
        if res.status_code == 204 or not res.content:
            return None
        return res.json()

    def post(self, path: str, **kwargs: Any) -> Any:
        """Perform POST request and parse JSON response."""
        res = self.request("POST", path, **kwargs)
        if res.status_code == 204 or not res.content:
            return None
        return res.json()

    def put(self, path: str, **kwargs: Any) -> Any:
        """Perform PUT request and parse JSON response."""
        res = self.request("PUT", path, **kwargs)
        if res.status_code == 204 or not res.content:
            return None
        return res.json()

    def patch(self, path: str, **kwargs: Any) -> Any:
        """Perform PATCH request and parse JSON response."""
        res = self.request("PATCH", path, **kwargs)
        if res.status_code == 204 or not res.content:
            return None
        return res.json()

    def delete(self, path: str, **kwargs: Any) -> Any:
        """Perform DELETE request."""
        res = self.request("DELETE", path, **kwargs)
        if res.status_code == 204 or not res.content:
            return None
        return res.json()

    # --- Domain Specific Endpoints ---

    def login(self, username: str, password: str) -> dict[str, Any]:
        """Authenticate with email and password, returning TokenResponse dict and setting token."""
        data = cast(
            dict[str, Any],
            self.post(
                "/api/v1/auth/login",
                json={"username": username, "password": password},
            ),
        )
        token = data.get("access_token")
        if token:
            self.set_token(token)
        return data

    def get_me(self) -> UserProfile:
        """Retrieve profile of currently authenticated user."""
        data = self.get("/api/v1/auth/me")
        return UserProfile(
            user_id=str(data["user_id"]),
            email=data["email"],
            full_name=data.get("full_name"),
            role=data["role"],
        )

    def get_status_strip_data(self) -> StatusStripData:
        """Fetch system metadata for the global status strip.

        Reads:
        - /api/v1/kpis/overview -> data_as_of
        - /api/v1/models -> champion forecast model
        - /api/v1/pipelines/runs -> last pipeline status and timestamp
        """
        data_as_of: str | None = None
        champion_model: str | None = None
        last_status: str | None = None
        last_time: str | None = None
        is_stale = False
        stale_reason: str | None = None

        try:
            kpis = self.get("/api/v1/kpis/overview")
            data_as_of = kpis.get("data_as_of")
        except Exception as exc:
            logger.debug("Failed to fetch KPIs for status strip: %s", exc)

        try:
            models = self.get("/api/v1/models", params={"task": "forecast"})
            items = models.get("items", [])
            for m in items:
                if m.get("is_champion"):
                    algo = m.get("algorithm", "Unknown")
                    champion_model = algo.capitalize() if isinstance(algo, str) else str(algo)
                    break
        except Exception as exc:
            logger.debug("Failed to fetch models for status strip: %s", exc)

        try:
            pipe_runs = self.get("/api/v1/pipelines/runs", params={"page_size": 5})
            items = pipe_runs.get("items", [])
            if items:
                latest = items[0]
                last_status = latest.get("status", "unknown").upper()
                last_time = latest.get("queued_at") or latest.get("started_at")
        except Exception as exc:
            logger.debug("Failed to fetch pipeline runs for status strip: %s", exc)

        return StatusStripData(
            data_as_of=data_as_of,
            champion_model=champion_model,
            last_pipeline_status=last_status,
            last_pipeline_time=last_time,
            is_stale_warning=is_stale,
            stale_reason=stale_reason,
        )

    def close(self) -> None:
        """Close underlying HTTP transport connections."""
        self._http_client.close()

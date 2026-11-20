"""Health and readiness diagnostic probes.

Implements P5-01, TRD §7, PRD §7.7 (FR-API-04), and API §2.8:
- GET /health/live: Kubernetes liveness probe (process is up).
- GET /health/ready: Kubernetes readiness probe. Verifies database connectivity
  and Alembic migrations at head. MLflow and Prefect are checked asynchronously
  and reported as degraded, not failing the readiness probe.
- GET /health/startup: Kubernetes startup probe.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import httpx
import sqlalchemy as sa
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from retailpulse.api.errors import create_problem_response
from retailpulse.config import Settings, get_settings
from retailpulse.db.session import check_database, get_engine
from retailpulse.logging import get_logger

logger = get_logger("retailpulse.api.health")

router = APIRouter(prefix="/health", tags=["Health"])


async def _check_service_reachability(url: str, path: str = "/health") -> str:
    """Check HTTP health endpoint of an external service with tight 1s timeout."""
    if not url:
        return "degraded"

    target_url = f"{url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(target_url)
            return "ready" if resp.status_code < 400 else "degraded"
    except Exception:
        return "degraded"


@router.get(
    "/live",
    summary="Liveness Probe",
    response_description="Process is up and serving requests",
    status_code=status.HTTP_200_OK,
)
async def health_live() -> dict[str, str]:
    """Process liveness probe indicating application process is active."""
    return {"status": "ok"}


@router.get(
    "/startup",
    summary="Startup Probe",
    response_description="Application startup complete",
    status_code=status.HTTP_200_OK,
)
async def health_startup() -> dict[str, str]:
    """Startup probe indicating container initialization is complete."""
    return {"status": "ok"}


@router.get(
    "",
    summary="Root Health Probe",
    response_description="Service status overview",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def health_root() -> dict[str, str]:
    """Alias for root /health route."""
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Readiness Probe",
    response_description="Database reachable, migrations current, external services evaluated",
)
async def health_ready(
    request: Request,
    engine: Annotated[sa.Engine, Depends(get_engine)],
) -> JSONResponse:
    """Readiness probe evaluating DB connection, migrations at head, and external services."""
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()

    db_connected = False
    migrations_current = False
    current_rev: str | None = None
    head_rev: str | None = None
    db_error_msg: str | None = None

    # 1. Evaluate database connectivity and Alembic migrations state
    try:
        db_check = await asyncio.to_thread(check_database, engine)
        db_connected = bool(db_check.get("connected", False))
        migrations_current = bool(db_check.get("is_up_to_date", False))
        current_rev = db_check.get("current_revision")
        head_rev = db_check.get("head_revision")
        if not db_connected:
            db_error_msg = db_check.get("error") or "Database connection verification failed"
        elif not migrations_current:
            db_error_msg = f"Migrations not at head (current={current_rev}, head={head_rev})"
    except Exception as exc:
        db_connected = False
        migrations_current = False
        db_error_msg = "Database connectivity check failed"
        logger.error("Readiness database check failed", error=str(exc))

    # 2. Evaluate optional dependencies concurrently (degraded, not failed per API §2.8)
    mlflow_task = _check_service_reachability(settings.mlflow_tracking_uri, path="/health")
    prefect_task = _check_service_reachability(settings.prefect_api_url, path="/health")
    mlflow_status, prefect_status = await asyncio.gather(mlflow_task, prefect_task)

    is_ready = db_connected and migrations_current

    payload: dict[str, Any] = {
        "status": "ready" if is_ready else "not_ready",
        "database": {
            "connected": db_connected,
            "migrations": "current" if migrations_current else "outdated",
            "current_revision": current_rev,
            "head_revision": head_rev,
        },
        "dependencies": {
            "mlflow": mlflow_status,
            "prefect": prefect_status,
        },
    }

    if not is_ready:
        errors = []
        if not db_connected:
            errors.append({"field": "database.connected", "message": "Database is unreachable"})
        if not migrations_current:
            errors.append(
                {
                    "field": "database.migrations",
                    "message": f"Migrations not at head: current={current_rev}, head={head_rev}",
                }
            )

        req_id = getattr(request.state, "request_id", None)
        return create_problem_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            title="Service Not Ready",
            detail=db_error_msg or "Service database or migration requirements are not satisfied",
            instance=request.url.path,
            request_id=req_id,
            errors=errors,
            type_suffix="model_unavailable",
        )

    return JSONResponse(
        content=payload,
        status_code=status.HTTP_200_OK,
    )

"""Pipelines orchestration and execution trigger API router.

Implements P5-08, API §2.7, §3.10, and SECURITY §4 (SEC-AUTHZ-02):
- GET /api/v1/pipelines/runs: Paginated list of pipeline runs with filters (Viewer+).
- GET /api/v1/pipelines/runs/{run_id}: Pipeline run execution details (Viewer+).
- POST /api/v1/pipelines/{flow_name}/trigger: Concurrency-gated pipeline trigger with RBAC
  (Analyst+ for etl and score_all; Admin only for training and drift_check).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import get_current_user, require_role
from retailpulse.api.pagination import PaginatedResponse, PaginationParams
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.ops import PipelineRun
from retailpulse.db.session import get_db
from retailpulse.errors import (
    ConflictError,
    DomainValidationError,
    NotFoundError,
    PermissionDeniedError,
    ServiceUnavailableError,
)
from retailpulse.logging import get_logger
from retailpulse.pipelines.trigger import PipelineTrigger, get_pipeline_trigger
from retailpulse.security.audit import log_audit
from retailpulse.timeutil import utc_now

logger = get_logger(__name__)

router = APIRouter(tags=["Pipelines"])

ALLOWED_FLOWS: frozenset[str] = frozenset(
    {
        "etl",
        "score_all",
        "train_forecast",
        "train_segmentation",
        "train_churn",
        "drift_check",
    }
)

ANALYST_ALLOWED_FLOWS: frozenset[str] = frozenset({"etl", "score_all"})

ALLOWED_SORT_FIELDS: frozenset[str] = frozenset(
    {"queued_at", "started_at", "finished_at", "status", "flow_name"}
)


# --- Request and Response Models ---


class TriggerPipelineRequest(BaseModel):
    """Optional parameters supplied when triggering a pipeline flow."""

    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any] = Field(default_factory=dict)


class TriggerPipelineResponse(BaseModel):
    """Pipeline trigger dispatch acceptance response matching API §3.10."""

    model_config = ConfigDict(extra="forbid")

    pipeline_run_id: uuid.UUID
    flow_name: str
    status: str


class PipelineRunItem(BaseModel):
    """Detailed pipeline execution run record."""

    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    flow_name: str
    trigger: str
    status: str
    params: dict[str, Any]
    error: str | None = None
    queued_at: datetime.datetime
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    prefect_flow_run_id: str | None = None


# --- Endpoints ---


@router.get(
    "/pipelines/runs",
    response_model=PaginatedResponse[PipelineRunItem],
    operation_id="list_pipeline_runs",
    summary="List pipeline execution runs (Viewer+)",
)
def list_pipeline_runs(
    pagination: Annotated[PaginationParams, Depends()],
    session: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[AppUser, Depends(require_role("viewer"))],
    flow_name: str | None = None,
    status: str | None = None,
    sort: str = "queued_at:desc",
) -> PaginatedResponse[PipelineRunItem]:
    """Retrieve a paginated list of pipeline execution runs with optional filters."""
    stmt = sa.select(PipelineRun)

    if flow_name:
        clean_flow = flow_name.strip().lower()
        if clean_flow not in ALLOWED_FLOWS:
            raise DomainValidationError(
                f"Invalid flow_name '{flow_name}'. Allowed flows: {sorted(ALLOWED_FLOWS)}"
            )
        stmt = stmt.where(PipelineRun.flow_name == clean_flow)

    if status:
        stmt = stmt.where(PipelineRun.status == status.strip().lower())

    # Sorting
    sort_parts = sort.strip().split(":")
    sort_col_name = sort_parts[0].lower()
    sort_dir = sort_parts[1].lower() if len(sort_parts) > 1 else "desc"

    if sort_col_name not in ALLOWED_SORT_FIELDS:
        raise DomainValidationError(
            f"Invalid sort field '{sort_col_name}'. Allowed: {sorted(ALLOWED_SORT_FIELDS)}"
        )

    col = getattr(PipelineRun, sort_col_name)
    stmt = stmt.order_by(col.desc() if sort_dir == "desc" else col.asc())

    # Count total
    count_stmt = sa.select(sa.func.count()).select_from(stmt.subquery())
    total = session.execute(count_stmt).scalar_one()

    # Pagination
    items_stmt = stmt.offset(pagination.offset).limit(pagination.limit)
    rows = session.execute(items_stmt).scalars().all()

    items = [
        PipelineRunItem(
            run_id=r.run_id,
            flow_name=r.flow_name,
            trigger=r.trigger,
            status=r.status,
            params=r.params,
            error=r.error,
            queued_at=r.queued_at,
            started_at=r.started_at,
            finished_at=r.finished_at,
            prefect_flow_run_id=r.prefect_flow_run_id,
        )
        for r in rows
    ]

    return PaginatedResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )


@router.get(
    "/pipelines/runs/{run_id}",
    response_model=PipelineRunItem,
    operation_id="get_pipeline_run",
    summary="Get details of a specific pipeline run (Viewer+)",
)
def get_pipeline_run(
    run_id: uuid.UUID,
    session: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[AppUser, Depends(require_role("viewer"))],
) -> PipelineRunItem:
    """Retrieve execution state, timing, and errors for a specific pipeline run."""
    run = session.get(PipelineRun, run_id)
    if run is None:
        raise NotFoundError(f"Pipeline run {run_id} not found")

    return PipelineRunItem(
        run_id=run.run_id,
        flow_name=run.flow_name,
        trigger=run.trigger,
        status=run.status,
        params=run.params,
        error=run.error,
        queued_at=run.queued_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        prefect_flow_run_id=run.prefect_flow_run_id,
    )


@router.post(
    "/pipelines/{flow_name}/trigger",
    response_model=TriggerPipelineResponse,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="trigger_pipeline_flow",
    summary="Trigger an orchestration pipeline run (RBAC enforced per flow)",
)
async def trigger_pipeline(  # noqa: PLR0917
    flow_name: str,
    response: Response,
    current_user: Annotated[AppUser, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_db)],
    trigger: Annotated[PipelineTrigger, Depends(get_pipeline_trigger)],
    payload: TriggerPipelineRequest | None = None,
) -> TriggerPipelineResponse:
    """Trigger an asynchronous workflow with concurrency gating and flow-specific RBAC."""
    clean_flow = flow_name.strip().lower()
    if clean_flow not in ALLOWED_FLOWS:
        raise DomainValidationError(
            f"Invalid flow name '{flow_name}'. Allowed flows: {sorted(ALLOWED_FLOWS)}"
        )

    # SEC-AUTHZ-02: Flow-specific authorization check
    if clean_flow in ANALYST_ALLOWED_FLOWS:
        if current_user.role not in ("analyst", "admin"):
            raise PermissionDeniedError(
                f"Role 'analyst' or 'admin' required to trigger flow '{clean_flow}'"
            )
    elif current_user.role != "admin":
        raise PermissionDeniedError(f"Role 'admin' required to trigger flow '{clean_flow}'")

    # API §1.1 & §2.7: Concurrency gate (return 409 if already queued or running)
    active_stmt = sa.select(PipelineRun).where(
        PipelineRun.flow_name == clean_flow,
        PipelineRun.status.in_(["queued", "running"]),
    )
    active_run = session.execute(active_stmt).scalar_one_or_none()
    if active_run is not None:
        raise ConflictError(
            f"Pipeline flow '{clean_flow}' is already active (run_id={active_run.run_id})"
        )

    run_params = payload.params if payload else {}
    run_id = uuid.uuid4()

    pipe_run = PipelineRun(
        run_id=run_id,
        flow_name=clean_flow,
        trigger="manual",
        status="queued",
        params=run_params,
        requested_by=current_user.user_id,
        queued_at=utc_now(),
    )
    session.add(pipe_run)
    session.flush()

    # Dispatch via trigger abstraction
    trigger_ok = await trigger.trigger(
        flow_name=clean_flow,
        run_id=run_id,
        params=run_params,
    )
    if not trigger_ok:
        pipe_run.status = "failed"
        pipe_run.error = "Pipeline trigger failed to dispatch workflow"
        session.commit()
        raise ServiceUnavailableError(
            f"Pipeline trigger service unavailable for flow '{clean_flow}'"
        )

    # Audit logging
    log_audit(
        session,
        action="trigger",
        user_id=current_user.user_id,
        resource=f"pipeline:{clean_flow}:{run_id}",
        details={
            "flow_name": clean_flow,
            "run_id": str(run_id),
            "trigger": "manual",
            "params": run_params,
        },
    )
    session.commit()

    response.headers["Location"] = f"/api/v1/pipelines/runs/{run_id}"
    return TriggerPipelineResponse(
        pipeline_run_id=run_id,
        flow_name=clean_flow,
        status="queued",
    )

"""Model catalog, training dispatch, and champion promotion API router.

Implements API §2.6, §3.10-§3.11:
- GET /api/v1/models: Filtered and paginated model runs (Viewer+).
- GET /api/v1/models/{run_id}: Model run details; mlflow_url shown to Admins only (Viewer+).
- POST /api/v1/models/{task}/train: Dispatch training pipeline, returns 202 Accepted (Admin only).
- POST /api/v1/models/{run_id}/promote: Promote champion model, triggers score_all (Admin only).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import require_role
from retailpulse.api.limiter import limiter
from retailpulse.api.pagination import (
    PaginatedResponse,
    PaginationParams,
    apply_sort,
    get_pagination_params,
    parse_sort,
)
from retailpulse.config import get_settings
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.ml import ModelRun
from retailpulse.db.models.ops import PipelineRun
from retailpulse.db.session import get_db
from retailpulse.errors import ConflictError, DomainValidationError, NotFoundError
from retailpulse.logging import get_logger
from retailpulse.security.audit import log_audit

router = APIRouter(tags=["Models"])
logger = get_logger(__name__)

VALID_TASKS: frozenset[str] = frozenset({"forecast", "segmentation", "churn"})
MODEL_SORT_WHITELIST: dict[str, Any] = {
    "trained_at": ModelRun.trained_at,
    "algorithm": ModelRun.algorithm,
    "status": ModelRun.status,
    "task": ModelRun.task,
}


# --- Schemas ---


class ModelListItem(BaseModel):
    """Model run item in paginated listing."""

    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    task: str
    algorithm: str
    status: str
    is_champion: bool
    trained_at: datetime.datetime
    data_end: datetime.date | None = None
    metrics: dict[str, Any]
    params: dict[str, Any]
    git_sha: str
    dataset_hash: str
    seed: int
    mlflow_url: str | None = None


class ModelDetailResponse(BaseModel):
    """Model run details matching API §3.11."""

    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    task: str
    algorithm: str
    status: str
    is_champion: bool
    trained_at: datetime.datetime
    data_end: datetime.date | None = None
    metrics: dict[str, Any]
    params: dict[str, Any]
    git_sha: str
    dataset_hash: str
    seed: int
    mlflow_url: str | None = None


class TrainModelRequest(BaseModel):
    """Request payload to initiate a training pipeline."""

    model_config = ConfigDict(extra="forbid")

    algorithms: list[str] | None = None
    params: dict[str, Any] | None = None


class TrainModelResponse(BaseModel):
    """Accepted training dispatch response matching API §3.10."""

    model_config = ConfigDict(extra="forbid")

    pipeline_run_id: uuid.UUID
    flow_name: str
    status: str


class PromoteModelResponse(BaseModel):
    """Model promotion response matching API §3.10."""

    model_config = ConfigDict(extra="forbid")

    task: str
    run_id: uuid.UUID
    previous_champion_run_id: uuid.UUID | None = None
    score_all_pipeline_run_id: uuid.UUID


# --- Helpers ---


def _build_mlflow_url(mlflow_run_id: str | None) -> str | None:
    """Construct external MLflow URL if configured and run ID present."""
    if not mlflow_run_id:
        return None
    settings = get_settings()
    base_uri = settings.mlflow_tracking_uri.rstrip("/")
    return f"{base_uri}/#/experiments/0/runs/{mlflow_run_id}"


def _map_model_to_item(model: ModelRun, is_admin: bool) -> ModelListItem:
    """Map DB ModelRun entity to ModelListItem, protecting MLflow URL from non-admins."""
    mlflow_url = _build_mlflow_url(model.mlflow_run_id) if is_admin else None
    return ModelListItem(
        run_id=model.run_id,
        task=model.task,
        algorithm=model.algorithm,
        status=model.status,
        is_champion=model.is_champion,
        trained_at=model.trained_at,
        data_end=model.data_end,
        metrics=model.metrics,
        params=model.params,
        git_sha=model.git_sha,
        dataset_hash=model.dataset_hash,
        seed=model.seed,
        mlflow_url=mlflow_url,
    )


# --- Endpoints ---


@router.get(
    "/models",
    response_model=PaginatedResponse[ModelListItem],
    operation_id="list_models",
    summary="List trained model runs",
)
def list_models(  # noqa: PLR0917
    session: Annotated[Session, Depends(get_db)],
    current_user: Annotated[AppUser, Depends(require_role("viewer"))],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    task: Annotated[
        str | None,
        Query(description="Filter by task (forecast, segmentation, churn)"),
    ] = None,
    status_filter: Annotated[
        str | None,
        Query(alias="status", description="Filter by status (succeeded, failed, running)"),
    ] = None,
    is_champion: Annotated[
        bool | None,
        Query(description="Filter by champion status"),
    ] = None,
    sort: Annotated[
        str,
        Query(description="Sort expression. Default: trained_at:desc"),
    ] = "trained_at:desc",
) -> PaginatedResponse[ModelListItem]:
    """List and filter model runs with pagination and sorting."""
    query: sa.Select[Any] = sa.select(ModelRun)

    if task:
        task_clean = task.strip().lower()
        if task_clean not in VALID_TASKS:
            raise DomainValidationError(
                f"Invalid task '{task}'. Allowed tasks: {sorted(VALID_TASKS)}"
            )
        query = query.where(ModelRun.task == task_clean)

    if status_filter:
        query = query.where(ModelRun.status == status_filter.strip().lower())

    if is_champion is not None:
        query = query.where(ModelRun.is_champion.is_(is_champion))

    # Total count
    count_stmt = sa.select(sa.func.count()).select_from(query.subquery())
    total = session.scalar(count_stmt) or 0

    # Sort & Paginate
    sort_clauses = parse_sort(
        sort,
        allowed_fields=MODEL_SORT_WHITELIST.keys(),
        default="trained_at:desc",
    )
    query = apply_sort(query, sort_clauses, MODEL_SORT_WHITELIST)
    offset = (pagination.page - 1) * pagination.page_size
    query = query.offset(offset).limit(pagination.page_size)

    rows = session.execute(query).scalars().all()
    is_admin = current_user.role == "admin"
    items = [_map_model_to_item(m, is_admin) for m in rows]

    return PaginatedResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )


@router.get(
    "/models/{run_id}",
    response_model=ModelDetailResponse,
    operation_id="get_model_run",
    summary="Get details of a specific model run",
)
def get_model_run(
    run_id: uuid.UUID,
    session: Annotated[Session, Depends(get_db)],
    current_user: Annotated[AppUser, Depends(require_role("viewer"))],
) -> ModelDetailResponse:
    """Retrieve run details and evaluation metrics. MLflow link is shown to admins only."""
    model = session.get(ModelRun, run_id)
    if model is None:
        raise NotFoundError(f"Model run {run_id} not found")

    is_admin = current_user.role == "admin"
    mlflow_url = _build_mlflow_url(model.mlflow_run_id) if is_admin else None

    return ModelDetailResponse(
        run_id=model.run_id,
        task=model.task,
        algorithm=model.algorithm,
        status=model.status,
        is_champion=model.is_champion,
        trained_at=model.trained_at,
        data_end=model.data_end,
        metrics=model.metrics,
        params=model.params,
        git_sha=model.git_sha,
        dataset_hash=model.dataset_hash,
        seed=model.seed,
        mlflow_url=mlflow_url,
    )


@router.post(
    "/models/{task}/train",
    response_model=TrainModelResponse,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="dispatch_training_pipeline",
    summary="Trigger model training pipeline (Admin only)",
)
@limiter.limit("20/hour")
def train_model(  # noqa: PLR0917
    request: Request,
    task: str,
    payload: TrainModelRequest,
    current_user: Annotated[AppUser, Depends(require_role("admin"))],
    session: Annotated[Session, Depends(get_db)],
    response: Response,
) -> TrainModelResponse:
    """Trigger an asynchronous model training pipeline run."""
    del request
    task_clean = task.strip().lower()
    if task_clean not in VALID_TASKS:
        raise DomainValidationError(f"Invalid task '{task}'. Allowed tasks: {sorted(VALID_TASKS)}")

    flow_name = f"train_{task_clean}"

    # Check for active pipeline run of this flow
    active_stmt = sa.select(PipelineRun).where(
        PipelineRun.flow_name == flow_name,
        PipelineRun.status.in_(["queued", "running"]),
    )
    active_run = session.execute(active_stmt).scalar_one_or_none()
    if active_run is not None:
        raise ConflictError(
            f"Pipeline run for {flow_name} is already active (run_id={active_run.run_id})"
        )

    # Create new PipelineRun record
    pipe_run = PipelineRun(
        run_id=uuid.uuid4(),
        flow_name=flow_name,
        trigger="manual",
        status="queued",
        params=payload.model_dump(),
        requested_by=current_user.user_id,
    )
    session.add(pipe_run)

    # SEC-LOG-01: Audit log trigger action
    log_audit(
        session=session,
        action="trigger",
        user_id=current_user.user_id,
        resource=f"pipeline:{flow_name}",
        details={"pipeline_run_id": str(pipe_run.run_id), "task": task_clean},
    )

    session.commit()
    session.refresh(pipe_run)

    location_url = f"/api/v1/pipelines/runs/{pipe_run.run_id}"
    response.headers["Location"] = location_url

    return TrainModelResponse(
        pipeline_run_id=pipe_run.run_id,
        flow_name=flow_name,
        status="queued",
    )


@router.post(
    "/models/{run_id}/promote",
    response_model=PromoteModelResponse,
    operation_id="promote_model_champion",
    summary="Promote model run to champion and trigger score_all (Admin only)",
)
@limiter.limit("20/hour")
def promote_model(
    request: Request,
    run_id: uuid.UUID,
    current_user: Annotated[AppUser, Depends(require_role("admin"))],
    session: Annotated[Session, Depends(get_db)],
) -> PromoteModelResponse:
    """Promote a successful, gate-passed model run to champion status."""
    del request
    model = session.get(ModelRun, run_id)
    if model is None:
        raise NotFoundError(f"Model run {run_id} not found")

    # 1. Verify status is succeeded
    if model.status != "succeeded":
        raise ConflictError(
            f"Cannot promote model run {run_id} with status '{model.status}'; "
            "only succeeded runs may be promoted."
        )

    # 2. Verify model passed its evaluation gate
    gate_passed = model.metrics.get("gate_passed", True)
    if not gate_passed:
        gate_reason = model.metrics.get("gate_reason", "Failed evaluation gate")
        raise ConflictError(f"Model run {run_id} did not pass the promotion gate: {gate_reason}")

    task = model.task

    # 3. Find current champion for this task
    prev_champ = session.execute(
        sa.select(ModelRun).where(
            ModelRun.task == task,
            ModelRun.is_champion.is_(True),
        )
    ).scalar_one_or_none()

    prev_champ_id: uuid.UUID | None = None
    if prev_champ is not None:
        prev_champ_id = prev_champ.run_id
        if prev_champ.run_id != model.run_id:
            prev_champ.is_champion = False

    model.is_champion = True

    # 4. Trigger score_all pipeline run
    score_active = session.execute(
        sa.select(PipelineRun).where(
            PipelineRun.flow_name == "score_all",
            PipelineRun.status.in_(["queued", "running"]),
        )
    ).scalar_one_or_none()

    if score_active is not None:
        score_all_id = score_active.run_id
    else:
        score_run = PipelineRun(
            run_id=uuid.uuid4(),
            flow_name="score_all",
            trigger="promote",
            status="queued",
            params={"promoted_run_id": str(model.run_id), "task": task},
            requested_by=current_user.user_id,
        )
        session.add(score_run)
        score_all_id = score_run.run_id

    # 5. SEC-LOG-01: Audit log promote action
    log_audit(
        session=session,
        action="promote",
        user_id=current_user.user_id,
        resource=f"model:{task}:{model.run_id}",
        details={
            "previous_champion_run_id": str(prev_champ_id) if prev_champ_id else None,
            "score_all_pipeline_run_id": str(score_all_id),
        },
    )

    session.commit()

    return PromoteModelResponse(
        task=task,
        run_id=model.run_id,
        previous_champion_run_id=prev_champ_id,
        score_all_pipeline_run_id=score_all_id,
    )

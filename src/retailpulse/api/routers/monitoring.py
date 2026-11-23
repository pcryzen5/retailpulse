"""Monitoring API endpoints for data/prediction drift reports and realized performance snapshots.

Implements API §2.6:
- GET /api/v1/monitoring/drift: List drift reports (task, kind, pagination) (Viewer+).
- GET /api/v1/monitoring/drift/{report_id}: Report summary JSON (Viewer+).
- GET /api/v1/monitoring/drift/{report_id}/html: Stored Evidently HTML with sandbox CSP (Viewer+).
- GET /api/v1/monitoring/performance: Realized-error performance snapshots (Viewer+).
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import require_role
from retailpulse.api.pagination import (
    PaginatedResponse,
    PaginationParams,
    apply_sort,
    get_pagination_params,
    parse_sort,
)
from retailpulse.config import get_settings
from retailpulse.db.models.ops import DriftReport, ModelPerformance
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError, NotFoundError

router = APIRouter(tags=["Monitoring"])

VALID_TASKS: frozenset[str] = frozenset({"forecast", "segmentation", "churn"})
VALID_DRIFT_KINDS: frozenset[str] = frozenset({"data", "prediction"})

DRIFT_SORT_WHITELIST: dict[str, Any] = {
    "created_at": DriftReport.created_at,
    "task": DriftReport.task,
    "kind": DriftReport.kind,
}

PERF_SORT_WHITELIST: dict[str, Any] = {
    "created_at": ModelPerformance.created_at,
    "window_start": ModelPerformance.window_start,
    "metric": ModelPerformance.metric,
    "value": ModelPerformance.value,
}


# --- Schemas ---


class DriftReportListItem(BaseModel):
    """Summary item in drift reports list."""

    model_config = ConfigDict(extra="forbid")

    report_id: uuid.UUID
    task: str
    model_run_id: uuid.UUID | None = None
    kind: str
    reference_start: datetime.date
    reference_end: datetime.date
    current_start: datetime.date
    current_end: datetime.date
    dataset_drift: bool
    share_drifted_features: float | None = None
    created_at: datetime.datetime


class DriftReportDetailResponse(BaseModel):
    """Full drift report metadata and test summary JSON."""

    model_config = ConfigDict(extra="forbid")

    report_id: uuid.UUID
    task: str
    model_run_id: uuid.UUID | None = None
    kind: str
    reference_start: datetime.date
    reference_end: datetime.date
    current_start: datetime.date
    current_end: datetime.date
    dataset_drift: bool
    share_drifted_features: float | None = None
    summary: dict[str, Any]
    report_path: str
    created_at: datetime.datetime


class ModelPerformanceItem(BaseModel):
    """Realized error and performance snapshot matching API §2.6."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: int
    model_run_id: uuid.UUID
    task: str
    window_start: datetime.date
    window_end: datetime.date
    metric: str
    value: float
    baseline_value: float | None = None
    n_observations: int
    created_at: datetime.datetime


# --- Endpoints ---


@router.get(
    "/monitoring/drift",
    response_model=PaginatedResponse[DriftReportListItem],
    operation_id="list_drift_reports",
    summary="List Evidently drift reports",
    dependencies=[Depends(require_role("viewer"))],
)
def list_drift_reports(
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    task: Annotated[
        str | None,
        Query(description="Filter by task (forecast, segmentation, churn)"),
    ] = None,
    kind: Annotated[
        str | None,
        Query(description="Filter by drift kind (data, prediction)"),
    ] = None,
    sort: Annotated[
        str,
        Query(description="Sort expression. Default: created_at:desc"),
    ] = "created_at:desc",
) -> PaginatedResponse[DriftReportListItem]:
    """Retrieve filtered and paginated drift reports."""
    query: sa.Select[Any] = sa.select(DriftReport)

    if task:
        task_clean = task.strip().lower()
        if task_clean not in VALID_TASKS:
            raise DomainValidationError(
                f"Invalid task '{task}'. Allowed tasks: {sorted(VALID_TASKS)}"
            )
        query = query.where(DriftReport.task == task_clean)

    if kind:
        kind_clean = kind.strip().lower()
        if kind_clean not in VALID_DRIFT_KINDS:
            raise DomainValidationError(
                f"Invalid kind '{kind}'. Allowed kinds: {sorted(VALID_DRIFT_KINDS)}"
            )
        query = query.where(DriftReport.kind == kind_clean)

    count_stmt = sa.select(sa.func.count()).select_from(query.subquery())
    total = session.scalar(count_stmt) or 0

    sort_clauses = parse_sort(
        sort,
        allowed_fields=DRIFT_SORT_WHITELIST.keys(),
        default="created_at:desc",
    )
    query = apply_sort(query, sort_clauses, DRIFT_SORT_WHITELIST)
    offset = (pagination.page - 1) * pagination.page_size
    query = query.offset(offset).limit(pagination.page_size)

    rows = session.execute(query).scalars().all()
    items = [
        DriftReportListItem(
            report_id=r.report_id,
            task=r.task,
            model_run_id=r.model_run_id,
            kind=r.kind,
            reference_start=r.reference_start,
            reference_end=r.reference_end,
            current_start=r.current_start,
            current_end=r.current_end,
            dataset_drift=r.dataset_drift,
            share_drifted_features=(
                float(r.share_drifted_features) if r.share_drifted_features is not None else None
            ),
            created_at=r.created_at,
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
    "/monitoring/drift/{report_id}",
    response_model=DriftReportDetailResponse,
    operation_id="get_drift_report",
    summary="Get drift report summary JSON",
    dependencies=[Depends(require_role("viewer"))],
)
def get_drift_report(
    report_id: uuid.UUID,
    session: Annotated[Session, Depends(get_db)],
) -> DriftReportDetailResponse:
    """Retrieve full drift report metadata and test results summary."""
    report = session.get(DriftReport, report_id)
    if report is None:
        raise NotFoundError(f"Drift report {report_id} not found")

    return DriftReportDetailResponse(
        report_id=report.report_id,
        task=report.task,
        model_run_id=report.model_run_id,
        kind=report.kind,
        reference_start=report.reference_start,
        reference_end=report.reference_end,
        current_start=report.current_start,
        current_end=report.current_end,
        dataset_drift=report.dataset_drift,
        share_drifted_features=(
            float(report.share_drifted_features)
            if report.share_drifted_features is not None
            else None
        ),
        summary=report.summary,
        report_path=report.report_path,
        created_at=report.created_at,
    )


@router.get(
    "/monitoring/drift/{report_id}/html",
    response_class=HTMLResponse,
    operation_id="get_drift_report_html",
    summary="Serve stored Evidently HTML report with sandbox security headers",
    dependencies=[Depends(require_role("viewer"))],
)
def get_drift_report_html(
    report_id: uuid.UUID,
    session: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    """Serve stored Evidently visual HTML dashboard.

    Enforces API §2.6 security headers:
    - Content-Security-Policy: sandbox
    - X-Content-Type-Options: nosniff
    """
    report = session.get(DriftReport, report_id)
    if report is None:
        raise NotFoundError(f"Drift report {report_id} not found")

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    reports_dir = data_dir / "reports"
    file_path = (reports_dir / report.report_path).resolve()

    html_content: str | None = None
    if file_path.is_file():
        html_content = file_path.read_text(encoding="utf-8")
    elif "html_content" in report.summary:
        html_content = str(report.summary["html_content"])

    if html_content is None:
        raise NotFoundError(
            f"HTML report file for {report_id} not found on disk at {report.report_path}"
        )

    headers = {
        "Content-Security-Policy": "sandbox",
        "X-Content-Type-Options": "nosniff",
    }
    return HTMLResponse(content=html_content, headers=headers)


@router.get(
    "/monitoring/performance",
    response_model=PaginatedResponse[ModelPerformanceItem],
    operation_id="list_model_performance",
    summary="List realized-error model performance snapshots",
    dependencies=[Depends(require_role("viewer"))],
)
def list_model_performance(  # noqa: PLR0917
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    task: Annotated[
        str | None,
        Query(description="Filter by task (forecast, segmentation, churn)"),
    ] = None,
    model_run_id: Annotated[
        uuid.UUID | None,
        Query(description="Filter by model run UUID"),
    ] = None,
    metric: Annotated[
        str | None,
        Query(description="Filter by metric name (e.g. realized_wape)"),
    ] = None,
    sort: Annotated[
        str,
        Query(description="Sort expression. Default: created_at:desc"),
    ] = "created_at:desc",
) -> PaginatedResponse[ModelPerformanceItem]:
    """Retrieve realized model evaluation and ground-truth comparison snapshots."""
    query: sa.Select[Any] = sa.select(ModelPerformance)

    if task:
        task_clean = task.strip().lower()
        if task_clean not in VALID_TASKS:
            raise DomainValidationError(
                f"Invalid task '{task}'. Allowed tasks: {sorted(VALID_TASKS)}"
            )
        query = query.where(ModelPerformance.task == task_clean)

    if model_run_id:
        query = query.where(ModelPerformance.model_run_id == model_run_id)

    if metric:
        query = query.where(ModelPerformance.metric == metric.strip())

    count_stmt = sa.select(sa.func.count()).select_from(query.subquery())
    total = session.scalar(count_stmt) or 0

    sort_clauses = parse_sort(
        sort,
        allowed_fields=PERF_SORT_WHITELIST.keys(),
        default="created_at:desc",
    )
    query = apply_sort(query, sort_clauses, PERF_SORT_WHITELIST)
    offset = (pagination.page - 1) * pagination.page_size
    query = query.offset(offset).limit(pagination.page_size)

    rows = session.execute(query).scalars().all()
    items = [
        ModelPerformanceItem(
            snapshot_id=p.snapshot_id,
            model_run_id=p.model_run_id,
            task=p.task,
            window_start=p.window_start,
            window_end=p.window_end,
            metric=p.metric,
            value=p.value,
            baseline_value=p.baseline_value,
            n_observations=p.n_observations,
            created_at=p.created_at,
        )
        for p in rows
    ]

    return PaginatedResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )

"""Uploads API endpoints for sales data ingestion.

Implements P5-08, API §3.9, and SECURITY §5:
- POST /api/v1/uploads/sales: Upload sales file with streaming validation, SHA-256 deduplication,
  magic bytes check, and ETL flow triggering (Analyst+).
- GET /api/v1/uploads: Paginated list of ingestion batches (Analyst+).
- GET /api/v1/uploads/{batch_id}: Detailed ingestion batch status and quality results (Analyst+).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import require_role
from retailpulse.api.pagination import PaginatedResponse, PaginationParams
from retailpulse.config import Settings, get_settings
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.ops import IngestionBatch, PipelineRun
from retailpulse.db.session import get_db
from retailpulse.errors import (
    ConflictError,
    DomainValidationError,
    NotFoundError,
    ServiceUnavailableError,
)
from retailpulse.ingestion.validation import (
    finalize_upload_storage,
    stream_and_validate_upload,
)
from retailpulse.logging import get_logger
from retailpulse.pipelines.trigger import PipelineTrigger, get_pipeline_trigger
from retailpulse.security.audit import log_audit
from retailpulse.timeutil import utc_now

logger = get_logger(__name__)

router = APIRouter(tags=["Uploads"])

ALLOWED_SORT_FIELDS: frozenset[str] = frozenset(
    {"created_at", "status", "source_name", "started_at", "finished_at"}
)


def get_app_settings(request: Request) -> Settings:
    """Retrieve app settings from FastAPI state with fallback to get_settings()."""
    return getattr(request.app.state, "settings", None) or get_settings()


# --- Response Models ---


class UploadSalesResponse(BaseModel):
    """Response returned upon sales file upload (API §3.9)."""

    model_config = ConfigDict(extra="forbid")

    batch_id: uuid.UUID
    status: str
    pipeline_run_id: uuid.UUID | None = None


class IngestionBatchListItem(BaseModel):
    """List item for ingestion batches."""

    model_config = ConfigDict(extra="forbid")

    batch_id: uuid.UUID
    source_name: str
    original_filename: str | None = None
    status: str
    row_count_in: int | None = None
    row_count_loaded: int | None = None
    row_count_rejected: int | None = None
    created_at: datetime.datetime
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    error: str | None = None


class DataQualityItem(BaseModel):
    """Data quality validation check result within a batch."""

    model_config = ConfigDict(extra="forbid")

    check_name: str
    severity: str
    passed: bool
    affected_rows: int


class IngestionBatchDetailResponse(BaseModel):
    """Detailed response for a single ingestion batch (API §3.9)."""

    model_config = ConfigDict(extra="forbid")

    batch_id: uuid.UUID
    status: str
    source_name: str
    row_count_in: int | None = None
    row_count_loaded: int | None = None
    row_count_rejected: int | None = None
    quality: list[DataQualityItem] = []
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    error: str | None = None


# --- Endpoints ---


@router.post(
    "/uploads/sales",
    response_model=UploadSalesResponse,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="upload_sales_file",
    summary="Upload sales data file for ingestion (Analyst+)",
)
async def upload_sales(  # noqa: PLR0917
    response: Response,
    file: Annotated[UploadFile, File(description="CSV or XLSX sales data file")],
    current_user: Annotated[AppUser, Depends(require_role("analyst"))],
    session: Annotated[Session, Depends(get_db)],
    trigger: Annotated[PipelineTrigger, Depends(get_pipeline_trigger)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    source_name: Annotated[
        str, Form(description="Source adapter: online_retail_ii or generic_csv")
    ] = "generic_csv",
) -> UploadSalesResponse:
    """Validate, deduplicate, store, and trigger ETL ingestion for a sales file."""
    # 1. Stream, calculate SHA-256, verify size limit & magic bytes into temp file
    sha256_hex, temp_path, total_bytes, ext = await stream_and_validate_upload(
        upload_file=file,
        source_name=source_name,
        settings=settings,
    )

    # 2. Duplicate file detection per API §1.1 & §3.9
    existing_stmt = sa.select(IngestionBatch).where(
        IngestionBatch.source_name == source_name,
        IngestionBatch.file_sha256 == sha256_hex,
    )
    existing_batch = session.execute(existing_stmt).scalar_one_or_none()

    if existing_batch is not None:
        temp_path.unlink(missing_ok=True)
        # Find associated pipeline run if one was dispatched
        runs = (
            session.execute(
                sa.select(PipelineRun)
                .where(PipelineRun.flow_name == "etl")
                .order_by(PipelineRun.queued_at.desc())
            )
            .scalars()
            .all()
        )
        existing_run = next(
            (r for r in runs if str(r.params.get("batch_id")) == str(existing_batch.batch_id)),
            None,
        )
        response.status_code = status.HTTP_200_OK
        return UploadSalesResponse(
            batch_id=existing_batch.batch_id,
            status=existing_batch.status,
            pipeline_run_id=existing_run.run_id if existing_run else None,
        )

    # 3. Concurrency check: Ensure no active ETL flow is already queued or running
    active_etl = session.execute(
        sa.select(PipelineRun).where(
            PipelineRun.flow_name == "etl",
            PipelineRun.status.in_(["queued", "running"]),
        )
    ).scalar_one_or_none()

    if active_etl is not None:
        temp_path.unlink(missing_ok=True)
        raise ConflictError(
            f"An ETL pipeline run is already active (run_id={active_etl.run_id}). "
            "Please wait for it to finish."
        )

    # 4. Finalize storage: header validation and atomic move with mode 0600
    batch_id = uuid.uuid4()
    validated = finalize_upload_storage(
        temp_path=temp_path,
        batch_id=batch_id,
        ext=ext,
        source_name=source_name,
        original_filename=file.filename or "upload.csv",
        settings=settings,
        total_bytes=total_bytes,
        file_sha256=sha256_hex,
    )

    # 5. Persist IngestionBatch and PipelineRun
    batch = IngestionBatch(
        batch_id=batch_id,
        source_name=source_name,
        source_file=str(validated.saved_path),
        original_filename=validated.original_filename,
        file_sha256=sha256_hex,
        status="queued",
        uploaded_by=current_user.user_id,
        created_at=utc_now(),
    )
    session.add(batch)

    pipe_run_id = uuid.uuid4()
    pipe_run = PipelineRun(
        run_id=pipe_run_id,
        flow_name="etl",
        trigger="upload",
        status="queued",
        params={
            "batch_id": str(batch_id),
            "source_name": source_name,
            "file_path": str(validated.saved_path),
        },
        requested_by=current_user.user_id,
        queued_at=utc_now(),
    )
    session.add(pipe_run)
    session.flush()

    # 6. Trigger pipeline execution via PipelineTrigger abstraction
    trigger_ok = await trigger.trigger(
        flow_name="etl",
        run_id=pipe_run.run_id,
        params=pipe_run.params,
    )
    if not trigger_ok:
        pipe_run.status = "failed"
        pipe_run.error = "Pipeline trigger failed to dispatch ETL flow"
        session.commit()
        raise ServiceUnavailableError("Pipeline trigger failed to dispatch ETL flow")

    # 7. Audit log (SEC-LOG-01)
    log_audit(
        session,
        action="upload",
        user_id=current_user.user_id,
        resource=f"upload:{source_name}:{batch_id}",
        details={
            "batch_id": str(batch_id),
            "filename": validated.original_filename,
            "sha256": sha256_hex,
            "size_bytes": total_bytes,
            "pipeline_run_id": str(pipe_run.run_id),
        },
    )
    session.commit()

    response.headers["Location"] = f"/api/v1/uploads/{batch_id}"
    return UploadSalesResponse(
        batch_id=batch_id,
        status="queued",
        pipeline_run_id=pipe_run.run_id,
    )


@router.get(
    "/uploads",
    response_model=PaginatedResponse[IngestionBatchListItem],
    operation_id="list_uploads",
    summary="List file ingestion batches (Analyst+)",
)
def list_uploads(
    pagination: Annotated[PaginationParams, Depends()],
    session: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[AppUser, Depends(require_role("analyst"))],
    status: str | None = None,
    source_name: str | None = None,
    sort: str = "created_at:desc",
) -> PaginatedResponse[IngestionBatchListItem]:
    """Retrieve paginated list of file ingestion batches."""
    stmt = sa.select(IngestionBatch)

    if status:
        stmt = stmt.where(IngestionBatch.status == status.strip().lower())
    if source_name:
        stmt = stmt.where(IngestionBatch.source_name == source_name.strip().lower())

    # Sorting
    sort_parts = sort.strip().split(":")
    sort_col_name = sort_parts[0].lower()
    sort_dir = sort_parts[1].lower() if len(sort_parts) > 1 else "desc"

    if sort_col_name not in ALLOWED_SORT_FIELDS:
        raise DomainValidationError(
            f"Invalid sort field '{sort_col_name}'. Allowed: {sorted(ALLOWED_SORT_FIELDS)}"
        )

    col = getattr(IngestionBatch, sort_col_name)
    stmt = stmt.order_by(col.desc() if sort_dir == "desc" else col.asc())

    # Count total
    count_stmt = sa.select(sa.func.count()).select_from(stmt.subquery())
    total = session.execute(count_stmt).scalar_one()

    # Pagination
    items_stmt = stmt.offset(pagination.offset).limit(pagination.limit)
    rows = session.execute(items_stmt).scalars().all()

    items = [
        IngestionBatchListItem(
            batch_id=b.batch_id,
            source_name=b.source_name,
            original_filename=b.original_filename,
            status=b.status,
            row_count_in=b.row_count_in,
            row_count_loaded=b.row_count_loaded,
            row_count_rejected=b.row_count_rejected,
            created_at=b.created_at,
            started_at=b.started_at,
            finished_at=b.finished_at,
            error=b.error,
        )
        for b in rows
    ]

    return PaginatedResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )


@router.get(
    "/uploads/{batch_id}",
    response_model=IngestionBatchDetailResponse,
    operation_id="get_upload_batch",
    summary="Get detailed status of an ingestion batch (Analyst+)",
)
def get_upload_batch(
    batch_id: uuid.UUID,
    session: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[AppUser, Depends(require_role("analyst"))],
) -> IngestionBatchDetailResponse:
    """Retrieve details and quality check results for a specific ingestion batch."""
    batch = session.get(IngestionBatch, batch_id)
    if batch is None:
        raise NotFoundError(f"Ingestion batch {batch_id} not found")

    quality_items = [
        DataQualityItem(
            check_name=qr.check_name,
            severity=qr.severity,
            passed=qr.passed,
            affected_rows=qr.affected_rows,
        )
        for qr in batch.quality_results
    ]

    return IngestionBatchDetailResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        source_name=batch.source_name,
        row_count_in=batch.row_count_in,
        row_count_loaded=batch.row_count_loaded,
        row_count_rejected=batch.row_count_rejected,
        quality=quality_items,
        started_at=batch.started_at,
        finished_at=batch.finished_at,
        error=batch.error,
    )

"""Operations and pipeline observability models."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from retailpulse.db.base import Base


class IngestionBatch(Base):
    """File ingestion tracking batch stored in ops.ingestion_batch."""

    __tablename__ = "ingestion_batch"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'rejected')",
            name="ck_ingestion_batch_status",
        ),
        UniqueConstraint("source_name", "file_sha256", name="uq_ingestion_batch_source_sha256"),
        {"schema": "ops"},
    )

    batch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    file_sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    row_count_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    row_count_loaded: Mapped[int | None] = mapped_column(Integer, nullable=True)
    row_count_rejected: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("app.app_user.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    quality_results: Mapped[list[DataQualityResult]] = relationship(
        "DataQualityResult", back_populates="batch", cascade="all, delete-orphan"
    )


class PipelineRun(Base):
    """Execution state of orchestration workflows in ops.pipeline_run."""

    __tablename__ = "pipeline_run"
    __table_args__ = (
        CheckConstraint(
            "flow_name IN ('etl', 'score_all', 'train_forecast', 'train_segmentation', "
            "'train_churn', 'drift_check')",
            name="ck_pipeline_run_flow_name",
        ),
        CheckConstraint(
            "trigger IN ('schedule', 'manual', 'upload', 'alert', 'promote')",
            name="ck_pipeline_run_trigger",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_pipeline_run_status",
        ),
        Index(
            "uq_pipeline_run_active",
            "flow_name",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        Index("ix_pipeline_run_flow_time", "flow_name", text("queued_at DESC")),
        {"schema": "ops"},
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    flow_name: Mapped[str] = mapped_column(String, nullable=False)
    trigger: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=text("'{}'"), nullable=False
    )
    prefect_flow_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("app.app_user.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    queued_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class DataQualityResult(Base):
    """Validation and gate findings per batch in ops.data_quality_result."""

    __tablename__ = "data_quality_result"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name="ck_data_quality_result_severity",
        ),
        Index("ix_dq_batch", "batch_id"),
        {"schema": "ops"},
    )

    result_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        Identity(always=True),
        primary_key=True,
        autoincrement=True,
    )
    batch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ops.ingestion_batch.batch_id", ondelete="CASCADE"),
        nullable=False,
    )
    check_name: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    affected_rows: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=text("'{}'"), nullable=False
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    batch: Mapped[IngestionBatch] = relationship("IngestionBatch", back_populates="quality_results")

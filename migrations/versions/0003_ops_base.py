"""Create ops base tables: ingestion_batch, pipeline_run, data_quality_result.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ops base tables."""
    # 1. ops.ingestion_batch
    op.create_table(
        "ingestion_batch",
        sa.Column(
            "batch_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("source_file", sa.Text(), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=True),
        sa.Column("file_sha256", sa.CHAR(length=64), nullable=False),
        sa.Column("row_count_in", sa.Integer(), nullable=True),
        sa.Column("row_count_loaded", sa.Integer(), nullable=True),
        sa.Column("row_count_rejected", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("uploaded_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["uploaded_by"],
            ["app.app_user.user_id"],
            name=op.f("fk_ingestion_batch_uploaded_by_app_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("batch_id", name=op.f("pk_ingestion_batch")),
        sa.UniqueConstraint("source_name", "file_sha256", name="uq_ingestion_batch_source_sha256"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'rejected')",
            name=op.f("ck_ingestion_batch_status"),
        ),
        schema="ops",
    )

    # 2. ops.pipeline_run
    op.create_table(
        "pipeline_run",
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("flow_name", sa.Text(), nullable=False),
        sa.Column("trigger", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("prefect_flow_run_id", sa.Text(), nullable=True),
        sa.Column("requested_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "queued_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["requested_by"],
            ["app.app_user.user_id"],
            name=op.f("fk_pipeline_run_requested_by_app_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_pipeline_run")),
        sa.CheckConstraint(
            "flow_name IN ('etl', 'score_all', 'train_forecast', 'train_segmentation', "
            "'train_churn', 'drift_check')",
            name=op.f("ck_pipeline_run_flow_name"),
        ),
        sa.CheckConstraint(
            "trigger IN ('schedule', 'manual', 'upload', 'alert', 'promote')",
            name=op.f("ck_pipeline_run_trigger"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_pipeline_run_status"),
        ),
        schema="ops",
    )
    op.create_index(
        "uq_pipeline_run_active",
        "pipeline_run",
        ["flow_name"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
        schema="ops",
    )
    op.create_index(
        "ix_pipeline_run_flow_time",
        "pipeline_run",
        ["flow_name", sa.text("queued_at DESC")],
        unique=False,
        schema="ops",
    )

    # 3. ops.data_quality_result
    op.create_table(
        "data_quality_result",
        sa.Column(
            "result_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("batch_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("check_name", sa.Text(), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("affected_rows", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["ops.ingestion_batch.batch_id"],
            name=op.f("fk_data_quality_result_batch_id_ingestion_batch"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("result_id", name=op.f("pk_data_quality_result")),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'error')",
            name=op.f("ck_data_quality_result_severity"),
        ),
        schema="ops",
    )
    op.create_index(
        "ix_dq_batch",
        "data_quality_result",
        ["batch_id"],
        unique=False,
        schema="ops",
    )


def downgrade() -> None:
    """Drop ops base tables and indexes."""
    op.drop_index("ix_dq_batch", table_name="data_quality_result", schema="ops")
    op.drop_table("data_quality_result", schema="ops")
    op.drop_index("ix_pipeline_run_flow_time", table_name="pipeline_run", schema="ops")
    op.drop_index("uq_pipeline_run_active", table_name="pipeline_run", schema="ops")
    op.drop_table("pipeline_run", schema="ops")
    op.drop_table("ingestion_batch", schema="ops")

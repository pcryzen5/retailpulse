"""Create ops monitoring and audit tables: drift_report, model_performance, alert, audit_log.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ops.drift_report, ops.model_performance, ops.alert, ops.audit_log."""
    # 1. ops.drift_report
    op.create_table(
        "drift_report",
        sa.Column(
            "report_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("reference_start", sa.Date(), nullable=False),
        sa.Column("reference_end", sa.Date(), nullable=False),
        sa.Column("current_start", sa.Date(), nullable=False),
        sa.Column("current_end", sa.Date(), nullable=False),
        sa.Column("dataset_drift", sa.Boolean(), nullable=False),
        sa.Column("share_drifted_features", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column(
            "summary",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("report_path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "task IN ('forecast', 'segmentation', 'churn')",
            name=op.f("ck_drift_report_task"),
        ),
        sa.CheckConstraint(
            "kind IN ('data', 'prediction')",
            name=op.f("ck_drift_report_kind"),
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_drift_report_model_run_id_model_run"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("report_id", name=op.f("pk_drift_report")),
        schema="ops",
    )

    # 2. ops.model_performance
    op.create_table(
        "model_performance",
        sa.Column(
            "snapshot_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("window_start", sa.Date(), nullable=False),
        sa.Column("window_end", sa.Date(), nullable=False),
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("baseline_value", sa.Float(), nullable=True),
        sa.Column("n_observations", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_model_performance_model_run_id_model_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("snapshot_id", name=op.f("pk_model_performance")),
        sa.UniqueConstraint(
            "model_run_id",
            "metric",
            "window_start",
            "window_end",
            name=op.f("uq_model_performance_model_run_id_metric_window"),
        ),
        schema="ops",
    )

    # 3. ops.alert
    op.create_table(
        "alert",
        sa.Column(
            "alert_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("dedupe_key", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.CheckConstraint(
            "severity IN ('info', 'warning', 'critical')",
            name=op.f("ck_alert_severity"),
        ),
        sa.ForeignKeyConstraint(
            ["acknowledged_by"],
            ["app.app_user.user_id"],
            name=op.f("fk_alert_acknowledged_by_app_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("alert_id", name=op.f("pk_alert")),
        schema="ops",
    )
    op.create_index(
        "ix_alert_open",
        "alert",
        [sa.text("created_at DESC")],
        unique=False,
        schema="ops",
        postgresql_where=sa.text("acknowledged_at IS NULL"),
    )
    op.create_index(
        "uq_alert_open_dedupe",
        "alert",
        ["dedupe_key"],
        unique=True,
        schema="ops",
        postgresql_where=sa.text("acknowledged_at IS NULL AND dedupe_key IS NOT NULL"),
    )

    # 4. ops.audit_log
    op.create_table(
        "audit_log",
        sa.Column(
            "audit_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("resource", sa.Text(), nullable=True),
        sa.Column("ip_address", postgresql.INET(), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["app.app_user.user_id"],
            name=op.f("fk_audit_log_user_id_app_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("audit_id", name=op.f("pk_audit_log")),
        schema="ops",
    )
    op.create_index(
        "ix_audit_ts",
        "audit_log",
        [sa.text("ts DESC")],
        unique=False,
        schema="ops",
    )
    op.create_index(
        "ix_audit_user",
        "audit_log",
        ["user_id", sa.text("ts DESC")],
        unique=False,
        schema="ops",
    )


def downgrade() -> None:
    """Drop ops.audit_log, ops.alert, ops.model_performance, ops.drift_report."""
    op.drop_index("ix_audit_user", table_name="audit_log", schema="ops")
    op.drop_index("ix_audit_ts", table_name="audit_log", schema="ops")
    op.drop_table("audit_log", schema="ops")

    op.drop_index(
        "uq_alert_open_dedupe",
        table_name="alert",
        schema="ops",
        postgresql_where=sa.text("acknowledged_at IS NULL AND dedupe_key IS NOT NULL"),
    )
    op.drop_index(
        "ix_alert_open",
        table_name="alert",
        schema="ops",
        postgresql_where=sa.text("acknowledged_at IS NULL"),
    )
    op.drop_table("alert", schema="ops")

    op.drop_table("model_performance", schema="ops")
    op.drop_table("drift_report", schema="ops")

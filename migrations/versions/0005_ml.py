"""Create ml schema tables and indexes.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all machine learning tables and constraints."""
    # 1. ml.model_run
    op.create_table(
        "model_run",
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("algorithm", sa.Text(), nullable=False),
        sa.Column("mlflow_run_id", sa.Text(), nullable=False),
        sa.Column("mlflow_model_name", sa.Text(), nullable=True),
        sa.Column("mlflow_model_version", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("is_champion", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("data_start", sa.Date(), nullable=True),
        sa.Column("data_end", sa.Date(), nullable=True),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("git_sha", sa.Text(), nullable=True),
        sa.Column("dataset_hash", sa.Text(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("trained_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.CheckConstraint(
            "task IN ('forecast', 'segmentation', 'churn')",
            name=op.f("ck_model_run_task"),
        ),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'archived')",
            name=op.f("ck_model_run_status"),
        ),
        sa.ForeignKeyConstraint(
            ["trained_by"],
            ["app.app_user.user_id"],
            name=op.f("fk_model_run_trained_by_app_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_model_run")),
        schema="ml",
    )
    op.create_index(
        "ix_model_run_task_time",
        "model_run",
        ["task", sa.text("trained_at DESC")],
        unique=False,
        schema="ml",
    )
    op.create_index(
        "uq_model_run_champion",
        "model_run",
        ["task"],
        unique=True,
        postgresql_where=sa.text("is_champion"),
        schema="ml",
    )

    # 2. ml.forecast
    op.create_table(
        "forecast",
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("store_id", sa.SmallInteger(), nullable=False),
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("origin_date", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("horizon_day", sa.SmallInteger(), nullable=False),
        sa.Column("yhat", sa.Numeric(precision=14, scale=4), nullable=False),
        sa.Column("yhat_lower", sa.Numeric(precision=14, scale=4), nullable=False),
        sa.Column("yhat_upper", sa.Numeric(precision=14, scale=4), nullable=False),
        sa.Column(
            "interval_level",
            sa.Numeric(precision=3, scale=2),
            server_default=sa.text("0.80"),
            nullable=False,
        ),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("horizon_day >= 1", name=op.f("ck_forecast_horizon_day")),
        sa.CheckConstraint("yhat >= 0", name=op.f("ck_forecast_yhat_nonneg")),
        sa.CheckConstraint("yhat_lower >= 0", name=op.f("ck_forecast_lower_nonneg")),
        sa.CheckConstraint(
            "yhat_lower <= yhat AND yhat <= yhat_upper",
            name=op.f("ck_forecast_bounds"),
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_forecast_model_run_id_model_run"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["core.dim_product.product_id"],
            name=op.f("fk_forecast_product_id_dim_product"),
        ),
        sa.ForeignKeyConstraint(
            ["store_id"],
            ["core.dim_store.store_id"],
            name=op.f("fk_forecast_store_id_dim_store"),
        ),
        sa.PrimaryKeyConstraint(
            "model_run_id",
            "store_id",
            "product_id",
            "origin_date",
            "target_date",
            name=op.f("pk_forecast"),
        ),
        schema="ml",
    )
    op.create_index(
        "ix_forecast_lookup",
        "forecast",
        ["product_id", "store_id", "target_date"],
        unique=False,
        schema="ml",
    )

    # 3. ml.forecast_metric
    op.create_table(
        "forecast_metric",
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("scope_key", sa.Text(), nullable=False),
        sa.Column("metric", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("n_series", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "scope IN ('global', 'abc', 'horizon', 'sku')",
            name=op.f("ck_forecast_metric_scope"),
        ),
        sa.CheckConstraint(
            "metric IN ('wape', 'smape', 'mase', 'rmsse', 'bias', 'rmse', 'mae', "
            "'coverage', 'pinball')",
            name=op.f("ck_forecast_metric_type"),
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_forecast_metric_model_run_id_model_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "model_run_id", "scope", "scope_key", "metric", name=op.f("pk_forecast_metric")
        ),
        schema="ml",
    )

    # 4. ml.segment_profile
    op.create_table(
        "segment_profile",
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("segment_id", sa.SmallInteger(), nullable=False),
        sa.Column("segment_label", sa.Text(), nullable=False),
        sa.Column("n_customers", sa.Integer(), nullable=False),
        sa.Column("avg_recency_days", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("avg_frequency", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.Column("avg_monetary", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("revenue_share", sa.Numeric(precision=6, scale=5), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_segment_profile_model_run_id_model_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "model_run_id", "as_of_date", "segment_id", name=op.f("pk_segment_profile")
        ),
        schema="ml",
    )

    # 5. ml.customer_segment
    op.create_table(
        "customer_segment",
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=False),
        sa.Column("segment_id", sa.SmallInteger(), nullable=False),
        sa.Column("recency_days", sa.Integer(), nullable=False),
        sa.Column("frequency", sa.Integer(), nullable=False),
        sa.Column("monetary", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("r_score", sa.SmallInteger(), nullable=True),
        sa.Column("f_score", sa.SmallInteger(), nullable=True),
        sa.Column("m_score", sa.SmallInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["core.dim_customer.customer_id"],
            name=op.f("fk_customer_segment_customer_id_dim_customer"),
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id", "as_of_date", "segment_id"],
            [
                "ml.segment_profile.model_run_id",
                "ml.segment_profile.as_of_date",
                "ml.segment_profile.segment_id",
            ],
            name=op.f("fk_customer_segment_profile"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "model_run_id", "as_of_date", "customer_id", name=op.f("pk_customer_segment")
        ),
        schema="ml",
    )
    op.create_index(
        "ix_customer_segment_lookup",
        "customer_segment",
        ["customer_id", sa.text("as_of_date DESC")],
        unique=False,
        schema="ml",
    )

    # 6. ml.churn_score
    op.create_table(
        "churn_score",
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=False),
        sa.Column("churn_probability", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("risk_band", sa.Text(), nullable=False),
        sa.Column(
            "top_drivers",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "churn_probability BETWEEN 0 AND 1",
            name=op.f("ck_churn_score_prob_range"),
        ),
        sa.CheckConstraint(
            "risk_band IN ('low', 'medium', 'high')",
            name=op.f("ck_churn_score_risk_band"),
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["core.dim_customer.customer_id"],
            name=op.f("fk_churn_score_customer_id_dim_customer"),
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_churn_score_model_run_id_model_run"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "model_run_id", "as_of_date", "customer_id", name=op.f("pk_churn_score")
        ),
        schema="ml",
    )
    op.create_index(
        "ix_churn_score_lookup",
        "churn_score",
        ["customer_id", sa.text("as_of_date DESC")],
        unique=False,
        schema="ml",
    )

    # 7. ml.inventory_recommendation
    op.create_table(
        "inventory_recommendation",
        sa.Column(
            "recommendation_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("forecast_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("store_id", sa.SmallInteger(), nullable=False),
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("lead_time_days", sa.SmallInteger(), nullable=False),
        sa.Column("service_level", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.Column("reorder_point", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("order_quantity", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("safety_stock", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reorder_urgency", sa.SmallInteger(), nullable=False),
        sa.Column("expected_stockout_prob", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("holding_cost_rate", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("order_cost", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("unit_cost", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('ok', 'reorder_now', 'stockout_risk', 'no_data')",
            name=op.f("ck_inventory_rec_status"),
        ),
        sa.CheckConstraint(
            "reorder_urgency BETWEEN 0 AND 100",
            name=op.f("ck_inventory_rec_urgency"),
        ),
        sa.ForeignKeyConstraint(
            ["forecast_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_inventory_recommendation_forecast_run_id_model_run"),
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["core.dim_product.product_id"],
            name=op.f("fk_inventory_recommendation_product_id_dim_product"),
        ),
        sa.ForeignKeyConstraint(
            ["store_id"],
            ["core.dim_store.store_id"],
            name=op.f("fk_inventory_recommendation_store_id_dim_store"),
        ),
        sa.PrimaryKeyConstraint("recommendation_id", name=op.f("pk_inventory_recommendation")),
        sa.UniqueConstraint(
            "forecast_run_id",
            "store_id",
            "product_id",
            "as_of_date",
            name=op.f("uq_inventory_rec_tuple"),
        ),
        schema="ml",
    )
    op.create_index(
        "ix_inv_rec_urgency",
        "inventory_recommendation",
        ["as_of_date", sa.text("reorder_urgency DESC")],
        unique=False,
        schema="ml",
    )

    # 8. ml.active_output
    op.create_table(
        "active_output",
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("data_as_of", sa.Date(), nullable=False),
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "task IN ('forecast', 'segmentation', 'churn', 'inventory')",
            name=op.f("ck_active_output_task"),
        ),
        sa.ForeignKeyConstraint(
            ["model_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_active_output_model_run_id_model_run"),
        ),
        sa.PrimaryKeyConstraint("task", name=op.f("pk_active_output")),
        schema="ml",
    )


def downgrade() -> None:
    """Drop all machine learning tables in reverse order."""
    op.drop_table("active_output", schema="ml")
    op.drop_index("ix_inv_rec_urgency", table_name="inventory_recommendation", schema="ml")
    op.drop_table("inventory_recommendation", schema="ml")
    op.drop_index("ix_churn_score_lookup", table_name="churn_score", schema="ml")
    op.drop_table("churn_score", schema="ml")
    op.drop_index("ix_customer_segment_lookup", table_name="customer_segment", schema="ml")
    op.drop_table("customer_segment", schema="ml")
    op.drop_table("segment_profile", schema="ml")
    op.drop_table("forecast_metric", schema="ml")
    op.drop_index("ix_forecast_lookup", table_name="forecast", schema="ml")
    op.drop_table("forecast", schema="ml")
    op.drop_index("uq_model_run_champion", table_name="model_run", schema="ml")
    op.drop_index("ix_model_run_task_time", table_name="model_run", schema="ml")
    op.drop_table("model_run", schema="ml")

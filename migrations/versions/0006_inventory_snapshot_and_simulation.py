"""Create core.inventory_snapshot and ml.policy_simulation tables.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create inventory snapshot and policy simulation tables."""
    # 1. core.inventory_snapshot
    op.create_table(
        "inventory_snapshot",
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("store_id", sa.SmallInteger(), nullable=False),
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("on_hand", sa.Integer(), nullable=False),
        sa.Column(
            "on_order",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.Text(),
            server_default=sa.text("'synthetic'"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_date"],
            ["core.dim_date.date_key"],
            name=op.f("fk_inventory_snapshot_snapshot_date_dim_date"),
        ),
        sa.ForeignKeyConstraint(
            ["store_id"],
            ["core.dim_store.store_id"],
            name=op.f("fk_inventory_snapshot_store_id_dim_store"),
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["core.dim_product.product_id"],
            name=op.f("fk_inventory_snapshot_product_id_dim_product"),
        ),
        sa.PrimaryKeyConstraint(
            "snapshot_date", "store_id", "product_id", name=op.f("pk_inventory_snapshot")
        ),
        sa.CheckConstraint("on_hand >= 0", name=op.f("ck_inventory_snapshot_on_hand")),
        sa.CheckConstraint("on_order >= 0", name=op.f("ck_inventory_snapshot_on_order")),
        sa.CheckConstraint(
            "source IN ('synthetic', 'upload')",
            name=op.f("ck_inventory_snapshot_source"),
        ),
        schema="core",
    )

    # 2. ml.policy_simulation
    op.create_table(
        "policy_simulation",
        sa.Column(
            "simulation_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("forecast_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("window_start", sa.Date(), nullable=False),
        sa.Column("window_end", sa.Date(), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "results",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "uses_synthetic_params",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["forecast_run_id"],
            ["ml.model_run.run_id"],
            name=op.f("fk_policy_simulation_forecast_run_id_model_run"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["app.app_user.user_id"],
            name=op.f("fk_policy_simulation_created_by_app_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("simulation_id", name=op.f("pk_policy_simulation")),
        schema="ml",
    )

    # 3. Alter ml.inventory_recommendation to add inventory context columns
    op.add_column(
        "inventory_recommendation",
        sa.Column("on_hand", sa.Integer(), server_default=sa.text("0"), nullable=False),
        schema="ml",
    )
    op.add_column(
        "inventory_recommendation",
        sa.Column("on_order", sa.Integer(), server_default=sa.text("0"), nullable=False),
        schema="ml",
    )
    op.add_column(
        "inventory_recommendation",
        sa.Column("demand_mean_lt", sa.Numeric(14, 4), nullable=True),
        schema="ml",
    )
    op.add_column(
        "inventory_recommendation",
        sa.Column("demand_std_lt", sa.Numeric(14, 4), nullable=True),
        schema="ml",
    )
    op.add_column(
        "inventory_recommendation",
        sa.Column("days_of_cover", sa.Numeric(8, 2), nullable=True),
        schema="ml",
    )
    op.add_column(
        "inventory_recommendation",
        sa.Column(
            "params_source", sa.Text(), server_default=sa.text("'synthetic'"), nullable=False
        ),
        schema="ml",
    )
    op.drop_constraint(
        "ck_inventory_rec_status", "inventory_recommendation", schema="ml", type_="check"
    )
    op.create_check_constraint(
        "ck_inventory_rec_status",
        "inventory_recommendation",
        "status IN ('ok', 'reorder_now', 'overstock', 'stockout_risk', 'no_data')",
        schema="ml",
    )


def downgrade() -> None:
    """Drop inventory snapshot and policy simulation tables and revert recommendation columns."""
    op.drop_constraint(
        "ck_inventory_rec_status", "inventory_recommendation", schema="ml", type_="check"
    )
    op.create_check_constraint(
        "ck_inventory_rec_status",
        "inventory_recommendation",
        "status IN ('ok', 'reorder_now', 'stockout_risk', 'no_data')",
        schema="ml",
    )
    op.drop_column("inventory_recommendation", "params_source", schema="ml")
    op.drop_column("inventory_recommendation", "days_of_cover", schema="ml")
    op.drop_column("inventory_recommendation", "demand_std_lt", schema="ml")
    op.drop_column("inventory_recommendation", "demand_mean_lt", schema="ml")
    op.drop_column("inventory_recommendation", "on_order", schema="ml")
    op.drop_column("inventory_recommendation", "on_hand", schema="ml")
    op.drop_table("policy_simulation", schema="ml")
    op.drop_table("inventory_snapshot", schema="core")

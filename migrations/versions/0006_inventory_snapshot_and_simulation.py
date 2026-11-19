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


def downgrade() -> None:
    """Drop inventory snapshot and policy simulation tables."""
    op.drop_table("policy_simulation", schema="ml")
    op.drop_table("inventory_snapshot", schema="core")


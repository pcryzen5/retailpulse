"""Create core schema tables and indexes.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create core star schema tables, indexes, and initial store seed."""
    # 1. core.dim_store
    op.create_table(
        "dim_store",
        sa.Column(
            "store_id",
            sa.SmallInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("store_code", sa.Text(), nullable=False),
        sa.Column("store_name", sa.Text(), nullable=False),
        sa.Column("country", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("store_id", name=op.f("pk_dim_store")),
        sa.UniqueConstraint("store_code", name=op.f("uq_dim_store_store_code")),
        schema="core",
    )
    # Seed default store ONLINE per DATABASE.md §6
    op.execute(
        "INSERT INTO core.dim_store (store_code, store_name, country) "
        "VALUES ('ONLINE', 'Online store', 'GB');"
    )

    # 2. core.dim_product
    op.create_table(
        "dim_product",
        sa.Column(
            "product_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("sku", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("is_product", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("unit_cost_est", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("abc_class", sa.CHAR(length=1), nullable=True),
        sa.Column("xyz_class", sa.CHAR(length=1), nullable=True),
        sa.Column("first_sale_date", sa.Date(), nullable=True),
        sa.Column("last_sale_date", sa.Date(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("product_id", name=op.f("pk_dim_product")),
        sa.UniqueConstraint("sku", name=op.f("uq_dim_product_sku")),
        sa.CheckConstraint("abc_class IN ('A', 'B', 'C')", name=op.f("ck_dim_product_abc_class")),
        sa.CheckConstraint("xyz_class IN ('X', 'Y', 'Z')", name=op.f("ck_dim_product_xyz_class")),
        schema="core",
    )
    op.create_index(
        "ix_dim_product_desc_trgm",
        "dim_product",
        ["description"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"description": "gin_trgm_ops"},
        schema="core",
    )
    op.create_index(
        "ix_dim_product_abc",
        "dim_product",
        ["abc_class"],
        unique=False,
        postgresql_where=sa.text("is_product"),
        schema="core",
    )

    # 3. core.dim_customer
    op.create_table(
        "dim_customer",
        sa.Column(
            "customer_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("customer_key", sa.Text(), nullable=False),
        sa.Column("country", sa.Text(), nullable=True),
        sa.Column("first_order_date", sa.Date(), nullable=True),
        sa.Column("last_order_date", sa.Date(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("customer_id", name=op.f("pk_dim_customer")),
        sa.UniqueConstraint("customer_key", name=op.f("uq_dim_customer_customer_key")),
        schema="core",
    )

    # 4. core.dim_date
    op.create_table(
        "dim_date",
        sa.Column("date_key", sa.Date(), nullable=False),
        sa.Column("year", sa.SmallInteger(), nullable=False),
        sa.Column("quarter", sa.SmallInteger(), nullable=False),
        sa.Column("month", sa.SmallInteger(), nullable=False),
        sa.Column("iso_week", sa.SmallInteger(), nullable=False),
        sa.Column("day_of_week", sa.SmallInteger(), nullable=False),
        sa.Column("day_of_month", sa.SmallInteger(), nullable=False),
        sa.Column("is_weekend", sa.Boolean(), nullable=False),
        sa.Column("is_holiday", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("holiday_name", sa.Text(), nullable=True),
        sa.Column("is_trading_day", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_complete", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.PrimaryKeyConstraint("date_key", name=op.f("pk_dim_date")),
        schema="core",
    )

    # 5. core.fact_sales_line
    op.create_table(
        "fact_sales_line",
        sa.Column(
            "sales_line_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            nullable=False,
        ),
        sa.Column("batch_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_row_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("invoice_no", sa.Text(), nullable=False),
        sa.Column("invoice_ts", sa.DateTime(timezone=False), nullable=False),
        sa.Column("date_key", sa.Date(), nullable=False),
        sa.Column("store_id", sa.SmallInteger(), nullable=False),
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_price", sa.Numeric(precision=12, scale=4), nullable=False),
        sa.Column(
            "line_amount",
            sa.Numeric(precision=14, scale=4),
            sa.Computed("quantity * unit_price", persisted=True),
            nullable=False,
        ),
        sa.Column("is_return", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_outlier", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("currency", sa.CHAR(length=3), server_default=sa.text("'GBP'"), nullable=False),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["ops.ingestion_batch.batch_id"],
            name=op.f("fk_fact_sales_line_batch_id_ingestion_batch"),
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["core.dim_customer.customer_id"],
            name=op.f("fk_fact_sales_line_customer_id_dim_customer"),
        ),
        sa.ForeignKeyConstraint(
            ["date_key"],
            ["core.dim_date.date_key"],
            name=op.f("fk_fact_sales_line_date_key_dim_date"),
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["core.dim_product.product_id"],
            name=op.f("fk_fact_sales_line_product_id_dim_product"),
        ),
        sa.ForeignKeyConstraint(
            ["store_id"],
            ["core.dim_store.store_id"],
            name=op.f("fk_fact_sales_line_store_id_dim_store"),
        ),
        sa.PrimaryKeyConstraint("sales_line_id", name=op.f("pk_fact_sales_line")),
        sa.UniqueConstraint("source_row_hash", name=op.f("uq_fact_sales_line_source_row_hash")),
        sa.CheckConstraint("unit_price > 0", name=op.f("ck_fact_sales_line_unit_price")),
        sa.CheckConstraint(
            "((is_return AND quantity < 0) OR (NOT is_return AND quantity > 0))",
            name=op.f("ck_fact_sales_line_return_qty"),
        ),
        schema="core",
    )
    op.create_index("ix_fsl_date", "fact_sales_line", ["date_key"], unique=False, schema="core")
    op.create_index(
        "ix_fsl_product_date",
        "fact_sales_line",
        ["product_id", "date_key"],
        unique=False,
        schema="core",
    )
    op.create_index(
        "ix_fsl_customer_date",
        "fact_sales_line",
        ["customer_id", "date_key"],
        unique=False,
        postgresql_where=sa.text("customer_id IS NOT NULL"),
        schema="core",
    )
    op.create_index("ix_fsl_batch", "fact_sales_line", ["batch_id"], unique=False, schema="core")

    # 6. core.fact_daily_sales
    op.create_table(
        "fact_daily_sales",
        sa.Column("date_key", sa.Date(), nullable=False),
        sa.Column("store_id", sa.SmallInteger(), nullable=False),
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("units_sold", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("units_returned", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "net_units",
            sa.Integer(),
            sa.Computed("units_sold - units_returned", persisted=True),
            nullable=False,
        ),
        sa.Column(
            "revenue",
            sa.Numeric(precision=14, scale=4),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("avg_unit_price", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("n_orders", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.ForeignKeyConstraint(
            ["date_key"],
            ["core.dim_date.date_key"],
            name=op.f("fk_fact_daily_sales_date_key_dim_date"),
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["core.dim_product.product_id"],
            name=op.f("fk_fact_daily_sales_product_id_dim_product"),
        ),
        sa.ForeignKeyConstraint(
            ["store_id"],
            ["core.dim_store.store_id"],
            name=op.f("fk_fact_daily_sales_store_id_dim_store"),
        ),
        sa.PrimaryKeyConstraint(
            "date_key", "store_id", "product_id", name=op.f("pk_fact_daily_sales")
        ),
        schema="core",
    )
    op.create_index(
        "ix_fds_product_date",
        "fact_daily_sales",
        ["product_id", "store_id", "date_key"],
        unique=False,
        schema="core",
    )

    # 7. core.inventory_param
    op.create_table(
        "inventory_param",
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("store_id", sa.SmallInteger(), nullable=False),
        sa.Column("lead_time_days", sa.Numeric(precision=5, scale=1), nullable=False),
        sa.Column(
            "lead_time_std_days",
            sa.Numeric(precision=5, scale=2),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "service_level",
            sa.Numeric(precision=4, scale=3),
            server_default=sa.text("0.950"),
            nullable=False,
        ),
        sa.Column("unit_cost", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column(
            "holding_cost_rate",
            sa.Numeric(precision=5, scale=4),
            server_default=sa.text("0.2000"),
            nullable=False,
        ),
        sa.Column(
            "order_cost",
            sa.Numeric(precision=12, scale=4),
            server_default=sa.text("25"),
            nullable=False,
        ),
        sa.Column("moq", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "source",
            sa.Text(),
            server_default=sa.text("'synthetic'"),
            nullable=False,
        ),
        sa.Column("updated_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["core.dim_product.product_id"],
            name=op.f("fk_inventory_param_product_id_dim_product"),
        ),
        sa.ForeignKeyConstraint(
            ["store_id"],
            ["core.dim_store.store_id"],
            name=op.f("fk_inventory_param_store_id_dim_store"),
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"],
            ["app.app_user.user_id"],
            name=op.f("fk_inventory_param_updated_by_app_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("product_id", "store_id", name=op.f("pk_inventory_param")),
        sa.CheckConstraint("lead_time_days > 0", name=op.f("ck_inventory_param_lead_time_days")),
        sa.CheckConstraint(
            "lead_time_std_days >= 0",
            name=op.f("ck_inventory_param_lead_time_std_days"),
        ),
        sa.CheckConstraint(
            "service_level > 0.5 AND service_level < 1",
            name=op.f("ck_inventory_param_service_level"),
        ),
        sa.CheckConstraint("unit_cost >= 0", name=op.f("ck_inventory_param_unit_cost")),
        sa.CheckConstraint(
            "holding_cost_rate >= 0",
            name=op.f("ck_inventory_param_holding_cost_rate"),
        ),
        sa.CheckConstraint("order_cost >= 0", name=op.f("ck_inventory_param_order_cost")),
        sa.CheckConstraint("moq >= 1", name=op.f("ck_inventory_param_moq")),
        sa.CheckConstraint(
            "source IN ('synthetic', 'user', 'upload')",
            name=op.f("ck_inventory_param_source"),
        ),
        schema="core",
    )


def downgrade() -> None:
    """Drop core schema tables and indexes."""
    op.drop_table("inventory_param", schema="core")
    op.drop_index("ix_fds_product_date", table_name="fact_daily_sales", schema="core")
    op.drop_table("fact_daily_sales", schema="core")
    op.drop_index("ix_fsl_batch", table_name="fact_sales_line", schema="core")
    op.drop_index("ix_fsl_customer_date", table_name="fact_sales_line", schema="core")
    op.drop_index("ix_fsl_product_date", table_name="fact_sales_line", schema="core")
    op.drop_index("ix_fsl_date", table_name="fact_sales_line", schema="core")
    op.drop_table("fact_sales_line", schema="core")
    op.drop_table("dim_date", schema="core")
    op.drop_table("dim_customer", schema="core")
    op.drop_index("ix_dim_product_abc", table_name="dim_product", schema="core")
    op.drop_index("ix_dim_product_desc_trgm", table_name="dim_product", schema="core")
    op.drop_table("dim_product", schema="core")
    op.drop_table("dim_store", schema="core")

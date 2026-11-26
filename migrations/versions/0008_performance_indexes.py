"""Create performance indexes for ETL aggregation and high-frequency API endpoints.

Implements P9-01 Performance Optimization:
- core.dim_customer: index on last_order_date DESC for 90d active customer KPI scan.
- core.fact_sales_line: partial composite index on (date_key, product_id, store_id)
  WHERE NOT is_outlier for ETL daily sales aggregation.
- ml.churn_score: composite index on (model_run_id, as_of_date, risk_band, churn_probability DESC)
  for churn risk filtering and KPI counts.
- ml.inventory_recommendation: composite index on (forecast_run_id, as_of_date, status)
  for replenishment status aggregation and reorder alert listing.
- ml.forecast: composite index on (model_run_id, product_id, store_id, target_date ASC)
  for instant SKU horizon lookup and zero-sort ordering.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create performance indexes across core and ml schemas."""
    # 1. core.dim_customer: active customer 90d scan optimization
    op.create_index(
        "ix_dim_customer_last_order_date",
        "dim_customer",
        [sa.text("last_order_date DESC")],
        unique=False,
        schema="core",
    )

    # 2. core.fact_sales_line: partial composite index for ETL daily sales aggregation
    op.create_index(
        "ix_fsl_date_agg",
        "fact_sales_line",
        ["date_key", "product_id", "store_id"],
        unique=False,
        postgresql_where=sa.text("NOT is_outlier"),
        schema="core",
    )

    # 3. ml.churn_score: composite index for risk band filtering and sorting
    op.create_index(
        "ix_churn_score_run_date_band",
        "churn_score",
        ["model_run_id", "as_of_date", "risk_band", sa.text("churn_probability DESC")],
        unique=False,
        schema="ml",
    )

    # 4. ml.inventory_recommendation: composite index for status counts and alert filtering
    op.create_index(
        "ix_inv_rec_run_date_status",
        "inventory_recommendation",
        ["forecast_run_id", "as_of_date", "status"],
        unique=False,
        schema="ml",
    )

    # 5. ml.forecast: composite index for SKU horizon retrieval matching ORDER BY target_date ASC
    op.create_index(
        "ix_forecast_run_prod_store_target",
        "forecast",
        ["model_run_id", "product_id", "store_id", "target_date"],
        unique=False,
        schema="ml",
    )


def downgrade() -> None:
    """Drop performance indexes in reverse order."""
    op.drop_index(
        "ix_forecast_run_prod_store_target",
        table_name="forecast",
        schema="ml",
    )
    op.drop_index(
        "ix_inv_rec_run_date_status",
        table_name="inventory_recommendation",
        schema="ml",
    )
    op.drop_index(
        "ix_churn_score_run_date_band",
        table_name="churn_score",
        schema="ml",
    )
    op.drop_index(
        "ix_fsl_date_agg",
        table_name="fact_sales_line",
        schema="core",
    )
    op.drop_index(
        "ix_dim_customer_last_order_date",
        table_name="dim_customer",
        schema="core",
    )

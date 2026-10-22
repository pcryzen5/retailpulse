"""Core retail star schema and inventory parameter models."""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from retailpulse.db.base import Base


class DimStore(Base):
    """Store dimension in core.dim_store."""

    __tablename__ = "dim_store"
    __table_args__ = ({"schema": "core"},)

    store_id: Mapped[int] = mapped_column(SmallInteger, Identity(always=True), primary_key=True)
    store_code: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    store_name: Mapped[str] = mapped_column(String, nullable=False)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DimProduct(Base):
    """Product dimension in core.dim_product."""

    __tablename__ = "dim_product"
    __table_args__ = (
        CheckConstraint("abc_class IN ('A', 'B', 'C')", name="ck_dim_product_abc_class"),
        CheckConstraint("xyz_class IN ('X', 'Y', 'Z')", name="ck_dim_product_xyz_class"),
        Index(
            "ix_dim_product_desc_trgm",
            "description",
            postgresql_using="gin",
            postgresql_ops={"description": "gin_trgm_ops"},
        ),
        Index(
            "ix_dim_product_abc",
            "abc_class",
            postgresql_where=text("is_product"),
        ),
        {"schema": "core"},
    )

    product_id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    sku: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    is_product: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    unit_cost_est: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    abc_class: Mapped[str | None] = mapped_column(CHAR(1), nullable=True)
    xyz_class: Mapped[str | None] = mapped_column(CHAR(1), nullable=True)
    first_sale_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    last_sale_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class DimCustomer(Base):
    """Pseudonymous customer dimension in core.dim_customer."""

    __tablename__ = "dim_customer"
    __table_args__ = ({"schema": "core"},)

    customer_id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    customer_key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    first_order_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    last_order_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class DimDate(Base):
    """Calendar and trading day dimension in core.dim_date."""

    __tablename__ = "dim_date"
    __table_args__ = ({"schema": "core"},)

    date_key: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    quarter: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    month: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    iso_week: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    day_of_week: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    day_of_month: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    is_weekend: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_holiday: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    holiday_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_trading_day: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )
    is_complete: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )


class FactSalesLine(Base):
    """Granular sales lines in core.fact_sales_line."""

    __tablename__ = "fact_sales_line"
    __table_args__ = (
        CheckConstraint("unit_price > 0", name="ck_fact_sales_line_unit_price"),
        CheckConstraint(
            "((is_return AND quantity < 0) OR (NOT is_return AND quantity > 0))",
            name="ck_fact_sales_line_return_qty",
        ),
        Index("ix_fsl_date", "date_key"),
        Index("ix_fsl_product_date", "product_id", "date_key"),
        Index(
            "ix_fsl_customer_date",
            "customer_id",
            "date_key",
            postgresql_where=text("customer_id IS NOT NULL"),
        ),
        Index("ix_fsl_batch", "batch_id"),
        {"schema": "core"},
    )

    sales_line_id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    batch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ops.ingestion_batch.batch_id"),
        nullable=False,
    )
    source_row_hash: Mapped[str] = mapped_column(CHAR(64), unique=True, nullable=False)
    invoice_no: Mapped[str] = mapped_column(String, nullable=False)
    invoice_ts: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    date_key: Mapped[datetime.date] = mapped_column(
        Date, ForeignKey("core.dim_date.date_key"), nullable=False
    )
    store_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("core.dim_store.store_id"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("core.dim_product.product_id"), nullable=False
    )
    customer_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("core.dim_customer.customer_id"), nullable=True
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    line_amount: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        Computed("quantity * unit_price", persisted=True),
        nullable=False,
    )
    is_return: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    is_outlier: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    currency: Mapped[str] = mapped_column(
        CHAR(3), default="GBP", server_default="'GBP'", nullable=False
    )


class FactDailySales(Base):
    """Daily aggregated sales per product/store in core.fact_daily_sales."""

    __tablename__ = "fact_daily_sales"
    __table_args__ = (
        Index("ix_fds_product_date", "product_id", "store_id", "date_key"),
        {"schema": "core"},
    )

    date_key: Mapped[datetime.date] = mapped_column(
        Date, ForeignKey("core.dim_date.date_key"), primary_key=True
    )
    store_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("core.dim_store.store_id"), primary_key=True
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("core.dim_product.product_id"), primary_key=True
    )
    units_sold: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    units_returned: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    net_units: Mapped[int] = mapped_column(
        Integer,
        Computed("units_sold - units_returned", persisted=True),
        nullable=False,
    )
    revenue: Mapped[Decimal] = mapped_column(
        Numeric(14, 4), default=0, server_default="0", nullable=False
    )
    avg_unit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    n_orders: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)


class InventoryParam(Base):
    """Inventory control policy parameters in core.inventory_param."""

    __tablename__ = "inventory_param"
    __table_args__ = (
        CheckConstraint("lead_time_days > 0", name="ck_inventory_param_lead_time_days"),
        CheckConstraint("lead_time_std_days >= 0", name="ck_inventory_param_lead_time_std_days"),
        CheckConstraint(
            "service_level > 0.5 AND service_level < 1",
            name="ck_inventory_param_service_level",
        ),
        CheckConstraint("unit_cost >= 0", name="ck_inventory_param_unit_cost"),
        CheckConstraint("holding_cost_rate >= 0", name="ck_inventory_param_holding_cost_rate"),
        CheckConstraint("order_cost >= 0", name="ck_inventory_param_order_cost"),
        CheckConstraint("moq >= 1", name="ck_inventory_param_moq"),
        CheckConstraint(
            "source IN ('synthetic', 'user', 'upload')",
            name="ck_inventory_param_source",
        ),
        {"schema": "core"},
    )

    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("core.dim_product.product_id"), primary_key=True
    )
    store_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("core.dim_store.store_id"), primary_key=True
    )
    lead_time_days: Mapped[Decimal] = mapped_column(Numeric(5, 1), nullable=False)
    lead_time_std_days: Mapped[Decimal] = mapped_column(
        Numeric(5, 2), default=0, server_default="0", nullable=False
    )
    service_level: Mapped[Decimal] = mapped_column(
        Numeric(4, 3), default=Decimal("0.950"), server_default="0.950", nullable=False
    )
    unit_cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    holding_cost_rate: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        default=Decimal("0.2000"),
        server_default="0.2000",
        nullable=False,
    )
    order_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 4), default=Decimal("25"), server_default="25", nullable=False
    )
    moq: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    source: Mapped[str] = mapped_column(
        String, default="synthetic", server_default="'synthetic'", nullable=False
    )
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("app.app_user.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

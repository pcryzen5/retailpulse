"""Machine learning schema models for runs, forecasts, metrics, and active outputs."""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from retailpulse.db.base import Base


class ModelRun(Base):
    """Execution record of an ML training run stored in ml.model_run."""

    __tablename__ = "model_run"
    __table_args__ = (
        CheckConstraint(
            "task IN ('forecast', 'segmentation', 'churn')",
            name="ck_model_run_task",
        ),
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'archived')",
            name="ck_model_run_status",
        ),
        Index("ix_model_run_task_time", "task", text("trained_at DESC")),
        {"schema": "ml"},
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    task: Mapped[str] = mapped_column(String, nullable=False)
    algorithm: Mapped[str] = mapped_column(String, nullable=False)
    mlflow_run_id: Mapped[str] = mapped_column(String, nullable=False)
    mlflow_model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    mlflow_model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    is_champion: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    data_start: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    data_end: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        server_default=text("'{}'"),
        nullable=False,
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        server_default=text("'{}'"),
        nullable=False,
    )
    git_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trained_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    trained_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("app.app_user.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    forecast_metrics: Mapped[list[ForecastMetric]] = relationship(
        "ForecastMetric", back_populates="model_run", cascade="all, delete-orphan"
    )
    forecasts: Mapped[list[Forecast]] = relationship(
        "Forecast", back_populates="model_run", cascade="all, delete-orphan"
    )


class ForecastMetric(Base):
    """Scoped evaluation metrics per forecast model run in ml.forecast_metric."""

    __tablename__ = "forecast_metric"
    __table_args__ = (
        CheckConstraint(
            "scope IN ('global', 'abc', 'horizon', 'sku')",
            name="ck_forecast_metric_scope",
        ),
        CheckConstraint(
            "metric IN ('wape', 'smape', 'mase', 'rmsse', 'bias', 'rmse', 'mae', "
            "'coverage', 'pinball')",
            name="ck_forecast_metric_type",
        ),
        {"schema": "ml"},
    )

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml.model_run.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    scope: Mapped[str] = mapped_column(String, primary_key=True)
    scope_key: Mapped[str] = mapped_column(String, primary_key=True)
    metric: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    n_series: Mapped[int | None] = mapped_column(Integer, nullable=True)

    model_run: Mapped[ModelRun] = relationship("ModelRun", back_populates="forecast_metrics")


class Forecast(Base):
    """Point and interval demand forecasts stored in ml.forecast."""

    __tablename__ = "forecast"
    __table_args__ = (
        CheckConstraint("horizon_day >= 1", name="ck_forecast_horizon_day"),
        CheckConstraint("yhat >= 0", name="ck_forecast_yhat_nonneg"),
        CheckConstraint("yhat_lower >= 0", name="ck_forecast_lower_nonneg"),
        CheckConstraint(
            "yhat_lower <= yhat AND yhat <= yhat_upper",
            name="ck_forecast_bounds",
        ),
        Index("ix_forecast_lookup", "product_id", "store_id", "target_date"),
        Index(
            "ix_forecast_run_prod_store_target",
            "model_run_id",
            "product_id",
            "store_id",
            "target_date",
        ),
        {"schema": "ml"},
    )

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml.model_run.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    store_id: Mapped[int] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"),
        ForeignKey("core.dim_store.store_id"),
        primary_key=True,
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("core.dim_product.product_id"),
        primary_key=True,
    )
    origin_date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    target_date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    horizon_day: Mapped[int] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    yhat: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    yhat_lower: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    yhat_upper: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    interval_level: Mapped[Decimal] = mapped_column(
        Numeric(3, 2), server_default="0.80", default=Decimal("0.80"), nullable=False
    )
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model_run: Mapped[ModelRun] = relationship("ModelRun", back_populates="forecasts")


class ActiveOutput(Base):
    """Pointers to the active outputs currently served by APIs in ml.active_output."""

    __tablename__ = "active_output"
    __table_args__ = (
        CheckConstraint(
            "task IN ('forecast', 'segmentation', 'churn', 'inventory')",
            name="ck_active_output_task",
        ),
        {"schema": "ml"},
    )

    task: Mapped[str] = mapped_column(String, primary_key=True)
    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ml.model_run.run_id"), nullable=False
    )
    data_as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    scored_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SegmentProfile(Base):
    """Customer segment profiles and personas stored in ml.segment_profile."""

    __tablename__ = "segment_profile"
    __table_args__ = ({"schema": "ml"},)

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml.model_run.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    as_of_date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    segment_id: Mapped[int] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"), primary_key=True
    )
    segment_label: Mapped[str] = mapped_column(String, nullable=False)
    n_customers: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_recency_days: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    avg_frequency: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False)
    avg_monetary: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    revenue_share: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)


class CustomerSegment(Base):
    """Customer assignment to RFM segments in ml.customer_segment."""

    __tablename__ = "customer_segment"
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_run_id", "as_of_date", "segment_id"],
            [
                "ml.segment_profile.model_run_id",
                "ml.segment_profile.as_of_date",
                "ml.segment_profile.segment_id",
            ],
            ondelete="CASCADE",
            name="fk_customer_segment_profile",
        ),
        Index("ix_customer_segment_lookup", "customer_id", text("as_of_date DESC")),
        {"schema": "ml"},
    )

    model_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    as_of_date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("core.dim_customer.customer_id"),
        primary_key=True,
    )
    segment_id: Mapped[int] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    recency_days: Mapped[int] = mapped_column(Integer, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, nullable=False)
    monetary: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    r_score: Mapped[int | None] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    f_score: Mapped[int | None] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"), nullable=True
    )
    m_score: Mapped[int | None] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"), nullable=True
    )


class ChurnScore(Base):
    """Customer churn predictions and top drivers in ml.churn_score."""

    __tablename__ = "churn_score"
    __table_args__ = (
        CheckConstraint("churn_probability BETWEEN 0 AND 1", name="ck_churn_score_prob_range"),
        CheckConstraint("risk_band IN ('low', 'medium', 'high')", name="ck_churn_score_risk_band"),
        Index("ix_churn_score_lookup", "customer_id", text("as_of_date DESC")),
        Index(
            "ix_churn_score_run_date_band",
            "model_run_id",
            "as_of_date",
            "risk_band",
            text("churn_probability DESC"),
        ),
        {"schema": "ml"},
    )

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml.model_run.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    as_of_date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("core.dim_customer.customer_id"),
        primary_key=True,
    )
    churn_probability: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    risk_band: Mapped[str] = mapped_column(String, nullable=False)
    top_drivers: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        server_default=text("'[]'"),
        nullable=False,
    )
    scored_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class InventoryRecommendation(Base):
    """Inventory replenishment recommendations stored in ml.inventory_recommendation."""

    __tablename__ = "inventory_recommendation"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ok', 'reorder_now', 'overstock', 'stockout_risk', 'no_data')",
            name="ck_inventory_rec_status",
        ),
        CheckConstraint("reorder_urgency BETWEEN 0 AND 100", name="ck_inventory_rec_urgency"),
        UniqueConstraint(
            "forecast_run_id",
            "store_id",
            "product_id",
            "as_of_date",
            name="uq_inventory_rec_tuple",
        ),
        Index("ix_inv_rec_urgency", "as_of_date", text("reorder_urgency DESC")),
        Index("ix_inv_rec_run_date_status", "forecast_run_id", "as_of_date", "status"),
        {"schema": "ml"},
    )

    recommendation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    forecast_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ml.model_run.run_id"), nullable=False
    )
    as_of_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    store_id: Mapped[int] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"),
        ForeignKey("core.dim_store.store_id"),
        nullable=False,
    )
    product_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("core.dim_product.product_id"),
        nullable=False,
    )
    on_hand: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    on_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    lead_time_days: Mapped[Decimal] = mapped_column(
        Numeric(5, 1).with_variant(Numeric(5, 1), "sqlite"), nullable=False
    )
    service_level: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    demand_mean_lt: Mapped[Decimal | None] = mapped_column(Numeric(14, 4), nullable=True)
    demand_std_lt: Mapped[Decimal | None] = mapped_column(Numeric(14, 4), nullable=True)
    reorder_point: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    order_quantity: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    safety_stock: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    reorder_urgency: Mapped[int] = mapped_column(
        SmallInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    expected_stockout_prob: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    days_of_cover: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    holding_cost_rate: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    order_cost: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    unit_cost: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    params_source: Mapped[str] = mapped_column(
        String, default="synthetic", server_default="'synthetic'", nullable=False
    )
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    @property
    def recommended_order_qty(self) -> Decimal:
        """Alias for order_quantity."""
        return self.order_quantity

    @recommended_order_qty.setter
    def recommended_order_qty(self, value: Decimal) -> None:
        self.order_quantity = value

    @property
    def stockout_risk(self) -> Decimal:
        """Alias for expected_stockout_prob."""
        return self.expected_stockout_prob

    @stockout_risk.setter
    def stockout_risk(self, value: Decimal) -> None:
        self.expected_stockout_prob = value


class PolicySimulation(Base):
    """Inventory policy simulation run results in ml.policy_simulation."""

    __tablename__ = "policy_simulation"
    __table_args__ = ({"schema": "ml"},)

    simulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    forecast_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml.model_run.run_id", ondelete="SET NULL"),
        nullable=True,
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("app.app_user.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    window_start: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    window_end: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=False,
    )
    results: Mapped[dict[str, Any]] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=False,
    )
    uses_synthetic_params: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )

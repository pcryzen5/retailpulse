"""Overview KPI dashboard metrics API endpoint.

Implements API §2.3 and §3.2:
- GET /api/v1/kpis/overview (and /api/v1/kpis/summary alias):
  - Trailing 30d revenue, prior 30d revenue, and change percentage.
  - Active customer count (90d).
  - Forecast champion summary (run_id, algorithm, wape, baseline_wape).
  - Inventory alerts (skus_reorder_now).
  - Churn risk summary (high_risk_customers, expected_revenue_at_risk).
  - Last pipeline execution status.
  - In-process 60-second caching (API §1.1).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from retailpulse.api.cache import api_cache
from retailpulse.api.dependencies import require_role
from retailpulse.db.models.core import DimCustomer, FactDailySales
from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    ForecastMetric,
    InventoryRecommendation,
    ModelRun,
)
from retailpulse.db.models.ops import PipelineRun
from retailpulse.db.session import get_db

router = APIRouter(prefix="/kpis", tags=["KPIs"])


class ForecastKPISummary(BaseModel):
    """Summary of active champion forecast model."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    algorithm: str
    wape: float | None = None
    baseline_wape: float | None = None


class LastPipelineRunSummary(BaseModel):
    """Latest pipeline execution status."""

    model_config = ConfigDict(extra="forbid")

    flow_name: str
    status: str
    finished_at: datetime.datetime | None = None


class OverviewKPIsResponse(BaseModel):
    """High-level dashboard KPIs model adhering to API §3.2."""

    model_config = ConfigDict(extra="forbid")

    data_as_of: datetime.date | None = None
    currency: str = "GBP"
    revenue_30d: Decimal = Decimal("0.00")
    revenue_prev_30d: Decimal = Decimal("0.00")
    revenue_change_pct: float | None = None
    active_customers_90d: int = 0
    forecast: ForecastKPISummary | None = None
    skus_reorder_now: int = 0
    high_risk_customers: int = 0
    expected_revenue_at_risk: Decimal = Decimal("0.00")
    open_alerts: int = 0
    last_pipeline_run: LastPipelineRunSummary | None = None


def compute_overview_kpis(session: Session) -> OverviewKPIsResponse:
    """Compute high-level dashboard KPIs across core, ML, and operations."""
    # 1. Determine data_as_of
    latest_sales_date = session.scalar(sa.select(sa.func.max(FactDailySales.date_key)))
    if latest_sales_date is None:
        return OverviewKPIsResponse()

    as_of = latest_sales_date
    d30_start = as_of - datetime.timedelta(days=30)
    d60_start = as_of - datetime.timedelta(days=60)

    # 2. Revenue trailing 30d
    rev_30d = session.scalar(
        sa.select(sa.func.sum(FactDailySales.revenue)).where(
            FactDailySales.date_key > d30_start,
            FactDailySales.date_key <= as_of,
        )
    ) or Decimal("0.00")

    # 3. Revenue prior 30d
    rev_prev_30d = session.scalar(
        sa.select(sa.func.sum(FactDailySales.revenue)).where(
            FactDailySales.date_key > d60_start,
            FactDailySales.date_key <= d30_start,
        )
    ) or Decimal("0.00")

    # Revenue change %
    if rev_prev_30d > Decimal("0.00"):
        change_pct = float(round(((rev_30d - rev_prev_30d) / rev_prev_30d) * 100, 2))
    else:
        change_pct = None

    # 4. Active customers (90d)
    d90_start = as_of - datetime.timedelta(days=90)
    active_custs = (
        session.scalar(
            sa.select(sa.func.count(DimCustomer.customer_id)).where(
                DimCustomer.last_order_date > d90_start
            )
        )
        or 0
    )

    # 5. Forecast summary
    forecast_kpi: ForecastKPISummary | None = None
    active_forecast = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "forecast")
    ).scalar_one_or_none()

    if active_forecast is not None:
        run = session.execute(
            sa.select(ModelRun).where(ModelRun.run_id == active_forecast.model_run_id)
        ).scalar_one_or_none()
        if run is not None:
            # Query global WAPE from ForecastMetric
            wape_val = session.scalar(
                sa.select(ForecastMetric.value).where(
                    ForecastMetric.model_run_id == run.run_id,
                    ForecastMetric.scope == "global",
                    ForecastMetric.metric == "wape",
                )
            )
            base_wape = run.metrics.get("baselines", {}).get("seasonal_naive", {}).get("wape")
            forecast_kpi = ForecastKPISummary(
                run_id=str(run.run_id),
                algorithm=run.algorithm,
                wape=float(wape_val) if wape_val is not None else None,
                baseline_wape=float(base_wape) if base_wape is not None else None,
            )

    # 6. Inventory: skus_reorder_now
    active_inv = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "inventory")
    ).scalar_one_or_none()
    reorder_now_count = 0
    if active_inv is not None:
        reorder_now_count = (
            session.scalar(
                sa.select(sa.func.count(InventoryRecommendation.recommendation_id)).where(
                    InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
                    InventoryRecommendation.as_of_date == active_inv.data_as_of,
                    InventoryRecommendation.status == "reorder_now",
                )
            )
            or 0
        )

    # 7. Churn risk: high_risk_customers and expected_revenue_at_risk
    active_churn = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "churn")
    ).scalar_one_or_none()
    high_risk_count = 0
    expected_rev_at_risk = Decimal("0.00")
    if active_churn is not None:
        high_risk_count = (
            session.scalar(
                sa.select(sa.func.count(ChurnScore.customer_id)).where(
                    ChurnScore.model_run_id == active_churn.model_run_id,
                    ChurnScore.as_of_date == active_churn.data_as_of,
                    ChurnScore.risk_band == "high",
                )
            )
            or 0
        )
        churn_run = session.execute(
            sa.select(ModelRun).where(ModelRun.run_id == active_churn.model_run_id)
        ).scalar_one_or_none()
        if churn_run is not None and churn_run.metrics:
            risk_val = churn_run.metrics.get(
                "expected_revenue_at_risk",
                churn_run.metrics.get("revenue_at_risk", 0.0),
            )
            expected_rev_at_risk = Decimal(str(risk_val))

    # 8. Last completed pipeline run
    last_run = session.execute(
        sa.select(PipelineRun)
        .where(PipelineRun.status.in_(("succeeded", "failed")))
        .order_by(PipelineRun.finished_at.desc().nulls_last())
        .limit(1)
    ).scalar_one_or_none()

    last_run_summary: LastPipelineRunSummary | None = None
    if last_run is not None:
        last_run_summary = LastPipelineRunSummary(
            flow_name=last_run.flow_name,
            status=last_run.status,
            finished_at=last_run.finished_at,
        )

    return OverviewKPIsResponse(
        data_as_of=as_of,
        currency="GBP",
        revenue_30d=Decimal(str(round(rev_30d, 2))),
        revenue_prev_30d=Decimal(str(round(rev_prev_30d, 2))),
        revenue_change_pct=change_pct,
        active_customers_90d=active_custs,
        forecast=forecast_kpi,
        skus_reorder_now=reorder_now_count,
        high_risk_customers=high_risk_count,
        expected_revenue_at_risk=Decimal(str(round(expected_rev_at_risk, 2))),
        open_alerts=0,
        last_pipeline_run=last_run_summary,
    )


@router.get(
    "/overview",
    summary="Overview KPIs",
    response_model=OverviewKPIsResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def get_kpis_overview(
    session: Annotated[Session, Depends(get_db)],
    response: Response,
) -> Any:
    """Retrieve top-level business and predictive KPIs, cached in-process for 60 seconds."""
    # Enforce non-caching by intermediate shared proxies
    response.headers["Cache-Control"] = "no-store"

    cache_key = "kpis_overview"
    cached = api_cache.get(cache_key)
    if cached is not None:
        return cached

    kpis = compute_overview_kpis(session)
    api_cache.set(cache_key, kpis, ttl_seconds=60.0)
    return kpis


@router.get(
    "/summary",
    summary="Overview KPIs Alias",
    response_model=OverviewKPIsResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    dependencies=[Depends(require_role("viewer"))],
)
def get_kpis_summary_alias(
    session: Annotated[Session, Depends(get_db)],
    response: Response,
) -> Any:
    """Alias for /overview."""
    return get_kpis_overview(session=session, response=response)

"""Demand forecasting and model accuracy API endpoints.

Implements API §2.3, §3.3, §3.4:
- GET /api/v1/forecasts: Point & interval demand forecast for a SKU over horizon (1-28 days).
- GET /api/v1/forecasts/accuracy: Champion backtest accuracy
  (global, ABC, horizon buckets, baselines, gate).
- Error handling: 404 unknown product, 422 ineligible product, 503 before first scoring.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from retailpulse.api.cache import api_cache
from retailpulse.api.dependencies import require_role
from retailpulse.api.routers.reference import check_product_eligibility
from retailpulse.db.models.core import DimProduct, DimStore
from retailpulse.db.models.ml import ActiveOutput, Forecast, ForecastMetric, ModelRun
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError, NotFoundError, ServiceUnavailableError

router = APIRouter(prefix="/forecasts", tags=["Forecasts"])


# --- Pydantic Response Models ---


class ModelMeta(BaseModel):
    """Metadata regarding generating forecast model."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    algorithm: str
    is_champion: bool
    trained_at: datetime.datetime


class ProductMeta(BaseModel):
    """Product identification header."""

    model_config = ConfigDict(extra="forbid")

    product_id: int
    sku: str
    description: str | None = None
    abc_class: str | None = None


class ForecastPoint(BaseModel):
    """Daily forecast point with prediction intervals."""

    model_config = ConfigDict(extra="forbid")

    date: datetime.date
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    """Complete forecast endpoint response matching API §3.3."""

    model_config = ConfigDict(extra="forbid")

    model: ModelMeta
    product: ProductMeta
    store_id: int
    origin_date: datetime.date
    interval_level: float
    points: list[ForecastPoint]


class AccuracyMetrics(BaseModel):
    """Core evaluation metrics."""

    model_config = ConfigDict(extra="forbid")

    wape: float | None = None
    mase: float | None = None
    rmsse: float | None = None
    bias: float | None = None
    coverage: float | None = None


class HorizonBucketAccuracy(BaseModel):
    """Accuracy segmented by lead time horizon bucket."""

    model_config = ConfigDict(extra="forbid")

    bucket: str
    wape: float | None = None


class ChampionGateInfo(BaseModel):
    """Champion gate decision rationale."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason: str


class ForecastAccuracyResponse(BaseModel):
    """Champion forecast accuracy breakdown matching API §3.4."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    algorithm: str
    metrics: AccuracyMetrics
    by_abc: dict[str, dict[str, float]]
    by_horizon: list[HorizonBucketAccuracy]
    baselines: dict[str, dict[str, float]]
    gate: ChampionGateInfo | None = None


# --- Endpoints ---


@router.get(
    "",
    summary="Get Demand Forecast",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def get_forecast(
    product_id: Annotated[int, Query(description="Product ID to forecast")],
    session: Annotated[Session, Depends(get_db)],
    store_id: Annotated[int | None, Query(description="Store ID (defaults to ONLINE)")] = None,
    horizon: Annotated[int, Query(ge=1, le=28, description="Horizon days (1 to 28)")] = 28,
) -> ForecastResponse:
    """Retrieve point and interval forecast for a product over a 1 to 28 day horizon."""
    # 1. Validate product exists
    product = session.execute(
        sa.select(DimProduct).where(DimProduct.product_id == product_id)
    ).scalar_one_or_none()

    if product is None:
        raise NotFoundError(f"Product {product_id} not found")

    # 2. Check for active forecast output (503 if no active output)
    active = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "forecast")
    ).scalar_one_or_none()

    if active is None:
        raise ServiceUnavailableError(
            "No active forecast model available. "
            "Run score_all or await initial pipeline completion."
        )

    # 3. Resolve store
    eff_store_id = store_id
    if eff_store_id is None:
        online_store = session.execute(
            sa.select(DimStore).where(DimStore.store_code == "ONLINE")
        ).scalar_one_or_none()
        eff_store_id = online_store.store_id if online_store is not None else 1

    # 4. Fetch generating model run
    model_run = session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == active.model_run_id)
    ).scalar_one_or_none()

    if model_run is None:
        raise ServiceUnavailableError("Active model run definition not found in registry")

    # 5. Query forecast rows
    forecast_rows = (
        session.execute(
            sa.select(Forecast)
            .where(
                Forecast.model_run_id == active.model_run_id,
                Forecast.product_id == product_id,
                Forecast.store_id == eff_store_id,
            )
            .order_by(Forecast.target_date.asc())
            .limit(horizon)
        )
        .scalars()
        .all()
    )

    # 6. If no rows found: determine why and explain
    if not forecast_rows:
        is_eligible, reason = check_product_eligibility(product, session=session)
        if not is_eligible:
            raise DomainValidationError(
                f"Product {product.sku} is not eligible for forecasting: {reason}"
            )
        raise NotFoundError(
            f"No active forecast points available for product {product.sku} "
            f"(origin {active.data_as_of})"
        )

    interval_level = float(forecast_rows[0].interval_level or Decimal("0.80"))

    points = [
        ForecastPoint(
            date=row.target_date,
            yhat=float(row.yhat),
            yhat_lower=float(row.yhat_lower),
            yhat_upper=float(row.yhat_upper),
        )
        for row in forecast_rows
    ]

    return ForecastResponse(
        model=ModelMeta(
            run_id=str(model_run.run_id),
            algorithm=model_run.algorithm,
            is_champion=model_run.is_champion,
            trained_at=model_run.trained_at,
        ),
        product=ProductMeta(
            product_id=product.product_id,
            sku=product.sku,
            description=product.description,
            abc_class=product.abc_class,
        ),
        store_id=eff_store_id,
        origin_date=active.data_as_of,
        interval_level=interval_level,
        points=points,
    )


@router.get(
    "/accuracy",
    summary="Forecast Model Accuracy",
    response_model=ForecastAccuracyResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def get_forecast_accuracy(
    session: Annotated[Session, Depends(get_db)],
    response: Response,
) -> Any:
    """Retrieve backtest evaluation metrics and gate rationale (cached for 60s)."""
    response.headers["Cache-Control"] = "no-store"

    # 1. Resolve active forecast output
    active = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "forecast")
    ).scalar_one_or_none()

    if active is None:
        raise ServiceUnavailableError("No active forecast model available")

    cache_key = f"forecast_accuracy_{active.model_run_id}"
    cached = api_cache.get(cache_key)
    if cached is not None:
        return cached

    # 2. Query model run
    model_run = session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == active.model_run_id)
    ).scalar_one_or_none()

    if model_run is None:
        raise ServiceUnavailableError("Active model run metadata not found")

    # 3. Query metrics from ForecastMetric table
    metrics_records = (
        session.execute(
            sa.select(ForecastMetric).where(ForecastMetric.model_run_id == active.model_run_id)
        )
        .scalars()
        .all()
    )

    global_metrics: dict[str, float] = {}
    by_abc: dict[str, dict[str, float]] = {"A": {}, "B": {}, "C": {}}
    by_horizon_map: dict[str, float] = {}

    for m in metrics_records:
        if m.scope == "global":
            global_metrics[m.metric] = float(m.value)
        elif m.scope == "abc":
            if m.scope_key not in by_abc:
                by_abc[m.scope_key] = {}
            by_abc[m.scope_key][m.metric] = float(m.value)
        elif m.scope == "horizon":
            by_horizon_map[m.scope_key] = float(m.value)

    # Fallback to model_run.metrics JSON if table records are empty
    run_metrics = model_run.metrics or {}
    if not global_metrics:
        global_metrics = {
            k: float(v) for k, v in run_metrics.items() if isinstance(v, (int, float))
        }

    accuracy = AccuracyMetrics(
        wape=global_metrics.get("wape"),
        mase=global_metrics.get("mase"),
        rmsse=global_metrics.get("rmsse"),
        bias=global_metrics.get("bias"),
        coverage=global_metrics.get("coverage"),
    )

    # Format horizon list
    standard_buckets = ["1-7", "8-14", "15-28"]
    by_horizon = [
        HorizonBucketAccuracy(
            bucket=b,
            wape=by_horizon_map.get(b, run_metrics.get("by_horizon", {}).get(b)),
        )
        for b in standard_buckets
    ]

    # Baseline metrics and gate
    baselines = run_metrics.get("baselines", {})
    gate_data = run_metrics.get("gate")
    gate_info: ChampionGateInfo | None = None
    if isinstance(gate_data, dict):
        gate_info = ChampionGateInfo(
            passed=bool(gate_data.get("passed", False)),
            reason=str(gate_data.get("reason", "")),
        )

    result = ForecastAccuracyResponse(
        run_id=str(model_run.run_id),
        algorithm=model_run.algorithm,
        metrics=accuracy,
        by_abc=by_abc,
        by_horizon=by_horizon,
        baselines=baselines,
        gate=gate_info,
    )

    api_cache.set(cache_key, result, ttl_seconds=60.0)
    return result

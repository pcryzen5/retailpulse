"""Batch inventory replenishment recommendations and active output management.

Generates SKU replenishment recommendations using:
  - Forecast point demand predictions from ml.forecast
  - Lead-time demand variance using backtest RMSE from ml.forecast_metric
  - Synthetic or user inventory parameters from core.inventory_param
  - Current physical inventory positions from core.inventory_snapshot
  - Status classification across ('reorder_now', 'overstock', 'ok', 'no_data')
  - Atomic persistence to ml.inventory_recommendation and ml.active_output
"""

from __future__ import annotations

import datetime
import math
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import (
    DimDate,
    DimProduct,
    FactDailySales,
    InventoryParam,
    InventorySnapshot,
)
from retailpulse.db.models.ml import (
    ActiveOutput,
    Forecast,
    ForecastMetric,
    InventoryRecommendation,
    ModelRun,
)
from retailpulse.logging import get_logger
from retailpulse.ml.inventory.policy import (
    InventoryPolicyEvaluation,
    evaluate_inventory_policy,
)

logger = get_logger(__name__)

TASK_FORECAST = "forecast"
TASK_INVENTORY = "inventory"


@dataclass(frozen=True)
class InventoryScoringResult:
    """Summary of batch inventory recommendation generation."""

    forecast_run_id: uuid.UUID
    store_id: int
    data_as_of: datetime.date
    n_recommendations: int
    status_counts: dict[str, int]
    mean_stockout_risk: float
    total_recommended_order_qty: int
    active_output_updated: bool


def _resolve_forecast_run(
    session: Session,
    forecast_run_id: uuid.UUID | None = None,
) -> tuple[ModelRun, datetime.date | None]:
    """Resolve the active or specified forecast model run."""
    if forecast_run_id is not None:
        run = session.get(ModelRun, forecast_run_id)
        if run is None:
            raise ValueError(f"Forecast model run {forecast_run_id} not found.")
        return run, None

    # 1. Check active_output for 'forecast'
    active_ptr = session.get(ActiveOutput, TASK_FORECAST)
    if active_ptr is not None:
        run = session.get(ModelRun, active_ptr.model_run_id)
        if run is not None:
            return run, active_ptr.data_as_of

    # 2. Check for champion forecast run
    stmt = (
        sa.select(ModelRun)
        .where(
            ModelRun.task == TASK_FORECAST,
            ModelRun.is_champion.is_(True),
        )
        .order_by(ModelRun.trained_at.desc())
    )
    champion_run = session.execute(stmt).scalars().first()
    if champion_run is not None:
        return champion_run, None

    # 3. Fallback to latest forecast run
    stmt_latest = (
        sa.select(ModelRun)
        .where(ModelRun.task == TASK_FORECAST)
        .order_by(ModelRun.trained_at.desc())
    )
    latest_run = session.execute(stmt_latest).scalars().first()
    if latest_run is not None:
        return latest_run, None

    raise ValueError(
        "No completed forecast model run found. Please run 'retailpulse score forecast' first."
    )


def _resolve_data_as_of(
    session: Session,
    data_as_of: datetime.date | None,
    run_as_of: datetime.date | None,
) -> datetime.date:
    """Determine effective cutoff date for inventory recommendations."""
    if data_as_of is not None:
        return data_as_of

    if run_as_of is not None:
        return run_as_of

    latest_date = session.execute(
        sa.select(sa.func.max(DimDate.date_key)).where(DimDate.is_complete.is_(True))
    ).scalar_one_or_none()
    if latest_date is not None:
        return latest_date

    max_sales_date = session.execute(
        sa.select(sa.func.max(FactDailySales.date_key))
    ).scalar_one_or_none()
    if max_sales_date is not None:
        return max_sales_date

    return datetime.date(2011, 12, 8)


def _load_forecast_error_stds(
    session: Session,
    forecast_run_id: uuid.UUID,
) -> tuple[dict[str, float], float]:
    """Retrieve backtest RMSE metrics by ABC class and globally from ml.forecast_metric."""
    stmt = sa.select(
        ForecastMetric.scope,
        ForecastMetric.scope_key,
        ForecastMetric.value,
    ).where(
        ForecastMetric.model_run_id == forecast_run_id,
        ForecastMetric.metric == "rmse",
    )
    rows = session.execute(stmt).all()

    sigma_e_by_abc: dict[str, float] = {}
    global_sigma_e = 1.0

    for r in rows:
        val = max(1e-4, float(r.value))
        if r.scope == "abc":
            sigma_e_by_abc[str(r.scope_key).strip().upper()] = val
        elif r.scope == "global":
            global_sigma_e = val

    return sigma_e_by_abc, global_sigma_e


def _load_forecasts_by_product(
    session: Session,
    forecast_run_id: uuid.UUID,
    store_id: int,
    data_as_of: datetime.date,
) -> dict[int, list[float]]:
    """Load daily forecast values yhat ordered by horizon day for each SKU."""
    stmt = (
        sa.select(
            Forecast.product_id,
            Forecast.horizon_day,
            Forecast.yhat,
        )
        .where(
            Forecast.model_run_id == forecast_run_id,
            Forecast.store_id == store_id,
            Forecast.origin_date == data_as_of,
        )
        .order_by(Forecast.product_id, Forecast.horizon_day.asc())
    )
    rows = session.execute(stmt).all()

    forecasts: dict[int, list[float]] = {}
    for r in rows:
        pid = int(r.product_id)
        if pid not in forecasts:
            forecasts[pid] = []
        forecasts[pid].append(max(0.0, float(r.yhat)))

    return forecasts


def _load_inventory_params(
    session: Session,
    store_id: int,
    product_ids: Sequence[int],
) -> dict[int, InventoryParam]:
    """Load inventory parameters for products in store."""
    if not product_ids:
        return {}

    stmt = sa.select(InventoryParam).where(
        InventoryParam.store_id == store_id,
        InventoryParam.product_id.in_(product_ids),
    )
    rows = session.execute(stmt).scalars().all()
    return {int(p.product_id): p for p in rows}


def _load_inventory_snapshots(
    session: Session,
    store_id: int,
    data_as_of: datetime.date,
    product_ids: Sequence[int],
) -> dict[int, tuple[int, int]]:
    """Load (on_hand, on_order) for products at as_of date."""
    if not product_ids:
        return {}

    stmt = sa.select(
        InventorySnapshot.product_id,
        InventorySnapshot.on_hand,
        InventorySnapshot.on_order,
    ).where(
        InventorySnapshot.store_id == store_id,
        InventorySnapshot.snapshot_date == data_as_of,
        InventorySnapshot.product_id.in_(product_ids),
    )
    rows = session.execute(stmt).all()
    return {int(r.product_id): (int(r.on_hand), int(r.on_order)) for r in rows}


def _build_product_recommendation(
    *,
    prod: DimProduct,
    param: InventoryParam | None,
    on_hand: int,
    on_order: int,
    yhat_list: list[float],
    sigma_e: float,
    eff_overstock_days: float,
    eff_run_id: uuid.UUID,
    store_id: int,
    eff_as_of: datetime.date,
) -> tuple[InventoryRecommendation, InventoryPolicyEvaluation]:
    """Compute policy metrics and construct an InventoryRecommendation record."""
    pid = int(prod.product_id)
    abc = str(prod.abc_class).strip().upper() if prod.abc_class else "C"

    # Parameter values (fallback to defaults if param is missing)
    if param is not None:
        lt_days = float(param.lead_time_days)
        lt_std_days = float(param.lead_time_std_days)
        service_level = float(param.service_level)
        unit_cost = (
            float(param.unit_cost)
            if param.unit_cost is not None and param.unit_cost > 0
            else (float(prod.unit_cost_est) if prod.unit_cost_est else 10.0)
        )
        holding_cost_rate = float(param.holding_cost_rate)
        order_cost = float(param.order_cost)
        moq = int(param.moq)
        params_source = param.source
    else:
        # Default fallback parameters when none seeded
        lt_days = 7.0 if abc == "A" else (14.0 if abc == "B" else 21.0)
        lt_std_days = round(0.2 * lt_days, 2)
        service_level = 0.98 if abc == "A" else (0.95 if abc == "B" else 0.90)
        unit_cost = float(prod.unit_cost_est) if prod.unit_cost_est else 10.0
        holding_cost_rate = 0.20
        order_cost = 25.0
        moq = 1
        params_source = "synthetic"

    # Calculate forecast demand over lead time
    if yhat_list:
        l_ceil = max(1, math.ceil(lt_days))
        window = yhat_list[:l_ceil]
        mu_l = sum(window)
        mu_d = mu_l / len(window)
        if l_ceil > len(yhat_list):
            mu_l = mu_d * lt_days
        is_eligible = True
    else:
        mu_d = 0.0
        mu_l = 0.0
        is_eligible = False

    # Policy evaluation
    eval_res = evaluate_inventory_policy(
        on_hand=on_hand,
        on_order=on_order,
        lead_time_days=lt_days,
        lead_time_std_days=lt_std_days,
        mu_d=mu_d,
        sigma_e=sigma_e,
        unit_cost=unit_cost,
        holding_cost_rate=holding_cost_rate,
        order_cost=order_cost,
        service_level=service_level,
        moq=moq,
        overstock_days=eff_overstock_days,
        is_eligible=is_eligible,
    )

    rec = InventoryRecommendation(
        recommendation_id=uuid.uuid4(),
        forecast_run_id=eff_run_id,
        store_id=store_id,
        product_id=pid,
        as_of_date=eff_as_of,
        on_hand=on_hand,
        on_order=on_order,
        lead_time_days=Decimal(str(round(lt_days, 1))),
        service_level=Decimal(str(round(service_level, 3))),
        demand_mean_lt=Decimal(str(round(eval_res.mu_l, 4))),
        demand_std_lt=Decimal(str(round(eval_res.sigma_l, 4))),
        safety_stock=Decimal(str(round(eval_res.safety_stock, 2))),
        reorder_point=Decimal(str(round(eval_res.reorder_point, 2))),
        order_quantity=Decimal(str(eval_res.recommended_order_qty)),
        status=eval_res.status,
        reorder_urgency=eval_res.reorder_urgency,
        expected_stockout_prob=Decimal(str(round(eval_res.stockout_risk, 4))),
        days_of_cover=(
            Decimal(str(round(eval_res.days_of_cover, 2)))
            if eval_res.days_of_cover is not None
            else None
        ),
        holding_cost_rate=Decimal(str(round(holding_cost_rate, 4))),
        order_cost=Decimal(str(round(order_cost, 2))),
        unit_cost=Decimal(str(round(unit_cost, 4))),
        params_source=params_source,
    )
    return rec, eval_res


def generate_inventory_recommendations(
    session: Session,
    store_id: int = 1,
    *,
    data_as_of: datetime.date | None = None,
    forecast_run_id: uuid.UUID | None = None,
    overstock_days: float | None = None,
    product_ids: Sequence[int] | None = None,
) -> InventoryScoringResult:
    """Generate and persist inventory replenishment recommendations and update active pointer.

    Args:
        session: Active SQLAlchemy database session.
        store_id: Store identifier (default 1).
        data_as_of: Cutoff origin date.
        forecast_run_id: Optional explicit forecast model run ID.
        overstock_days: Threshold days of cover for overstock status.
        product_ids: Optional list of product IDs to restrict recommendations to.

    Returns:
        InventoryScoringResult with recommendation counts and risk statistics.
    """
    settings = get_settings()
    eff_overstock_days = (
        overstock_days if overstock_days is not None else float(settings.overstock_days)
    )

    # 1. Resolve model run and as_of date
    forecast_run, run_as_of = _resolve_forecast_run(session, forecast_run_id)
    eff_as_of = _resolve_data_as_of(session, data_as_of, run_as_of)
    eff_run_id = forecast_run.run_id

    # 2. Query product dimension
    prod_stmt = sa.select(DimProduct).where(DimProduct.is_product.is_(True))
    if product_ids is not None:
        prod_stmt = prod_stmt.where(DimProduct.product_id.in_(product_ids))
    products = session.execute(prod_stmt).scalars().all()

    if not products:
        logger.warning("No active products found for inventory scoring.")
        return InventoryScoringResult(
            forecast_run_id=eff_run_id,
            store_id=store_id,
            data_as_of=eff_as_of,
            n_recommendations=0,
            status_counts={},
            mean_stockout_risk=0.0,
            total_recommended_order_qty=0,
            active_output_updated=False,
        )

    all_pids = [int(p.product_id) for p in products]

    # 3. Load forecasts, error stds, params, and snapshots
    forecasts = _load_forecasts_by_product(session, eff_run_id, store_id, eff_as_of)
    sigma_e_by_abc, global_sigma_e = _load_forecast_error_stds(session, eff_run_id)
    params = _load_inventory_params(session, store_id, all_pids)
    snapshots = _load_inventory_snapshots(session, store_id, eff_as_of, all_pids)

    # 4. Generate recommendations across products
    recs: list[InventoryRecommendation] = []
    status_counter: Counter[str] = Counter()
    total_risk = 0.0
    total_rec_order = 0

    for prod in products:
        pid = int(prod.product_id)
        abc = str(prod.abc_class).strip().upper() if prod.abc_class else "C"
        param = params.get(pid)
        on_hand, on_order = snapshots.get(pid, (0, 0))
        yhat_list = forecasts.get(pid, [])
        sigma_e = sigma_e_by_abc.get(abc, global_sigma_e)

        rec, eval_res = _build_product_recommendation(
            prod=prod,
            param=param,
            on_hand=on_hand,
            on_order=on_order,
            yhat_list=yhat_list,
            sigma_e=sigma_e,
            eff_overstock_days=eff_overstock_days,
            eff_run_id=eff_run_id,
            store_id=store_id,
            eff_as_of=eff_as_of,
        )

        recs.append(rec)
        status_counter[eval_res.status] += 1
        total_risk += eval_res.stockout_risk
        total_rec_order += eval_res.recommended_order_qty

    # 5. Persist recommendations & update active pointer atomically
    with session.begin_nested():
        # Clear existing recommendations for this run, store, and as_of_date
        session.execute(
            sa.delete(InventoryRecommendation).where(
                InventoryRecommendation.forecast_run_id == eff_run_id,
                InventoryRecommendation.store_id == store_id,
                InventoryRecommendation.as_of_date == eff_as_of,
                InventoryRecommendation.product_id.in_(all_pids),
            )
        )
        session.add_all(recs)

        # Update ml.active_output pointer for inventory
        session.execute(sa.delete(ActiveOutput).where(ActiveOutput.task == TASK_INVENTORY))
        active_rec = ActiveOutput(
            task=TASK_INVENTORY,
            model_run_id=eff_run_id,
            data_as_of=eff_as_of,
        )
        session.add(active_rec)

    session.commit()

    n_recs = len(recs)
    mean_risk = total_risk / n_recs if n_recs > 0 else 0.0

    return InventoryScoringResult(
        forecast_run_id=eff_run_id,
        store_id=store_id,
        data_as_of=eff_as_of,
        n_recommendations=n_recs,
        status_counts=dict(status_counter),
        mean_stockout_risk=mean_risk,
        total_recommended_order_qty=total_rec_order,
        active_output_updated=True,
    )

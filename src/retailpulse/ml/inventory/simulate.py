"""Inventory policy simulation engine (P4-04).

Implements TRD §5.4, PRD §7.5 FR-INV-05, DATABASE §7, API §3.8:
- Day-by-day lost-sales historical replay over a specified window (default 28-day span).
- Stochastic lead times drawn from N(L, sigma_LT) (deterministic via random seed).
- Comparison between:
    1. RetailPulse policy: forecast-driven ROP = mu_L + SS with EOQ / MOQ.
    2. Baseline policy: ROP = 28-day moving average * (L + 14) with EOQ / MOQ.
    3. Infinite-stock policy mode: verifying ideal 1.0 fill rate invariant.
- Standardized initial inventory: both policies start with on_hand = ceil(ROP_baseline).
- Detailed metric reporting:
    - fill_rate (units served / units demanded)
    - stockout_day_rate (days with unmet lost sales / total SKU-days)
    - avg_on_hand_units & avg_on_hand_value
    - holding_cost (holding_cost_rate / 365 per day)
    - n_orders placed
- Synchronous execution supporting up to 200 SKUs.
- Persistence into ml.policy_simulation table.
"""

from __future__ import annotations

import datetime
import math
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import (
    DimDate,
    DimProduct,
    FactDailySales,
    InventoryParam,
)
from retailpulse.db.models.ml import (
    ActiveOutput,
    Forecast,
    ForecastMetric,
    ModelRun,
    PolicySimulation,
)
from retailpulse.logging import get_logger
from retailpulse.ml.inventory.policy import (
    compute_eoq,
    compute_lead_time_demand_std,
    compute_reorder_point,
    compute_safety_stock,
    round_order_quantity_to_moq,
)

logger = get_logger(__name__)

TASK_FORECAST = "forecast"
MAX_SIMULATION_SKUS = 200
DEFAULT_COVER_DAYS_EXTRA = 14
DEFAULT_MOVING_AVG_DAYS = 28


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration options for an inventory policy simulation run."""

    store_id: int = 1
    window_start: datetime.date | None = None
    window_end: datetime.date | None = None
    abc_class: str | None = None
    product_ids: Sequence[int] | None = None
    limit: int = 200
    service_level: float | None = None
    baseline_type: str = "moving_average_28"
    cover_days_extra: int = DEFAULT_COVER_DAYS_EXTRA
    seed: int | None = None
    initial_on_hand_mode: str = "baseline_rop"
    run_infinite_stock: bool = False
    persist: bool = True


@dataclass(frozen=True)
class PolicyMetrics:
    """Summary performance metrics for a single inventory replenishment policy."""

    fill_rate: float
    stockout_day_rate: float
    avg_on_hand_units: float
    avg_on_hand_value: float
    holding_cost: float
    n_orders: int
    total_demand: int
    total_served: int
    total_lost: int
    stockout_days: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return asdict(self)


@dataclass
class PendingOrder:
    """Purchase order scheduled to arrive in a future simulation day."""

    product_id: int
    quantity: int
    arrival_date: datetime.date


@dataclass(frozen=True)
class SimulationResult:
    """Consolidated simulation result matching API §3.8 and database persistence."""

    simulation_id: uuid.UUID
    forecast_run_id: uuid.UUID | None
    created_by: uuid.UUID | None
    window_start: datetime.date
    window_end: datetime.date
    uses_synthetic_params: bool
    config: dict[str, Any]
    results: dict[str, Any]
    created_at: datetime.datetime | None = None
    sku_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "simulation_id": str(self.simulation_id),
            "forecast_run_id": str(self.forecast_run_id) if self.forecast_run_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "window": {
                "start": self.window_start.isoformat(),
                "end": self.window_end.isoformat(),
            },
            "uses_synthetic_params": self.uses_synthetic_params,
            "config": self.config,
            "results": self.results,
            "sku_details": self.sku_details,
        }


def _resolve_forecast_run(
    session: Session,
    forecast_run_id: uuid.UUID | None = None,
) -> tuple[ModelRun | None, datetime.date | None]:
    """Resolve the active or specified forecast model run."""
    if forecast_run_id is not None:
        run = session.get(ModelRun, forecast_run_id)
        if run is None:
            raise ValueError(f"Forecast model run {forecast_run_id} not found.")
        return run, None

    active_ptr = session.get(ActiveOutput, TASK_FORECAST)
    if active_ptr is not None:
        run = session.get(ModelRun, active_ptr.model_run_id)
        if run is not None:
            return run, active_ptr.data_as_of

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

    stmt_latest = (
        sa.select(ModelRun)
        .where(ModelRun.task == TASK_FORECAST)
        .order_by(ModelRun.trained_at.desc())
    )
    latest_run = session.execute(stmt_latest).scalars().first()
    return latest_run, None


def _resolve_simulation_dates(
    session: Session,
    window_start: datetime.date | None,
    window_end: datetime.date | None,
    run_as_of: datetime.date | None,
) -> tuple[datetime.date, datetime.date, list[datetime.date]]:
    """Determine effective simulation window start and end dates and generate sorted date list."""
    if window_end is not None:
        eff_end = window_end
    elif run_as_of is not None:
        eff_end = run_as_of
    else:
        max_sales_date = session.execute(
            sa.select(sa.func.max(FactDailySales.date_key))
        ).scalar_one_or_none()
        if max_sales_date is not None:
            eff_end = max_sales_date
        else:
            latest_dim_date = session.execute(
                sa.select(sa.func.max(DimDate.date_key)).where(DimDate.is_complete.is_(True))
            ).scalar_one_or_none()
            eff_end = latest_dim_date if latest_dim_date is not None else datetime.date(2011, 12, 8)

    if window_start is not None:
        eff_start = window_start
    else:
        eff_start = eff_end - datetime.timedelta(days=DEFAULT_MOVING_AVG_DAYS - 1)

    if eff_start > eff_end:
        raise ValueError(f"window_start ({eff_start}) cannot be after window_end ({eff_end}).")

    num_days = (eff_end - eff_start).days + 1
    dates = [eff_start + datetime.timedelta(days=i) for i in range(num_days)]
    return eff_start, eff_end, dates


def _load_simulation_products(
    session: Session,
    abc_class: str | None,
    product_ids: Sequence[int] | None,
    limit: int,
) -> list[DimProduct]:
    """Retrieve filtered and active products for simulation, enforcing <= 200 SKU limit."""
    if product_ids is not None and len(product_ids) > MAX_SIMULATION_SKUS:
        raise ValueError(
            f"Simulation scope limit cannot exceed {MAX_SIMULATION_SKUS} SKUs "
            f"(got {len(product_ids)})."
        )
    if limit > MAX_SIMULATION_SKUS:
        raise ValueError(
            f"Simulation scope limit cannot exceed {MAX_SIMULATION_SKUS} SKUs (got {limit})."
        )

    stmt = sa.select(DimProduct).where(DimProduct.is_product.is_(True))
    if abc_class is not None and abc_class.strip().upper() != "ALL":
        stmt = stmt.where(DimProduct.abc_class == abc_class.strip().upper())
    if product_ids is not None:
        stmt = stmt.where(DimProduct.product_id.in_(product_ids))

    stmt = stmt.order_by(DimProduct.product_id).limit(limit)
    products = list(session.execute(stmt).scalars().all())
    return products


def _load_forecast_error_stds(
    session: Session,
    forecast_run_id: uuid.UUID | None,
) -> tuple[dict[str, float], float]:
    """Retrieve backtest RMSE metrics by ABC class and global from ml.forecast_metric."""
    if forecast_run_id is None:
        return {"A": 1.0, "B": 2.0, "C": 3.0}, 2.0

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
        if r.scope == "abc" and r.scope_key:
            sigma_e_by_abc[str(r.scope_key).strip().upper()] = val
        elif r.scope == "global":
            global_sigma_e = val

    return sigma_e_by_abc, global_sigma_e


def _load_demands_and_history(
    session: Session,
    store_id: int,
    product_ids: Sequence[int],
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[dict[tuple[int, datetime.date], int], dict[tuple[int, datetime.date], int]]:
    """Load daily sales demands for simulation span and prior 28 days for moving average."""
    history_start = start_date - datetime.timedelta(days=DEFAULT_MOVING_AVG_DAYS)
    stmt = (
        sa.select(
            FactDailySales.product_id,
            FactDailySales.date_key,
            sa.func.sum(FactDailySales.units_sold).label("units_sold"),
        )
        .where(
            FactDailySales.store_id == store_id,
            FactDailySales.product_id.in_(product_ids),
            FactDailySales.date_key >= history_start,
            FactDailySales.date_key <= end_date,
        )
        .group_by(FactDailySales.product_id, FactDailySales.date_key)
    )

    rows = session.execute(stmt).all()

    sim_demands: dict[tuple[int, datetime.date], int] = {}
    history_demands: dict[tuple[int, datetime.date], int] = {}

    for r in rows:
        pid = int(r.product_id)
        dt = r.date_key
        qty = max(0, int(r.units_sold))
        history_demands[(pid, dt)] = qty
        if dt >= start_date:
            sim_demands[(pid, dt)] = qty

    return sim_demands, history_demands


def _precompute_lead_times(
    products: Sequence[DimProduct],
    params: Mapping[int, InventoryParam | None],
    dates: Sequence[datetime.date],
    seed: int,
) -> dict[tuple[int, datetime.date], int]:
    """Pre-draw deterministic lead times for each SKU and order date using Common Random Numbers."""
    rng = np.random.RandomState(seed)
    lead_times: dict[tuple[int, datetime.date], int] = {}

    for prod in products:
        pid = int(prod.product_id)
        param = params.get(pid)
        abc = str(prod.abc_class).strip().upper() if prod.abc_class else "C"
        lt_mean = float(param.lead_time_days) if param else (7.0 if abc == "A" else 14.0)
        lt_std = float(param.lead_time_std_days) if param else round(0.2 * lt_mean, 2)

        for dt in dates:
            if lt_std > 0:
                draw = rng.normal(lt_mean, lt_std)
                lt = max(1, round(draw))
            else:
                lt = max(1, round(lt_mean))
            lead_times[(pid, dt)] = lt

    return lead_times


def _compute_ma28(
    history_demands: dict[tuple[int, datetime.date], int],
    pid: int,
    current_date: datetime.date,
) -> float:
    """Compute 28-day moving average demand preceding current_date."""
    total_qty = 0
    for i in range(1, DEFAULT_MOVING_AVG_DAYS + 1):
        prev_dt = current_date - datetime.timedelta(days=i)
        total_qty += history_demands.get((pid, prev_dt), 0)
    return total_qty / float(DEFAULT_MOVING_AVG_DAYS)


def _compute_sku_order_quantity(
    *,
    pol: str,
    prod: DimProduct,
    param: InventoryParam | None,
    ip: int,
    current_date: datetime.date,
    config: SimulationConfig,
    history_demands: dict[tuple[int, datetime.date], int],
    forecast_rates: dict[int, float],
    sigma_e_by_abc: dict[str, float],
    global_sigma_e: float,
) -> int:
    """Evaluate replenishment policy logic and return order quantity for a SKU."""
    if pol == "infinite_stock":
        return 0

    pid = int(prod.product_id)
    abc = str(prod.abc_class).strip().upper() if prod.abc_class else "C"

    lt_mean = float(param.lead_time_days) if param else (7.0 if abc == "A" else 14.0)
    lt_std = float(param.lead_time_std_days) if param else round(0.2 * lt_mean, 2)
    srv_lvl = (
        config.service_level
        if config.service_level is not None
        else (float(param.service_level) if param else 0.95)
    )
    unit_cost = (
        float(param.unit_cost)
        if param and param.unit_cost is not None and param.unit_cost > 0
        else (float(prod.unit_cost_est) if prod.unit_cost_est else 10.0)
    )
    holding_cost_rate = float(param.holding_cost_rate) if param else 0.20
    order_cost = float(param.order_cost) if param else 25.0
    moq = int(param.moq) if param else 1

    if pol == "retailpulse":
        mu_d = forecast_rates.get(pid, 0.0)
        sigma_e = sigma_e_by_abc.get(abc, global_sigma_e)
        sigma_l = compute_lead_time_demand_std(
            lead_time_days=lt_mean,
            sigma_e=sigma_e,
            mu_d=mu_d,
            lead_time_std_days=lt_std,
        )
        ss = compute_safety_stock(sigma_l=sigma_l, service_level=srv_lvl)
        rop = compute_reorder_point(mu_l=mu_d * lt_mean, safety_stock=ss)
        eoq_val = compute_eoq(
            annual_demand=365.0 * mu_d,
            order_cost=order_cost,
            holding_cost_rate=holding_cost_rate,
            unit_cost=unit_cost,
        )
        batch_q = round_order_quantity_to_moq(eoq_val, moq)

        if ip <= rop and batch_q > 0:
            deficit = rop - ip
            multiplier = max(1, math.ceil(deficit / batch_q))
            return multiplier * batch_q
        return 0

    if pol == "baseline":
        ma28 = _compute_ma28(history_demands, pid, current_date)
        rop = ma28 * (lt_mean + float(config.cover_days_extra))
        eoq_val = compute_eoq(
            annual_demand=365.0 * ma28,
            order_cost=order_cost,
            holding_cost_rate=holding_cost_rate,
            unit_cost=unit_cost,
        )
        batch_q = round_order_quantity_to_moq(eoq_val, moq)

        if ip <= rop and batch_q > 0:
            deficit = rop - ip
            multiplier = max(1, math.ceil(deficit / batch_q))
            return multiplier * batch_q
        return 0

    return 0


def _deliver_orders(
    pending_orders: list[PendingOrder],
    current_date: datetime.date,
    on_hand: dict[int, int],
    on_order: dict[int, int],
) -> list[PendingOrder]:
    """Process order arrivals on current_date and return remaining pending orders."""
    remaining: list[PendingOrder] = []
    for order in pending_orders:
        if order.arrival_date <= current_date:
            on_hand[order.product_id] += order.quantity
            on_order[order.product_id] = max(0, on_order[order.product_id] - order.quantity)
        else:
            remaining.append(order)
    return remaining


def _fulfill_demands(
    *,
    products: Sequence[DimProduct],
    params: Mapping[int, InventoryParam | None],
    current_date: datetime.date,
    sim_demands: dict[tuple[int, datetime.date], int],
    on_hand: dict[int, int],
    sku_stats: dict[int, dict[str, float]],
) -> tuple[int, int, int, int, float, float, float]:
    """Fulfill daily demand for all SKUs and update inventories and stats."""
    dem_tot, srv_tot, lost_tot, stk_days = 0, 0, 0, 0
    oh_units_tot, oh_val_tot, hold_cost_tot = 0.0, 0.0, 0.0

    for prod in products:
        pid = int(prod.product_id)
        param = params.get(pid)
        unit_cost = (
            float(param.unit_cost)
            if param and param.unit_cost is not None and param.unit_cost > 0
            else (float(prod.unit_cost_est) if prod.unit_cost_est else 10.0)
        )
        holding_cost_rate = float(param.holding_cost_rate) if param else 0.20

        demand = sim_demands.get((pid, current_date), 0)
        dem_tot += demand
        sku_stats[pid]["demand"] += demand

        if on_hand[pid] >= demand:
            served = demand
            lost = 0
            on_hand[pid] -= demand
        else:
            served = on_hand[pid]
            lost = demand - on_hand[pid]
            on_hand[pid] = 0

        srv_tot += served
        lost_tot += lost
        sku_stats[pid]["served"] += served
        sku_stats[pid]["lost"] += lost

        if lost > 0:
            stk_days += 1
            sku_stats[pid]["stockout_days"] += 1

        ending_oh = on_hand[pid]
        oh_units_tot += ending_oh
        oh_val_tot += ending_oh * unit_cost
        hold_cost_tot += ending_oh * unit_cost * (holding_cost_rate / 365.0)
        sku_stats[pid]["on_hand_units"] += ending_oh

    return dem_tot, srv_tot, lost_tot, stk_days, oh_units_tot, oh_val_tot, hold_cost_tot


def _run_single_policy(
    *,
    pol: str,
    dates: Sequence[datetime.date],
    products: Sequence[DimProduct],
    params: Mapping[int, InventoryParam | None],
    initial_on_hand: dict[int, int],
    sim_demands: dict[tuple[int, datetime.date], int],
    history_demands: dict[tuple[int, datetime.date], int],
    forecast_rates: dict[int, float],
    sigma_e_by_abc: dict[str, float],
    global_sigma_e: float,
    lead_times: dict[tuple[int, datetime.date], int],
    config: SimulationConfig,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Execute simulation for a single replenishment policy."""
    n_days = len(dates)
    n_skus = len(products)

    on_hand: dict[int, int] = {}
    on_order: dict[int, int] = {int(p.product_id): 0 for p in products}
    pending_orders: list[PendingOrder] = []

    for p in products:
        pid = int(p.product_id)
        on_hand[pid] = 1_000_000_000 if pol == "infinite_stock" else initial_on_hand[pid]

    total_demand = 0
    total_served = 0
    total_lost = 0
    stockout_days = 0
    total_on_hand_units = 0.0
    total_on_hand_value = 0.0
    total_holding_cost = 0.0
    n_orders = 0

    sku_stats: dict[int, dict[str, float]] = {
        int(p.product_id): {
            "demand": 0.0,
            "served": 0.0,
            "lost": 0.0,
            "stockout_days": 0.0,
            "on_hand_units": 0.0,
            "orders": 0.0,
        }
        for p in products
    }

    for current_date in dates:
        # A. Deliveries arrive at start of day
        pending_orders = _deliver_orders(pending_orders, current_date, on_hand, on_order)

        # B. Demand fulfillment and lost sales
        d_tot, s_tot, l_tot, stk, oh_u, oh_v, h_cost = _fulfill_demands(
            products=products,
            params=params,
            current_date=current_date,
            sim_demands=sim_demands,
            on_hand=on_hand,
            sku_stats=sku_stats,
        )
        total_demand += d_tot
        total_served += s_tot
        total_lost += l_tot
        stockout_days += stk
        total_on_hand_units += oh_u
        total_on_hand_value += oh_v
        total_holding_cost += h_cost

        # C. Review and ordering decisions at end of day
        for prod in products:
            pid = int(prod.product_id)
            param = params.get(pid)
            ip = on_hand[pid] + on_order[pid]

            order_qty = _compute_sku_order_quantity(
                pol=pol,
                prod=prod,
                param=param,
                ip=ip,
                current_date=current_date,
                config=config,
                history_demands=history_demands,
                forecast_rates=forecast_rates,
                sigma_e_by_abc=sigma_e_by_abc,
                global_sigma_e=global_sigma_e,
            )

            if order_qty > 0:
                lt = lead_times[(pid, current_date)]
                arrival_dt = current_date + datetime.timedelta(days=lt)
                on_order[pid] += order_qty
                pending_orders.append(
                    PendingOrder(
                        product_id=pid,
                        quantity=order_qty,
                        arrival_date=arrival_dt,
                    )
                )
                n_orders += 1
                sku_stats[pid]["orders"] += 1

    fill_rate = total_served / total_demand if total_demand > 0 else 1.0
    total_sku_days = n_skus * n_days
    stockout_day_rate = stockout_days / total_sku_days if total_sku_days > 0 else 0.0
    avg_oh_units = total_on_hand_units / n_days if n_days > 0 else 0.0
    avg_oh_value = total_on_hand_value / n_days if n_days > 0 else 0.0

    metrics = PolicyMetrics(
        fill_rate=round(fill_rate, 4),
        stockout_day_rate=round(stockout_day_rate, 4),
        avg_on_hand_units=round(avg_oh_units, 1),
        avg_on_hand_value=round(avg_oh_value, 2),
        holding_cost=round(total_holding_cost, 2),
        n_orders=n_orders,
        total_demand=total_demand,
        total_served=total_served,
        total_lost=total_lost,
        stockout_days=stockout_days,
    ).to_dict()

    per_sku: dict[int, dict[str, Any]] = {}
    for pid, s in sku_stats.items():
        per_sku[pid] = {
            "fill_rate": round(s["served"] / s["demand"], 4) if s["demand"] > 0 else 1.0,
            "avg_on_hand_units": round(s["on_hand_units"] / n_days, 1) if n_days > 0 else 0.0,
            "stockout_days": int(s["stockout_days"]),
            "orders": int(s["orders"]),
        }

    return metrics, per_sku


def simulate_inventory_replay(
    *,
    dates: Sequence[datetime.date],
    sim_demands: dict[tuple[int, datetime.date], int],
    history_demands: dict[tuple[int, datetime.date], int],
    products: Sequence[DimProduct],
    params: Mapping[int, InventoryParam | None],
    forecast_rates: dict[int, float],
    sigma_e_by_abc: dict[str, float],
    global_sigma_e: float,
    lead_times: dict[tuple[int, datetime.date], int],
    config: SimulationConfig,
) -> dict[str, Any]:
    """Execute pure in-memory day-by-day lost-sales simulation across policies."""
    policies = ["retailpulse", "baseline"]
    if config.run_infinite_stock:
        policies.append("infinite_stock")

    first_date = dates[0] if dates else datetime.date(2011, 12, 8)

    # 1. Compute initial on-hand at window start (baseline ROP)
    initial_on_hand: dict[int, int] = {}
    for prod in products:
        pid = int(prod.product_id)
        param = params.get(pid)
        abc = str(prod.abc_class).strip().upper() if prod.abc_class else "C"
        lt_mean = float(param.lead_time_days) if param else (7.0 if abc == "A" else 14.0)

        ma28 = _compute_ma28(history_demands, pid, first_date)
        rop_baseline = ma28 * (lt_mean + float(config.cover_days_extra))
        initial_on_hand[pid] = max(0, math.ceil(rop_baseline))

    policy_results: dict[str, dict[str, Any]] = {}
    sku_performance: defaultdict[int, dict[str, Any]] = defaultdict(dict)

    # 2. Run simulation loop for each policy
    for pol in policies:
        pm_dict, per_sku = _run_single_policy(
            pol=pol,
            dates=dates,
            products=products,
            params=params,
            initial_on_hand=initial_on_hand,
            sim_demands=sim_demands,
            history_demands=history_demands,
            forecast_rates=forecast_rates,
            sigma_e_by_abc=sigma_e_by_abc,
            global_sigma_e=global_sigma_e,
            lead_times=lead_times,
            config=config,
        )
        policy_results[pol] = pm_dict
        for pid, sku_res in per_sku.items():
            sku_performance[pid][pol] = sku_res

    # 3. Compute delta between retailpulse and baseline
    rp_res = policy_results["retailpulse"]
    base_res = policy_results["baseline"]
    delta = {
        "fill_rate_diff": round(rp_res["fill_rate"] - base_res["fill_rate"], 4),
        "stockout_day_rate_diff": round(
            rp_res["stockout_day_rate"] - base_res["stockout_day_rate"], 4
        ),
        "avg_on_hand_units_diff": round(
            rp_res["avg_on_hand_units"] - base_res["avg_on_hand_units"], 1
        ),
        "avg_on_hand_value_diff": round(
            rp_res["avg_on_hand_value"] - base_res["avg_on_hand_value"], 2
        ),
        "holding_cost_diff": round(rp_res["holding_cost"] - base_res["holding_cost"], 2),
        "n_orders_diff": rp_res["n_orders"] - base_res["n_orders"],
    }
    policy_results["delta"] = delta

    sku_details = [
        {
            "product_id": pid,
            "retailpulse": perf.get("retailpulse", {}),
            "baseline": perf.get("baseline", {}),
        }
        for pid, perf in sku_performance.items()
    ]

    return {
        "results": policy_results,
        "sku_details": sku_details,
    }


def run_policy_simulation(
    session: Session,
    config: SimulationConfig | None = None,
    created_by: uuid.UUID | None = None,
    forecast_run_id: uuid.UUID | None = None,
) -> SimulationResult:
    """Run an inventory policy simulation and optionally persist to ml.policy_simulation.

    Args:
        session: Active SQLAlchemy database session.
        config: SimulationConfig options (uses defaults if None).
        created_by: Optional user UUID who initiated the simulation.
        forecast_run_id: Explicit forecast run ID (defaults to active champion).

    Returns:
        SimulationResult containing simulation metrics, comparison, and metadata.
    """
    settings = get_settings()
    cfg = config or SimulationConfig()
    eff_seed = cfg.seed if cfg.seed is not None else settings.seed

    # 1. Resolve forecast model run and simulation window
    forecast_run, run_as_of = _resolve_forecast_run(session, forecast_run_id)
    eff_run_id = forecast_run.run_id if forecast_run is not None else None

    win_start, win_end, dates = _resolve_simulation_dates(
        session=session,
        window_start=cfg.window_start,
        window_end=cfg.window_end,
        run_as_of=run_as_of,
    )

    # 2. Select products within scope (max 200)
    products = _load_simulation_products(
        session=session,
        abc_class=cfg.abc_class,
        product_ids=cfg.product_ids,
        limit=cfg.limit,
    )

    if not products:
        logger.warning("No active products matched simulation scope.")
        empty_metrics = PolicyMetrics(
            fill_rate=1.0,
            stockout_day_rate=0.0,
            avg_on_hand_units=0.0,
            avg_on_hand_value=0.0,
            holding_cost=0.0,
            n_orders=0,
            total_demand=0,
            total_served=0,
            total_lost=0,
            stockout_days=0,
        ).to_dict()
        empty_res = {
            "retailpulse": empty_metrics,
            "baseline": empty_metrics,
            "delta": {
                "fill_rate_diff": 0.0,
                "stockout_day_rate_diff": 0.0,
                "avg_on_hand_units_diff": 0.0,
                "avg_on_hand_value_diff": 0.0,
                "holding_cost_diff": 0.0,
                "n_orders_diff": 0,
            },
        }
        return SimulationResult(
            simulation_id=uuid.uuid4(),
            forecast_run_id=eff_run_id,
            created_by=created_by,
            window_start=win_start,
            window_end=win_end,
            uses_synthetic_params=True,
            config={
                "store_id": cfg.store_id,
                "scope": {"abc_class": cfg.abc_class, "limit": cfg.limit},
                "policy": {"service_level": cfg.service_level},
                "baseline": {
                    "type": cfg.baseline_type,
                    "cover_days_extra": cfg.cover_days_extra,
                },
                "seed": eff_seed,
            },
            results=empty_res,
            created_at=None,
            sku_details=[],
        )

    all_pids = [int(p.product_id) for p in products]

    # 3. Load parameters, error stds, demands, and forecasts
    param_stmt = sa.select(InventoryParam).where(
        InventoryParam.store_id == cfg.store_id,
        InventoryParam.product_id.in_(all_pids),
    )
    param_rows = session.execute(param_stmt).scalars().all()
    params_by_pid = {int(p.product_id): p for p in param_rows}

    uses_synthetic = any(p.source == "synthetic" for p in params_by_pid.values()) or len(
        params_by_pid
    ) < len(all_pids)

    sigma_e_by_abc, global_sigma_e = _load_forecast_error_stds(session, eff_run_id)
    sim_demands, history_demands = _load_demands_and_history(
        session=session,
        store_id=cfg.store_id,
        product_ids=all_pids,
        start_date=win_start,
        end_date=win_end,
    )

    # Calculate average forecast daily rate per product
    forecast_rates: dict[int, float] = {}
    if eff_run_id is not None:
        fc_stmt = (
            sa.select(
                Forecast.product_id,
                sa.func.avg(Forecast.yhat).label("mean_yhat"),
            )
            .where(
                Forecast.model_run_id == eff_run_id,
                Forecast.store_id == cfg.store_id,
                Forecast.product_id.in_(all_pids),
            )
            .group_by(Forecast.product_id)
        )
        fc_rows = session.execute(fc_stmt).all()
        for r in fc_rows:
            forecast_rates[int(r.product_id)] = max(0.0, float(r.mean_yhat))

    for pid in all_pids:
        if pid not in forecast_rates:
            forecast_rates[pid] = _compute_ma28(history_demands, pid, win_start)

    # 4. Precompute deterministic lead times using Common Random Numbers
    lead_times = _precompute_lead_times(
        products=products,
        params=params_by_pid,
        dates=dates,
        seed=eff_seed,
    )

    # 5. Run replay simulation
    sim_output = simulate_inventory_replay(
        dates=dates,
        sim_demands=sim_demands,
        history_demands=history_demands,
        products=products,
        params=params_by_pid,
        forecast_rates=forecast_rates,
        sigma_e_by_abc=sigma_e_by_abc,
        global_sigma_e=global_sigma_e,
        lead_times=lead_times,
        config=cfg,
    )

    eff_srv_lvl = (
        cfg.service_level if cfg.service_level is not None else settings.default_service_level
    )
    config_dict = {
        "store_id": cfg.store_id,
        "scope": {
            "abc_class": cfg.abc_class,
            "limit": cfg.limit,
            "product_ids": list(cfg.product_ids) if cfg.product_ids else None,
        },
        "policy": {
            "service_level": eff_srv_lvl,
        },
        "baseline": {
            "type": cfg.baseline_type,
            "cover_days_extra": cfg.cover_days_extra,
        },
        "seed": eff_seed,
    }

    sim_id = uuid.uuid4()
    created_at: datetime.datetime | None = None

    # 6. Persist to ml.policy_simulation if requested
    if cfg.persist:
        sim_record = PolicySimulation(
            simulation_id=sim_id,
            forecast_run_id=eff_run_id,
            created_by=created_by,
            window_start=win_start,
            window_end=win_end,
            config=config_dict,
            results=sim_output["results"],
            uses_synthetic_params=uses_synthetic,
        )
        session.add(sim_record)
        session.commit()
        session.refresh(sim_record)
        created_at = sim_record.created_at

    return SimulationResult(
        simulation_id=sim_id,
        forecast_run_id=eff_run_id,
        created_by=created_by,
        window_start=win_start,
        window_end=win_end,
        uses_synthetic_params=uses_synthetic,
        config=config_dict,
        results=sim_output["results"],
        created_at=created_at,
        sku_details=sim_output["sku_details"],
    )

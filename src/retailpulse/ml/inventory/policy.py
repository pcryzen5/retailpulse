"""Pure inventory policy functions for (s, Q) replenishment and stockout risk.

Implements inventory optimization policies specified in TRD §5.4, PRD §7.5 (FR-INV-02,
FR-INV-03, FR-INV-06), and TESTING §3.2:
  - Lead-time demand standard deviation (sigma_l) with lead-time uncertainty.
  - Safety stock (SS = z * sigma_l).
  - Reorder point (ROP = mu_l + SS).
  - Economic Order Quantity (EOQ) and batch size Q rounded up to MOQ multiples.
  - Recommended order quantity when inventory position <= ROP.
  - Stockout risk under normal approximation and empirical quantile fallback.
  - Days of cover and inventory status classification (reorder_now, overstock, ok, no_data).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class InventoryPolicyEvaluation:
    """Consolidated inventory policy evaluation result for a SKU."""

    mu_d: float
    mu_l: float
    sigma_e: float
    sigma_l: float
    safety_stock: float
    reorder_point: float
    inventory_position: int
    eoq: float
    batch_size_q: int
    recommended_order_qty: int
    stockout_risk: float
    days_of_cover: float | None
    status: str
    reorder_urgency: int


def compute_lead_time_demand_std(
    lead_time_days: float,
    sigma_e: float,
    mu_d: float,
    lead_time_std_days: float = 0.0,
) -> float:
    """Compute standard deviation of total demand over lead time (sigma_l).

    Formula (TRD §5.4):
      sigma_l = sqrt( L * sigma_e^2 + mu_d^2 * sigma_LT^2 )

    Args:
        lead_time_days: Expected lead time in days (L >= 0).
        sigma_e: Daily forecast error standard deviation (sigma_e >= 0).
        mu_d: Average daily demand during lead time (mu_d >= 0).
        lead_time_std_days: Lead time standard deviation in days (sigma_LT >= 0).

    Returns:
        Combined lead time demand standard deviation floored at 0.0.
    """
    l_eff = max(0.0, float(lead_time_days))
    s_e = max(0.0, float(sigma_e))
    m_d = max(0.0, float(mu_d))
    s_lt = max(0.0, float(lead_time_std_days))

    var_demand = l_eff * (s_e**2)
    var_lt = (m_d**2) * (s_lt**2)

    return math.sqrt(max(0.0, var_demand + var_lt))


def compute_safety_stock(
    sigma_l: float,
    service_level: float = 0.95,
) -> float:
    """Compute safety stock (SS) under standard normal approximation.

    Formula (TRD §5.4):
      z = Phi^-1(service_level)
      SS = z * sigma_l (floored at 0.0 when service_level >= 0.5)

    Args:
        sigma_l: Standard deviation of lead time demand.
        service_level: Target cycle service level in (0, 1).

    Returns:
        Safety stock units (>= 0.0 for service_level >= 0.5).
    """
    s_l = max(0.0, float(sigma_l))
    sl = min(max(float(service_level), 1e-4), 1.0 - 1e-4)

    z = float(stats.norm.ppf(sl))
    if sl >= 0.5:
        return max(0.0, z * s_l)
    return z * s_l


def compute_reorder_point(
    mu_l: float,
    safety_stock: float,
) -> float:
    """Compute reorder point (ROP).

    Formula (TRD §5.4):
      ROP = mu_l + SS

    Args:
        mu_l: Expected total demand over lead time (units).
        safety_stock: Safety stock buffer (units).

    Returns:
        Reorder point units (>= mu_l when safety_stock >= 0).
    """
    m_l = max(0.0, float(mu_l))
    ss = float(safety_stock)
    return max(0.0, m_l + ss)


def compute_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_rate: float,
    unit_cost: float,
) -> float:
    """Compute classic Economic Order Quantity (EOQ).

    Formula (TRD §5.4):
      h = holding_cost_rate * unit_cost
      EOQ = sqrt( (2 * annual_demand * order_cost) / h )

    Args:
        annual_demand: Expected annual unit demand (365 * mu_d).
        order_cost: Fixed cost per replenishment order (K >= 0).
        holding_cost_rate: Annual holding cost rate fraction (h_rate >= 0).
        unit_cost: Unit purchase/production cost (C > 0).

    Returns:
        Unconstrained EOQ batch size in units (>= 0.0).
    """
    d = max(0.0, float(annual_demand))
    k = max(0.0, float(order_cost))
    h_rate = max(0.0, float(holding_cost_rate))
    c = max(0.0, float(unit_cost))

    h = h_rate * c
    if h <= 0.0 or d <= 0.0 or k <= 0.0:
        return 0.0

    return math.sqrt((2.0 * d * k) / h)


def round_order_quantity_to_moq(
    eoq: float,
    moq: int = 1,
) -> int:
    """Round EOQ up to a multiple of Minimum Order Quantity (at least MOQ).

    Formula (TRD §5.4):
      Q = ceil(EOQ / MOQ) * MOQ, at least MOQ

    Args:
        eoq: Computed EOQ value.
        moq: Minimum order quantity integer (>= 1).

    Returns:
        Batch order quantity Q (integer multiple of MOQ, at least MOQ).
    """
    moq_eff = max(1, int(moq))
    eoq_eff = max(0.0, float(eoq))

    if eoq_eff <= moq_eff:
        return moq_eff

    multiple = math.ceil(eoq_eff / moq_eff)
    return int(multiple * moq_eff)


def compute_recommended_order_quantity(
    inventory_position: float,
    reorder_point: float,
    batch_size_q: int,
) -> int:
    """Compute recommended replenishment order quantity.

    Formula (TRD §5.4):
      when IP <= ROP:
        recommended_order_qty = Q * max(1, ceil((ROP - IP) / Q))
      else:
        0

    Args:
        inventory_position: on_hand + on_order.
        reorder_point: Reorder point threshold ROP.
        batch_size_q: Batch order quantity Q (>= 1).

    Returns:
        Order quantity in integer units (0 if no reorder needed).
    """
    ip = float(inventory_position)
    rop = float(reorder_point)
    q = max(1, int(batch_size_q))

    if ip <= rop:
        deficit = rop - ip
        batches = max(1, math.ceil(deficit / q)) if deficit > 0 else 1
        return int(batches * q)

    return 0


def compute_stockout_risk(
    inventory_position: float,
    mu_l: float,
    sigma_l: float,
) -> float:
    """Compute stockout probability over lead time under normal approximation.

    Formula (TRD §5.4):
      stockout_risk = 1 - Phi( (IP - mu_l) / sigma_l )

    Boundary conditions:
      if sigma_l == 0: 1.0 if IP < mu_l else 0.0.

    Args:
        inventory_position: Current inventory position (on_hand + on_order).
        mu_l: Expected lead time demand.
        sigma_l: Standard deviation of lead time demand.

    Returns:
        Stockout probability in [0.0, 1.0].
    """
    ip = float(inventory_position)
    m_l = float(mu_l)
    s_l = float(sigma_l)

    if s_l <= 0.0:
        return 1.0 if ip < m_l else 0.0

    z = (ip - m_l) / s_l
    risk = float(1.0 - stats.norm.cdf(z))
    return min(max(risk, 0.0), 1.0)


def compute_empirical_stockout_risk(
    inventory_position: float,
    demand_samples: Sequence[float],
) -> float:
    """Compute empirical stockout risk for intermittent SKUs (FR-INV-06).

    Calculates the proportion of historical or simulated lead-time demand
    observations that strictly exceed the inventory position.

    Args:
        inventory_position: Current inventory position (on_hand + on_order).
        demand_samples: Sequence of realized or simulated lead-time demand values.

    Returns:
        Empirical stockout probability in [0.0, 1.0].
    """
    if not demand_samples:
        return 0.0

    ip = float(inventory_position)
    stockout_count = sum(1 for d in demand_samples if float(d) > ip)
    return stockout_count / len(demand_samples)


def compute_days_of_cover(
    on_hand: float,
    mu_d: float,
) -> float | None:
    """Compute inventory days of cover.

    Formula (TRD §5.4):
      days_of_cover = on_hand / mu_d

    Args:
        on_hand: Physical stock on hand.
        mu_d: Expected daily sales rate.

    Returns:
        Days of cover float (or None if daily demand rate is 0).
    """
    oh = max(0.0, float(on_hand))
    m_d = float(mu_d)

    if m_d <= 0.0:
        return None

    return max(0.0, oh / m_d)


def classify_inventory_status(
    inventory_position: float,
    reorder_point: float,
    days_of_cover: float | None,
    is_eligible: bool = True,
    overstock_days: float = 90.0,
) -> str:
    """Classify SKU replenishment status (TRD §5.4, PRD §7.5 FR-INV-03).

    Rules:
      1. 'no_data': SKU is not eligible or has no forecast data.
      2. 'reorder_now': IP <= ROP.
      3. 'overstock': days_of_cover > overstock_days (and IP > ROP).
      4. 'ok': otherwise.

    Args:
        inventory_position: Current inventory position.
        reorder_point: Calculated reorder point.
        days_of_cover: Current days of cover (or None).
        is_eligible: Whether the product is eligible for forecasting/inventory control.
        overstock_days: Threshold days of cover triggering overstock (default 90).

    Returns:
        Status string: 'reorder_now', 'overstock', 'ok', or 'no_data'.
    """
    if not is_eligible:
        return "no_data"

    if inventory_position <= reorder_point:
        return "reorder_now"

    if days_of_cover is not None and days_of_cover > overstock_days:
        return "overstock"

    return "ok"


def evaluate_inventory_policy(  # noqa: PLR0917
    on_hand: int,
    on_order: int,
    lead_time_days: float,
    lead_time_std_days: float,
    mu_d: float,
    sigma_e: float,
    unit_cost: float,
    holding_cost_rate: float = 0.20,
    order_cost: float = 25.0,
    service_level: float = 0.95,
    moq: int = 1,
    overstock_days: float = 90.0,
    is_eligible: bool = True,
) -> InventoryPolicyEvaluation:
    """Evaluate full (s, Q) inventory policy for a single product/store.

    Args:
        on_hand: Units currently on hand.
        on_order: Units currently on order.
        lead_time_days: Lead time in days.
        lead_time_std_days: Lead time standard deviation in days.
        mu_d: Daily demand rate.
        sigma_e: Forecast error daily standard deviation.
        unit_cost: Unit purchase cost.
        holding_cost_rate: Annual holding cost rate (default 0.20).
        order_cost: Cost per order (default 25.0).
        service_level: Target service level (default 0.95).
        moq: Minimum order quantity (default 1).
        overstock_days: Days of cover threshold for overstock (default 90.0).
        is_eligible: SKU forecasting eligibility flag.

    Returns:
        InventoryPolicyEvaluation dataclass instance.
    """
    ip = on_hand + on_order
    mu_l = mu_d * lead_time_days

    sigma_l = compute_lead_time_demand_std(
        lead_time_days=lead_time_days,
        sigma_e=sigma_e,
        mu_d=mu_d,
        lead_time_std_days=lead_time_std_days,
    )

    safety_stock = compute_safety_stock(sigma_l=sigma_l, service_level=service_level)
    rop = compute_reorder_point(mu_l=mu_l, safety_stock=safety_stock)

    annual_demand = 365.0 * mu_d
    eoq = compute_eoq(
        annual_demand=annual_demand,
        order_cost=order_cost,
        holding_cost_rate=holding_cost_rate,
        unit_cost=unit_cost,
    )
    batch_q = round_order_quantity_to_moq(eoq=eoq, moq=moq)

    rec_order_qty = compute_recommended_order_quantity(
        inventory_position=ip,
        reorder_point=rop,
        batch_size_q=batch_q,
    )

    stockout_risk = compute_stockout_risk(
        inventory_position=ip,
        mu_l=mu_l,
        sigma_l=sigma_l,
    )

    days_of_cover = compute_days_of_cover(on_hand=on_hand, mu_d=mu_d)

    status = classify_inventory_status(
        inventory_position=ip,
        reorder_point=rop,
        days_of_cover=days_of_cover,
        is_eligible=is_eligible,
        overstock_days=overstock_days,
    )

    urgency = int(min(100, max(0, round(stockout_risk * 100))))

    return InventoryPolicyEvaluation(
        mu_d=mu_d,
        mu_l=mu_l,
        sigma_e=sigma_e,
        sigma_l=sigma_l,
        safety_stock=safety_stock,
        reorder_point=rop,
        inventory_position=ip,
        eoq=eoq,
        batch_size_q=batch_q,
        recommended_order_qty=rec_order_qty if is_eligible else 0,
        stockout_risk=stockout_risk,
        days_of_cover=days_of_cover,
        status=status,
        reorder_urgency=urgency,
    )

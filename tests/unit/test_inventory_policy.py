"""Unit and property-based tests for inventory policy functions (P4-02).

Validates pure inventory optimization formulas per TRD §5.4, PRD §7.5, and TESTING §3.2:
  - Safety stock (SS >= 0, monotone in service level and sigma).
  - Reorder point (ROP >= mu_l).
  - EOQ and MOQ multiple rounding.
  - Recommended order quantity respects MOQ and clears ROP deficit.
  - Stockout risk bounded in [0, 1] and monotonically decreasing in inventory position.
  - Empirical stockout risk for intermittent series.
  - Status classification across all 4 states (reorder_now, overstock, ok, no_data).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retailpulse.ml.inventory.policy import (
    classify_inventory_status,
    compute_days_of_cover,
    compute_empirical_stockout_risk,
    compute_eoq,
    compute_lead_time_demand_std,
    compute_recommended_order_quantity,
    compute_reorder_point,
    compute_safety_stock,
    compute_stockout_risk,
    evaluate_inventory_policy,
    round_order_quantity_to_moq,
)

pytestmark = pytest.mark.unit


# =====================================================================
# Hypothesis Property-Based Tests
# =====================================================================


@given(
    sigma_l=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    service_level=st.floats(min_value=0.5, max_value=0.999, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_safety_stock_non_negative(sigma_l: float, service_level: float) -> None:
    """Safety stock must be non-negative for standard service levels (SL >= 0.50)."""
    ss = compute_safety_stock(sigma_l=sigma_l, service_level=service_level)
    assert ss >= 0.0


@given(
    mu_l=st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False),
    safety_stock=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_reorder_point_ge_mu_l(mu_l: float, safety_stock: float) -> None:
    """Reorder point must be greater than or equal to lead-time demand (ROP >= mu_l)."""
    rop = compute_reorder_point(mu_l=mu_l, safety_stock=safety_stock)
    assert rop >= mu_l


@given(
    sigma_l=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
    sl1=st.floats(min_value=0.50, max_value=0.99, allow_nan=False, allow_infinity=False),
    sl2=st.floats(min_value=0.50, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_safety_stock_monotone_in_service_level(
    sigma_l: float, sl1: float, sl2: float
) -> None:
    """Safety stock must be monotonically non-decreasing in service level."""
    ss1 = compute_safety_stock(sigma_l=sigma_l, service_level=min(sl1, sl2))
    ss2 = compute_safety_stock(sigma_l=sigma_l, service_level=max(sl1, sl2))
    assert ss1 <= ss2 + 1e-9


@given(
    lead_time=st.floats(min_value=1.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    se1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    se2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    mu_d=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    s_lt=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_sigma_l_monotone_in_sigma_e(
    lead_time: float, se1: float, se2: float, mu_d: float, s_lt: float
) -> None:
    """Lead-time demand std (sigma_l) must be monotonically non-decreasing in sigma_e."""
    s_l1 = compute_lead_time_demand_std(lead_time, min(se1, se2), mu_d, s_lt)
    s_l2 = compute_lead_time_demand_std(lead_time, max(se1, se2), mu_d, s_lt)
    assert s_l1 <= s_l2 + 1e-9


@given(
    annual_demand=st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False),
    order_cost=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    holding_cost_rate=st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    unit_cost=st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_eoq_non_negative(
    annual_demand: float,
    order_cost: float,
    holding_cost_rate: float,
    unit_cost: float,
) -> None:
    """EOQ must always be non-negative."""
    eoq = compute_eoq(annual_demand, order_cost, holding_cost_rate, unit_cost)
    assert eoq >= 0.0


@given(
    eoq=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    moq=st.integers(min_value=1, max_value=100),
)
def test_hypothesis_moq_rounding_properties(eoq: float, moq: int) -> None:
    """Batch size Q must be >= MOQ, >= EOQ, and an exact integer multiple of MOQ."""
    q = round_order_quantity_to_moq(eoq=eoq, moq=moq)
    assert q >= moq
    assert q % moq == 0
    if eoq > moq:
        assert q >= eoq


@given(
    ip=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rop=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    eoq=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    moq=st.integers(min_value=1, max_value=50),
)
def test_hypothesis_order_quantity_respects_moq_and_deficit(
    ip: float, rop: float, eoq: float, moq: int
) -> None:
    """When reorder is triggered, recommended order qty must be >= MOQ and an exact multiple."""
    batch_q = round_order_quantity_to_moq(eoq=eoq, moq=moq)
    rec_qty = compute_recommended_order_quantity(
        inventory_position=ip, reorder_point=rop, batch_size_q=batch_q
    )

    if ip <= rop:
        assert rec_qty >= moq
        assert rec_qty % moq == 0
        assert rec_qty % batch_q == 0
        assert ip + rec_qty >= rop
    else:
        assert rec_qty == 0


@given(
    mu_l=st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
    sigma_l=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    ip1=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    ip2=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_stockout_risk_bounds_and_monotone(
    mu_l: float, sigma_l: float, ip1: float, ip2: float
) -> None:
    """Stockout risk must be in [0, 1] and monotonically non-increasing in inventory position."""
    lower_ip = min(ip1, ip2)
    higher_ip = max(ip1, ip2)

    risk_lower = compute_stockout_risk(lower_ip, mu_l, sigma_l)
    risk_higher = compute_stockout_risk(higher_ip, mu_l, sigma_l)

    assert 0.0 <= risk_lower <= 1.0
    assert 0.0 <= risk_higher <= 1.0
    assert risk_lower >= risk_higher - 1e-9


# =====================================================================
# Deterministic Known-Value Unit Tests
# =====================================================================


def test_lead_time_demand_std_known_calculation() -> None:
    """Verify sigma_l matches hand-calculated values from TRD §5.4.

    Given:
      L = 7 days
      sigma_e = 2.0 units/day
      mu_d = 10.0 units/day
      sigma_LT = 1.4 days (0.2 * L)

    var = L * sigma_e^2 + mu_d^2 * sigma_LT^2
        = 7 * 4.0 + 100.0 * 1.96
        = 28.0 + 196.0 = 224.0
    sigma_l = sqrt(224.0) = 14.9666295...
    """
    sigma_l = compute_lead_time_demand_std(
        lead_time_days=7.0,
        sigma_e=2.0,
        mu_d=10.0,
        lead_time_std_days=1.4,
    )
    expected = math.sqrt(7.0 * 4.0 + 100.0 * 1.96)
    assert math.isclose(sigma_l, expected, rel_tol=1e-6)


def test_safety_stock_and_rop_known_calculation() -> None:
    """Verify safety stock and ROP for 95% service level."""
    sigma_l = 14.96663
    ss = compute_safety_stock(sigma_l=sigma_l, service_level=0.95)
    # z_0.95 ~ 1.6448536 -> SS ~ 24.6179
    assert math.isclose(ss, 24.6179, rel_tol=1e-3)

    mu_l = 70.0
    rop = compute_reorder_point(mu_l=mu_l, safety_stock=ss)
    assert math.isclose(rop, 94.6179, rel_tol=1e-3)


def test_eoq_and_moq_rounding_known_calculation() -> None:
    """Verify EOQ and batch rounding on known inputs.

    Annual demand = 365 * 10 = 3650 units
    Order cost K = 25.0
    Holding cost rate = 0.20
    Unit cost = 10.0 -> h = 2.0
    EOQ = sqrt((2 * 3650 * 25) / 2.0) = sqrt(91250) = 302.076...
    With MOQ = 6:
      ceil(302.076 / 6) = 51 -> Q = 51 * 6 = 306.
    """
    eoq = compute_eoq(
        annual_demand=3650.0,
        order_cost=25.0,
        holding_cost_rate=0.20,
        unit_cost=10.0,
    )
    expected_eoq = math.sqrt((2.0 * 3650.0 * 25.0) / 2.0)
    assert math.isclose(eoq, expected_eoq, rel_tol=1e-6)

    q = round_order_quantity_to_moq(eoq=eoq, moq=6)
    assert q == 306


def test_recommended_order_quantity_scenarios() -> None:
    """Verify order quantity when IP > ROP, IP <= ROP, and deep deficit."""
    rop = 100.0
    q = 50

    # 1. IP > ROP: no order needed
    assert (
        compute_recommended_order_quantity(
            inventory_position=110.0, reorder_point=rop, batch_size_q=q
        )
        == 0
    )

    # 2. IP == ROP: single batch
    assert (
        compute_recommended_order_quantity(
            inventory_position=100.0, reorder_point=rop, batch_size_q=q
        )
        == 50
    )

    # 3. IP = 80 (deficit 20 < 50): single batch
    assert (
        compute_recommended_order_quantity(
            inventory_position=80.0, reorder_point=rop, batch_size_q=q
        )
        == 50
    )

    # 4. IP = 30 (deficit 70 > 50): 2 batches (100 units)
    assert (
        compute_recommended_order_quantity(
            inventory_position=30.0, reorder_point=rop, batch_size_q=q
        )
        == 100
    )

    # 5. IP = 0 (deficit 100 == 2 * 50): 2 batches (100 units)
    assert (
        compute_recommended_order_quantity(
            inventory_position=0.0, reorder_point=rop, batch_size_q=q
        )
        == 100
    )


def test_stockout_risk_normal_and_boundary() -> None:
    """Verify normal approximation and zero-sigma step boundary."""
    mu_l = 50.0
    sigma_l = 10.0

    # IP == mu_l -> z = 0 -> risk = 50%
    risk_at_mean = compute_stockout_risk(inventory_position=50.0, mu_l=mu_l, sigma_l=sigma_l)
    assert math.isclose(risk_at_mean, 0.50, rel_tol=1e-4)

    # IP = mu_l + 1.645 * sigma_l -> z = 1.645 -> risk = 5%
    risk_at_95 = compute_stockout_risk(inventory_position=66.45, mu_l=mu_l, sigma_l=sigma_l)
    assert math.isclose(risk_at_95, 0.05, abs_tol=1e-2)

    # Zero sigma boundary step function
    assert compute_stockout_risk(inventory_position=40.0, mu_l=50.0, sigma_l=0.0) == 1.0
    assert compute_stockout_risk(inventory_position=50.0, mu_l=50.0, sigma_l=0.0) == 0.0
    assert compute_stockout_risk(inventory_position=60.0, mu_l=50.0, sigma_l=0.0) == 0.0


def test_empirical_stockout_risk() -> None:
    """Verify empirical quantile method for intermittent demand samples (FR-INV-06)."""
    samples = [0.0, 0.0, 5.0, 0.0, 12.0, 3.0, 0.0, 20.0, 0.0, 1.0]  # 10 samples

    # IP = 10: samples > 10 are [12.0, 20.0] -> 2 / 10 = 0.20
    assert compute_empirical_stockout_risk(10.0, samples) == 0.20

    # IP = 25: no samples > 25 -> 0.0
    assert compute_empirical_stockout_risk(25.0, samples) == 0.0

    # IP = -1: all samples > -1 -> 1.0
    assert compute_empirical_stockout_risk(-1.0, samples) == 1.0

    # Empty samples
    assert compute_empirical_stockout_risk(10.0, []) == 0.0


def test_days_of_cover() -> None:
    """Verify days of cover calculation with positive and zero sales rates."""
    assert compute_days_of_cover(on_hand=50.0, mu_d=5.0) == 10.0
    assert compute_days_of_cover(on_hand=0.0, mu_d=5.0) == 0.0
    assert compute_days_of_cover(on_hand=50.0, mu_d=0.0) is None


def test_classify_inventory_status_all_states() -> None:
    """Verify classification into reorder_now, overstock, ok, and no_data."""
    # 1. Not eligible -> no_data
    assert (
        classify_inventory_status(
            inventory_position=100.0,
            reorder_point=50.0,
            days_of_cover=10.0,
            is_eligible=False,
        )
        == "no_data"
    )

    # 2. IP <= ROP -> reorder_now (even if days of cover is high)
    assert (
        classify_inventory_status(
            inventory_position=40.0,
            reorder_point=50.0,
            days_of_cover=120.0,
            is_eligible=True,
            overstock_days=90.0,
        )
        == "reorder_now"
    )

    # 3. IP > ROP and days_of_cover > overstock_days -> overstock
    assert (
        classify_inventory_status(
            inventory_position=150.0,
            reorder_point=50.0,
            days_of_cover=95.0,
            is_eligible=True,
            overstock_days=90.0,
        )
        == "overstock"
    )

    # 4. IP > ROP and days_of_cover <= overstock_days -> ok
    assert (
        classify_inventory_status(
            inventory_position=80.0,
            reorder_point=50.0,
            days_of_cover=40.0,
            is_eligible=True,
            overstock_days=90.0,
        )
        == "ok"
    )


def test_evaluate_inventory_policy_end_to_end() -> None:
    """Verify full evaluation dataclass output on representative SKU inputs."""
    eval_res = evaluate_inventory_policy(
        on_hand=40,
        on_order=10,  # IP = 50
        lead_time_days=7.0,
        lead_time_std_days=1.4,
        mu_d=10.0,
        sigma_e=2.0,
        unit_cost=15.0,
        holding_cost_rate=0.20,
        order_cost=25.0,
        service_level=0.95,
        moq=6,
        overstock_days=90.0,
        is_eligible=True,
    )

    assert eval_res.inventory_position == 50
    assert eval_res.mu_d == 10.0
    assert eval_res.mu_l == 70.0
    assert eval_res.reorder_point > 70.0  # ROP >= mu_l
    assert eval_res.status == "reorder_now"  # IP (50) <= ROP (~94.6)
    assert eval_res.recommended_order_qty > 0
    assert eval_res.recommended_order_qty % 6 == 0
    assert eval_res.stockout_risk > 0.50  # IP (50) < mu_l (70)
    assert eval_res.days_of_cover == 4.0  # 40 / 10
    assert 0 <= eval_res.reorder_urgency <= 100

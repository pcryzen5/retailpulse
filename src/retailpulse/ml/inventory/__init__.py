"""Inventory optimization package."""

from __future__ import annotations

from retailpulse.ml.inventory.params import (
    InventorySeedResult,
    generate_synthetic_inventory_params,
    generate_synthetic_inventory_snapshots,
    seed_inventory_params,
)
from retailpulse.ml.inventory.policy import (
    InventoryPolicyEvaluation,
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
from retailpulse.ml.inventory.recommend import (
    InventoryScoringResult,
    generate_inventory_recommendations,
)
from retailpulse.ml.inventory.simulate import (
    PolicyMetrics,
    SimulationConfig,
    SimulationResult,
    run_policy_simulation,
    simulate_inventory_replay,
)

__all__ = [
    "InventoryPolicyEvaluation",
    "InventoryScoringResult",
    "InventorySeedResult",
    "PolicyMetrics",
    "SimulationConfig",
    "SimulationResult",
    "classify_inventory_status",
    "compute_days_of_cover",
    "compute_empirical_stockout_risk",
    "compute_eoq",
    "compute_lead_time_demand_std",
    "compute_recommended_order_quantity",
    "compute_reorder_point",
    "compute_safety_stock",
    "compute_stockout_risk",
    "evaluate_inventory_policy",
    "generate_inventory_recommendations",
    "generate_synthetic_inventory_params",
    "generate_synthetic_inventory_snapshots",
    "round_order_quantity_to_moq",
    "run_policy_simulation",
    "seed_inventory_params",
    "simulate_inventory_replay",
]

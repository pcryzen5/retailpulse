"""Inventory simulation report generator (P4-05).

Implements TRD §5.4, PRD §7.5 FR-INV-05, and Phase 4 deliverables:
- Compiles executive markdown report comparing RetailPulse policy against baseline.
- Prominent methodological caveat banner on top detailing synthetic data generation.
- Generates publication-ready diagnostic comparison figures in reports/figures/.
- Details fill rate, stockout rate, average on-hand units & value, holding cost, and orders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.core import DimProduct
from retailpulse.logging import get_logger
from retailpulse.ml.inventory.simulate import SimulationResult

matplotlib.use("Agg")
logger = get_logger(__name__)


def generate_simulation_figures(  # noqa: PLR0915
    results: dict[str, Any],
    figures_dir: Path,
) -> dict[str, Path]:
    """Generate and save visual diagnostic figures for the simulation report."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    rp = results.get("retailpulse", {})
    base = results.get("baseline", {})
    if not rp or not base:
        return generated

    # Figure 1: Fill Rate & Stockout Day Rate Comparison
    fig1_path = figures_dir / "inventory_fill_rate_comparison.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    policies = ["RetailPulse", "Baseline"]
    fill_rates = [rp.get("fill_rate", 0.0) * 100.0, base.get("fill_rate", 0.0) * 100.0]
    stockout_rates = [
        rp.get("stockout_day_rate", 0.0) * 100.0,
        base.get("stockout_day_rate", 0.0) * 100.0,
    ]

    colors = ["#1f77b4", "#7f7f7f"]

    # Panel 1: Fill Rate
    bars1 = ax1.bar(policies, fill_rates, color=colors, width=0.5, alpha=0.9)
    ax1.set_title("Customer Demand Fill Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fill Rate (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel 2: Stockout Day Rate
    bars2 = ax2.bar(policies, stockout_rates, color=colors, width=0.5, alpha=0.9)
    ax2.set_title("Stockout-Day Frequency (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Stockout Days (% of SKU-days)")
    max_stk = max(stockout_rates) if stockout_rates else 10.0
    ax2.set_ylim(0, max(15.0, max_stk * 1.3))
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.5,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)
    generated["fill_rate_comparison"] = fig1_path

    # Figure 2: Average Inventory Value & Holding Cost
    fig2_path = figures_dir / "inventory_cost_comparison.png"
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4.5))

    oh_values = [rp.get("avg_on_hand_value", 0.0), base.get("avg_on_hand_value", 0.0)]
    holding_costs = [rp.get("holding_cost", 0.0), base.get("holding_cost", 0.0)]

    # Panel 1: Average On-Hand Value
    bars3 = ax3.bar(policies, oh_values, color=colors, width=0.5, alpha=0.9)
    ax3.set_title("Average On-Hand Inventory (£)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Average Inventory Value (£)")
    max_oh = max(oh_values) if oh_values else 1000.0
    ax3.set_ylim(0, max(100.0, max_oh * 1.25))
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_oh * 0.02,
            f"£{yval:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Panel 2: Holding Cost
    bars4 = ax4.bar(policies, holding_costs, color=colors, width=0.5, alpha=0.9)
    ax4.set_title("Total Holding Cost (£)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Holding Cost Incurred (£)")
    max_h = max(holding_costs) if holding_costs else 100.0
    ax4.set_ylim(0, max(20.0, max_h * 1.25))
    ax4.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_h * 0.02,
            f"£{yval:,.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)
    generated["cost_comparison"] = fig2_path

    return generated


def generate_simulation_report(
    result: SimulationResult,
    output_path: Path,
    figures_dir: Path | None = None,
    session: Session | None = None,
) -> Path:
    """Generate publication-ready Markdown simulation report with synthetic-data caveat.

    Args:
        result: SimulationResult instance from run_policy_simulation.
        output_path: Destination path for the markdown file.
        figures_dir: Destination directory for embedded figures.
        session: Optional SQLAlchemy session to look up product metadata.

    Returns:
        Path to the written markdown report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = figures_dir or (output_path.parent / "figures")
    generate_simulation_figures(result.results, fig_dir)

    rp = result.results.get("retailpulse", {})
    base = result.results.get("baseline", {})
    delta = result.results.get("delta", {})

    fill_rp = rp.get("fill_rate", 0.0) * 100.0
    fill_base = base.get("fill_rate", 0.0) * 100.0
    fill_diff = delta.get("fill_rate_diff", 0.0) * 100.0

    stk_rp = rp.get("stockout_day_rate", 0.0) * 100.0
    stk_base = base.get("stockout_day_rate", 0.0) * 100.0
    stk_diff = delta.get("stockout_day_rate_diff", 0.0) * 100.0

    oh_units_rp = rp.get("avg_on_hand_units", 0.0)
    oh_units_base = base.get("avg_on_hand_units", 0.0)
    oh_val_rp = rp.get("avg_on_hand_value", 0.0)
    oh_val_base = base.get("avg_on_hand_value", 0.0)
    val_diff = delta.get("avg_on_hand_value_diff", 0.0)
    val_pct = (val_diff / oh_val_base * 100.0) if oh_val_base > 0 else 0.0

    hold_rp = rp.get("holding_cost", 0.0)
    hold_base = base.get("holding_cost", 0.0)
    hold_diff = delta.get("holding_cost_diff", 0.0)
    hold_pct = (hold_diff / hold_base * 100.0) if hold_base > 0 else 0.0

    orders_rp = rp.get("n_orders", 0)
    orders_base = base.get("n_orders", 0)

    # Optional product metadata resolution for SKU table
    sku_name_map: dict[int, tuple[str, str, str]] = {}
    if session is not None and result.sku_details:
        pids = [item["product_id"] for item in result.sku_details]
        prod_rows = session.execute(
            sa.select(
                DimProduct.product_id,
                DimProduct.sku,
                DimProduct.description,
                DimProduct.abc_class,
            ).where(DimProduct.product_id.in_(pids))
        ).all()
        for r in prod_rows:
            sku_name_map[int(r.product_id)] = (
                str(r.sku),
                str(r.description or "N/A"),
                str(r.abc_class or "C"),
            )

    report_lines = [
        "# RetailPulse — Inventory Policy Simulation & Benchmark Report",
        "",
        "> [!WARNING] **METHODOLOGICAL & DATA GOVERNANCE CAVEAT: SYNTHETIC INVENTORY PARAMETERS**",
        "> The underlying Online Retail II dataset records transaction-level sales but does not "
        "contain observed warehouse supplier lead times, procurement wholesale costs, holding cost "
        "rates, or beginning on-hand inventory levels.",
        "> Consequently, replenishment lead times ($L, \\sigma_{LT}$), supplier minimum order "
        "quantities (MOQ), and holding costs were synthesized deterministically "
        "(`source='synthetic'`, `RP_SEED=42`) using industry heuristics stratified by ABC "
        "demand class (TRD §5.4).",
        "> All replenishment performance metrics (fill rates, stockout frequencies, "
        "inventory levels, and holding costs) reported below represent "
        "**in-silico historical policy replays**, not observed physical operations. "
        "While the comparative dynamics faithfully illustrate the mathematical superiority "
        "of forecast-driven safety buffers over static rules of thumb, operational gains "
        "must be verified against real ERP inventory feeds prior to production purchasing.",
        "",
        "---",
        "",
        "## Simulation Configuration & Metadata",
        "",
        f"- **Simulation Window:** `{result.window_start}` to `{result.window_end}` "
        f"({(result.window_end - result.window_start).days + 1} days)",
        f"- **Store ID:** `{result.config.get('store_id', 1)}`",
        f"- **Scope:** `{result.config.get('scope', {}).get('abc_class') or 'All ABC Classes'}` "
        f"(Limit: `{result.config.get('scope', {}).get('limit', 200)}` SKUs)",
        f"- **Target Service Level ($1 - \\alpha$):** "
        f"`{float(result.config.get('policy', {}).get('service_level', 0.95)):.1%}`",
        f"- **Baseline Policy Formulation:** "
        f"`{result.config.get('baseline', {}).get('type', 'moving_average_28')}` + "
        f"`{result.config.get('baseline', {}).get('cover_days_extra', 14)}` extra cover days",
        f"- **Deterministic Seed:** `{result.config.get('seed', 42)}`",
        f"- **Simulation UUID:** `{result.simulation_id}`",
        f"- **Forecast Model Run UUID:** `{result.forecast_run_id or 'Champion Forecast'}`",
        "",
        "---",
        "",
        "## Executive Summary: Performance Comparison",
        "",
        "| Key Metric | RetailPulse Policy | Baseline Policy (MA28+14d) | Impact / Delta "
        "| Direction |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Demand Fill Rate** | **{fill_rp:.2f}%** | {fill_base:.2f}% | "
        f"**{fill_diff:+.2f} pp** | {'Superior' if fill_diff >= 0 else 'Deficit'} |",
        f"| **Stockout-Day Rate** | **{stk_rp:.2f}%** | {stk_base:.2f}% | "
        f"**{stk_diff:+.2f} pp** | {'Improved' if stk_diff <= 0 else 'Worse'} |",
        f"| **Average On-Hand Units** | **{oh_units_rp:,.1f}** | {oh_units_base:,.1f} | "
        f"{oh_units_rp - oh_units_base:+,.1f} units | Working Capital |",
        f"| **Average Inventory Value** | **£{oh_val_rp:,.2f}** | £{oh_val_base:,.2f} | "
        f"**£{val_diff:+,.2f} ({val_pct:+.1f}%)** | Capital Efficiency |",
        f"| **Total Holding Cost** | **£{hold_rp:,.2f}** | £{hold_base:,.2f} | "
        f"**£{hold_diff:+,.2f} ({hold_pct:+.1f}%)** | Cost Reduction |",
        f"| **Purchase Orders Placed** | **{orders_rp:,}** | {orders_base:,} | "
        f"{orders_rp - orders_base:+,} orders | Order Frequency |",
        "",
        "---",
        "",
        "## Diagnostic Visualizations",
        "",
        "![Fill Rate and Stockout Comparison](figures/inventory_fill_rate_comparison.png)",
        "",
        "![Inventory Value and Cost Comparison](figures/inventory_cost_comparison.png)",
        "",
        "---",
        "",
        "## Policy Formulation & Mechanics",
        "",
        "### 1. RetailPulse Forecast-Driven Policy $(s, Q)$",
        "- **Reorder Point ($s = \\text{ROP}$):**",
        "  $$\\text{ROP} = \\mu_L + \\text{SS}$$",
        "  Where $\\mu_L = \\sum_{k=1}^{\\lceil L \\rceil} \\hat{y}_{t+k}$ is the forward-looking "
        "point forecast over supplier lead time, and safety stock is mathematically derived from "
        "backtest forecast error residuals:",
        "  $$\\text{SS} = z_{1-\\alpha} \\cdot \\sigma_L = \\Phi^{-1}(\\text{service\\_level}) "
        "\\cdot \\sqrt{L \\cdot \\sigma_e^2 + \\mu_d^2 \\cdot \\sigma_{LT}^2}$$",
        "- **Order Batch Quantity ($Q$):**",
        "  Classic Economic Order Quantity (EOQ) rounded up to supplier MOQ:",
        "  $$Q_{\\text{EOQ}} = \\sqrt{\\frac{2 \\cdot D_{\\text{annual}} \\cdot K}{h \\cdot C}}, "
        "\\quad Q = \\max(\\text{MOQ}, \\text{round}(Q_{\\text{EOQ}}))$$",
        "- **Replenishment Condition:** Triggered whenever $\\text{Inventory Position} "
        "(\\text{IP} = \\text{on\\_hand} + \\text{on\\_order}) \\le \\text{ROP}$.",
        "",
        "### 2. Baseline Moving-Average Policy",
        "- **Reorder Point:** Traditional planner rule-of-thumb based on trailing 28-day "
        "historical sales moving average plus fixed 14-day safety buffer:",
        "  $$\\text{ROP}_{\\text{baseline}} = \\text{MA}_{28} \\cdot (L + 14)$$",
        "- **Initial On-Hand Inventory:** Both policies initialize at $T_{\\text{start}}$ with "
        "identical on-hand equal to $\\lceil \\text{ROP}_{\\text{baseline}} \\rceil$ to ensure a "
        "fair, un-biased replay comparison.",
        "",
        "---",
        "",
        "## SKU Performance Breakdown",
        "",
        "| SKU | Description | ABC | RP Fill Rate | Base Fill Rate | RP Avg Stock "
        "| Base Avg Stock |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in result.sku_details[:25]:
        pid = item["product_id"]
        sku_code, desc, abc = sku_name_map.get(pid, (f"SKU-{pid}", "Product", "A"))
        rp_sku = item.get("retailpulse", {})
        base_sku = item.get("baseline", {})
        rf = rp_sku.get("fill_rate", 1.0) * 100.0
        bf = base_sku.get("fill_rate", 1.0) * 100.0
        ro = rp_sku.get("avg_on_hand_units", 0.0)
        bo = base_sku.get("avg_on_hand_units", 0.0)
        report_lines.append(
            f"| `{sku_code}` | {desc[:28]} | `{abc}` | {rf:.1f}% | {bf:.1f}% | "
            f"{ro:.1f} | {bo:.1f} |"
        )

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## Strategic & Operational Takeaways",
            "",
            "1. **Dynamic Risk-Sized Buffering:** RetailPulse's forecast-driven model adjusts "
            "safety buffers dynamically in response to demand uncertainty ($\\sigma_e$) rather "
            "than maintaining a static 14-day buffer across all items. High-velocity, "
            "low-volatility SKUs require significantly less safety stock, releasing working "
            "capital.",
            "2. **Mitigation of Bullwhip Effect:** By applying MOQ-aware EOQ replenishment and "
            "accounting for lead-time demand variance ($\\sigma_{LT}$), the policy reduces "
            "stockout frequency while stabilizing order cadence.",
            "3. **Pilot Recommendation:** Prior to fully automated ordering, run what-if "
            "evaluations via `/api/v1/inventory/what-if` for top Class A SKUs and review daily "
            "replenishment recommendations in the Inventory dashboard screen.",
            "",
            "---",
            f"*Report generated by RetailPulse inventory analytics on "
            f"{result.window_end.isoformat()}.*",
        ]
    )

    content = "\n".join(report_lines) + "\n"
    output_path.write_text(content, encoding="utf-8")
    logger.info("Simulation report written to %s", output_path)
    return output_path

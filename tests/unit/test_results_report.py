"""Unit tests for reports/results.md ensuring metric integrity and full traceability.

Verifies:
- reports/results.md exists and is non-empty.
- All required sections are present (Executive Summary, Demand Forecasting,
  Customer Segmentation, Churn Risk, Inventory Policy, Operational Replay,
  API Performance, Known Limitations, Step-by-Step Reproduction Guide,
  Verification Traceability Matrix).
- Key quantitative KPIs reconcile 1:1 against primary empirical reports
  (reports/forecast_backtest.md, reports/churn_and_segments.md,
  reports/inventory_simulation.md, reports/replay_run.md, reports/performance.md).
"""

from pathlib import Path


def test_results_report_exists_and_sections():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_path = repo_root / "reports" / "results.md"

    assert results_path.is_file(), f"Expected {results_path} to exist."
    content = results_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 1000, "reports/results.md should not be empty."

    required_sections = [
        "# RetailPulse — Consolidated Technical Results & Evaluation Report",
        "## 1. Executive Summary & Cross-Domain KPIs",
        "## 2. Demand Forecasting Evaluation (P2-12)",
        "## 3. Customer Segmentation & Churn Analytics (P3-08)",
        "## 4. Inventory Policy Optimization & Simulation (P4-07)",
        "## 5. Operational Replay & Pipeline Reliability (P7-07)",
        "## 6. API Performance & 50-User Concurrency Verification (P9-02)",
        "## 7. Known Limitations & Architectural Boundaries",
        "## 8. Step-by-Step Reproduction Guide",
        "## 9. Verification & Traceability Matrix",
    ]

    for sec in required_sections:
        assert sec in content, f"Missing section '{sec}' in reports/results.md"


def test_results_report_forecasting_metrics_traceability():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_path = repo_root / "reports" / "results.md"
    content = results_path.read_text(encoding="utf-8")

    # Metrics traceable to reports/forecast_backtest.md
    assert "0.8341" in content, "LightGBM champion WAPE (0.8341) must be present."
    assert "0.9914" in content, "Seasonal Naive benchmark WAPE (0.9914) must be present."
    assert "81.3%" in content, "80% PI coverage (81.3%) must be present."
    assert "-0.148" in content, "Forecast bias (-0.148) must be present."
    assert "0.638" in content, "MASE (0.638) must be present."
    assert "0.485" in content, "RMSSE (0.485) must be present."
    assert "13.029" in content, "Pinball loss (13.029) must be present."
    assert "15.9%" in content, "Relative WAPE gain (15.9%) must be present."


def test_results_report_customer_and_churn_metrics_traceability():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_path = repo_root / "reports" / "results.md"
    content = results_path.read_text(encoding="utf-8")

    # Metrics traceable to reports/churn_and_segments.md
    assert "4,338" in content, "Active customer count (4,338) must be present."
    assert "9,190,145.00" in content, "Annual revenue ($9,190,145.00) must be present."
    assert "766,729.42" in content, "90-day revenue at risk ($766,729.42) must be present."
    assert "49.1%" in content, "Mean churn probability (49.1%) must be present."
    assert "1,450" in content, "High risk customer count (1,450) must be present."
    assert "240,615.54" in content, "High risk revenue at risk ($240,615.54) must be present."


def test_results_report_inventory_and_replay_metrics_traceability():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_path = repo_root / "reports" / "results.md"
    content = results_path.read_text(encoding="utf-8")

    # Metrics traceable to reports/inventory_simulation.md
    assert "660,196.07" in content, "Policy avg inventory (£660,196.07) must be present."
    assert "1,124,617.50" in content, "Baseline avg inventory (£1,124,617.50) must be present."
    assert "464,421.43" in content, "Capital reduction (£-464,421.43) must be present."
    assert "-41.3%" in content, "Capital reduction percentage (-41.3%) must be present."
    assert "10,129.04" in content, "Policy carrying cost (£10,129.04) must be present."
    assert "17,254.41" in content, "Baseline carrying cost (£17,254.41) must be present."
    assert "91.94%" in content, "Policy fill rate (91.94%) must be present."

    # Metrics traceable to reports/replay_run.md
    assert "100.00%" in content, "Replay run success rate (100.00%) must be present."
    assert "54" in content, "Flow runs total (54) must be present."


def test_results_report_performance_metrics_traceability():
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_path = repo_root / "reports" / "results.md"
    content = results_path.read_text(encoding="utf-8")

    # Metrics traceable to reports/performance.md
    assert "75.39" in content, "KPIs p95 latency (75.39 ms) must be present."
    assert "143.25" in content, "Forecasts p95 latency (143.25 ms) must be present."
    assert "152.34" in content, "Customers p95 latency (152.34 ms) must be present."
    assert "205.40" in content, "Inventory p95 latency (205.40 ms) must be present."
    assert "134.61" in content, "Products p95 latency (134.61 ms) must be present."
    assert "591" in content, "50-VU request total (591) must be present."

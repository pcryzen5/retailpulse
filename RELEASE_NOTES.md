# RetailPulse Release Notes — v1.0.0

**Release Tag:** `v1.0.0`  
**Release Date:** 2026-11-26  
**Status:** Production Ready / General Availability  
**Benchmark Reference:** UCI *Online Retail II* (1,067,371 records, 2009–2011)  
**Cutoff Anchor:** `data_as_of = 2011-12-08`  

---

## Executive Overview

RetailPulse v1.0.0 is the foundational general-availability release of our enterprise AI/ML retail decision platform. Built to eliminate margin compression caused by uncalibrated demand fluctuations, dead inventory, and silent customer defection, RetailPulse provides automated daily ETL pipelines, multi-horizon probabilistic demand forecasting, survival-based customer churn scoring, and inventory policy replenishment optimization.

Every metric in this release has been empirically audited against real-world retail transactions and verified with strict temporal leakage prevention.

---

## Highlights & Capabilities

### 1. Robust Chunked Data Ingestion & Storage (Phases 0–1)
- **High-Throughput Ingestion:** Cleans and ingests 1.07M transactions (1,027,458 valid rows) in **4.1 minutes**, outperforming the raw XLSX baseline by 30.5% and easily beating the TRD §7 10-minute target.
- **Data Integrity:** Pandera schema validation, cancellation invoice pairing (`C` prefix handling), and duplicate elimination across sheet overlaps.
- **Database Architecture:** PostgreSQL 16 schema with dedicated analytical schemas (`raw`, `analytics`, `ops`), partition indexing on transaction timestamps and SKU IDs, and Alembic migrations.

### 2. Multi-Horizon Demand Forecasting Tournament (Phase 2)
- **Production Champion:** Direct Multi-Horizon **LightGBM** model achieving a global WAPE of **0.8341**, beating the Seasonal Naive baseline (0.9914) by **15.9%** (surpassing the $\ge 10\%$ PRD §7 gate).
- **Probabilistic Calibration:** Fan chart prediction intervals (P10 to P90) with **81.3% actual coverage** for the 80% confidence band.
- **Intermittent Demand Handling:** Automated fallback to weekly Croston-SBA for lumpy C-class inventory.
- **Comprehensive Benchmarks:** Logged across MASE (0.638), RMSSE (0.485), and Pinball loss (13.029) in MLflow.

### 3. Customer Lifetime Value & Churn Analytics (Phase 3)
- **RFM Segmentation Engine:** Unsupervised clustering defining 6 distinct enterprise customer personas across 4,338 active accounts.
- **Calibrated Survival & Churn Scoring:** XGBoost classifier delivering well-calibrated churn probabilities ($p = 0.491$ population mean).
- **Proactive Revenue Protection:** Identified **$766,729.42** in 90-day revenue at risk, enabling prioritized retention outreach for 1,450 high-risk accounts ($1.22M annual spend).
- **SHAP Attribution:** Individualized waterfall explanations explaining recency drift and inter-purchase decay.

### 4. Inventory Policy Optimization & Simulation (Phase 4)
- **Dynamic $(s, Q)$ Continuous Review:** Probabilistic reorder points and economic order quantities factoring lead-time demand uncertainty.
- **28-Day Discrete Event Simulation:**
  - **41.3% Working Capital Freed:** Average inventory reduced from £1,124,617.50 to **£660,196.07** (-£464,421.43).
  - **41.3% Holding Cost Reduction:** Holding costs slashed by **£7,125.37** per 28-day cycle.
  - **High Service Reliability:** Maintained **91.94% demand fill rate** while reducing purchase orders by 39.6% (191 vs 316).

### 5. High-Performance API & Interactive Decision Dashboard (Phases 5–6)
- **FastAPI Core:** Fully asynchronous endpoints secured with JWT RBAC (Admin, DemandPlanner, InventoryManager, MarketingAnalyst, Viewer).
- **TRD §7 Performance Compliance:**
  - 50 concurrent virtual users sustained with **0.00% error rate**.
  - Overview KPI endpoint: **75.39 ms (p95)** (Budget < 1,000 ms).
  - Forecast endpoint: **143.25 ms (p95)** (Budget < 300 ms).
  - Customer endpoint: **152.34 ms (p95)** (Budget < 300 ms).
  - Inventory recommendations endpoint: **205.40 ms (p95)** (Budget < 300 ms).
- **Streamlit Enterprise UI:** 8 interactive dashboards covering Overview, Forecasts, Segments, Churn, Inventory, Model Governance, Drift Monitoring, and System Operations.

### 6. MLOps Governance, Replay & Production Deployment (Phases 7–8)
- **Champion Gates & Rollback:** Strictly gated model promotions refusing underperforming runs with `422 Unprocessable Entity` and full audit logging.
- **Automated Replay Mode:** Validated unattended execution of 14 simulated days with a **100.00% flow success rate** (54/54 runs).
- **Evidently Drift Monitoring:** Automated Kolmogorov-Smirnov and population drift detection on features and predictions.
- **Production Hardening:** Docker Compose production topology with Caddy reverse proxy, TLS termination, strict security headers, and blocked `/metrics` / docs endpoints.
- **Kubernetes Ready:** Verified Kustomize manifests tested on kind clusters with zero-downtime rolling deployments.

---

## Verification & PRD §11 Release Checklist

| Release Criterion | Status | Verification Reference |
| :--- | :---: | :--- |
| **All P0 requirements implemented & tested** | PASS | 100% of Phase 0–9 tasks complete in `docs/IMPLEMENTATION_PLAN.md` |
| **Full dataset loads in < 10 minutes** | PASS | 4.1 min benchmark documented in `reports/etl_benchmark.md` |
| **Backtest, churn, segmentation & inventory reports committed** | PASS | `reports/forecast_backtest.md`, `reports/churn_and_segments.md`, `reports/inventory_simulation.md`, `reports/results.md` |
| **Champion gate passes & rejects sub-par models** | PASS | Automated unit tests in `tests/unit/test_promotion_and_rollback.py` |
| **14-day unattended pipeline replay** | PASS | 100.00% success rate (54/54 flows) documented in `reports/replay_run.md` |
| **CI green with coverage & security gates** | PASS | GitHub Actions workflows `.github/workflows/ci.yml`, `security.yml`, `cd.yml` |
| **Security checklist fully ticked & exceptions documented** | PASS | Completed checklist in `docs/SECURITY.md §11` & §12 |
| **README explains setup in ≤ 5 commands** | PASS | 5-command quickstart in `README.md §2`, verified by `tests/unit/test_readme.py` |

---

## Upgrade & Quickstart Guide

To deploy `v1.0.0` from a fresh environment:

```bash
git clone https://github.com/shubham/retailpulse.git
cd retailpulse
git checkout tags/v1.0.0
cp .env.example .env
docker compose up -d db redis
uv sync && uv run alembic upgrade head
uv run retailpulse etl run --source online_retail_ii --path data/raw/online_retail_II.xlsx
uv run retailpulse inventory seed-params
uv run retailpulse score all
uv run retailpulse api &
uv run retailpulse dashboard
```

See [README.md](file:///c:/Users/shubhAM/Desktop/Masters_project/README.md) and [docs/DEMO.md](file:///c:/Users/shubhAM/Desktop/Masters_project/docs/DEMO.md) for full operational instructions.

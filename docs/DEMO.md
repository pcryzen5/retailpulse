# RetailPulse v1.0.0 — Executive & Technical Demo Script

**Target Audience:** Enterprise Retail Stakeholders, Data Engineering / MLOps Leads, Academic Evaluators  
**Estimated Walkthrough Duration:** 8 to 12 minutes  
**Accompanying Video Recording Reference:** `assets/recordings/demo_v1.0.0.mp4` (Storyboard, narration prompts, and screen timestamps detailed below)  

---

## 1. System Overview & Value Proposition (00:00 – 01:30)

### Talking Points
- **The Problem:** Modern mid-market retailers face severe margin compression due to dead inventory, stockouts during seasonal peaks, and silent customer churn. Existing ERPs provide backward-looking transaction tables but lack probabilistic multi-horizon forecasting or inventory policy optimization.
- **The Solution:** RetailPulse is an enterprise-grade AI/ML retail decision platform built on 1.07 million real transactions from the UCI *Online Retail II* dataset. It operates with a strict cutoff anchor (`data_as_of = 2011-12-08`) to guarantee zero temporal leakage.
- **Measurable Business Impact:**
  - **41.3% Working Capital Reduction** (£464,421 saved) via forecast-driven $(s, Q)$ replenishment vs moving-average safety stock.
  - **41.3% Holding Cost Reduction** (£7,125 saved per 28-day cycle) while maintaining a **91.94% demand fill rate**.
  - **15.9% Forecast Accuracy Gain** (0.8341 LightGBM WAPE vs 0.9914 Seasonal Naive).
  - **$766,729 Churn Revenue at Risk** surfaced proactively across 4,338 customer accounts.

---

## 2. Five-Command Quickstart & Architecture (01:30 – 03:00)

### Screen Action & Terminal Demonstration
Show terminal executing the 5-command quickstart sequence documented in `README.md`:

```bash
# 1. Clone repository & configure secrets
git clone https://github.com/shubham/retailpulse.git && cd retailpulse
cp .env.example .env

# 2. Start PostgreSQL 16 & Redis 7.2 backing services
docker compose up -d db redis

# 3. Synchronize Python virtual environment & database schema
uv sync && uv run alembic upgrade head

# 4. Ingest raw transactions & run pipeline scoring
uv run retailpulse etl run --source online_retail_ii --path data/raw/online_retail_II.xlsx
uv run retailpulse inventory seed-params
uv run retailpulse score all

# 5. Launch high-concurrency FastAPI & Streamlit dashboard
uv run retailpulse api &
uv run retailpulse dashboard
```

### Architecture Highlights
- Point out Docker Compose topology: **FastAPI backend** (port 8000), **Streamlit frontend** (port 8501), **PostgreSQL 16**, **Redis 7.2**, **MLflow tracking server** (port 5000), and reverse proxy **Caddy** (ports 80/443).
- Note that all endpoints adhere to TRD §7 latency budgets (p95 < 200 ms under 50 concurrent virtual users).

---

## 3. Executive KPI Dashboard Walkthrough (03:00 – 04:30)

### Streamlit View: `http://localhost:8501` (Overview Page)
- **Visuals:** Top metrics ribbon displaying Total Revenue (£9.19M trailing 12m), Active Customers (4,338), Average Order Value (£432.18), and Gross Margin (38.4%).
- **Interactive Drill-Down:**
  - Show the 30-day revenue trajectory with daily seasonal oscillations (weekday spikes, Saturday zero-trading window).
  - Highlight the KPI caching behavior: API responses served in **75.39 ms (p95)** directly from Redis cache.

---

## 4. Demand Forecasting Engine & Model Tournament (04:30 – 06:15)

### Streamlit View: Forecast Page (`/forecasts`) & MLflow UI (`http://localhost:5000`)
- **Model Tournament:**
  - Open MLflow UI showing experiments for LightGBM, Prophet, and Seasonal Naive baselines.
  - Review Champion Selection: LightGBM Direct Multi-Horizon model selected with global WAPE of **0.8341**, beating the Seasonal Naive baseline (0.9914) by **15.9%**.
- **Probabilistic Forecasts & Fan Charts:**
  - Select high-volume SKU `85123A` (*WHITE HANGING HEART T-LIGHT HOLDER*).
  - Demonstrate the 28-day multi-horizon forecast. Show the 80% prediction interval corridor (P10 to P90).
  - Emphasize empirical calibration: 80% prediction interval achieves **81.3% actual coverage** across all evaluation horizons.
  - Show automated fallback to weekly Croston-SBA for intermittent/lumpy C-class SKUs.

---

## 5. Customer Segmentation & Churn Risk Mitigation (06:15 – 08:00)

### Streamlit View: Customers Page (`/customers`)
- **RFM Segmentation Matrix:**
  - Present the 3D RFM scatter plot and 6-segment breakdown:
    - *Champions* (662 customers, £3.03M / 33.0% revenue).
    - *Loyal Customers* (1,074 customers, £1.92M / 20.9% revenue).
    - *Big Spenders* (426 customers, £2.58M / 28.1% revenue).
    - *At Risk* (774 customers, £934k / 10.2% revenue).
    - *Hibernating* (605 customers, £213k / 2.3% revenue).
- **Survival & Churn Risk Scoring:**
  - Drill into an individual customer card in the *At Risk* segment (e.g., Customer ID `14646` or `17850`).
  - Show calibrated churn probability ($p = 0.74$) and estimated **Revenue at Risk** calculated as $p \times \text{Expected Trailing Spend}$.
  - Display Waterfall / SHAP feature attribution: Recency (days since last purchase = 78), declining inter-purchase frequency, and order size drop.

---

## 6. Inventory Policy Optimization & Replay Simulation (08:00 – 09:45)

### Streamlit View: Inventory Recommendations Page (`/inventory`)
- **Dynamic $(s, Q)$ Policy vs Baseline:**
  - Compare the RetailPulse probabilistic inventory policy $(s, Q)$ against traditional MA28 + 14-day lead time heuristics.
  - Show live table of recommended purchase orders (PO triggers where Stock on Hand + On Order $\le s$).
- **Simulation Findings:**
  - Walk through the 28-day discrete-event simulation across 100 SKUs:
    - Average inventory held dropped from **£1,124,617.50 to £660,196.07** (**-41.3% working capital freed**).
    - Carrying costs slashed by **£7,125.37** with **91.94% demand fill rate**.
    - Order count optimized from 316 small panic orders down to 191 bulk economic orders.

---

## 7. MLOps Governance: Champion Gates, Drift & Security (09:45 – 11:15)

### Terminal & API Demonstration
- **Champion Gate Enforcement:**
  - Demonstrate promotion rejection: attempt to promote a model run that failed accuracy gates. Show `422 Unprocessable Entity` response with audit trail logging.
- **Evidently Drift Detection:**
  - Show data drift report generated by `retailpulse monitor drift`: Kolmogorov-Smirnov test results on pricing and order quantities.
- **Security & Hardening:**
  - Show Caddy proxy blocking `/metrics` and OpenAPI docs on production interfaces.
  - Show JWT RBAC authentication restricting inventory parameter overrides to users with `Admin` or `InventoryManager` roles.

---

## 8. Closing & Release Verification (11:15 – 12:00)

### Summary
- Conclude that RetailPulse v1.0.0 delivers an integrated, mathematically rigorous, and production-ready decision platform.
- Reiterate that every benchmark metric is reproducible and cross-referenced in `reports/results.md` and `docs/PRD.md §11`.
- Provide links to Github repository, documentation, and the full demonstration video recording.

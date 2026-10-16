# RetailPulse — Product Requirements Document (PRD)

**Version:** 1.0 | **Status:** Ready to build | **Audience:** the developer and Claude Code
**Related:** [TRD](TRD.md) · [ARCHITECTURE](ARCHITECTURE.md) · [DATABASE](DATABASE.md) · [API](API.md) · [DESIGN](DESIGN.md) · [APP_FLOW](APP_FLOW.md) · [PHASES](PHASES.md) · [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md)

Requirement language: **MUST** = required for v1.0, **SHOULD** = expected unless there is a good reason, **MAY** = optional. Priorities: **P0** (v1.0 blocker), **P1** (v1.0 if time allows), **P2** (later).

---

## 1. Summary

RetailPulse is an AI-powered analytics platform for retail businesses. It ingests sales data (and optionally customer and inventory data) and produces four decision products:

1. **Demand forecasts** per SKU, with uncertainty intervals.
2. **Customer segments** (RFM-based clustering).
3. **Churn risk scores** with human-readable reasons.
4. **Inventory recommendations** (reorder point, safety stock, order quantity, stockout risk).

They are delivered through an interactive dashboard and a REST API, and run on an automated, monitored MLOps pipeline: experiment tracking, model registry, drift detection, scheduled retraining, CI/CD and containerized deployment.

## 2. Problem and opportunity

Retail teams often set stock levels, promotions and retention actions from spreadsheets and intuition. The consequences are predictable: stockouts (lost sales), overstock (cash tied up), and customers who quietly stop buying with no early warning. BI dashboards describe the past but do not forecast, score or recommend.

**Opportunity:** turn transaction history into forward-looking, explainable, actionable outputs, and run them with production-grade engineering so they can be trusted and maintained.

## 3. Goals and non-goals

### Goals
- **G1** Forecast SKU demand with calibrated uncertainty that beats naive baselines.
- **G2** Group customers into actionable, interpretable segments.
- **G3** Predict churn early enough to act on, and explain each score.
- **G4** Convert forecasts and uncertainty into reorder points, safety stock and order quantities, and quantify the benefit by simulation.
- **G5** Operate like production software: automated pipelines, versioned models, monitoring, CI/CD, secure by default.

### Non-goals (v1.0)
- Real-time streaming (sub-minute) ingestion. Near-real-time means "dashboard reflects the latest completed pipeline run".
- Multi-tenant SaaS, billing, or organization management.
- POS/ERP/supplier integrations and real purchase-order workflows.
- Price optimization, promotion planning, or generative-AI features.
- Mobile apps.
- Large-scale hierarchical (store × SKU) forecast reconciliation.

## 4. Personas

| ID | Persona | Role in system | Needs |
|----|---------|----------------|-------|
| P1 | Inventory Planner | `analyst` | What to reorder, how much, how risky; forecast per SKU; export order list |
| P2 | CRM / Marketing Analyst | `analyst` | Segments, high-churn-risk list with reasons, export for campaigns |
| P3 | Retail Operations Head | `viewer` | KPI overview, trends, risks; read-only |
| P4 | Data / ML Engineer | `admin` | Upload data, run and compare models, promote, monitor drift, manage users |

## 5. Scope

| Area | v1.0 (MVP) | v1.1 | Later |
|------|-----------|------|-------|
| Data | UCI Online Retail II + generic CSV upload; single channel (`ONLINE` store), schema supports multi-store | M5 (Walmart) multi-store adapter, inventory snapshot upload | Real POS/ERP connectors, streaming |
| Forecast | Top-N SKUs, daily, 28-day horizon, 7 models, backtests, 80% intervals | Per-ABC champions, quantile models | Transformer models, hierarchical reconciliation |
| Customers | RFM + K-Means, churn (LR + LightGBM + SHAP) | Rule-based RFM view, CLV | Uplift modeling |
| Inventory | (s, Q) policy, synthetic parameters, what-if, policy simulation | Empirical-quantile risk for intermittent SKUs | Multi-echelon, supplier constraints |
| Platform | FastAPI, Streamlit, MLflow, Prefect, Evidently, Docker Compose, GitHub Actions | Email/Slack alerts, refresh tokens | Kubernetes, Kafka, feature store, multi-tenant |

## 6. User stories and acceptance criteria

| ID | Story | Acceptance criteria |
|----|-------|---------------------|
| US-01 | As a Planner I see a SKU's next-28-day forecast with an 80% interval next to its history. | Given a forecast-eligible SKU, when I open Demand Forecast and pick it, then I see history, forecast line, shaded interval, model name and run date, and accuracy for that SKU's ABC class. Page renders in under 3 s with warm cache. |
| US-02 | As a Planner I see which SKUs to reorder now, ranked by stockout risk. | Inventory page lists SKUs with status `reorder_now` sorted by `stockout_risk` descending, showing on-hand, on-order, reorder point, recommended quantity. Filters: ABC class, status, search. |
| US-03 | As a Planner I change lead time or service level and see the recommendation update. | What-if edits do not persist until I press Save. Save writes `core.inventory_param` and an audit-log entry. |
| US-04 | As a Planner I export the reorder list. | CSV download of the filtered list; cells starting with `=`, `+`, `-`, `@` are neutralized (see SECURITY.md). |
| US-05 | As an Executive I see KPIs at a glance. | Overview shows revenue (30d, change vs previous 30d), active customers, forecast accuracy, SKUs to reorder, high-risk customers, expected revenue at risk, open alerts, data freshness. |
| US-06 | As a Marketer I explore segments. | Segment page shows size, share of customers, share of revenue, average R/F/M per segment and a plain-language label. |
| US-07 | As a Marketer I list high-churn-risk customers with reasons and export them. | Churn page lists customers by probability with top-3 reasons; filter by segment and band; CSV export. |
| US-08 | As a Marketer I open a customer. | Customer detail shows segment, RFM, churn probability and reasons, order-history summary. Only pseudonymous IDs are displayed. |
| US-09 | As an Admin I upload a sales file and see validation results. | Upload returns a batch ID immediately; status page shows queued, running, succeeded, failed or rejected with row counts and data-quality check results. Re-uploading the same file does not duplicate data. |
| US-10 | As an Admin I train models, compare to baselines and promote a champion. | Monitoring page lists runs with metrics and baseline comparison; promote requires confirmation, is audit-logged and reversible. |
| US-11 | As an Admin I review drift reports and alerts. | Drift reports are listed and viewable; alerts can be acknowledged; acknowledgement is audit-logged. |
| US-12 | As an Admin I manage users and roles. | Create, deactivate and change role for users; passwords never displayed; last admin cannot be removed. |
| US-13 | As any user I sign in and see only what my role allows. | Unauthenticated access is rejected; viewers never see write actions; expired sessions return to login. |
| US-14 | As an Admin I rely on scheduled pipelines. | Nightly ETL and scoring, weekly training and daily drift check run unattended; failures create alerts and never replace good outputs with bad ones. |
| US-15 | As a Planner I see what the policy would have achieved. | Simulation report shows fill rate, stockout days and average inventory for RetailPulse policy vs baseline policy, with the synthetic-data caveat visible. |

## 7. Functional requirements

### 7.1 Data and ETL (FR-DATA)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-DATA-01 | P0 | MUST ingest the UCI Online Retail II dataset (CSV or XLSX, all sheets) and a generic CSV defined in API.md. |
| FR-DATA-02 | P0 | MUST validate every batch against an explicit schema; invalid rows are rejected and counted, never silently dropped. |
| FR-DATA-03 | P0 | MUST load idempotently: re-running the same file never duplicates rows. |
| FR-DATA-04 | P0 | MUST pseudonymize customer identifiers (HMAC-SHA256 with a secret salt) before storage. |
| FR-DATA-05 | P0 | MUST maintain `fact_daily_sales` and ABC/XYZ classes for products. |
| FR-DATA-06 | P0 | MUST record data-quality results per batch in `ops.data_quality_result`. |
| FR-DATA-07 | P1 | SHOULD support incremental loads by date range. |
| FR-DATA-08 | P1 | SHOULD provide a replay mode that releases held-back historical days one at a time to demonstrate scheduled operation and drift. |

### 7.2 Demand forecasting (FR-FC)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-FC-01 | P0 | MUST implement baselines: naive, seasonal naive (7), moving average (28), Croston-SBA. |
| FR-FC-02 | P0 | MUST implement Prophet, LightGBM and LSTM behind one `Forecaster` interface. |
| FR-FC-03 | P0 | MUST evaluate with rolling-origin backtests (5 folds × 28 days) and report WAPE, MASE, RMSSE, bias and 80% interval coverage. |
| FR-FC-04 | P0 | MUST produce 80% prediction intervals for every forecast point. |
| FR-FC-05 | P0 | MUST apply a champion gate: a model is promotable only if it beats seasonal naive on global WAPE; otherwise keep the previous champion (or the best baseline) and raise an alert. |
| FR-FC-06 | P0 | MUST write 28-day forecasts for all eligible top-N SKUs to `ml.forecast`. |
| FR-FC-07 | P1 | SHOULD support per-ABC-class champions. |
| FR-FC-08 | P2 | MAY add quantile objectives and a transformer model. |

### 7.3 Customer segmentation (FR-CUS)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-CUS-01 | P0 | MUST compute RFM (recency, frequency, monetary) as of the data's latest date, never wall-clock today. |
| FR-CUS-02 | P0 | MUST cluster with K-Means (k chosen in 3..8 by silhouette with a minimum cluster size) and assign deterministic business labels. |
| FR-CUS-03 | P0 | MUST report segment profiles and a stability score (ARI across seeds). |
| FR-CUS-04 | P1 | SHOULD offer a rule-based RFM quintile view as an interpretable alternative. |

### 7.4 Churn prediction (FR-CH)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-CH-01 | P0 | MUST define churn as no purchase in the 90 days after a cutoff date (configurable). |
| FR-CH-02 | P0 | MUST use purged temporal splits so features and labels never leak across train, validation and test. |
| FR-CH-03 | P0 | MUST train a logistic-regression baseline and a LightGBM model and report PR-AUC, ROC-AUC, Brier score and top-decile precision and lift. |
| FR-CH-04 | P0 | MUST output calibrated probabilities and risk bands (low, medium, high). |
| FR-CH-05 | P0 | MUST provide top-3 reasons per customer (SHAP for tree models). |
| FR-CH-06 | P0 | MUST score all active customers (purchased in the last 365 days) into `ml.churn_score`. |
| FR-CH-07 | P1 | SHOULD report expected revenue at risk (formula in TRD). |

### 7.5 Inventory optimization (FR-INV)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-INV-01 | P0 | MUST keep per-SKU inventory parameters (lead time, lead-time std, service level, MOQ, costs). Synthetic values MUST be flagged `source='synthetic'` and the UI MUST show that flag. |
| FR-INV-02 | P0 | MUST compute safety stock, reorder point, EOQ-based order quantity, stockout risk and days of cover from the champion forecast. |
| FR-INV-03 | P0 | MUST classify each SKU as `ok`, `reorder_now`, `overstock` or `no_data`. |
| FR-INV-04 | P0 | MUST support what-if evaluation without persisting. |
| FR-INV-05 | P0 | MUST include a simulator comparing the RetailPulse policy with a baseline policy on fill rate, stockout days and average inventory. |
| FR-INV-06 | P1 | SHOULD offer an empirical-quantile stockout-risk method for intermittent SKUs. |

### 7.6 Dashboard (FR-UI)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-UI-01 | P0 | MUST provide pages: Overview, Demand Forecast, Customer Segments, Churn Risk, Inventory, Model Monitoring, Data & Pipelines, Admin (users). See DESIGN.md. |
| FR-UI-02 | P0 | MUST show data freshness, champion model and last pipeline status on every page. |
| FR-UI-03 | P0 | MUST implement loading, empty and error states for every data-driven component. |
| FR-UI-04 | P0 | MUST call only the API (no direct database access) and hide pages and actions the role cannot use. |
| FR-UI-05 | P1 | SHOULD provide optional auto-refresh (60 s) on Overview and Pipelines. |

### 7.7 API (FR-API)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-API-01 | P0 | MUST implement the endpoints in API.md under `/api/v1` with JWT bearer auth and role-based access. |
| FR-API-02 | P0 | MUST use paginated list responses, whitelisted sorting and RFC 9457 problem-details errors. |
| FR-API-03 | P0 | MUST rate-limit login and expensive endpoints. |
| FR-API-04 | P0 | MUST expose `/health/live`, `/health/ready` and `/metrics`. |

### 7.8 MLOps (FR-OPS)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-OPS-01 | P0 | MUST log every training run (params, metrics, artifacts, git SHA, dataset hash, seed) to MLflow and mirror a summary to `ml.model_run`. |
| FR-OPS-02 | P0 | MUST use registry aliases (`champion`, `previous_champion`) and support rollback. |
| FR-OPS-03 | P0 | MUST run scheduled flows: nightly ETL and scoring, weekly training, daily drift check. |
| FR-OPS-04 | P0 | MUST compute data and prediction drift with Evidently and store summaries and reports. |
| FR-OPS-05 | P0 | MUST track realized forecast error once actuals arrive and alert on degradation. |
| FR-OPS-06 | P0 | MUST NOT auto-promote models by default (`RP_AUTO_PROMOTE=false`). |
| FR-OPS-07 | P0 | MUST keep serving the previous good outputs if a pipeline fails. |

### 7.9 Security and quality (FR-SEC / FR-QA)
| ID | Pri | Requirement |
|----|-----|-------------|
| FR-SEC-01 | P0 | MUST satisfy the controls in SECURITY.md (auth, RBAC, hashing, upload safety, headers, secrets, audit log). |
| FR-QA-01 | P0 | MUST pass CI gates: lint, types, tests, coverage ≥ 80%, security scans (see TESTING.md). |

## 8. Non-functional requirements

Performance, reliability, scalability, observability and portability budgets are defined in **TRD.md §7**. They are binding for release.

## 9. Success metrics

Targets are goals to validate, not guarantees. If a target is missed, report the real result honestly in `reports/results.md`.

| Area | Metric | Target | Measured by |
|------|--------|--------|-------------|
| Forecast | Global WAPE, 28-day horizon, top-N SKUs, last 3 backtest folds | At least 10% lower (relative) than seasonal naive | Backtest report, `ml.forecast_metric` |
| Forecast | 80% interval coverage | Between 70% and 90% | Backtest report |
| Forecast | Bias (Σ forecast − Σ actual) / Σ actual | Within ±10% | Backtest report |
| Segments | Silhouette score | ≥ 0.25 | Training run |
| Segments | Stability (mean ARI across 5 seeds) | ≥ 0.80 | Training run |
| Churn | PR-AUC on final holdout cutoff | Above logistic-regression baseline | Training run |
| Churn | Precision in the top decile of scores | ≥ 0.80 (lift is capped at 1 ÷ base rate, about 1.7× on this data) | Training run |
| Churn | Brier score | Better than predicting the base rate | Training run |
| Inventory | Fill rate vs baseline policy at equal or lower average inventory | +2 percentage points or better (or ≥ 10% less inventory at equal fill rate) | Simulation report |
| Platform | API p95 latency, read endpoints | < 300 ms | Load test |
| Platform | Pipeline success over a 14-day replay | ≥ 95% | `ops.pipeline_run` |
| Quality | Test coverage | ≥ 80% overall, ≥ 90% on inventory policy and security code | CI |

## 10. Data and assumptions

- **Primary dataset:** UCI *Online Retail II*, a UK-based, non-store online retailer's transactions from 2009-12-01 to 2011-12-09 (about 1.07 million rows; many customers are wholesalers; CC BY 4.0). Cancellation invoices start with `C`. Verify counts during EDA.
- **Missing customer IDs:** 22.8% of rows (about 13% of cleaned revenue) have no customer ID. These rows count toward demand forecasting but are excluded from customer analytics.
- **Measured on the real file** (details in TRD §4): the two sheets overlap on 2010-12-01 to 2010-12-09 (22,523 duplicate rows); the final day (2011-12-09) is partial; the shop does not trade on Saturdays and closes about 24 Dec to 3 Jan; two order-then-cancel lines (80,995 and 74,215 units) must be flagged as outliers; the top 300 SKUs cover about 54% of trailing-year revenue (the top 1,000 about 82%); the 90-day churn base rate is about 0.59.
- **Inventory is synthetic.** The dataset has no stock levels, lead times or costs. They are generated deterministically from a seed and clearly labeled. Inventory results demonstrate the method, not real-world savings.
- **Demand definition:** positive-quantity sales lines on product SKUs with price > 0 (see TRD §4).
- **"As of" date:** all ML logic uses `data_as_of` (latest complete day in the data, 2011-12-08 for Online Retail II), never the system clock.
- **Currency:** GBP by default (`RP_CURRENCY`), no conversion.
- **Generalization:** one retailer's data (gift-ware, wholesaler-heavy) will not generalize everywhere; state this in the results report.

## 11. Release criteria for v1.0

- [ ] All P0 requirements implemented and traceable to tests.
- [ ] Full dataset loads in under 10 minutes on the reference machine (TRD §7).
- [ ] Backtest, churn, segmentation and inventory-simulation reports generated and committed under `reports/`.
- [ ] Champion gate demonstrated (a model passes, and a deliberately bad model is rejected in a test).
- [ ] Nightly and weekly flows run unattended for 14 simulated days (replay mode).
- [ ] CI green with coverage and security gates; images build and deploy to the target host.
- [ ] Security checklist in SECURITY.md fully ticked or exceptions documented.
- [ ] README explains setup in five commands or fewer.

## 12. Risks and mitigations

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | Inventory data is synthetic, so business claims are weak | Label clearly everywhere; present results as simulation; the schema already accepts real inventory (`source = 'upload'`) and the upload endpoint is planned for v1.1 |
| R2 | Intermittent (lumpy) demand makes SKU-level daily forecasting hard | Croston-SBA baseline, eligibility rules, WAPE on aggregates, weekly-aggregation fallback |
| R3 | Advanced models fail to beat baselines | Champion gate, honest reporting; baselines are valid champions |
| R4 | Prophet and LSTM training time | Top-N cap, dev mode (N=30), parallel jobs, time budgets in TRD |
| R5 | Scope creep (Kubernetes, streaming, LLM features) | Stretch list in PHASES.md; no stretch work before Phase 9 |
| R6 | Library API drift (Evidently, MLflow, Prefect, Prophet) | Pin versions; isolate each library behind one module; read docs for the pinned version |
| R7 | Data leakage in features or splits | Leakage unit tests (TESTING.md), purged splits, origin-based features |
| R8 | Privacy exposure if real data is used later | Pseudonymization, minimization, retention rules in SECURITY.md |

## 13. Open questions (defaults apply until decided)

| Question | Default |
|----------|---------|
| Hosting target for the demo | Single VM running Docker Compose behind Caddy (HTTPS) |
| Add the M5 multi-store dataset | Not in v1.0 |
| Display currency and locale | GBP, `en-GB` number formatting, configurable |
| Alert delivery | In-app only; webhook optional via env |

## 14. Glossary

**SKU** stock-keeping unit. **WAPE** weighted absolute percentage error = Σ|y−ŷ| / Σ|y|. **MASE** mean absolute scaled error, scaled by the in-sample seasonal-naive MAE. **RMSSE** root mean squared scaled error (M5 metric). **RFM** recency, frequency, monetary value. **ABC** classes by revenue contribution (A ≈ top 80%, B next 15%, C last 5%). **XYZ** classes by demand variability. **ROP** reorder point. **SS** safety stock. **EOQ** economic order quantity. **PR-AUC** area under the precision-recall curve (average precision). **Drift** change in input or prediction distributions versus a reference window. **Champion** the model version currently used for scoring. **MLOps** the practices that make ML systems reproducible, monitored and deployable.

# RetailPulse — Technical Requirements Document (TRD)

**Version:** 1.0 | **Related:** [PRD](PRD.md) · [ARCHITECTURE](ARCHITECTURE.md) · [DATABASE](DATABASE.md) · [API](API.md) · [SECURITY](SECURITY.md) · [TESTING](TESTING.md) · [CODE_STYLE](CODE_STYLE.md)

This document defines **how** RetailPulse is built: stack, configuration, data rules, ML specifications, budgets and interfaces. If this file conflicts with PRD.md on a technical point, this file wins; if it conflicts with SECURITY.md, SECURITY.md wins.

---

## 1. Technology stack

Pin exact versions in `uv.lock`. Before using any library API, read the docs for the **installed** version; do not code from memory for fast-moving libraries (Prefect, MLflow, Evidently, Streamlit, Prophet).

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | Python 3.11 | Broad wheel support for Prophet, PyTorch, LightGBM |
| Dependency mgmt | `uv` + `pyproject.toml` + lockfile | One lockfile, dependency groups: `core`, `ml`, `api`, `dashboard`, `dev` |
| Data | pandas, numpy, pyarrow, openpyxl | openpyxl only for reading the official XLSX; convert once to Parquet |
| Validation | pandera | DataFrame schemas for ingest and features |
| Database | PostgreSQL 16 | One server, three databases: `retailpulse`, `mlflow`, `prefect` |
| ORM / migrations | SQLAlchemy 2.0 (sync) + Alembic + psycopg 3 | Sync sessions; FastAPI endpoints are plain `def` |
| Forecasting | statsmodels (ETS optional), Prophet, LightGBM, PyTorch (CPU) | Thin internal `Forecaster` interface. Darts is **not** used |
| Segmentation / churn | scikit-learn, LightGBM, SHAP | |
| Optimization | numpy, scipy.stats | Closed-form formulas; no solver needed |
| Experiment tracking / registry | MLflow | Backend store: Postgres. Artifacts: volume (S3-compatible store optional) |
| Orchestration | Prefect 3.x | Server in compose; flows served by `worker` |
| Monitoring | Evidently (pinned) + custom realized-error tracking | Isolated in `monitoring/` because its API changes between releases |
| API | FastAPI, Pydantic v2, uvicorn (gunicorn optional), slowapi, PyJWT, argon2-cffi, structlog | |
| Dashboard | Streamlit (`st.navigation`), Plotly, httpx | Dashboard talks only to the API |
| CLI | Typer | `retailpulse` entry point |
| Quality | ruff, mypy, pytest, pytest-cov, hypothesis, bandit, pip-audit, gitleaks, pre-commit | |
| Containers | Docker (multi-stage), Docker Compose v2 | |
| CI/CD | GitHub Actions | |
| Reverse proxy (prod) | Caddy | Automatic HTTPS |

## 2. Environments

| Env | Purpose | Notes |
|-----|---------|-------|
| `dev` | Local work | `make up` starts Postgres, MLflow, Prefect; API and dashboard may run on host with reload |
| `test` | Automated tests | Ephemeral Postgres (compose or testcontainers); MLflow file store in a temp dir |
| `prod` | Demo / deployed | Compose behind Caddy; only the proxy is published; Postgres, MLflow, Prefect are internal-only |

## 3. Configuration

All configuration comes from environment variables read **once** by `retailpulse.config.Settings` (pydantic-settings, prefix `RP_`). No other module reads `os.environ`. Commit `.env.example`; never commit `.env`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `RP_ENV` | `dev` | `dev`, `test` or `prod` |
| `RP_DATABASE_URL` | none (required) | SQLAlchemy URL, `postgresql+psycopg://...` |
| `RP_MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Tracking server |
| `RP_PREFECT_API_URL` | `http://prefect:4200/api` | Prefect server |
| `RP_JWT_SECRET` | none (required, ≥ 32 bytes) | Signs access tokens |
| `RP_JWT_EXPIRE_MINUTES` | `30` | Access token lifetime |
| `RP_PSEUDONYM_SALT` | none (required, ≥ 32 bytes) | HMAC key for customer IDs |
| `RP_API_BASE_URL` | `http://api:8000` | Used by the dashboard |
| `RP_CORS_ORIGINS` | empty | Comma-separated; empty means CORS disabled |
| `RP_MAX_UPLOAD_MB` | `100` | Upload size cap |
| `RP_DATA_DIR` | `./data` | raw / interim / processed / reports |
| `RP_SEED` | `42` | Global random seed |
| `RP_FORECAST_TOP_N` | `300` | SKUs to forecast (use `30` in dev) |
| `RP_FORECAST_HORIZON_DAYS` | `28` | Horizon |
| `RP_BACKTEST_FOLDS` | `5` | Rolling-origin folds |
| `RP_INTERVAL_LEVEL` | `0.80` | Prediction interval level |
| `RP_CHURN_WINDOW_DAYS` | `90` | Churn label window |
| `RP_CHURN_BAND_MEDIUM` / `RP_CHURN_BAND_HIGH` | `0.30` / `0.60` | Risk band thresholds; tune after inspecting score distribution |
| `RP_RFM_WINDOW_DAYS` | `730` | Window for segmentation |
| `RP_DEDUPE_EXACT_ROWS` | `true` | Drop exact duplicate source rows during cleaning |
| `RP_DEFAULT_SERVICE_LEVEL` | `0.95` | Inventory default |
| `RP_OVERSTOCK_DAYS` | `90` | Days of cover that trigger `overstock` |
| `RP_AUTO_PROMOTE` | `false` | Never `true` by default |
| `RP_ENABLE_DOCS` | `false` in prod | Exposes `/docs` and `/openapi.json` |
| `RP_ALERT_WEBHOOK_URL` | empty | Optional; set only by an operator via env |
| `RP_CURRENCY` | `GBP` | Display currency |
| `RP_LOG_LEVEL` | `INFO` | |

## 4. Data contracts

### 4.1 Sources
1. **Online Retail II** (CSV or XLSX; all sheets concatenated). Official download: the UCI repository page for *Online Retail II* (dataset id 502), a zip containing `online_retail_II.xlsx` (about 43.5 MB, 1,067,371 rows). License CC BY 4.0; cite Chen, D. (2012), DOI 10.24432/C5CG6D. Column names vary between copies of the data (`Invoice` or `InvoiceNo`, `Price` or `UnitPrice`, `Customer ID` or `CustomerID`): the adapter normalizes case and spaces, maps these aliases to `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`, and fails with a clear error if a required column is missing. The official workbook has two sheets (`Year 2009-2010`, `Year 2010-2011`) with the same eight columns; read all of them.
2. **Generic CSV** (UTF-8, header required). Required: `invoice_no`, `invoice_date` (ISO 8601), `sku`, `quantity` (int), `unit_price` (decimal ≥ 0). Optional: `customer_id`, `store_code` (default `ONLINE`), `description`, `country`, `category`.

### 4.2 Canonical sales line
`invoice_no, invoice_ts, sku, description, quantity, unit_price, customer_key (nullable), country, store_code`.

### 4.3 Cleaning rules (applied in this order; every rule logs counts to `ops.data_quality_result`)
Counts marked "real file" were measured on the official `online_retail_II.xlsx` and are the expected values for the full-dataset test.

1. Trim strings; read `Invoice` and `StockCode` as text (the workbook mixes number and text cells); normalize `StockCode` (strip, uppercase; 173 codes in the real file differ only by case); canonicalize `Customer ID` to integer text with no decimal part (the workbook stores it as a number with blanks, so a naive string conversion gives `13085.0`); coerce types; reject rows that fail coercion (counted, written to a quarantine file under `data/interim/quarantine/`).
2. Remove duplicates in two steps. **(a) Cross-sheet overlap:** the two sheets both contain 2010-12-01 to 2010-12-09 (22,523 identical rows each). Drop rows that repeat when keyed by (all fields + occurrence index within their own sheet); this removes the second copy without touching legitimate repeats inside a sheet. Always on. **(b) Remaining exact duplicate rows** (11,812 in the real file) are dropped when `RP_DEDUPE_EXACT_ROWS=true` (default).
3. Mark `is_return = true` when the invoice starts with `C` (case-insensitive) and `quantity < 0`. A `C` invoice with `quantity > 0` is invalid: reject and count (one row in the real file). Negative quantity without a `C` invoice is a return only if it has a customer ID and `unit_price > 0`; otherwise it is a stock adjustment: drop and count (3,457 rows in the real file, none with a customer ID).
4. Mark non-product SKUs (`is_product = false`) from an explicit list in `config/non_product_skus.yaml`, never by pattern. Seen in the real file (rows): `POST` 2,122, `DOT` 1,446, `M` 1,421, `C2` 282, `D` 177, `S` 104, `BANK CHARGES` 102, `ADJUST` 67, `AMAZONFEE` 43, plus `PADS`, `CRUK`, `TEST001` and the `gift_0001_*` vouchers. `DCGS*` codes are **real products** (party bags, night lights, gum): keep them. Re-check the list on any new dataset.
5. Drop lines with `unit_price <= 0` (free or adjustment lines); count them.
6. Pseudonymize `Customer ID` → `customer_key = HMAC_SHA256(RP_PSEUDONYM_SALT, canonical_integer_text)` (hex). Rows with no customer ID (243,007 rows, 22.8% of the real file, about 13% of cleaned revenue) keep `customer_id = NULL`.
7. Compute `source_row_hash = sha256(all canonical fields joined by a unit separator)` for idempotent loading (unique).
8. **Batch gate:** if more than 20% of input rows are rejected, mark the batch `rejected` and load nothing (constant `MAX_REJECT_RATIO = 0.20` in `etl/quality.py`).
9. **Outlier flag:** lines with `quantity >= 50,000` (constant `OUTLIER_QTY` in `etl/quality.py`) get `is_outlier = true`. They stay in `core.fact_sales_line` but are excluded from `fact_daily_sales` and from customer features. The real file has two (80,995 and 74,215 units), each cancelled afterwards; left in, they would add about £245k of phantom revenue (about 1.2% of the total) and make one customer look like a £77k buyer.

Expected row counts for the real file: 1,067,371 read → 1,044,848 after step 2(a) → 1,033,036 after step 2(b).

### 4.4 Definitions
- **Demand (`units_sold`)** = sum of `quantity` over non-return, non-outlier lines, product SKUs, `unit_price > 0`, per (date, store, product), for complete days only.
- **Net revenue** = Σ line_amount over non-return, non-outlier lines minus returns.
- **Complete day:** the last calendar day of a dataset is partial if its last transaction time is earlier than the 10th percentile of the daily last-transaction times over the previous 28 trading days. The ETL sets `core.dim_date.is_complete = false` for it and excludes it from daily aggregates (the lines stay in `fact_sales_line`). In Online Retail II the final day, 2011-12-09, ends at 12:50 and is partial.
- **`data_as_of`** = the latest complete date (`is_complete = true`) in `core.fact_daily_sales`: 2011-12-08 for Online Retail II. Every feature, label, forecast origin and recency calculation uses `data_as_of`. **Never call `date.today()` or `now()` inside ML logic.**
- **Trading days:** a day is non-trading if (a) its weekday has no whole-catalog sales over the last 52 weeks (Saturday in Online Retail II; Sunday trades normally), or (b) it is a GB public holiday (`holidays` package), or (c) it lies inside a gap of 4 or more days between consecutive trading dates. The real data shows about 12 days at Christmas and New Year (last trading day 23 Dec, reopening 4 Jan, in both years) and 5 days at Easter. For future dates, the previous year's closure window, matched by month and day, is assumed to repeat. Series are modeled on trading days only; forecasts for non-trading days are written as 0.
- **Timestamps:** `invoice_ts` is source-local and timezone-naive; system timestamps are UTC `timestamptz`.

## 5. ML specifications

### 5.1 Forecasting
- **Granularity:** daily, per (store, product). MVP has one store (`ONLINE`).
- **Series universe:** top-N product SKUs by trailing-365-day revenue. **Eligible** if first sale ≥ 90 days ago and ≥ 20 non-zero sales days in the last 365. Ineligible SKUs get inventory status `no_data`.
- **Coverage and scale:** in Online Retail II the top 300 eligible SKUs cover about 54% of trailing-year revenue, the top 750 about 76% (roughly the A class: 744 SKUs reach 80% of eligible revenue) and the top 1,000 about 82%. Use 30 in dev and 300 for the first full run; raise `RP_FORECAST_TOP_N` toward 1,000 once the pipeline works if Prophet's runtime allows, or drop Prophet from the large run and say so in the report.
- **Lumpy wholesale orders:** training targets are clipped per series at the 99.5th percentile of the training window (evaluation always uses unclipped actuals). The real data has 321 sales lines of 1,000 units or more.
- **Closure calendar:** calendar features of the target date include `days_to_christmas` (signed, clipped to ±30) and an `in_closure_window` flag from §4.4, so models see the end-of-year closure.
- **Interface:**
  ```python
  class Forecaster(Protocol):
      name: str
      def fit(self, history: pd.DataFrame) -> "Forecaster": ...
      def predict(self, origin: date, horizon: int) -> pd.DataFrame: ...
  # history columns: series_id, date, y, price, is_holiday, abc_class (plus model-specific)
  # predict returns: series_id, date, yhat (>= 0)
  ```
- **Models:**
  | Name | Spec |
  |------|------|
  | `naive` | Last observed value |
  | `seasonal_naive` | Value 7 days earlier |
  | `moving_average` | Mean of last 28 trading days |
  | `croston_sba` | Croston with Syntetos-Boylan bias correction; for intermittent series |
  | `prophet` | One model per series; weekly + yearly seasonality, GB holidays; `joblib` parallel; series below eligibility fall back to `croston_sba` |
  | `lightgbm` | Global model, **direct strategy**: training rows are (origin, horizon_day h in 1..28) sampled weekly; all features computed from data ≤ origin; target is y(origin+h). Features: lags 1/7/14/28, rolling mean/std 7/28 (shifted), days since last sale, price relative to SKU median, calendar of the target date, `horizon_day`, `product_id` (categorical), `abc_class`. Try objectives `tweedie` and `regression`; keep the better by backtest |
  | `lstm` | Global model, PyTorch, CPU. Window 56 days → 28-day output. Inputs per step: scaled `log1p(units)`, day-of-week sin/cos, holiday flag, price ratio; static SKU embedding (dim 8). 1–2 layers, hidden 64. Loss: Huber on `log1p` scale; outputs `expm1` then clipped ≥ 0. Adam 1e-3, batch 256, ≤ 30 epochs, early stopping (patience 5) on the last 28 days of each training fold. Seeds fixed |
- **Backtest:** rolling origin, `RP_BACKTEST_FOLDS` folds, step = horizon = 28 days; training data strictly before each origin. Metrics: WAPE (global = Σ|e| / Σ y over all series and folds; also per ABC class and per horizon bucket 1–7, 8–14, 15–28), MASE (scale = in-sample seasonal-naive MAE, m=7; skip series with zero scale), RMSSE, bias, 80% coverage, pinball loss (q=0.1, 0.9).
- **Prediction intervals:** empirical residual quantiles from backtests, per (model, ABC class, horizon bucket). `lower = max(0, yhat + q10)`, `upper = yhat + q90`. Prophet native intervals are logged for comparison only.
- **Champion gate:** candidate must have global WAPE strictly lower than `seasonal_naive` and |bias| ≤ 0.15. Coverage outside 0.65–0.95 raises a warning alert, not a block. If nothing passes, the best baseline becomes champion and alert `forecast_no_model_beats_baseline` is created.
- **Registry note:** local models (baselines, Prophet) are refit at scoring time, so the registry stores config, metrics and backtest artifacts. Global models (LightGBM, LSTM) store weights as MLflow artifacts.
- **Outputs:** `ml.forecast` rows for horizon days 1..28, `origin_date = data_as_of`.

### 5.2 Customer segmentation
- Population: customers with ≥ 1 non-return purchase in the last `RP_RFM_WINDOW_DAYS` before `data_as_of`.
- Features: `recency_days = data_as_of − last_order_date`; `frequency` = distinct invoices; `monetary` = net revenue clipped ≥ 0. Transform: `log1p`, then `StandardScaler`.
- K-Means, `n_init=20`, `random_state=RP_SEED`; k in 3..8 chosen by highest silhouette with every cluster ≥ 2% of customers. Log all candidates' silhouette and Davies-Bouldin.
- Labels: deterministic rules on centroid ranks map clusters to business labels (`Champions`, `Loyal`, `New / Promising`, `Big Spenders`, `At Risk`, `Hibernating`). The mapping function is pure and unit-tested.
- Stability: refit with 5 seeds, report mean adjusted Rand index.

### 5.3 Churn
- **Cutoffs:** monthly, from the first cutoff with ≥ 6 months of history through the last cutoff `T` where `T + 90 days ≤ data_as_of`.
- **Population at cutoff:** customers with ≥ 1 non-return purchase in the 365 days before `T`.
- **Label:** `1` if no non-return purchase in `(T, T + RP_CHURN_WINDOW_DAYS]`, else `0`.
- **Features (data ≤ T only):** recency, orders in last 30/90/180/365 days, net revenue in last 90/365 days, tenure, mean and std of inter-purchase gap, overdue ratio (`recency / mean gap`), average order value, items per order, distinct SKUs, return rate, order-count trend (last 90 vs prior 90), country (categorical), month of `T`. Gap features are undefined for customers with a single order (28% of customers in the real data): LightGBM handles missing values natively; logistic regression uses median imputation plus a missing-indicator.
- **Split (purged, temporal):** train on earliest cutoffs, validate on the next block, test on the latest cutoff. A training cutoff `T_train` is allowed only if `T_train + 90 days ≤` the first validation cutoff, so label windows never overlap evaluation features.
- **Models:** logistic regression (standardized, class-weighted) as baseline; LightGBM with early stopping on validation. Calibrate on validation (isotonic or Platt, whichever has the lower Brier).
- **Metrics:** PR-AUC, ROC-AUC, Brier, log loss, lift and precision at top 10%, recall at top 20%, calibration curve artifact.
- **Base rate:** on Online Retail II the 90-day churn rate is about 0.59 (0.42 to 0.67 across monthly cutoffs, seasonal), so lift is capped at 1 ÷ base rate (about 1.7×) and PR-AUC starts high. Judge models mainly by ROC-AUC, Brier score, precision at the top decile and calibration.
- **Explanations:** SHAP `TreeExplainer` for LightGBM; coefficient × standardized value for logistic regression. Store top-3 `{feature, value, impact, direction}` per customer.
- **Bands:** `high` if p ≥ `RP_CHURN_BAND_HIGH`, `medium` if p ≥ `RP_CHURN_BAND_MEDIUM`, else `low`.
- **Expected revenue at risk:** Σ over active customers of `p_churn × trailing_12m_net_revenue × (RP_CHURN_WINDOW_DAYS / 365)`.
- **Scoring population:** customers with a purchase in the last 365 days before `data_as_of`.

### 5.4 Inventory policy ((s, Q), evaluated nightly)
Per (store, product), with `L` lead time (days), `σ_LT` lead-time std, `z = Φ⁻¹(service_level)`:

```
μ_d   = mean of champion forecast over the next L days (units/day)
μ_L   = Σ forecast over the next ceil(L) days
σ_e   = std of daily forecast errors from backtests (per ABC class × model; floor at small ε)
σ_L   = sqrt( L · σ_e² + μ_d² · σ_LT² )
SS    = z · σ_L                          # safety stock
ROP   = μ_L + SS                         # reorder point
IP    = on_hand + on_order               # inventory position
EOQ   = sqrt( 2 · (365 · μ_d) · K / h )  # K = order_cost, h = holding_cost_rate · unit_cost
Q     = EOQ rounded up to a multiple of MOQ (at least MOQ)
stockout_risk = 1 − Φ( (IP − μ_L) / σ_L )          # normal approximation; if σ_L = 0 use a step
days_of_cover = on_hand / μ_d
status: reorder_now if IP ≤ ROP; overstock if days_of_cover > RP_OVERSTOCK_DAYS; no_data if not eligible; else ok
recommended_order_qty = Q · max(1, ceil((ROP − IP) / Q)) when reorder_now, else 0
```
Intermittent SKUs violate the normal approximation; FR-INV-06 adds an empirical-quantile alternative.

**Simulator:** replay the last backtest span day by day with lost sales (unmet demand is lost). Orders are placed when `IP ≤ ROP` computed from the forecast available at that origin; they arrive after a lead time drawn from `N(L, σ_LT)` (rounded, seeded). Compare with a baseline policy (ROP = 28-day moving average × (L + 14)). Report fill rate (units served / units demanded), stockout-day rate, average on-hand units and value, holding cost, and number of orders.

**Synthetic parameter generation (deterministic by `RP_SEED`):** lead time drawn from a small discrete set (3, 7, 14, 21 days) by ABC class, `σ_LT = 0.2 · L`, MOQ from {1, 6, 12}, `unit_cost = 0.6 × median selling price`, `holding_cost_rate = 0.20`, `order_cost = 25`. Initial on-hand at the simulation start = `ROP` of the baseline policy. Everything is stored with `source = 'synthetic'`.

## 6. MLOps requirements

- **Experiments:** `retailpulse/forecast`, `retailpulse/segmentation`, `retailpulse/churn`. **Registered models:** `retailpulse-forecast`, `retailpulse-segmentation`, `retailpulse-churn`.
- **Every run logs:** params, metrics, artifacts (plots, backtest tables, feature list), tags `git_sha`, `dataset_hash`, `data_as_of`, `seed`, `library_versions`.
- **Aliases:** `champion` and `previous_champion`. Promotion = move alias + update `ml.model_run.is_champion` + audit log, in one service function. Rollback = promote the previous run.
- **Promotion is manual** unless `RP_AUTO_PROMOTE=true`; gates in §5 apply either way.
- **Drift:** Evidently data drift on forecast inputs (last 28 days vs training reference), churn features, and prediction drift on outputs. Start with the library's default thresholds; store the HTML report and a JSON summary.
- **Performance monitoring:** weekly job computes realized WAPE for forecasts whose target dates have passed. Alert if realized WAPE > 1.25 × backtest WAPE or is worse than seasonal naive over the same window. Churn: compute realized PR-AUC once the 90-day outcome window has elapsed.
- **Retraining:** weekly schedule, plus on drift or degradation alerts. A failed or gate-failing training run never changes the champion.
- **Reproducibility:** fixed seeds; deterministic LightGBM settings; `uv.lock`; dataset hash of the canonical input; training is a pure function of (data, config, seed, code version).

## 7. Non-functional requirements and budgets

Reference machine: 4 vCPU, 8 GB RAM, SSD, Docker. Budgets are targets to verify in Phase 9; if missed, optimize or document why.

| Area | Requirement |
|------|-------------|
| ETL | Full Online Retail II load (≈1.07M rows) < 10 min; incremental day < 2 min |
| Forecast training | Baselines + LightGBM backtest (300 SKUs) < 15 min; LSTM < 30 min on CPU; Prophet < 60 min with 4 jobs (use `RP_FORECAST_TOP_N=30` in dev) |
| Batch scoring (forecast + churn + segments + inventory) | < 15 min |
| API latency | p95 < 300 ms for indexed read endpoints; p95 < 1 s for cached KPI aggregates; 50 concurrent users |
| Dashboard | Page load p95 < 3 s with warm cache |
| Reliability | Flows are idempotent; retries 3× with exponential backoff; timeouts on every external call; writes of model outputs are atomic per run; API serves last good run on failure |
| Availability | Best effort (demo). Health endpoints and restart policies in compose |
| Scalability | Designed for 10× data without redesign (partition `fact_sales_line` by month, move features to DuckDB/Polars); path documented in ARCHITECTURE §10 |
| Observability | JSON logs with `request_id`; Prometheus `/metrics`; MLflow for ML; Evidently reports; `ops.pipeline_run` for flows |
| Maintainability | Module boundaries in ARCHITECTURE §4; mypy strict on `src/`; coverage gates in TESTING.md |
| Portability | Runs on Linux, macOS and Windows (WSL2) via Docker; Prophet may need Docker if native install fails |
| Browser support | Latest two versions of Chrome, Firefox, Edge and Safari |
| Accessibility | WCAG 2.1 AA targets in DESIGN.md |

## 8. Cross-cutting technical standards

- **Errors:** domain exceptions in `retailpulse/errors.py`; API maps them to RFC 9457 problem responses; no bare `except`.
- **Logging:** structlog JSON; no `print`; never log secrets, tokens, passwords or raw customer identifiers.
- **Time:** ML logic takes `as_of: date` as a parameter. UTC for system timestamps.
- **Idempotency:** every flow and loader is safe to retry; unique keys and upserts (see DATABASE.md).
- **Concurrency:** one run per flow at a time (advisory lock or status check); API returns 409 on conflict.
- **Versioning:** API under `/api/v1`; DB changes only through Alembic; every migration has `downgrade`.

## 9. Build and deploy requirements

- Multi-stage Dockerfiles, non-root user, pinned base image tags, healthchecks, no secrets in images.
- Compose files: `docker-compose.yml` (base), `docker-compose.override.yml` (dev), `docker-compose.prod.yml` (prod, Caddy, internal network, no published DB/MLflow/Prefect ports).
- CI: lint, types, unit and integration tests, coverage gate, security scans, image build. CD: build and push on tag, deploy, post-deploy smoke test.

## 10. Technical risks and gotchas

| Risk | Handling |
|------|----------|
| Prophet install problems on Windows or ARM | Run in Docker (`python:3.11-slim` image); keep Prophet behind `ml/forecasting/prophet_model.py` |
| MLflow tracking server host/CORS checks in newer releases | If requests via the compose service name are rejected, add the service name to the allowed hosts per the docs of the pinned version |
| Evidently API changes across versions | Pin; wrap in `monitoring/drift.py`; contract-test the wrapper |
| Prefect API changes | Confine to `retailpulse/pipelines/`; tests call the underlying functions directly |
| Pickle risk when loading models | Load only from this deployment's registry; never accept model files from users (SECURITY.md) |
| Large XLSX memory use | Convert once to Parquet; read in chunks |
| Wall-clock leakage | Lint rule and review checklist: no `today()`/`now()` in `ml/`, `features/`, `etl/` logic |
| Streamlit cache leaking between users | Include role in cache keys; never cache admin or user-specific responses (DESIGN.md) |

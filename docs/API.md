# RetailPulse — API Specification (v1)

**Version:** 1.0 | **Framework:** FastAPI + Pydantic v2 | **Related:** [DATABASE](DATABASE.md) · [SECURITY](SECURITY.md) · [TRD](TRD.md) · [APP_FLOW](APP_FLOW.md)

This is the contract between the API, the dashboard and any other client. All example numbers are illustrative.

---

## 1. Conventions

| Topic | Rule |
|-------|------|
| Base path | `/api/v1`. Health and metrics endpoints are at the root. |
| Format | JSON (`application/json`), UTF-8, `snake_case` keys. CSV only on `/export` endpoints. |
| Dates and times | Dates `YYYY-MM-DD`. Timestamps ISO 8601 UTC with `Z`. |
| Auth | `Authorization: Bearer <JWT>` on everything except login and health. |
| Roles | `viewer` < `analyst` < `admin`. Each endpoint lists its minimum role. |
| IDs | `product_id`, `customer_id`, `store_id` are integers. Runs, batches, reports, users are UUIDs. Customers appear only as pseudonymous surrogate IDs (`customer_id`) and `customer_key`. |
| Pagination | `?page=1&page_size=50` (default 50, max 200). Response envelope below. |
| Sorting | `?sort=field:asc` or `field:desc`, whitelisted per endpoint; unknown field → 422. |
| Request ID | Server generates `X-Request-ID` (or accepts a safe client value) and echoes it in responses and error bodies. |
| Idempotency | Uploads dedupe on file hash. Trigger endpoints return 409 if the flow is already queued or running. |
| Rate limits | Login 5/min per IP (plus account lockout). Uploads 10/hour per user. Triggers and simulate 20/hour per user. Everything else 120/min per user. 429 includes `Retry-After`. |
| Caching | KPI and champion metadata are cached in-process for 60 s. Responses are not cacheable by shared caches (`Cache-Control: no-store` for authenticated data). |
| Docs | `/docs` and `/openapi.json` are served only when `RP_ENABLE_DOCS=true` (default on in dev/test, off in prod). |
| Versioning | Additive changes stay in v1. Breaking changes require `/api/v2`. |

### Pagination envelope
```json
{ "items": [], "page": 1, "page_size": 50, "total": 1234 }
```

### Errors (RFC 9457 problem details, `application/problem+json`)
```json
{
  "type": "urn:retailpulse:problem:validation_error",
  "title": "Validation error",
  "status": 422,
  "detail": "horizon must be between 1 and 28",
  "instance": "/api/v1/forecasts",
  "request_id": "b1e5c0de-0000-4000-8000-000000000000",
  "errors": [{ "field": "horizon", "message": "must be <= 28" }]
}
```

| Status | `type` suffix | When |
|--------|---------------|------|
| 400 | `bad_request` | Malformed request that is not a validation issue |
| 401 | `unauthorized` | Missing, invalid or expired token; bad credentials (generic message) |
| 403 | `forbidden` | Role too low |
| 404 | `not_found` | Resource does not exist (never reveal whether a user exists) |
| 409 | `conflict` | Flow already running, duplicate resource, last-admin protection |
| 413 | `payload_too_large` | Upload over `RP_MAX_UPLOAD_MB` |
| 415 | `unsupported_media_type` | Wrong file type |
| 422 | `validation_error` | Schema or business validation failed |
| 429 | `rate_limited` | Rate limit hit |
| 503 | `model_unavailable` | No active outputs yet (e.g. before first scoring) or a dependency is down |
| 500 | `internal_error` | Unhandled error; generic message, details only in logs |

Errors never include stack traces, SQL, file paths or secrets.

## 2. Endpoint catalogue

Minimum role: **V** viewer, **A** analyst, **D** admin, **—** public.

### 2.1 Auth and users
| Method | Path | Role | Description | Success |
|--------|------|------|-------------|---------|
| POST | `/api/v1/auth/login` | — | Form fields `username` (email) and `password` → access token | 200 |
| GET | `/api/v1/auth/me` | V | Current user (`user_id`, `email`, `role`) | 200 |
| POST | `/api/v1/auth/change-password` | V | Body `current_password`, `new_password` (min 12 chars) | 204 |
| GET | `/api/v1/users` | D | List users (paginated) | 200 |
| POST | `/api/v1/users` | D | Create user (`email`, `full_name`, `role`, `password`) | 201 |
| PATCH | `/api/v1/users/{user_id}` | D | Change `role`, `is_active`, `full_name`, or set `password`; cannot demote or deactivate the last active admin (409) | 200 |

### 2.2 Reference data and sales
| Method | Path | Role | Description |
|--------|------|------|-------------|
| GET | `/api/v1/stores` | V | List stores |
| GET | `/api/v1/products` | V | Search products: `q`, `abc_class`, `forecast_eligible`, `is_product`, pagination |
| GET | `/api/v1/products/{product_id}` | V | Product detail incl. ABC/XYZ, first and last sale dates |
| GET | `/api/v1/sales/daily` | V | Daily series: `product_id` (optional; omit for catalog total), `store_id`, `from`, `to` (max 800 days) |

### 2.3 KPIs and forecasts
| Method | Path | Role | Description |
|--------|------|------|-------------|
| GET | `/api/v1/kpis/overview` | V | Overview KPIs (below) |
| GET | `/api/v1/forecasts` | V | `product_id` (required), `store_id` (default `ONLINE`), `horizon` (1–28, default 28) |
| GET | `/api/v1/forecasts/accuracy` | V | Backtest metrics for the champion: global, by ABC class, by horizon bucket, vs baselines |

### 2.4 Customers
| Method | Path | Role | Description |
|--------|------|------|-------------|
| GET | `/api/v1/segments` | V | Segment profiles for the active segmentation |
| GET | `/api/v1/customers` | V | Filters: `segment_id`, `risk_band`, `min_probability`, `q` (customer_key prefix); sort by `churn_probability`, `monetary`, `recency_days` |
| GET | `/api/v1/customers/{customer_id}` | V | Segment, RFM, churn probability and reasons, order-history summary |
| GET | `/api/v1/customers/export` | A | Same filters as the list; returns `text/csv` (formula-safe cells) |
| GET | `/api/v1/churn/summary` | V | Band counts, probability histogram, expected revenue at risk |

### 2.5 Inventory
| Method | Path | Role | Description |
|--------|------|------|-------------|
| GET | `/api/v1/inventory/summary` | V | Counts by status, stockout-risk distribution, share of parameters that are synthetic |
| GET | `/api/v1/inventory/recommendations` | V | Filters: `status`, `abc_class`, `q`; sort by `stockout_risk`, `days_of_cover` |
| GET | `/api/v1/inventory/recommendations/export` | A | CSV of the filtered list (formula-safe) |
| GET | `/api/v1/inventory/params/{product_id}` | V | Parameters for a product and store |
| PUT | `/api/v1/inventory/params/{product_id}` | A | Replace parameters (`source` becomes `user`); audit-logged |
| POST | `/api/v1/inventory/what-if` | A | Recompute one SKU with overrides; nothing persisted |
| POST | `/api/v1/inventory/simulate` | A | Run the policy simulation (≤ 200 SKUs, synchronous); result stored in `ml.policy_simulation` |
| GET | `/api/v1/inventory/simulations/latest` | V | Latest stored simulation |

### 2.6 Models, monitoring, alerts
| Method | Path | Role | Description |
|--------|------|------|-------------|
| GET | `/api/v1/models` | V | `task`, pagination; includes metrics and champion flag |
| GET | `/api/v1/models/{run_id}` | V | Run detail: params, metrics, gate result, MLflow link |
| POST | `/api/v1/models/{task}/train` | D | Body `algorithms` (optional) and `params` (optional); returns 202 with `pipeline_run_id` |
| POST | `/api/v1/models/{run_id}/promote` | D | Move `champion` alias; only runs that passed the gate (else 409); triggers `score_all`; audit-logged |
| GET | `/api/v1/monitoring/drift` | V | List drift reports (`task`, `kind`) |
| GET | `/api/v1/monitoring/drift/{report_id}` | V | Summary JSON |
| GET | `/api/v1/monitoring/drift/{report_id}/html` | V | Stored Evidently HTML, served with `Content-Security-Policy: sandbox` and `X-Content-Type-Options: nosniff` |
| GET | `/api/v1/monitoring/performance` | V | Realized-error snapshots (`task`, `model_run_id`) |
| GET | `/api/v1/alerts` | V | `status=open` or `all` |
| POST | `/api/v1/alerts/{alert_id}/ack` | A | Acknowledge; audit-logged |

### 2.7 Data and pipelines
| Method | Path | Role | Description |
|--------|------|------|-------------|
| POST | `/api/v1/uploads/sales` | A | Multipart: `file`, `source_name` (`online_retail_ii` or `generic_csv`). Returns 202 with `batch_id` |
| GET | `/api/v1/uploads` | A | List batches |
| GET | `/api/v1/uploads/{batch_id}` | A | Status, row counts, data-quality results, error |
| GET | `/api/v1/pipelines/runs` | V | `flow_name`, `status`, pagination |
| GET | `/api/v1/pipelines/runs/{run_id}` | V | Run detail |
| POST | `/api/v1/pipelines/{flow_name}/trigger` | A for `etl` and `score_all`; D for the rest | Returns 202 with `pipeline_run_id`; 409 if active |

### 2.8 Operations (root)
| Method | Path | Role | Description |
|--------|------|------|-------------|
| GET | `/health/live` | — | Process is up |
| GET | `/health/ready` | — | Database reachable and migrations at head (MLflow and Prefect reported as degraded, not failed) |
| GET | `/metrics` | internal | Prometheus metrics; blocked at the proxy in prod |

## 3. Request and response examples

### 3.1 Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=analyst@example.com&password=********
```
```json
{ "access_token": "eyJhbGciOi...", "token_type": "bearer", "expires_in": 1800 }
```
Failure always returns the same 401 body regardless of whether the email exists.

### 3.2 Overview KPIs
```json
{
  "data_as_of": "2011-12-08",
  "currency": "GBP",
  "revenue_30d": 612345.67,
  "revenue_prev_30d": 598210.12,
  "revenue_change_pct": 2.36,
  "active_customers_90d": 1840,
  "forecast": { "run_id": "6f1c...", "algorithm": "lightgbm", "wape": 0.42, "baseline_wape": 0.51 },
  "skus_reorder_now": 37,
  "high_risk_customers": 214,
  "expected_revenue_at_risk": 48210.55,
  "open_alerts": 2,
  "last_pipeline_run": { "flow_name": "score_all", "status": "succeeded", "finished_at": "2026-10-09T01:22:41Z" }
}
```

### 3.3 Forecast
`GET /api/v1/forecasts?product_id=123&horizon=3`
```json
{
  "model": { "run_id": "6f1c...", "algorithm": "lightgbm", "is_champion": true, "trained_at": "2026-10-04T02:41:10Z" },
  "product": { "product_id": 123, "sku": "85123A", "description": "EXAMPLE PRODUCT", "abc_class": "A" },
  "store_id": 1,
  "origin_date": "2011-12-08",
  "interval_level": 0.8,
  "points": [
    { "date": "2011-12-09", "yhat": 9.4, "yhat_lower": 3.0, "yhat_upper": 19.8 },
    { "date": "2011-12-10", "yhat": 0.0, "yhat_lower": 0.0, "yhat_upper": 0.0 },
    { "date": "2011-12-11", "yhat": 14.2, "yhat_lower": 6.0, "yhat_upper": 27.5 }
  ]
}
```
Errors: 404 unknown product; 422 product not forecast-eligible (`detail` explains why); 503 no active forecast.

### 3.4 Forecast accuracy
```json
{
  "run_id": "6f1c...",
  "algorithm": "lightgbm",
  "metrics": { "wape": 0.42, "mase": 0.88, "rmsse": 0.91, "bias": -0.03, "coverage": 0.78 },
  "by_abc": { "A": { "wape": 0.31 }, "B": { "wape": 0.46 }, "C": { "wape": 0.71 } },
  "by_horizon": [ { "bucket": "1-7", "wape": 0.36 }, { "bucket": "8-14", "wape": 0.41 }, { "bucket": "15-28", "wape": 0.47 } ],
  "baselines": { "seasonal_naive": { "wape": 0.51 }, "moving_average": { "wape": 0.55 } },
  "gate": { "passed": true, "reason": "wape 0.42 < seasonal_naive 0.51; |bias| 0.03 <= 0.15" }
}
```

### 3.5 Customer detail
```json
{
  "customer_id": 4812,
  "customer_key": "9c1f...e3",
  "country": "United Kingdom",
  "segment": { "segment_id": 2, "label": "At Risk" },
  "rfm": { "recency_days": 143, "frequency": 6, "monetary": 1840.5 },
  "churn": {
    "probability": 0.74, "risk_band": "high", "as_of_date": "2011-12-08",
    "top_reasons": [
      { "feature": "overdue_ratio", "value": 2.9, "impact": 0.31, "direction": "increases_risk" },
      { "feature": "orders_last_90d", "value": 0, "impact": 0.22, "direction": "increases_risk" },
      { "feature": "tenure_days", "value": 640, "impact": -0.08, "direction": "decreases_risk" }
    ]
  },
  "orders_summary": { "first_order_date": "2010-03-02", "last_order_date": "2011-07-19", "n_orders": 6, "avg_order_value": 306.75 }
}
```

### 3.6 Inventory recommendation item
```json
{
  "product_id": 123, "sku": "85123A", "description": "EXAMPLE PRODUCT", "abc_class": "A", "store_id": 1,
  "as_of_date": "2011-12-08", "on_hand": 120, "on_order": 0,
  "lead_time_days": 14.0, "service_level": 0.95,
  "demand_mean_lt": 310.4, "demand_std_lt": 88.2,
  "safety_stock": 145.1, "reorder_point": 455.5, "recommended_order_qty": 240,
  "stockout_risk": 0.97, "days_of_cover": 5.4, "status": "reorder_now",
  "params_source": "synthetic"
}
```
The response envelope for lists also includes `"params_synthetic_share"` so the UI can show the caveat.

### 3.7 What-if
`POST /api/v1/inventory/what-if`
```json
{ "product_id": 123, "store_id": 1, "overrides": { "lead_time_days": 7, "service_level": 0.98, "moq": 12 } }
```
Returns the same item shape as §3.6 plus `"persisted": false`.

### 3.8 Simulate
`POST /api/v1/inventory/simulate`
```json
{
  "scope": { "abc_class": "A", "limit": 100 },
  "policy": { "service_level": 0.95 },
  "baseline": { "type": "moving_average_28", "cover_days_extra": 14 }
}
```
```json
{
  "simulation_id": "0a7d...",
  "window": { "start": "2011-07-22", "end": "2011-12-08" },
  "uses_synthetic_params": true,
  "results": {
    "retailpulse": { "fill_rate": 0.962, "stockout_day_rate": 0.041, "avg_on_hand_units": 5120, "avg_on_hand_value": 31200.0, "n_orders": 410 },
    "baseline":    { "fill_rate": 0.941, "stockout_day_rate": 0.067, "avg_on_hand_units": 5480, "avg_on_hand_value": 33400.0, "n_orders": 388 }
  }
}
```
More than 200 SKUs → 422.

### 3.9 Upload
```http
POST /api/v1/uploads/sales
Content-Type: multipart/form-data

file=<sales.csv>; source_name=generic_csv
```
```json
{ "batch_id": "c2b8...", "status": "queued", "pipeline_run_id": "91ee..." }
```
Validation before queueing: size ≤ `RP_MAX_UPLOAD_MB` (413), extension and magic bytes are CSV or XLSX (415), header contains the required columns (422 listing missing columns). Re-uploading an identical file returns 200 with the existing batch.

`GET /api/v1/uploads/{batch_id}`
```json
{
  "batch_id": "c2b8...", "status": "succeeded", "source_name": "generic_csv",
  "row_count_in": 120000, "row_count_loaded": 118442, "row_count_rejected": 1558,
  "quality": [
    { "check_name": "duplicate_rows_removed", "severity": "info", "passed": true, "affected_rows": 1200 },
    { "check_name": "non_positive_price", "severity": "warning", "passed": true, "affected_rows": 358 }
  ],
  "started_at": "2026-10-09T01:00:04Z", "finished_at": "2026-10-09T01:03:51Z", "error": null
}
```

### 3.10 Trigger, train, promote
```json
{ "pipeline_run_id": "91ee...", "flow_name": "train_forecast", "status": "queued" }
```
Response headers: `Location: /api/v1/pipelines/runs/91ee...`.

`POST /api/v1/models/RUN_ID/promote` → 200
```json
{ "task": "forecast", "run_id": "6f1c...", "previous_champion_run_id": "2a90...", "score_all_pipeline_run_id": "55bd..." }
```
409 when the run did not pass its gate or is not `succeeded`.

### 3.11 Model run
```json
{
  "run_id": "6f1c...", "task": "forecast", "algorithm": "lightgbm", "status": "succeeded", "is_champion": true,
  "trained_at": "2026-10-04T02:41:10Z", "data_end": "2011-12-08",
  "metrics": { "wape": 0.42, "gate_passed": true, "gate_reason": "..." },
  "params": { "objective": "tweedie", "horizon": 28 },
  "git_sha": "abc1234", "dataset_hash": "5e2a...", "seed": 42,
  "mlflow_url": "http://mlflow:5000/#/experiments/1/runs/..."
}
```
`mlflow_url` is shown to admins only.

### 3.12 Alert
```json
{ "alert_id": 17, "severity": "warning", "kind": "forecast_degraded",
  "message": "Realized WAPE 0.63 exceeds 1.25 x backtest WAPE 0.42",
  "created_at": "2026-10-08T04:00:12Z", "acknowledged_at": null }
```

## 4. Implementation notes for Claude Code

1. Every endpoint declares a `response_model`; never return ORM objects or dicts built ad hoc. `password_hash` must not exist in any response model.
2. Authorization is a dependency (`require_role("analyst")`), declared on the router or endpoint, never inside the function body.
3. Routers contain no SQL or business logic: they validate, call a service or repository function, and map the result. One router file per area in §2.
4. User-facing reads go through the `ml.v_active_*` views (DATABASE.md §8). Unknown product, run or batch → 404.
5. Trigger endpoints: insert `ops.pipeline_run` (`queued`), call the `PipelineTrigger` abstraction, return 202. If the trigger fails, mark the run `failed` with the error and return 503.
6. CSV exports neutralize cells beginning with `=`, `+`, `-`, `@`, tab or carriage return (see `security/csv_safety.py`) and set `Content-Disposition: attachment`.
7. Pagination, sorting whitelists and error mapping are shared helpers (`api/pagination.py`, `api/errors.py`).
8. Audit-log (`ops.audit_log`): login success/failure, upload, trigger, promote, param update, user create/update, alert ack.
9. Use `operation_id` in `snake_case` (e.g. `get_forecast`) and add OpenAPI examples matching §3.
10. Add or change an endpoint only together with its test in `tests/integration/api/` and an update to this file.

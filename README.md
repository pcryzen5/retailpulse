# RetailPulse

RetailPulse is an enterprise AI-powered customer analytics and demand forecasting platform for retail businesses. It provides:
1. **Demand Forecasting**: Per-SKU demand forecasting with calibrated 80% uncertainty intervals across 7, 14, and 28-day horizons.
2. **Customer Segmentation**: RFM-based behavioral clustering with revenue and customer distribution insights.
3. **Churn Prediction**: XGBoost churn probability scoring with plain-language SHAP reason codes.
4. **Inventory Optimization**: Multi-echelon $(s, Q)$ policy recommendations, stockout risk bounds, and what-if parameter simulation.
5. **Model Registry & Monitoring**: Champion/challenger comparison, continuous realized WAPE tracking, and Evidently feature drift reports.
6. **Data Ingestion & Orchestration**: Streaming CSV upload validation, batch quality gates, and concurrency-locked Prefect workflow triggers.

Served through a high-performance **FastAPI** REST API and an interactive **Streamlit** dashboard, orchestrated by **Prefect**, tracked by **MLflow**, and stored in **PostgreSQL 16**.

---

## Dashboard Previews

| Executive Overview | Demand Forecasting (80% Intervals) |
|:---:|:---:|
| ![Overview](docs/assets/screenshots/overview.png) | ![Forecast](docs/assets/screenshots/demand_forecast.png) |

| Customer Segmentation (RFM) | Churn Risk & SHAP Explanations |
|:---:|:---:|
| ![Segments](docs/assets/screenshots/customer_segments.png) | ![Churn](docs/assets/screenshots/churn_risk.png) |

| Inventory Policy & Simulation | Model Registry & Evidently Drift |
|:---:|:---:|
| ![Inventory](docs/assets/screenshots/inventory_policy.png) | ![Monitoring](docs/assets/screenshots/model_monitoring.png) |

| Data Ingestion & Pipelines | Administrative User Management |
|:---:|:---:|
| ![Pipelines](docs/assets/screenshots/data_pipelines.png) | ![Users](docs/assets/screenshots/admin_users.png) |

---

## Project Status

- **Phase 0 (Foundations)**: Complete (`master`)
- **Phase 1 (Data & Storage)**: Complete (`master`)
- **Phase 2 (Demand Forecasting)**: Complete (`master`)
- **Phase 3 (Customer Analytics)**: Complete (`master`)
- **Phase 4 (Inventory Optimization)**: Complete (`master`)
- **Phase 5 (FastAPI REST Backend & Auth)**: Complete (`master`)
- **Phase 6 (Streamlit Dashboard & UX)**: Complete (`master`)
  - Full WCAG 2.1 AA accessibility compliance audit: [reports/accessibility_audit.md](reports/accessibility_audit.md).

For detailed delivery progress, see [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) and [docs/PHASES.md](docs/PHASES.md).

---

## Quickstart

```bash
# Setup virtual environment and pre-commit hooks
make setup

# Start backing services (PostgreSQL, MLflow, Prefect)
make up

# Run checks
make lint typecheck test

# Start FastAPI API server
uv run uvicorn retailpulse.api.main:app --reload

# Start Streamlit Dashboard
uv run streamlit run src/retailpulse/dashboard/app.py
```

---

## Secrets Handling & Rotation

RetailPulse implements strict secrets isolation and defense-in-depth controls per `docs/SECURITY.md`:

### 1. File Permissions & Environment Isolation (SEC-CFG-01)
- Live configuration files (e.g. `.env`) must be restricted to read/write by the owner only:
  ```bash
  chmod 0600 .env
  ```
- `.env` is explicitly ignored by git. Never commit `.env` or plain-text secrets into repository history.
- In production (`RP_ENV=prod`), application startup **fails immediately** if any cryptographic secret is missing, shorter than 32 characters, or identical to development template values (`SEC-CFG-02`).

### 2. Secret Generation (SEC-CFG-01)
Generate high-entropy cryptographically secure tokens (48+ url-safe bytes) using Python's `secrets` module:
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 3. Secret Rotation Procedures (SECURITY §12)
- **JWT Secret (`RP_JWT_SECRET`) Rotation**:
  1. Generate a new 48-byte secret and update `.env` or deployment secret store.
  2. Restart the `api` service (`docker compose -f docker-compose.prod.yml restart api`).
  3. All existing user sessions will be invalidated immediately, requiring users to log in again.
- **Database Passwords (`RP_DB_APP_PASSWORD`, `RP_DB_OWNER_PASSWORD`) Rotation**:
  1. Execute `ALTER ROLE rp_app WITH PASSWORD 'new_password';` in PostgreSQL.
  2. Update `RP_DATABASE_URL` in `.env` and restart `api` and `worker` services.
- **Customer Pseudonymization Salt (`RP_PSEUDONYM_SALT`) Rotation (SEC-DATA-01)**:
  1. The salt is used for HMAC-SHA256 pseudonymization of source customer identifiers.
  2. Rotating `RP_PSEUDONYM_SALT` requires executing the customer re-key migration to recompute surrogate hashes across `core.dim_customer`, `core.fact_sales_line`, and `ml.churn_score`. Schedule during planned maintenance.
- **Leaked Secret Incident Response (SEC-CFG-05, SECURITY §12)**:
  - If a secret is leaked, immediately rotate the credential in production, audit `ops.audit_log` to inspect access history, and remove the secret from git history using a rewrite tool.


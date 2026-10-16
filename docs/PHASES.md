# RetailPulse — Delivery Phases

**Version:** 1.0 | **Related:** [PRD](PRD.md) · [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) · [TESTING](TESTING.md)

Ten phases, each ending in something that runs and can be demonstrated. Task-level detail is in IMPLEMENTATION_PLAN.md (task IDs `P<phase>-<nn>`).

**Effort** is in *focused working days* for one developer using Claude Code. Calendar time depends on availability: at about three focused days per week, expect roughly 12 to 15 weeks in total. Treat estimates as planning aids, and re-estimate after Phase 2.

---

## 1. Overview

| Phase | Name | Focus days | Depends on | Demo checkpoint |
|-------|------|-----------:|------------|-----------------|
| 0 | Foundations | 1–2 | none | `make up` runs Postgres, MLflow, Prefect; CI green |
| 1 | Data layer and ETL | 3–4 | 0 | Full dataset loads idempotently; data-quality report |
| 2 | Demand forecasting | 5–7 | 1 | Backtest report; champion forecasts written |
| 3 | Customer analytics | 4–5 | 1 | Segments and churn scores with reasons |
| 4 | Inventory optimization | 3–4 | 2 | Reorder list and policy simulation report |
| 5 | API and auth | 4–5 | 1 (2–4 for full data) | Every endpoint in API.md with RBAC tests |
| 6 | Dashboard | 5–6 | 5 | All pages in DESIGN.md working end to end |
| 7 | MLOps and orchestration | 4–5 | 2–6 | Scheduled flows run unattended in replay mode; drift and alerts |
| 8 | CI/CD and deployment | 3–4 | 5–7 | Deployed on a VM behind HTTPS; tagged release builds images |
| 9 | Hardening and documentation | 3–4 | all | Budgets met, security checklist done, README and results report |
| | **Total** | **35–46** | | |

Phases 2 and 3 are independent after Phase 1 and can be swapped. Phase 5 can start (against fixtures) once Phase 1 is done, but its prediction endpoints need Phases 2 to 4 data.

## 2. Phase details

### Phase 0 — Foundations
- **Goal:** a reproducible, linted, tested skeleton with the platform services running.
- **Deliverables:** `pyproject.toml` with tooling configs, `src/retailpulse` skeleton (`config`, `logging`, `errors`, `cli`), Makefile, pre-commit, Dockerfiles, `docker-compose.yml` (Postgres, MLflow, Prefect), `.env.example`, CI workflow (lint, types, unit tests).
- **Exit criteria:** `make setup && make up` succeeds from a clean clone; `make lint typecheck test` pass; CI is green; MLflow UI reachable.

### Phase 1 — Data layer and ETL
- **Goal:** trustworthy, idempotent data in the star schema.
- **Deliverables:** Alembic migrations 0001–0004 (and the `ops` base tables), SQLAlchemy models, fixture generator and mini dataset, Online Retail II adapter, pandera schemas, cleaning rules (TRD §4.3), loaders, `fact_daily_sales` aggregation, ABC/XYZ classification, data-quality results, CLI `etl run`.
- **Exit criteria:** migrations up/down/up cycle passes; loading the fixture twice yields identical counts; the full dataset loads in under 10 minutes; row counts reconcile at each cleaning stage (TRD §4.3) and Σ `units_sold` equals Σ positive non-outlier product-line quantity on complete days; data-quality results are written for every batch.

### Phase 2 — Demand forecasting
- **Goal:** compare baselines and ML models honestly and publish champion forecasts.
- **Deliverables:** series builder with leakage-safe features, metrics module, `Forecaster` interface, four baselines, Prophet, LightGBM (direct strategy), LSTM, backtest harness, residual-based intervals, MLflow integration and `ml.model_run` mirror, champion gate, batch forecasts into `ml.forecast`, `reports/forecast_backtest.md`.
- **Exit criteria:** metrics unit-tested against hand-computed values; leakage test passes; a gate-passing model exists or the "no model beats baseline" path is demonstrated; interval coverage reported; runtime within TRD budgets for `RP_FORECAST_TOP_N=300` (or documented).

### Phase 3 — Customer analytics
- **Goal:** actionable segments and explainable churn scores.
- **Deliverables:** RFM builder, K-Means segmentation with label mapping and stability check, churn labels and features with purged temporal splits, logistic regression and LightGBM, calibration, SHAP reasons, batch scoring into `ml.*` tables, revenue-at-risk, `reports/churn_and_segments.md`.
- **Exit criteria:** leakage tests pass; LightGBM compared against the baseline on the final holdout; every scored customer has probability, band and three reasons; segment profiles sum to 100% of customers and revenue share.

### Phase 4 — Inventory optimization
- **Goal:** forecast-driven reorder recommendations and evidence of benefit.
- **Deliverables:** synthetic parameter and snapshot generator, pure policy functions with property tests, recommendation job, simulator, `reports/inventory_simulation.md` (with the synthetic-data caveat prominent).
- **Exit criteria:** policy properties hold (monotone in service level, non-negative, ROP ≥ lead-time demand); simulation reproducible by seed; recommendations written for all eligible SKUs.

### Phase 5 — API and auth
- **Goal:** secure, documented, tested REST API.
- **Deliverables:** app factory and middleware, RFC 9457 errors, auth (Argon2id, JWT, lockout), RBAC dependency, all routers in API.md, rate limiting, pagination and sort helpers, `/health` and `/metrics`, `PipelineTrigger` abstraction, CSV exports with formula-safe cells.
- **Exit criteria:** RBAC matrix test (every endpoint × every role) passes; `api/` coverage ≥ 85%; OpenAPI matches API.md; no endpoint returns an ORM object or `password_hash`.

### Phase 6 — Dashboard
- **Goal:** the nine screens in DESIGN.md, usable by each role.
- **Deliverables:** API client, login and session handling, role-based navigation, all pages, shared components and chart helpers, loading/empty/error states, AppTest smoke tests.
- **Exit criteria:** journeys J2–J9 in APP_FLOW.md can be completed by hand; each page has an AppTest smoke test using a stub client; accessibility checklist in DESIGN.md §8 ticked.

### Phase 7 — MLOps and orchestration
- **Goal:** the system operates itself and tells you when it is unhealthy.
- **Deliverables:** Prefect flows and schedules, `worker` service, pipeline-run bookkeeping, promotion and rollback service, Evidently drift, realized-error tracking, alert rules and optional webhook, replay mode, retention job.
- **Exit criteria:** in replay mode, 14 simulated days run unattended with ≥ 95% flow success; a forced failure produces an alert and leaves served outputs unchanged; promotion and rollback work end to end.

### Phase 8 — CI/CD and deployment
- **Goal:** one-command, repeatable deployment.
- **Deliverables:** full CI (integration tests with Postgres, coverage gate), security workflow (bandit, pip-audit, gitleaks, image scan), dependency update config, CD on tag (build and push images, deploy, smoke test), `docker-compose.prod.yml` with Caddy, internal-only platform services, backups.
- **Exit criteria:** tagging a release deploys to the VM; only ports 80 and 443 are reachable externally; post-deploy smoke test passes; a backup can be restored in a test.

### Phase 9 — Hardening and documentation
- **Goal:** meet the budgets, close the security checklist, and tell the story.
- **Deliverables:** profiling and index fixes, load test report, security checklist walk-through, README (five-command quick start, architecture diagram, screenshots), `reports/results.md`, demo script and recording.
- **Exit criteria:** every budget in TRD §7 met or documented with a reason; SECURITY.md checklist ticked; PRD §11 release criteria all checked.

## 3. Cut list if time runs short (drop from the bottom first)

1. K8s, Kafka, any stretch item (never started before Phase 9).
2. Admin user-management page (use the CLI).
3. LSTM (keep Prophet and LightGBM; state the reason in the report).
4. Policy simulator UI (keep the stored report).
5. Webhook alerts (keep in-app alerts).
6. Prophet (keep baselines and LightGBM).

Never cut: leakage tests, champion gate, RBAC tests, idempotent loading, synthetic-data labeling, security checklist.

## 4. Stretch backlog (after Phase 9)

- Kubernetes manifests with kustomize, test on kind; HPA for the API; CronJobs for flows.
- M5 (Walmart) adapter for multi-store forecasting.
- Per-ABC-class champions; quantile LightGBM; Temporal Fusion Transformer.
- Empirical-quantile stockout risk for intermittent SKUs.
- Slack or email alerts; refresh tokens; dark theme.
- Kafka-based micro-batch ingestion; feature store.
- Customer lifetime value model.

## 5. Phase gate checklist (run at the end of every phase)

- [ ] All tasks for the phase ticked in IMPLEMENTATION_PLAN.md with their verify commands passing.
- [ ] `make lint typecheck test` and CI green; coverage not lower than before.
- [ ] Docs changed where contracts changed (DATABASE, API, TRD).
- [ ] No TODO without a linked issue; no secrets in the diff.
- [ ] A short demo note (what works, what does not) added to the PR or `reports/`.
- [ ] Tag `phase-N-complete`.

# RetailPulse — Implementation Plan

**Version:** 1.0 | **Related:** [PHASES](PHASES.md) · [PRD](PRD.md) · [TRD](TRD.md) · [TESTING](TESTING.md) · [CODE_STYLE](CODE_STYLE.md)

This is the build checklist. Work top to bottom, one task at a time. Tick a box only when the task's **Verify** step passes. Task IDs are `P<phase>-<nn>`.

---

## 1. Working agreement (for Claude Code and for you)

**Before any task**
1. Read `CLAUDE.md`, then the sections this task cites (do not read every doc every time).
2. If tasks touch more than ~5 files or change a contract (DB schema, API, config), state a short plan first and wait for approval.

**While working**
3. Write tests alongside the code (tests first for pure logic).
4. Follow `docs/CODE_STYLE.md`. Keep functions small and typed.
5. One task per commit (or a few small commits). Conventional Commit format, task ID in the message: `feat(forecasting): add LightGBM forecaster (P2-06)`.

**After the task**
6. Run the task's Verify command plus `make lint typecheck test`. Show a short summary of results (failures in full, passes in one line).
7. Tick the checkbox in this file and commit.

**Stop and ask before**
- Adding or upgrading a dependency.
- Changing the database schema, API contracts, config variables or security settings.
- Deleting files, rewriting history, or running anything that costs money or calls an external service.
- Exposing a network port or weakening auth, CORS, TLS or validation.
- Any place where two docs disagree. Precedence if you must proceed: SECURITY → DATABASE/API → TRD → ARCHITECTURE → PRD. Record the decision in `docs/DECISIONS.md` (date, decision, reason).

**Never**
- Use `date.today()` or `datetime.now()` inside ML, feature or ETL logic (use `as_of`).
- Read environment variables outside `config.py`.
- Commit secrets, `.env`, datasets, model artifacts or MLflow data.
- Disable a failing test or lint rule to get green; fix the cause or ask.

### Definition of Done (every task)
- [ ] Acceptance criteria and Verify command satisfied.
- [ ] Tests added or updated; coverage not lower.
- [ ] Types and lint clean; no `# type: ignore` or `noqa` without a reason comment.
- [ ] Docs updated if a contract changed.
- [ ] No secrets, no TODO without an issue reference.

### Prompt templates

**Kickoff**
> Read CLAUDE.md, then docs/IMPLEMENTATION_PLAN.md and docs/PHASES.md. We are starting Phase 0. Summarize your plan for P0-01 to P0-03 in about 10 lines and wait for my OK. Then implement task by task: after each, run its Verify command, tick the box in docs/IMPLEMENTATION_PLAN.md, and commit.

**One task**
> Implement P2-06 from docs/IMPLEMENTATION_PLAN.md. First read docs/TRD.md §5.1 and docs/CODE_STYLE.md. Write tests first, then the code. Run the Verify command and show a summary.

**Resume**
> Run `git log --oneline -10`, open docs/IMPLEMENTATION_PLAN.md and tell me the next unchecked task and what it depends on.

**Review before merging a phase**
> Review the diff for Phase N against docs/PHASES.md exit criteria, docs/SECURITY.md checklist and docs/TESTING.md gates. List gaps as a checklist; do not fix anything yet.

### Using Claude Code efficiently
- One phase per session; start a fresh session for the next phase instead of carrying a long history.
- Point to exact docs and sections instead of asking it to read everything.
- Keep `CLAUDE.md` short; put detail in `docs/`.
- Ask for a plan before large changes. Avoid open-ended "improve everything" prompts.
- Develop on the fixture and `RP_FORECAST_TOP_N=30`; run the full dataset only at phase checkpoints.
- Ask for test output as a summary (failures only), and never paste large data into the chat; point to files.
- Commit often so any step can be reverted.

---

## 2. Task list

### Phase 0 — Foundations
- [x] **P0-01 Repo and tooling.** `pyproject.toml` (uv dependency groups `core`, `ml`, `api`, `dashboard`, `dev`; ruff, mypy, pytest, coverage settings from CODE_STYLE and TESTING), `.gitignore`, `.editorconfig`, `.pre-commit-config.yaml` (ruff, ruff-format, gitleaks, large-file check), README stub. *Verify:* `uv sync && uv run pre-commit run --all-files`.
- [x] **P0-02 Package skeleton.** `src/retailpulse/{__init__,config,logging,errors,timeutil}.py`. `Settings` implements TRD §3 (prod requires secrets ≥ 32 bytes); structlog JSON with `request_id` contextvar; exception hierarchy; `timeutil.utc_now()` as the only clock access. *Verify:* `pytest tests/unit/test_config.py`.
- [x] **P0-03 Makefile.** Targets in Appendix A; `make help` lists them. *Verify:* `make help && make lint typecheck test`.
- [x] **P0-04 CLI skeleton.** Typer app `retailpulse` with `version`; entry point in `pyproject.toml`. *Verify:* `uv run retailpulse --help`.
- [x] **P0-05 Dockerfiles.** `infra/docker/{api,dashboard,worker,mlflow}.Dockerfile`: multi-stage, non-root user, pinned base tags, healthchecks; API and dashboard images exclude heavy ML groups. *Verify:* `docker build` each; `docker run --rm <img> id` shows non-root.
- [x] **P0-06 Compose.** `docker-compose.yml` with `postgres` (init script creates DBs `retailpulse`, `mlflow`, `prefect` and roles per DATABASE §10), `mlflow`, `prefect`; healthchecks, named volumes, internal network; `.env.example`. *Verify:* `make up` then `curl -fs localhost:5000` and Prefect health.
- [x] **P0-07 CI skeleton.** `.github/workflows/ci.yml`: uv cache, lint, types, unit tests. *Verify:* green run on a branch.
- [x] **P0-08 Import contracts.** `import-linter` config enforcing ARCHITECTURE §4 plus the clock rule (`ml`, `features`, `etl` may not import `timeutil`); wired into `make lint`. *Verify:* a deliberate forbidden import fails `make lint`.

### Phase 1 — Data layer and ETL
- [x] **P1-01 Fixture.** `scripts/make_fixture.py` creates `tests/fixtures/mini_retail.csv` (≈3k rows, deterministic): returns, non-product SKUs (plus a real-looking `DCGS*` product), null customers, exact duplicates, a cross-sheet overlap, mixed number/text codes, case-variant SKUs, an outlier order, a cancellation with positive quantity, a partial last day, bad rows. *Verify:* running twice yields identical bytes.
- [x] **P1-02 Models and migrations.** SQLAlchemy models with naming convention; Alembic revisions 0001–0004 per DATABASE.md §9. *Verify:* `tests/integration/test_migrations.py` (up, down, up on empty DB).
- [x] **P1-03 DB session and repositories.** `db/session.py`, repository helpers, `retailpulse db check`. *Verify:* integration test connects and reports head revision.
- [x] **P1-04 Calendar.** `ensure_dim_date` (2009-01-01 to 2013-12-31, GB holidays), `dim_store` seed, trading-day and closure-window derivation and the partial-last-day flag `is_complete` (TRD §4.4). *Verify:* unit tests for holidays, closure detection and the partial-day rule.
- [x] **P1-05 Adapters.** `ingestion/adapters/online_retail_ii.py` (CSV and XLSX, all sheets) and `generic_csv.py` produce the canonical frame (TRD §4.2); the Online Retail II adapter accepts both column-name variants (TRD §4.1) reads `Invoice` and `StockCode` as text, and fails clearly on missing columns. *Verify:* unit tests on the fixture including both naming variants; XLSX test uses a tiny generated workbook.
- [x] **P1-06 Validation and quality.** pandera schemas and checks in `etl/quality.py`, including the batch gate (`MAX_REJECT_RATIO`), the outlier flag and the rejection of cancellation lines with positive quantity. *Verify:* one test per rule in TRD §4.3.
- [x] **P1-07 Cleaning.** `etl/clean.py` rules 1–7 and 9 (TRD §4.3); `security/pseudonymize.py` (HMAC-SHA256); `source_row_hash`. *Verify:* unit tests; raw customer IDs never appear in outputs.
- [x] **P1-08 Loaders.** `ingestion/loaders.py`: dimension upserts, chunked `fact_sales_line` insert with `ON CONFLICT DO NOTHING`, batch and DQ rows. *Verify:* loading the fixture twice gives identical row counts.
- [x] **P1-09 Aggregation.** `etl/aggregate.py`: `fact_daily_sales` for touched dates (excluding outlier lines and incomplete days), ABC/XYZ, first/last dates. *Verify:* reconciliation test (Σ `units_sold` = Σ positive, non-outlier product-line quantity on complete days).
- [x] **P1-10 ETL CLI and data guide.** `retailpulse etl run --source --path`; `data/README.md` (how to obtain the dataset from the UCI page named in TRD §4.1, license and citation, optional Parquet conversion). *Verify:* full-dataset load < 10 min; row counts in `reports/etl_benchmark.md` reconcile: 1,067,371 read → 1,044,848 after the cross-sheet overlap → 1,033,036 after exact duplicates.

### Phase 2 — Demand forecasting
- [x] **P2-01 Series builder.** `features/forecast_features.py`: top-N selection, eligibility, zero-filled trading-day series, leakage-safe lag/rolling/calendar features. *Verify:* leakage test (features at origin use no later data).
- [x] **P2-02 Metrics.** WAPE, MASE, RMSSE, bias, sMAPE, coverage, pinball with zero-scale handling. *Verify:* tests against hand-computed values.
- [x] **P2-03 Interface and baselines.** `Forecaster` protocol; naive, seasonal naive, moving average, Croston-SBA. *Verify:* unit tests on synthetic series with known answers.
- [x] **P2-04 Backtest harness.** Rolling-origin, model-agnostic, asserts training data < origin. *Verify:* test where seasonal naive is perfect on a weekly-periodic series.
- [x] **P2-05 Intervals.** Residual-quantile intervals per (model, ABC, horizon bucket). *Verify:* coverage within tolerance on synthetic noise.
- [x] **P2-06 LightGBM forecaster.** Direct strategy per TRD §5.1; compare `tweedie` vs `regression`; deterministic settings. *Verify:* beats naive on a synthetic trending-seasonal dataset; same seed gives same output.
- [x] **P2-07 Prophet forecaster.** Isolated in `prophet_model.py`; parallel fit; GB holidays; fallback to Croston-SBA for ineligible series. *Verify:* smoke test on 3 series (marked `slow`).
- [x] **P2-08 LSTM forecaster.** PyTorch, windowing, scaling, early stopping, fixed seeds, CPU. *Verify:* tiny-data test (loss decreases; deterministic).
- [x] **P2-09 MLflow wrapper.** `ml/registry.py`, `ml/runs.py`: log params, metrics, artifacts and tags; register models; alias helpers; mirror to `ml.model_run` and `ml.forecast_metric`. *Verify:* integration test with a temp file-store MLflow.
- [x] **P2-10 Training pipeline.** `forecasting/train.py`: run all models, aggregate metrics, apply the champion gate; CLI `retailpulse train forecast`. *Verify:* run on the fixture; a deliberately bad model is rejected by the gate (test).
- [x] **P2-11 Batch forecasting.** `forecasting/predict.py`: champion (or best baseline), intervals, write `ml.forecast` and move `ml.active_output` in one transaction; CLI `retailpulse score forecast`. *Verify:* integration test; failed run leaves the pointer unchanged.
- [x] **P2-12 Backtest report.** `scripts/report_forecast.py` writes `reports/forecast_backtest.md` with tables and plots and an honest discussion. *Verify:* file generated from a real run.

### Phase 3 — Customer analytics
- [x] **P3-01 RFM.** `features/rfm.py` with `as_of`. *Verify:* hand-checked tiny dataset.
- [x] **P3-02 Segmentation.** Training, k selection, deterministic label mapping, profiles, stability (ARI); MLflow; CLI `train segmentation` and `score segmentation`. *Verify:* profiles cover 100% of customers; labels deterministic across runs.
- [x] **P3-03 Churn data.** Cutoffs, labels, features (`features/churn_features.py`, `ml/churn/labels.py`) and the purged split. *Verify:* leakage tests (features ≤ T; label window (T, T+90]; no overlap across split boundary).
- [x] **P3-04 Churn models.** Logistic baseline, LightGBM, calibration, metrics, gate (must beat baseline PR-AUC on the test cutoff); CLI `train churn`. *Verify:* metrics logged; calibration artifact saved.
- [x] **P3-05 Explanations.** `ml/churn/explain.py` top-3 reasons; global importance artifact. *Verify:* every scored row has three reasons with stable feature keys.
- [x] **P3-06 Churn scoring.** `ml.churn_score`, bands, active pointer; CLI `score churn`. *Verify:* integration test; bands follow thresholds.
- [x] **P3-07 Revenue at risk and report.** Formula from TRD §5.3; `reports/churn_and_segments.md`. *Verify:* unit test on known inputs.

### Phase 4 — Inventory optimization
- [x] **P4-01 Synthetic parameters.** `ml/inventory/params.py` and `retailpulse inventory seed-params` (deterministic; `source='synthetic'`), plus snapshots. *Verify:* same seed gives identical rows.
- [x] **P4-02 Policy functions.** Pure functions for SS, ROP, EOQ, stockout risk, order quantity with MOQ rounding. *Verify:* Hypothesis tests: non-negative, ROP ≥ μ_L, monotone in service level.
- [x] **P4-03 Recommendations.** `recommend.py` using forecasts, σ_e from backtest residuals, params and snapshots; statuses; active pointer; CLI `score inventory`. *Verify:* integration test covering all four statuses.
- [x] **P4-04 Simulator.** `simulate.py`: lost-sales replay, seeded lead times, baseline policy; stores `ml.policy_simulation`. *Verify:* reproducible by seed; infinite-stock policy has fill rate 1.0 (test).
- [ ] **P4-05 Simulation report.** `retailpulse inventory simulate` and `reports/inventory_simulation.md` with the synthetic-data caveat on top. *Verify:* report generated.

### Phase 5 — API and auth
- [ ] **P5-01 App factory.** Settings, lifespan, middleware (request ID, security headers, timing), RFC 9457 error handlers, `/health/*`. *Verify:* tests for error shape and headers.
- [ ] **P5-02 Auth.** Argon2id hashing, JWT (fixed algorithm, `exp`, `iat`, `sub`, `role`), lockout, `/auth/*`, `require_role`, `retailpulse users` CLI. *Verify:* tests for expiry, tampering, wrong algorithm, lockout, generic failure message.
- [ ] **P5-03 Shared helpers.** Pagination, sort whitelist, envelope, `security/csv_safety.py`. *Verify:* unit tests including formula-injection cases.
- [ ] **P5-04 Reference, KPI and forecast routers.** *Verify:* integration tests against seeded DB; 503 before first scoring.
- [ ] **P5-05 Customer routers.** Segments, customers, churn summary, export. *Verify:* filters and exports tested; no raw identifiers.
- [ ] **P5-06 Inventory router.** Params, what-if, simulate (limit 200), recommendations, exports. *Verify:* what-if never persists (test).
- [ ] **P5-07 Models, monitoring and alerts routers.** Reads now; promote and train call services (full flows in Phase 7). *Verify:* RBAC tests.
- [ ] **P5-08 Uploads and pipelines routers.** Upload validation (size, type, magic bytes, headers), `PipelineTrigger` abstraction with Noop and Inline implementations. *Verify:* duplicate-file and 409 tests.
- [ ] **P5-09 Rate limiting, metrics, docs toggle.** slowapi limits, `/metrics`, OpenAPI tags and examples, `RP_ENABLE_DOCS`. *Verify:* 429 test; docs disabled in prod config test.
- [ ] **P5-10 API test suite.** RBAC matrix (every endpoint × role), pagination bounds, error shapes, sensitive-field scan. *Verify:* `pytest tests/integration/api` and `api/` coverage ≥ 85%.

### Phase 6 — Dashboard
- [ ] **P6-01 Scaffold.** `.streamlit/config.toml`, `theme.py` (palette, `style_figure`), `client.py` (timeouts, retries, problem mapping), login and session state, role-based `st.navigation`, status strip. *Verify:* AppTest: login success and failure; viewer sees no Admin page.
- [ ] **P6-02 Components.** KPI card, badges, server-paged table, loading/empty/error helpers, confirm dialog, chart helpers. *Verify:* component tests.
- [ ] **P6-03 Overview.** *Verify:* AppTest with stub client.
- [ ] **P6-04 Demand Forecast.** *Verify:* AppTest; ineligible SKU shows the reason.
- [ ] **P6-05 Segments.** *Verify:* AppTest.
- [ ] **P6-06 Churn Risk.** Filters, reasons chart, export (analyst+). *Verify:* AppTest; viewer has no export button.
- [ ] **P6-07 Inventory.** Table, what-if, save with confirmation, simulation, export, synthetic notice. *Verify:* AppTest; notice present when `params_source = synthetic`.
- [ ] **P6-08 Monitoring.** Models, performance, drift, alerts; admin-only actions. *Verify:* AppTest per role.
- [ ] **P6-09 Data and Pipelines.** Upload, status polling, triggers. *Verify:* AppTest with stub upload.
- [ ] **P6-10 Admin Users.** *Verify:* AppTest; last-admin protection message.
- [ ] **P6-11 Polish.** Accessibility checklist (DESIGN §8), screenshots for README. *Verify:* checklist ticked in the PR.

### Phase 7 — MLOps and orchestration
- [ ] **P7-01 Flows.** `pipelines/flows.py`: `etl`, `score_all`, `train_forecast`, `train_segmentation`, `train_churn`, `drift_check` wrapping existing functions, with `ops.pipeline_run` bookkeeping, retries, timeouts. *Verify:* tests call the underlying functions; failure marks the run `failed` and creates an alert.
- [ ] **P7-02 Worker and schedules.** Served deployments with the schedules in ARCHITECTURE §6; `PrefectTrigger`; compose wiring. *Verify:* triggering `etl` from the API completes end to end in compose.
- [ ] **P7-03 Promotion and rollback.** Service moves alias, flips `is_champion`, audit-logs, triggers `score_all`; refuses runs that failed the gate. *Verify:* tests for promote, rollback, refusal.
- [ ] **P7-04 Drift.** `monitoring/drift.py` Evidently wrapper; data and prediction drift; HTML plus summary row. *Verify:* contract test of the wrapper on the fixture.
- [ ] **P7-05 Performance monitoring.** Realized WAPE for elapsed forecasts; churn PR-AUC after the outcome window; `ops.model_performance`. *Verify:* test with injected actuals.
- [ ] **P7-06 Alerts.** Rules and dedupe in `monitoring/alerts.py`; optional webhook (operator-set URL, timeout, no redirects). *Verify:* dedupe and webhook tests.
- [ ] **P7-07 Replay mode.** `retailpulse replay start --holdback-days 60` and `step`; runs 14 simulated days. *Verify:* `reports/replay_run.md` shows ≥ 95% flow success.
- [ ] **P7-08 Retention and auto-promote.** Retention job per DATABASE §8; `RP_AUTO_PROMOTE` (default false) with tests. *Verify:* pruning keeps the active pointer's rows.

### Phase 8 — CI/CD and deployment
- [ ] **P8-01 Full CI.** Postgres service, integration tests, coverage gate 80%, concurrency cancel, caching. *Verify:* green PR; failing coverage blocks.
- [ ] **P8-02 Security workflow.** bandit, pip-audit, gitleaks, image scan; dependency update config; Actions pinned to commit SHAs. *Verify:* workflow runs; a seeded fake secret is caught on a test branch.
- [ ] **P8-03 Production compose and proxy.** `docker-compose.prod.yml`, `Caddyfile`: only 80/443 published; `/metrics` and docs blocked; security headers; restart policies, resource limits. *Verify:* port scan shows only 80/443.
- [ ] **P8-04 CD.** On tag: build and push images, deploy over SSH (compose pull and up), smoke test (`/health/ready`, login, one read). *Verify:* tagged release deploys.
- [ ] **P8-05 Backups.** Nightly `pg_dump`, retention, restore script. *Verify:* restore into a scratch database succeeds.
- [ ] **P8-06 Secrets handling.** Env files `0600`, complete `.env.example`, rotation notes in README. *Verify:* checklist in SECURITY §11.
- [ ] **P8-07 (Stretch) Kubernetes.** kustomize manifests tested on kind. *Verify:* all pods ready; smoke test passes.

### Phase 9 — Hardening and documentation
- [ ] **P9-01 Profiling pass.** ETL and the slowest API queries (`EXPLAIN (ANALYZE, BUFFERS)`); indexes with evidence. *Verify:* before/after table in `reports/performance.md`.
- [ ] **P9-02 Load test.** k6 or Locust scenarios (50 concurrent users). *Verify:* p95 numbers vs TRD §7 in `reports/performance.md`.
- [ ] **P9-03 Security walk-through.** Complete SECURITY.md §11 checklist; document exceptions. *Verify:* ticked checklist committed.
- [ ] **P9-04 README.** Five-command quick start, architecture diagram, screenshots, limitations, dataset citation and license, test instructions. *Verify:* a fresh clone follows it successfully.
- [ ] **P9-05 Results report.** `reports/results.md` consolidating forecast, segments, churn, inventory, replay, limitations and reproduction steps. *Verify:* every number traceable to an MLflow run or report file.
- [ ] **P9-06 Release.** Demo script and recording; tag `v1.0.0`. *Verify:* PRD §11 checklist ticked.

---

## Appendix A — Make targets

`help`, `setup` (uv sync, pre-commit install), `up`, `down`, `logs`, `migrate` (alembic upgrade head), `seed` (admin prompt, inventory params), `etl`, `train`, `score`, `api` (uvicorn with reload), `dashboard` (streamlit), `lint` (ruff check, ruff format --check, import-linter), `fmt`, `typecheck` (mypy), `test` (unit + integration), `test-unit`, `test-int`, `e2e`, `cov`, `security` (bandit, pip-audit, gitleaks), `clean`.

## Appendix B — CLI commands (`retailpulse`)

```
version
db check
users create | list | deactivate
etl run --source online_retail_ii --path PATH
inventory seed-params | simulate
train forecast | segmentation | churn   [--algorithms a,b] [--top-n N]
score forecast | segmentation | churn | inventory | all
monitor drift | performance
replay start --holdback-days 60 | step
```

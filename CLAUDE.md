# RetailPulse

AI-powered customer analytics and demand forecasting platform for retail: demand forecasting, customer
segmentation, churn prediction and inventory optimization, served through a FastAPI API and a Streamlit
dashboard, with MLflow, Prefect, Evidently and PostgreSQL. Python 3.11, managed with `uv`.

Primary dataset: UCI *Online Retail II*. Inventory data is **synthetic** (always labeled as such).

## Current status
Progress lives in the checkboxes of `docs/IMPLEMENTATION_PLAN.md`. Find the first unchecked task; that is the
next piece of work. Phases and exit criteria are in `docs/PHASES.md`.

## Docs map (open only what the task cites; this file does not import them to save context)
| Need | Open |
|------|------|
| Requirements, scope, metrics | `docs/PRD.md` |
| Stack, config variables, data rules, ML formulas, budgets | `docs/TRD.md` |
| Module boundaries, diagrams, ADRs, schedules | `docs/ARCHITECTURE.md` |
| Tables, constraints, views, migrations | `docs/DATABASE.md` |
| Endpoints, errors, roles, examples | `docs/API.md` |
| UI tokens, pages, components, states | `docs/DESIGN.md`, `docs/APP_FLOW.md` |
| Tests and gates | `docs/TESTING.md` |
| Security controls and checklist | `docs/SECURITY.md` |
| Style, git, review checklist | `docs/CODE_STYLE.md` |

If docs conflict: stop and ask. Precedence: SECURITY > DATABASE/API > TRD > ARCHITECTURE > PRD.
Record decisions in `docs/DECISIONS.md` (date, decision, reason).

## Commands
```
make setup        # uv sync + pre-commit install
make up / down    # compose: postgres, mlflow, prefect (dev)
make migrate      # alembic upgrade head
make seed         # create admin (prompt) + synthetic inventory parameters
make etl          # load dataset      make train / score   # train and score models
make api          # uvicorn --reload   make dashboard       # streamlit
make lint | fmt | typecheck | test | test-unit | test-int | e2e | cov | security
```
CLI: `uv run retailpulse --help` (commands listed in `docs/IMPLEMENTATION_PLAN.md`, Appendix B).

## Layout (details in `docs/ARCHITECTURE.md` §4)
`src/retailpulse/{config,logging,errors,timeutil,cli}.py`, `db/`, `security/`, `ingestion/`, `etl/`, `features/`,
`ml/{forecasting,segmentation,churn,inventory}/`, `monitoring/`, `pipelines/`, `api/`, `dashboard/`;
`tests/{unit,integration,e2e,fixtures}`, `migrations/`, `infra/`, `reports/`, `scripts/`.

## Workflow (every task)
1. Read the task in `docs/IMPLEMENTATION_PLAN.md` and the doc sections it cites.
2. If it touches more than ~5 files or changes a contract (DB, API, config, security), propose a short plan and wait.
3. Write tests with the code (tests first for pure logic). Keep the diff small and focused.
4. Run the task's **Verify** command and `make lint typecheck test`. Summarize results (full output only for failures).
5. Tick the checkbox, commit: `type(scope): subject (P2-06)` (Conventional Commits).

## Ask first
Adding or upgrading dependencies; schema, API, config or security changes; deleting files; anything that costs
money or calls an external service; exposing ports; any doc conflict.

## Hard rules
- **No clock in ML/ETL/features:** never `date.today()` or `datetime.now()`; pass `as_of`. `data_as_of` is the
  latest date in the data. Only `timeutil.py` reads the clock.
- **Env vars only in `config.py`** (`RP_` prefix, `SecretStr` for secrets). No default or hard-coded credentials.
- **Time-based splits, no leakage;** fit transforms on training data only; seeds from `Settings.seed`.
- **Champion gate:** promote only models that beat seasonal naive on WAPE; no auto-promote unless `RP_AUTO_PROMOTE=true`.
- **Light images:** heavy libraries (torch, prophet, lightgbm, shap, evidently, mlflow, prefect) are imported lazily and
  never in `__init__.py`; `api/` and `dashboard/` must not import model-training code.
- **API:** `response_model` on every endpoint, `require_role(...)` on every route, RFC 9457 errors, reads via `ml.v_active_*`.
- **Dashboard** talks only to the API; never `unsafe_allow_html` with API strings.
- **Security:** pseudonymize customers (HMAC) at ingestion; validate all input; no `eval`/`pickle.loads`/`shell=True` on
  external data; never log secrets, tokens or `customer_key`; never publish Postgres/MLflow/Prefect ports in prod.
- **Tests:** never weaken or skip a failing test to get green; real Postgres in integration tests.

## Gotchas
- Prophet may fail to install natively on Windows/ARM: run it in Docker. On Windows, develop in WSL2.
- Newer MLflow releases can reject requests by Host/CORS: add the compose service name to the allowed hosts
  (check the docs of the pinned version).
- Evidently, Prefect, MLflow and Streamlit APIs change between versions: pin, read the docs for the installed
  version, and keep each behind its one wrapper module.
- Online Retail II (checked on the real file): one XLSX, two sheets, 1,067,371 rows; convert once to Parquet. The sheets overlap on 2010-12-01 to 2010-12-09 (22,523 duplicate rows). Read `Invoice` and `StockCode` as text; `Customer ID` is numeric with blanks (22.8% missing), so canonicalize before hashing. 173 SKUs differ only by case; `DCGS*` codes are real products. The last day (2011-12-09) is partial, so `data_as_of` is 2011-12-08. No Saturday trading; closed about 24 Dec to 3 Jan. Two order-then-cancel lines (80,995 and 74,215 units) must be flagged as outliers. Details: TRD §4.
- Develop on `tests/fixtures/mini_retail.csv` and `RP_FORECAST_TOP_N=30`; run the full dataset at phase checkpoints.
- Streamlit `st.cache_data` is global across sessions: include `role` in cache keys; never cache admin or user-specific data.

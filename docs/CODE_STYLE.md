# RetailPulse — Code Style and Conventions

**Version:** 1.0 | **Related:** [TESTING](TESTING.md) · [ARCHITECTURE](ARCHITECTURE.md) · [SECURITY](SECURITY.md) · [IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md)

Tools enforce what they can (ruff, mypy, import-linter, pre-commit); this file covers the rest. When in doubt, choose the simplest readable option. Consistency beats cleverness.

---

## 1. Tooling

| Tool | Purpose | Command |
|------|---------|---------|
| ruff | lint + format (replaces black, isort, flake8, pyupgrade) | `make lint`, `make fmt` |
| mypy | static types, strict on `src/` | `make typecheck` |
| import-linter | architectural boundaries (ARCHITECTURE §4) | part of `make lint` |
| pre-commit | runs the above plus gitleaks on commit | `make setup` installs it |

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "RUF", "S", "PD", "NPY",
          "PL", "T20", "DTZ", "PT", "ARG", "ANN", "TID", "ERA"]
ignore = ["ANN401", "PLR0913", "PLR2004", "PD901"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "ANN", "ARG", "PLR2004", "S105", "S106"]
"scripts/**" = ["T20"]
"migrations/**" = ["N999", "ANN"]
"src/retailpulse/timeutil.py" = ["TID251"]

[tool.ruff.lint.flake8-tidy-imports]
ban-relative-imports = "all"

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"datetime.datetime.now".msg = "Use retailpulse.timeutil.utc_now() (system code only) or pass as_of."
"datetime.datetime.utcnow".msg = "Use retailpulse.timeutil.utc_now()."
"datetime.date.today".msg = "ML, feature and ETL logic must take as_of; never read the clock."

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["prophet.*", "lightgbm.*", "shap.*", "mlflow.*", "evidently.*", "plotly.*",
          "streamlit.*", "holidays.*", "sklearn.*", "scipy.*", "statsmodels.*", "torch.*",
          "pandera.*", "slowapi.*", "prefect.*"]
ignore_missing_imports = true
```

`retailpulse/timeutil.py` is the **only** place that reads the clock (`utc_now()`). Add an import-linter contract so `ml`, `features` and `etl` cannot import it.

## 2. Python conventions

**Naming**
- Modules and functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`; private helpers `_leading_underscore`.
- Functions are verbs (`build_features`, `load_batch`); booleans read as questions (`is_return`, `has_champion`).
- Put units in names: `lead_time_days`, `horizon_days`, `revenue` (currency comes from settings).
- DataFrame columns and DB columns: `snake_case`, same names in both.

**Imports**
- Absolute imports only (`from retailpulse.etl import clean`); no wildcard imports; ruff sorts them.
- Heavy libraries (torch, prophet, lightgbm, shap, evidently, mlflow, prefect) are imported **inside** the module that needs them or inside functions, never in `__init__.py` files (light-image rule, ARCHITECTURE §4).

**Typing**
- Every public function and method is fully annotated. Use built-in generics (`list[str]`, `dict[str, float]`) and `X | None`.
- Use `Protocol` for interfaces (`Forecaster`), frozen `dataclass` for internal records and config objects, Pydantic models at I/O boundaries (API, settings, files), `Literal` for small closed sets.
- No `Any` unless unavoidable; no `# type: ignore` without a comment explaining why.

**Docstrings**
- Google style. Required for public modules, classes and functions; first line is an imperative summary. Include Args, Returns and Raises when not obvious. Do not restate the code; explain intent, units and edge cases.

**Functions and structure**
- Small and single-purpose (typically ≤ 40 lines, ≤ 5 parameters; group related parameters in a dataclass).
- Prefer pure functions. Put I/O (database, MLflow, files, network) in thin functions at the edges so logic is testable without them.
- No mutable default arguments; no module-level mutable state; no hidden globals. Settings are passed in or obtained once via `get_settings()` at the edge.
- Return values over mutation. Early returns over deep nesting. No dead code, commented-out code or `print`.

**Errors**
- Domain exceptions live in `errors.py`: `RetailPulseError` → `ConfigError`, `DomainValidationError`, `DataQualityError`, `NotFoundError`, `ConflictError`, `AuthenticationError`, `PermissionDeniedError`, `PipelineError`, `ModelGateError`. The API maps them to problem responses.
- Catch specific exceptions. `except Exception` is allowed only at process boundaries (flow wrapper, request handler) and must log with context and re-raise or convert. Never `pass` in an `except`.

**Logging**
```python
import structlog

log = structlog.get_logger(__name__)

log.info("batch_loaded", batch_id=str(batch_id), rows_loaded=n_loaded, rows_rejected=n_rejected)
```
Event names are `snake_case` verbs-in-past-tense; context goes in key-value pairs, not f-strings. Never log secrets, tokens, `customer_key` or file contents.

**Configuration**
- `Settings` (pydantic-settings) is the only reader of environment variables. ML and ETL functions receive the values they need as arguments (tests pass explicit values).

**Concurrency**
- FastAPI endpoints are plain `def` (sync SQLAlchemy). Use `async def` only when awaiting async I/O. Heavy work runs in the worker, not in a request.

### Example module
```python
"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np

from retailpulse.errors import DomainValidationError


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Return the weighted absolute percentage error.

    Args:
        actual: Observed demand, shape (n,).
        forecast: Forecast demand, shape (n,).

    Returns:
        Sum of absolute errors divided by the sum of absolute actuals.

    Raises:
        DomainValidationError: If shapes differ or total actual demand is zero.
    """
    if actual.shape != forecast.shape:
        raise DomainValidationError("actual and forecast must have the same shape")
    denominator = float(np.abs(actual).sum())
    if denominator == 0.0:
        raise DomainValidationError("WAPE is undefined when total actual demand is zero")
    return float(np.abs(actual - forecast).sum() / denominator)
```

## 3. pandas and NumPy

- Vectorize. No `iterrows`, no `apply(axis=1)` on frames beyond a few thousand rows.
  ```python
  # Bad
  total = 0.0
  for _, row in df.iterrows():
      total += row["quantity"] * row["unit_price"]
  # Good
  total = float((df["quantity"] * df["unit_price"]).sum())
  ```
- No chained assignment; use `.loc[mask, col] = value`. Avoid `inplace=True`. Do not mutate function arguments; return new frames.
- Set dtypes explicitly at load (`category` for low-cardinality text, nullable `Int64` where NULLs exist). Name columns once, in a schema constant, and validate with pandera at module boundaries.
- Sort explicitly before time-dependent operations (`sort_values(["series_id", "date"])`); use `groupby(..., observed=True)`.
- Dates in frames are `datetime64[ns]` (naive) for `date`; the business timezone question is handled once, at load.
- Read only needed columns from Parquet; write to the database in chunks.
- Keep functions free of hidden state; do not rely on index alignment you did not create.

## 4. Machine-learning code rules

1. **Seeds** come from `Settings.seed` and are passed explicitly to every model and sampler.
2. **Time-based splits only.** Never random splits for time series or churn.
3. **Feature builders are pure functions** of `(data, as_of)` or `(data, origin)`; they read nothing newer than the cutoff.
4. **Fit on training data only**: scalers, encoders, calibrators, k selection thresholds, intervals.
5. **Hyperparameters live in frozen config dataclasses** that are logged to MLflow; no magic numbers inside training loops.
6. **Training functions return a result object** (metrics, artifact paths, run info), not just side effects.
7. **Metrics are defined once** in `ml/forecasting/metrics.py` (and its churn equivalent); never reimplemented inline.
8. **No model code in `api/` or `dashboard/`**. API reads stored predictions; the only ML in the API is the pure inventory policy.
9. **Log everything needed to reproduce**: git SHA, lockfile hash, dataset hash, seed, library versions.
10. **Notebooks** live in `notebooks/`, are exploratory only, are never imported by `src/`, and have outputs stripped before commit.
11. Report honestly: if a model does not beat the baseline, say so in the report; do not tune the evaluation.

## 5. SQL and SQLAlchemy

- SQLAlchemy 2.0 style: `select()`, `Session.execute`, typed `Mapped[...]` columns, naming convention from DATABASE.md §1.
- Repositories own queries. They return Pydantic or plain row objects, never Session-bound ORM instances, outside the repository layer.
- Transactions via context managers; one unit of work per request or per load chunk.
- Bulk writes with PostgreSQL `insert().on_conflict_do_nothing()` or `do_update()`; no row-by-row loops for large loads.
- No f-string SQL; use bound parameters. List explicit columns; avoid `SELECT *` in code (views for the API are the exception).
- Alembic: autogenerate then **review by hand**; one logical change per revision; every revision has a working `downgrade`; never edit an applied revision.

## 6. API conventions

- One router module per area (`routers/forecasts.py`), thin handlers: parse, authorize, call a service or repository, map to a response model.
- Dependencies via `Annotated[T, Depends(...)]`; authorization via `require_role(...)` at router or endpoint level.
- Schemas in `api/schemas/` named `XxxRead`, `XxxCreate`, `XxxUpdate`, `XxxList`; request models use `extra="forbid"`.
- Status codes and error types exactly as in API.md; `operation_id` in `snake_case`.
- Any endpoint change updates API.md and the RBAC matrix test in the same PR.

## 7. Dashboard conventions

- Each view is a module with `render()`; no business logic, no database access, no ML. Fetch through `client.py`, shape data, display.
- Session-state keys are constants in `state.py`. The only secret in session state is the auth token.
- Follow DESIGN.md for tokens, components, states and accessibility. Keep views under about 250 lines and extract shared widgets.

## 8. Git and pull requests

- Trunk-based: short-lived branches named `feat/p2-06-lightgbm`, `fix/…`, `chore/…`; squash-merge to `main`.
- **Conventional Commits:** `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`. Format: `type(scope): imperative subject (P2-06)`, subject ≤ 72 characters, body explains *why*.
- Never commit secrets, `.env`, datasets, MLflow data, model artifacts, notebooks with outputs, or large binaries.
- PR description: what and why, task IDs, how verified (paste summary), screenshots for UI, checklist from TESTING.md §10.
- Tags: `phase-N-complete` at phase gates; `vMAJOR.MINOR.PATCH` for releases (SemVer).

## 9. Docker, Compose and Makefile

- Dockerfiles: multi-stage, pinned base tags, non-root user, `COPY` only what is needed, `.dockerignore` present, one process per container, healthcheck defined.
- Compose: lowercase service names as in ARCHITECTURE §2, `env_file` for configuration, internal network, named volumes, no published ports for platform services in prod.
- Makefile: every target is `.PHONY` and documented with a trailing `## description` so `make help` lists them.

## 10. Documentation and comments

- Comments explain **why**, not what. Remove comments that restate the code.
- Update docs in the same PR as the code (DATABASE, API, TRD, ARCHITECTURE). Record deviations in `docs/DECISIONS.md`.
- Markdown: sentence-case headings, tables for specifications, fenced code with a language tag, line length not enforced.
- TODOs must reference an issue (`# TODO(#123): …`); otherwise do it now or do not write it.

## 11. Review checklist

- [ ] Follows module boundaries and the light-image rule.
- [ ] No clock access in ML, feature or ETL code; `as_of` is a parameter.
- [ ] Types complete; docstrings on public items; no leftover debug code.
- [ ] Errors are specific; logs are structured and free of sensitive data.
- [ ] Tests cover new behavior and edge cases (TESTING.md §10).
- [ ] No new dependency without approval; lockfile updated if one was approved.
- [ ] Docs and API/DB contracts updated.

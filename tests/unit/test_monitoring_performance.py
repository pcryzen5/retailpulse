"""Unit tests for realized model performance monitoring.

Implements P7-05 verification per ARCHITECTURE §7, TRD §6, and PRD §7.7:
- Realized WAPE calculation for elapsed forecasts joined against actual ground-truth demand.
- Comparison against backtest WAPE and baseline WAPE; degradation alerting.
- Realized PR-AUC calculation for customer churn after 90-day outcome window.
- Database persistence to `ops.model_performance` and idempotency on unique constraint.
- API compatibility with `/api/v1/monitoring/performance`.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from retailpulse.api.app import create_app
from retailpulse.db.base import Base
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.core import FactDailySales
from retailpulse.db.models.ml import Forecast, ModelRun
from retailpulse.db.models.ops import Alert, ModelPerformance
from retailpulse.db.session import get_db
from retailpulse.monitoring.performance import (
    ModelPerformanceSnapshot,
    compute_churn_realized_pr_auc,
    compute_forecast_realized_wape,
    save_performance_snapshot,
)
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password
from retailpulse.timeutil import utc_now


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with translated schemas."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    engine = engine.execution_options(
        schema_translate_map={"app": None, "core": None, "ops": None, "ml": None}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_forecast_run(db_session: Session) -> ModelRun:
    """Create a sample champion forecast ModelRun."""
    run = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="lightgbm",
        mlflow_run_id="mlf-fc-001",
        status="succeeded",
        is_champion=True,
        metrics={"wape": 0.20, "baseline_wape": 0.35, "mase": 0.85},
        trained_at=utc_now() - datetime.timedelta(days=30),
    )
    db_session.add(run)
    db_session.commit()
    return run


@pytest.fixture
def sample_churn_run(db_session: Session) -> ModelRun:
    """Create a sample champion customer churn ModelRun."""
    run = ModelRun(
        run_id=uuid.uuid4(),
        task="churn",
        algorithm="lightgbm",
        mlflow_run_id="mlf-churn-001",
        status="succeeded",
        is_champion=True,
        metrics={"pr_auc": 0.65, "baseline_pr_auc": 0.20, "roc_auc": 0.82},
        trained_at=utc_now() - datetime.timedelta(days=120),
    )
    db_session.add(run)
    db_session.commit()
    return run


# ==============================================================================
# 1. Forecast Realized WAPE Tests with Injected Actuals
# ==============================================================================


def test_forecast_realized_wape_with_injected_actuals_passing(
    db_session: Session,
    sample_forecast_run: ModelRun,
) -> None:
    """Verify realized WAPE matches actuals, beats baseline, and stores to ops.model_performance."""
    start_date = datetime.date(2026, 11, 1)
    end_date = datetime.date(2026, 11, 10)

    dates = [start_date + datetime.timedelta(days=i) for i in range(10)]
    forecasts_df = pd.DataFrame({
        "product_id": [101] * 10,
        "store_id": [1] * 10,
        "target_date": dates,
        "yhat": [20.0, 25.0, 22.0, 30.0, 28.0, 35.0, 18.0, 21.0, 29.0, 31.0],
    })

    # Injected actuals with slight variance (~10% error, well under backtest 0.20 and baseline 0.35)
    actuals_df = pd.DataFrame({
        "product_id": [101] * 10,
        "store_id": [1] * 10,
        "target_date": dates,
        "actual_qty": [19.0, 26.0, 23.0, 28.0, 30.0, 33.0, 19.0, 22.0, 27.0, 33.0],
    })

    snapshot = compute_forecast_realized_wape(
        session=db_session,
        model_run_id=sample_forecast_run.run_id,
        window_start=start_date,
        window_end=end_date,
        forecasts_df=forecasts_df,
        actuals_df=actuals_df,
        persist=True,
    )

    assert isinstance(snapshot, ModelPerformanceSnapshot)
    assert snapshot.task == "forecast"
    assert snapshot.metric == "realized_wape"
    assert snapshot.model_run_id == sample_forecast_run.run_id
    assert snapshot.n_observations == 10
    assert 0.0 < snapshot.value < 0.15
    assert snapshot.is_degraded is False
    assert snapshot.alert_created is False

    # Verify persisted row in ops.model_performance
    row = db_session.execute(
        sa.select(ModelPerformance).where(
            ModelPerformance.model_run_id == sample_forecast_run.run_id,
            ModelPerformance.metric == "realized_wape",
        )
    ).scalar_one_or_none()
    assert row is not None
    assert row.value == snapshot.value
    assert row.baseline_value == snapshot.baseline_value
    assert row.n_observations == 10


def test_forecast_realized_wape_degradation_alert_on_backtest_multiplier(
    db_session: Session,
    sample_forecast_run: ModelRun,
) -> None:
    """Verify alert triggers when realized WAPE exceeds 1.25x backtest WAPE."""
    start_date = datetime.date(2026, 11, 1)
    end_date = datetime.date(2026, 11, 5)
    dates = [start_date + datetime.timedelta(days=i) for i in range(5)]

    # Backtest WAPE is 0.20. 1.25x threshold is 0.25.
    # We inject huge errors so realized WAPE is ~0.60
    forecasts_df = pd.DataFrame({
        "product_id": [101] * 5,
        "store_id": [1] * 5,
        "target_date": dates,
        "yhat": [100.0, 120.0, 90.0, 110.0, 105.0],
    })
    actuals_df = pd.DataFrame({
        "product_id": [101] * 5,
        "store_id": [1] * 5,
        "target_date": dates,
        "actual_qty": [40.0, 50.0, 45.0, 55.0, 50.0],
    })

    snapshot = compute_forecast_realized_wape(
        session=db_session,
        model_run_id=sample_forecast_run.run_id,
        window_start=start_date,
        window_end=end_date,
        forecasts_df=forecasts_df,
        actuals_df=actuals_df,
        persist=True,
        alert_on_degradation=True,
    )

    assert snapshot.is_degraded is True
    assert snapshot.alert_created is True
    assert "exceeded 1.25x backtest WAPE" in (snapshot.degradation_reason or "")

    # Check alert was created in ops.alert
    alert = db_session.execute(
        sa.select(Alert).where(Alert.kind == "forecast_degraded")
    ).scalar_one_or_none()
    assert alert is not None
    assert alert.severity == "warning"
    assert "Realized WAPE" in alert.message
    assert alert.payload["model_run_id"] == str(sample_forecast_run.run_id)


def test_forecast_realized_wape_worse_than_baseline_alert(
    db_session: Session,
    sample_forecast_run: ModelRun,
) -> None:
    """Verify alert triggers when realized WAPE is worse than seasonal naive baseline."""
    start_date = datetime.date(2026, 11, 1)
    end_date = datetime.date(2026, 11, 4)
    dates = [start_date + datetime.timedelta(days=i) for i in range(4)]

    # Injected actuals: 50.0 demand
    actuals_df = pd.DataFrame({
        "product_id": [101] * 4,
        "store_id": [1] * 4,
        "target_date": dates,
        "actual_qty": [50.0, 50.0, 50.0, 50.0],
    })
    # Candidate forecast: 61.0 (WAPE = 44/200 = 0.22, < 1.25x backtest 0.25)
    forecasts_df = pd.DataFrame({
        "product_id": [101] * 4,
        "store_id": [1] * 4,
        "target_date": dates,
        "yhat": [61.0, 61.0, 61.0, 61.0],
    })
    # Baseline forecast: 55.0 (WAPE = 20/200 = 0.10)
    baseline_df = pd.DataFrame({
        "product_id": [101] * 4,
        "store_id": [1] * 4,
        "target_date": dates,
        "yhat": [55.0, 55.0, 55.0, 55.0],
    })

    snapshot = compute_forecast_realized_wape(
        session=db_session,
        model_run_id=sample_forecast_run.run_id,
        window_start=start_date,
        window_end=end_date,
        forecasts_df=forecasts_df,
        actuals_df=actuals_df,
        baseline_df=baseline_df,
        persist=True,
    )

    assert snapshot.is_degraded is True
    assert snapshot.value > (snapshot.baseline_value or 0.0)
    assert "worse than seasonal naive baseline" in (snapshot.degradation_reason or "")


# ==============================================================================
# 2. Customer Churn Realized PR-AUC Tests with Injected Actuals
# ==============================================================================


def test_churn_realized_pr_auc_with_injected_actuals_passing(
    db_session: Session,
    sample_churn_run: ModelRun,
) -> None:
    """Verify customer churn realized PR-AUC computation after elapsed outcome window."""
    today = datetime.date(2026, 11, 20)
    customer_ids = list(range(1001, 1021))  # 20 customers

    # High predicted churn probabilities for customers who actually churned
    churned_actual = [1 if i < 5 else 0 for i in range(20)]
    predicted_probs = [
        0.85, 0.90, 0.78, 0.82, 0.75,  # Top scored churners
        0.12, 0.08, 0.22, 0.15, 0.05, 0.18, 0.10, 0.14, 0.09, 0.20,
        0.11, 0.07, 0.16, 0.13, 0.06,
    ]

    scores_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": predicted_probs,
    })
    actuals_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churned": churned_actual,
    })

    snapshot = compute_churn_realized_pr_auc(
        session=db_session,
        model_run_id=sample_churn_run.run_id,
        as_of_date=today,
        outcome_window_days=90,
        scores_df=scores_df,
        actuals_df=actuals_df,
        persist=True,
    )

    assert isinstance(snapshot, ModelPerformanceSnapshot)
    assert snapshot.task == "churn"
    assert snapshot.metric == "realized_pr_auc"
    assert snapshot.n_observations == 20
    assert snapshot.value >= 0.80
    assert snapshot.is_degraded is False
    assert snapshot.alert_created is False

    # Check ops.model_performance row
    row = db_session.execute(
        sa.select(ModelPerformance).where(
            ModelPerformance.model_run_id == sample_churn_run.run_id,
            ModelPerformance.metric == "realized_pr_auc",
        )
    ).scalar_one_or_none()
    assert row is not None
    assert row.task == "churn"
    assert row.value == snapshot.value
    assert row.n_observations == 20


def test_churn_realized_pr_auc_degradation_alert(
    db_session: Session,
    sample_churn_run: ModelRun,
) -> None:
    """Verify alert triggers when realized PR-AUC degrades below baseline or backtest."""
    today = datetime.date(2026, 11, 20)
    customer_ids = list(range(1001, 1021))

    # Churned customers scored lowest probabilities (inverted model)
    churned_actual = [1 if i < 5 else 0 for i in range(20)]
    predicted_probs = [
        0.05, 0.08, 0.02, 0.06, 0.04,  # Churners scored near zero
        0.90, 0.85, 0.80, 0.75, 0.70, 0.82, 0.88, 0.79, 0.81, 0.83,
        0.72, 0.76, 0.74, 0.86, 0.84,
    ]

    scores_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": predicted_probs,
    })
    actuals_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churned": churned_actual,
    })

    snapshot = compute_churn_realized_pr_auc(
        session=db_session,
        model_run_id=sample_churn_run.run_id,
        as_of_date=today,
        outcome_window_days=90,
        scores_df=scores_df,
        actuals_df=actuals_df,
        persist=True,
        alert_on_degradation=True,
    )

    assert snapshot.is_degraded is True
    assert snapshot.alert_created is True
    assert (
        "fell below 80% of backtest PR-AUC" in (snapshot.degradation_reason or "")
        or "worse than baseline" in (snapshot.degradation_reason or "")
    )

    # Verify alert created in ops.alert
    alert = db_session.execute(
        sa.select(Alert).where(Alert.kind == "churn_degraded")
    ).scalar_one_or_none()
    assert alert is not None
    assert alert.severity == "warning"


# ==============================================================================
# 3. Persistence Idempotency and Database Querying Tests
# ==============================================================================


def test_save_performance_snapshot_idempotency(
    db_session: Session,
    sample_forecast_run: ModelRun,
) -> None:
    """Verify upsert idempotency on UNIQUE(model_run_id, metric, window_start, window_end)."""
    start_date = datetime.date(2026, 11, 1)
    end_date = datetime.date(2026, 11, 15)

    s1 = ModelPerformanceSnapshot(
        task="forecast",
        metric="realized_wape",
        model_run_id=sample_forecast_run.run_id,
        window_start=start_date,
        window_end=end_date,
        value=0.22,
        baseline_value=0.35,
        backtest_value=0.20,
        n_observations=50,
        is_degraded=False,
    )
    rec1 = save_performance_snapshot(db_session, s1)
    db_session.commit()
    assert rec1.value == 0.22

    # Update with new value
    s2 = ModelPerformanceSnapshot(
        task="forecast",
        metric="realized_wape",
        model_run_id=sample_forecast_run.run_id,
        window_start=start_date,
        window_end=end_date,
        value=0.19,
        baseline_value=0.35,
        backtest_value=0.20,
        n_observations=60,
        is_degraded=False,
    )
    rec2 = save_performance_snapshot(db_session, s2)
    db_session.commit()

    assert rec2.snapshot_id == rec1.snapshot_id
    assert rec2.value == 0.19
    assert rec2.n_observations == 60

    # Ensure exactly 1 row exists
    total = db_session.scalar(sa.select(sa.func.count(ModelPerformance.snapshot_id)))
    assert total == 1


def test_forecast_performance_query_from_db_tables(
    db_session: Session,
    sample_forecast_run: ModelRun,
) -> None:
    """Verify performance engine joins Forecast and FactDailySales directly from database."""
    target_d = datetime.date(2026, 10, 15)

    # Insert a Forecast row
    fc = Forecast(
        model_run_id=sample_forecast_run.run_id,
        store_id=1,
        product_id=501,
        origin_date=datetime.date(2026, 10, 1),
        target_date=target_d,
        horizon_day=14,
        yhat=Decimal("45.0"),
        yhat_lower=Decimal("35.0"),
        yhat_upper=Decimal("55.0"),
    )
    db_session.add(fc)

    # Insert corresponding FactDailySales row (net_units is computed, don't supply it)
    actual = FactDailySales(
        date_key=target_d,
        store_id=1,
        product_id=501,
        units_sold=42,
        units_returned=0,
        revenue=Decimal("420.0"),
    )
    db_session.add(actual)
    db_session.commit()

    snapshot = compute_forecast_realized_wape(
        session=db_session,
        model_run_id=sample_forecast_run.run_id,
        window_start=datetime.date(2026, 10, 1),
        window_end=datetime.date(2026, 10, 20),
        as_of_date=datetime.date(2026, 10, 20),
    )

    assert snapshot.n_observations == 1
    # |42 - 45| / 42 = 3 / 42 = ~0.0714
    assert 0.070 <= snapshot.value <= 0.073


# ==============================================================================
# 4. API Integration Test
# ==============================================================================


def test_api_list_model_performance(
    db_engine: sa.Engine,
    sample_forecast_run: ModelRun,
    sample_churn_run: ModelRun,
) -> None:
    """Verify GET /api/v1/monitoring/performance serves computed realized performance snapshots."""
    app = create_app()

    def override_get_db():
        session_cls = sessionmaker(bind=db_engine)
        session = session_cls()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    session_cls = sessionmaker(bind=db_engine)
    session = session_cls()

    # Create viewer
    viewer_id = uuid.uuid4()
    viewer = AppUser(
        user_id=viewer_id,
        email="viewer_perf@retailpulse.local",
        password_hash=hash_password("ViewerPass123!"),
        role="viewer",
        is_active=True,
    )
    session.add(viewer)
    session.commit()

    # Pre-populate two performance snapshots
    today = datetime.date(2026, 11, 20)
    p_fc = ModelPerformance(
        model_run_id=sample_forecast_run.run_id,
        task="forecast",
        window_start=today - datetime.timedelta(days=14),
        window_end=today,
        metric="realized_wape",
        value=0.185,
        baseline_value=0.320,
        n_observations=140,
    )
    p_churn = ModelPerformance(
        model_run_id=sample_churn_run.run_id,
        task="churn",
        window_start=today - datetime.timedelta(days=90),
        window_end=today,
        metric="realized_pr_auc",
        value=0.742,
        baseline_value=0.250,
        n_observations=500,
    )
    session.add_all([p_fc, p_churn])
    session.commit()
    session.close()

    client = TestClient(app)
    token = create_access_token(user_id=viewer_id, role="viewer")
    headers = {"Authorization": f"Bearer {token}"}

    # Query all
    res_all = client.get("/api/v1/monitoring/performance", headers=headers)
    assert res_all.status_code == 200
    body = res_all.json()
    assert body["total"] == 2

    # Query by task=forecast
    res_fc = client.get("/api/v1/monitoring/performance?task=forecast", headers=headers)
    assert res_fc.status_code == 200
    body_fc = res_fc.json()
    assert body_fc["total"] == 1
    assert body_fc["items"][0]["task"] == "forecast"
    assert body_fc["items"][0]["metric"] == "realized_wape"
    assert body_fc["items"][0]["value"] == 0.185

    # Query by task=churn
    res_ch = client.get("/api/v1/monitoring/performance?task=churn", headers=headers)
    assert res_ch.status_code == 200
    body_ch = res_ch.json()
    assert body_ch["total"] == 1
    assert body_ch["items"][0]["task"] == "churn"
    assert body_ch["items"][0]["metric"] == "realized_pr_auc"
    assert body_ch["items"][0]["value"] == 0.742

    app.dependency_overrides.clear()

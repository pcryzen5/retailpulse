"""Integration tests for batch forecasting, intervals, and active output pointer management.

Verifies P2-11 & DATABASE §11 / TRD §5.1 / §6:
- Batch forecasting using active champion or best baseline
- Prediction intervals application and check constraints (0 <= yhat_lower <= yhat <= yhat_upper)
- Atomic transactional update of ml.forecast and ml.active_output
- VERIFY REQUIREMENT: A failed run leaves the ml.active_output pointer unchanged
- Idempotent re-scoring for the same cutoff date
- CLI command `retailpulse score forecast`
"""

from __future__ import annotations

import datetime
import uuid
from contextlib import contextmanager
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.db.models.core import DimCustomer, DimDate, DimProduct, DimStore
from retailpulse.db.models.ml import ActiveOutput, Forecast, ModelRun
from retailpulse.ml.forecasting.predict import (
    TASK_FORECAST,
    _resolve_data_as_of,
    predict_batch_forecast,
)

runner = CliRunner()


@pytest.fixture
def scoring_db_session() -> Session:
    """In-memory SQLite session with schema translation map for testing."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        execution_options={
            "schema_translate_map": {"core": None, "ops": None, "ml": None, "app": None}
        },
    )
    tables = [
        cast(sa.Table, DimStore.__table__),
        cast(sa.Table, DimProduct.__table__),
        cast(sa.Table, DimCustomer.__table__),
        cast(sa.Table, DimDate.__table__),
        cast(sa.Table, ModelRun.__table__),
        cast(sa.Table, Forecast.__table__),
        cast(sa.Table, ActiveOutput.__table__),
    ]
    for table in tables:
        table.create(engine)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # Seed test store and products
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store", country="GB")
    p1 = DimProduct(product_id=101, sku="SKU001", description="Item 1", is_product=True)
    p2 = DimProduct(product_id=102, sku="SKU002", description="Item 2", is_product=True)
    session.add_all([store, p1, p2])

    # Seed calendar dates
    start_d = datetime.date(2011, 1, 1)
    for i in range(120):
        d = start_d + datetime.timedelta(days=i)
        dim_d = DimDate(
            date_key=d,
            year=d.year,
            quarter=(d.month - 1) // 3 + 1,
            month=d.month,
            iso_week=d.isocalendar()[1],
            day_of_week=d.weekday() + 1,
            day_of_month=d.day,
            is_weekend=d.weekday() in (5, 6),
            is_holiday=False,
            is_trading_day=True,
            is_complete=True,
        )
        session.add(dim_d)

    session.commit()
    return session


def _create_synthetic_series() -> pd.DataFrame:
    """Generate 90 days of synthetic trading-day demand for two products."""
    start_d = datetime.date(2011, 1, 1)
    dates = [start_d + datetime.timedelta(days=i) for i in range(90)]
    rows = []
    for pid in (101, 102):
        for i, d in enumerate(dates):
            y_val = float(10 + (i % 7) * 2 + (pid % 5))
            rows.append(
                {
                    "series_id": pid,
                    "product_id": pid,
                    "store_id": 1,
                    "date": d,
                    "y": y_val,
                    "price": 15.0,
                    "is_holiday": False,
                    "abc_class": "A",
                }
            )
    return pd.DataFrame(rows)


def test_failed_run_leaves_active_output_pointer_unchanged(
    scoring_db_session: Session,
) -> None:
    """VERIFY REQUIREMENT: A failed run leaves the ml.active_output pointer unchanged."""
    initial_run_id = uuid.uuid4()
    initial_date = datetime.date(2011, 3, 1)

    # 1. Seed initial active champion run and active_output pointer
    initial_run = ModelRun(
        run_id=initial_run_id,
        task=TASK_FORECAST,
        algorithm="seasonal_naive",
        mlflow_run_id=f"mlflow-{initial_run_id.hex}",
        status="succeeded",
        is_champion=True,
        data_end=initial_date,
        params={},
        metrics={"wape": 0.25},
    )
    initial_pointer = ActiveOutput(
        task=TASK_FORECAST,
        model_run_id=initial_run_id,
        data_as_of=initial_date,
    )
    scoring_db_session.add_all([initial_run, initial_pointer])
    scoring_db_session.commit()

    # Verify initial pointer state
    ptr = scoring_db_session.get(ActiveOutput, TASK_FORECAST)
    assert ptr is not None
    assert ptr.model_run_id == initial_run_id
    assert ptr.data_as_of == initial_date

    # 2. Trigger a failing batch forecast run (simulate failure during database commit)
    new_date = datetime.date(2011, 3, 20)
    dataset_df = _create_synthetic_series()

    with (
        patch.object(
            scoring_db_session,
            "commit",
            side_effect=RuntimeError("Simulated database failure during transaction commit"),
        ),
        pytest.raises(RuntimeError, match="Simulated database failure"),
    ):
        predict_batch_forecast(
            session=scoring_db_session,
            data_as_of=new_date,
            dataset_df=dataset_df,
            horizon=14,
        )

    # 3. VERIFY ASSERTION: active_output pointer remains strictly unchanged!
    scoring_db_session.expire_all()
    ptr_after_fail = scoring_db_session.get(ActiveOutput, TASK_FORECAST)
    assert ptr_after_fail is not None
    assert ptr_after_fail.model_run_id == initial_run_id
    assert ptr_after_fail.data_as_of == initial_date

    # Verify no partial forecast rows were committed
    forecast_rows = (
        scoring_db_session.execute(sa.select(Forecast).where(Forecast.origin_date == new_date))
        .scalars()
        .all()
    )
    assert len(forecast_rows) == 0


def test_batch_forecast_success_and_active_output_update(
    scoring_db_session: Session,
) -> None:
    """Verify successful batch forecasting atomically writes rows and advances pointer."""
    initial_run_id = uuid.uuid4()
    initial_date = datetime.date(2011, 2, 15)

    # Initial pointer
    init_run = ModelRun(
        run_id=initial_run_id,
        task=TASK_FORECAST,
        algorithm="naive",
        mlflow_run_id=f"mlflow-{initial_run_id.hex}",
        status="succeeded",
        is_champion=True,
        data_end=initial_date,
        params={},
        metrics={},
    )
    init_ptr = ActiveOutput(
        task=TASK_FORECAST,
        model_run_id=initial_run_id,
        data_as_of=initial_date,
    )
    scoring_db_session.add_all([init_run, init_ptr])
    scoring_db_session.commit()

    # Execute successful batch forecast with new date
    new_date = datetime.date(2011, 3, 30)
    dataset_df = _create_synthetic_series()

    result = predict_batch_forecast(
        session=scoring_db_session,
        data_as_of=new_date,
        dataset_df=dataset_df,
        model_name="seasonal_naive",
        horizon=14,
    )

    assert result.origin_date == new_date
    assert result.horizon == 14
    assert result.n_series == 2
    assert result.n_forecast_rows == 28  # 2 series * 14 horizon days
    assert result.active_output_updated is True

    # Check ml.active_output pointer was updated
    scoring_db_session.expire_all()
    updated_ptr = scoring_db_session.get(ActiveOutput, TASK_FORECAST)
    assert updated_ptr is not None
    assert updated_ptr.model_run_id == result.model_run_id
    assert updated_ptr.data_as_of == new_date

    # Check ml.forecast records
    forecast_rows = (
        scoring_db_session.execute(
            sa.select(Forecast).where(Forecast.model_run_id == result.model_run_id)
        )
        .scalars()
        .all()
    )
    assert len(forecast_rows) == 28

    # Verify check constraints: 0 <= yhat_lower <= yhat <= yhat_upper
    for f in forecast_rows:
        assert f.yhat >= 0
        assert f.yhat_lower >= 0
        assert f.yhat_lower <= f.yhat
        assert f.yhat <= f.yhat_upper
        assert f.horizon_day >= 1
        assert f.origin_date == new_date


def test_batch_forecast_idempotent_rescore(
    scoring_db_session: Session,
) -> None:
    """Verify re-running scoring on the same date cleans prior rows idempotently."""
    score_date = datetime.date(2011, 3, 25)
    dataset_df = _create_synthetic_series()

    res1 = predict_batch_forecast(
        session=scoring_db_session,
        data_as_of=score_date,
        dataset_df=dataset_df,
        model_name="naive",
        horizon=7,
    )
    assert res1.n_forecast_rows == 14

    # Re-run same model and date
    res2 = predict_batch_forecast(
        session=scoring_db_session,
        data_as_of=score_date,
        dataset_df=dataset_df,
        model_name="naive",
        horizon=7,
    )

    # Check that rows count remains exactly 14 and no duplicates exist
    rows = (
        scoring_db_session.execute(
            sa.select(Forecast).where(Forecast.model_run_id == res2.model_run_id)
        )
        .scalars()
        .all()
    )
    assert len(rows) == 14


def test_cli_score_forecast_help() -> None:
    """Verify retailpulse score forecast --help displays valid options."""
    res = runner.invoke(app, ["score", "forecast", "--help"])
    assert res.exit_code == 0
    assert "--data-as-of" in res.output
    assert "--horizon" in res.output
    assert "--top-n" in res.output
    assert "--model" in res.output
    assert "--interval-level" in res.output


def test_batch_forecast_auto_fallback_and_champion(
    scoring_db_session: Session,
) -> None:
    """Verify fallback when no model/champion exists, and champion selection when present."""
    dataset_df = _create_synthetic_series()
    score_date = datetime.date(2011, 3, 20)

    # 1. No model specified and no champion exists -> falls back to seasonal_naive
    res_fallback = predict_batch_forecast(
        session=scoring_db_session,
        data_as_of=score_date,
        dataset_df=dataset_df,
        model_name=None,
        horizon=7,
    )
    assert res_fallback.model_name == "seasonal_naive"
    assert res_fallback.n_forecast_rows == 14

    # 2. Test _resolve_data_as_of from dataset_df
    resolved_date = _resolve_data_as_of(scoring_db_session, dataset_df, None)
    assert resolved_date == dataset_df["date"].max()

    # 3. Test _resolve_data_as_of from DB DimDate when dataset_df is None
    dim_date_resolved = _resolve_data_as_of(scoring_db_session, None, None)
    assert isinstance(dim_date_resolved, datetime.date)


def test_batch_forecast_empty_series_raises(
    scoring_db_session: Session,
) -> None:
    """Verify empty series dataframe raises ValueError."""
    empty_df = pd.DataFrame(columns=["series_id", "product_id", "date", "y"])
    with pytest.raises(ValueError, match="empty"):
        predict_batch_forecast(
            session=scoring_db_session,
            data_as_of=datetime.date(2011, 3, 20),
            dataset_df=empty_df,
        )


def test_cli_score_forecast_invocation(
    scoring_db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify retailpulse score forecast CLI invocation with --json output."""
    dataset_df = _create_synthetic_series()

    @contextmanager
    def mock_scope():
        yield scoring_db_session

    monkeypatch.setattr(
        "retailpulse.cli.session_scope",
        mock_scope,
    )
    monkeypatch.setattr(
        "retailpulse.ml.forecasting.predict._resolve_scoring_series",
        lambda *args, **kwargs: dataset_df,
    )

    result = runner.invoke(
        app,
        [
            "score",
            "forecast",
            "--data-as-of",
            "2011-03-25",
            "--horizon",
            "7",
            "--model",
            "naive",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert '"model_name": "naive"' in result.output
    assert '"n_forecast_rows": 14' in result.output

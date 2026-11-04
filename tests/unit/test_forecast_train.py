"""Unit tests for forecasting training pipeline, backtest orchestration, and champion gate.

Verifies P2-10 & PRD FR-FC-05 / TRD §5.1:
- Multi-model backtest execution across baselines and ML models
- Champion gate logic:
  - Candidate must have global WAPE strictly lower than seasonal_naive
  - Candidate must have |bias| <= 0.15
  - Coverage outside [0.65, 0.95] triggers warning, not block
  - If no model passes, best baseline becomes champion with alert 'forecast_no_model_beats_baseline'
- VERIFY REQUIREMENT: Run on fixture; a deliberately bad model is rejected by the gate
- MLflow tracking, parameter/metric logging, artifact saving, and model registration
- Auto-promotion behavior
- CLI command `retailpulse train forecast`
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.db.models.core import DimCustomer, DimDate, DimProduct, DimStore
from retailpulse.db.models.ml import ForecastMetric, ModelRun
from retailpulse.ml.forecasting.backtest import BacktestResult
from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.train import (
    ALERT_NO_MODEL_BEATS_BASELINE,
    _instantiate_model,
    evaluate_champion_gate,
    train_forecast_pipeline,
)
from retailpulse.ml.registry import (
    get_champion_version,
)

pytestmark = pytest.mark.unit
runner = CliRunner()


@pytest.fixture
def temp_mlflow_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create an isolated tracking URI for testing."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    return mlflow_dir.as_uri()


@pytest.fixture
def in_memory_session() -> Session:
    """SQLite database session with schema translate map for testing."""
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
        cast(sa.Table, ForecastMetric.__table__),
    ]
    for table in tables:
        table.create(engine)

    session_factory = sessionmaker(bind=engine)
    return session_factory()


def _make_dummy_backtest_result(
    model_name: str,
    *,
    wape: float,
    bias: float,
    smape: float = 0.20,
    n_folds: int = 3,
    horizon: int = 28,
) -> BacktestResult:
    """Construct a minimal BacktestResult with specified WAPE and bias."""
    dates = [datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(14)]
    preds = pd.DataFrame(
        {
            "series_id": [1] * 14,
            "product_id": [1] * 14,
            "origin": [datetime.date(2011, 1, 1)] * 14,
            "target_date": dates,
            "horizon_day": list(range(1, 15)),
            "y": [10.0] * 14,
            "yhat": [10.0 * (1.0 + bias)] * 14,
            "abc_class": ["A"] * 14,
        }
    )
    metrics_df = pd.DataFrame(
        [
            {
                "scope": "global",
                "scope_value": "all",
                "metric": "wape",
                "value": wape,
                "n_series": 1,
            },
            {
                "scope": "global",
                "scope_value": "all",
                "metric": "bias",
                "value": bias,
                "n_series": 1,
            },
        ]
    )
    return BacktestResult(
        model_name=model_name,
        n_folds=n_folds,
        horizon=horizon,
        origins=[datetime.date(2011, 1, 1)],
        predictions=preds,
        metrics=metrics_df,
        global_wape=wape,
        global_bias=bias,
        global_smape=smape,
        global_mae=1.5,
        global_rmse=2.0,
    )


class DeliberatelyBadForecaster:
    """Forecaster that deliberately predicts garbage values to test quality gate rejection."""

    name: str = "bad_model"

    def __init__(self, multiplier: float = 100.0) -> None:
        self.multiplier = multiplier

    def fit(self, history: pd.DataFrame) -> Forecaster:
        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        dates = [origin + datetime.timedelta(days=i) for i in range(1, horizon + 1)]
        return pd.DataFrame(
            {
                "series_id": [1] * horizon,
                "product_id": [1] * horizon,
                "date": dates,
                "yhat": [9999.0 * self.multiplier] * horizon,
            }
        )


def test_evaluate_champion_gate_candidate_passes() -> None:
    """Verify that a candidate with lower WAPE and acceptable bias passes the gate."""
    results = {
        "seasonal_naive": _make_dummy_backtest_result("seasonal_naive", wape=0.30, bias=0.05),
        "naive": _make_dummy_backtest_result("naive", wape=0.35, bias=0.08),
        "lightgbm": _make_dummy_backtest_result("lightgbm", wape=0.22, bias=-0.03),
    }

    gate = evaluate_champion_gate(results, baseline_name="seasonal_naive")
    assert gate.passed_gate is True
    assert gate.selected_champion == "lightgbm"
    assert gate.candidate_wape == pytest.approx(0.22)
    assert gate.baseline_wape == pytest.approx(0.30)
    assert gate.alert is None
    assert gate.rejection_reasons == []


def test_evaluate_champion_gate_worse_wape_rejected() -> None:
    """Verify candidate is rejected if WAPE is not strictly lower than seasonal_naive."""
    results = {
        "seasonal_naive": _make_dummy_backtest_result("seasonal_naive", wape=0.25, bias=0.02),
        "lightgbm": _make_dummy_backtest_result("lightgbm", wape=0.28, bias=0.01),
    }

    gate = evaluate_champion_gate(results, baseline_name="seasonal_naive")
    assert gate.passed_gate is False
    assert gate.selected_champion == "seasonal_naive"
    assert gate.alert == ALERT_NO_MODEL_BEATS_BASELINE
    assert len(gate.rejection_reasons) > 0
    assert "WAPE" in gate.rejection_reasons[0]


def test_evaluate_champion_gate_excessive_bias_rejected() -> None:
    """Verify candidate is rejected if |bias| exceeds 0.15 even if WAPE improved."""
    results = {
        "seasonal_naive": _make_dummy_backtest_result("seasonal_naive", wape=0.30, bias=0.02),
        "lightgbm": _make_dummy_backtest_result("lightgbm", wape=0.20, bias=0.22),
    }

    gate = evaluate_champion_gate(results, baseline_name="seasonal_naive")
    assert gate.passed_gate is False
    assert gate.selected_champion == "seasonal_naive"
    assert gate.alert == ALERT_NO_MODEL_BEATS_BASELINE
    assert any("bias" in r for r in gate.rejection_reasons)


def test_evaluate_champion_gate_coverage_warning_does_not_block() -> None:
    """Verify coverage outside [0.65, 0.95] triggers a warning but does NOT block promotion."""
    results = {
        "seasonal_naive": _make_dummy_backtest_result("seasonal_naive", wape=0.30, bias=0.02),
        "lightgbm": _make_dummy_backtest_result("lightgbm", wape=0.21, bias=0.04),
    }
    intervals = {
        "lightgbm": {"coverage": 0.55},  # Low coverage (< 0.65)
    }

    gate = evaluate_champion_gate(
        results,
        baseline_name="seasonal_naive",
        interval_evaluations=intervals,
    )
    assert gate.passed_gate is True
    assert gate.selected_champion == "lightgbm"
    assert gate.coverage_warning is True
    assert gate.candidate_coverage == pytest.approx(0.55)


def test_evaluate_champion_gate_multiple_candidates_picks_best() -> None:
    """Verify that among multiple passing candidates, the one with lowest WAPE is selected."""
    results = {
        "seasonal_naive": _make_dummy_backtest_result("seasonal_naive", wape=0.30, bias=0.01),
        "lightgbm": _make_dummy_backtest_result("lightgbm", wape=0.22, bias=0.02),
        "lstm": _make_dummy_backtest_result("lstm", wape=0.19, bias=-0.01),
    }

    gate = evaluate_champion_gate(results, baseline_name="seasonal_naive")
    assert gate.passed_gate is True
    assert gate.selected_champion == "lstm"
    assert gate.candidate_wape == pytest.approx(0.19)


def test_evaluate_champion_gate_no_candidates_defaults_to_baseline() -> None:
    """Verify that when only baselines are provided, best baseline is returned with alert."""
    results = {
        "seasonal_naive": _make_dummy_backtest_result("seasonal_naive", wape=0.28, bias=0.01),
        "naive": _make_dummy_backtest_result("naive", wape=0.34, bias=0.04),
    }

    gate = evaluate_champion_gate(results, baseline_name="seasonal_naive")
    assert gate.passed_gate is False
    assert gate.selected_champion == "seasonal_naive"
    assert gate.alert == ALERT_NO_MODEL_BEATS_BASELINE


def test_train_pipeline_and_deliberately_bad_model_rejected(
    temp_mlflow_uri: str,
    in_memory_session: Session,
) -> None:
    """VERIFY REQUIREMENT: Run on synthetic fixture; deliberately bad model is rejected by gate."""
    # Construct synthetic series: weekly-periodic pattern for 120 days
    pattern = [10.0, 15.0, 12.0, 14.0, 20.0, 25.0, 5.0]
    dates = [datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(112)]
    y_vals = (pattern * 16)[:112]

    df = pd.DataFrame(
        {
            "series_id": [1] * 112,
            "product_id": [1] * 112,
            "date": dates,
            "y": y_vals,
            "price": [12.5] * 112,
            "is_holiday": [False] * 112,
            "abc_class": ["A"] * 112,
        }
    )

    # 1. Run pipeline with standard baselines and a candidate model
    # On perfectly weekly-periodic data, seasonal naive achieves ~0.0 WAPE
    res = train_forecast_pipeline(
        dataset_df=df,
        session=in_memory_session,
        models_to_run=["seasonal_naive", "naive", "lightgbm"],
        n_folds=2,
        horizon=14,
        auto_promote=True,
        tracking_uri=temp_mlflow_uri,
    )

    assert res.total_series == 1
    assert "seasonal_naive" in res.backtest_results
    assert "lightgbm" in res.backtest_results
    assert res.intervals_table is not None

    # Check that seasonal_naive achieved very low WAPE on periodic data
    sn_wape = res.backtest_results["seasonal_naive"].global_wape
    assert sn_wape < 0.05

    # 2. Test champion gate directly with a deliberately bad candidate model
    bad_res = _make_dummy_backtest_result("bad_model", wape=15.0, bias=2.5)
    combined_results = {
        "seasonal_naive": res.backtest_results["seasonal_naive"],
        "bad_model": bad_res,
    }

    gate_verdict = evaluate_champion_gate(
        combined_results,
        baseline_name="seasonal_naive",
        candidate_names=["bad_model"],
    )

    # VERIFY ASSERTION: Deliberately bad model is rejected by the gate
    assert gate_verdict.passed_gate is False
    assert gate_verdict.selected_champion == "seasonal_naive"
    assert gate_verdict.alert == ALERT_NO_MODEL_BEATS_BASELINE
    assert gate_verdict.wape_improved is False
    assert gate_verdict.bias_within_tolerance is False
    assert len(gate_verdict.rejection_reasons) >= 2


def test_train_forecast_pipeline_auto_promote_and_db_mirroring(
    temp_mlflow_uri: str,
    in_memory_session: Session,
) -> None:
    """Verify that winning candidate is registered, promoted, and mirrored to database."""
    # Synthetic trending series
    dates = [datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(120)]
    y_vals = [float(5 + (i % 7) * 3 + int(i / 10)) for i in range(120)]

    df = pd.DataFrame(
        {
            "series_id": [101] * 120,
            "product_id": [101] * 120,
            "date": dates,
            "y": y_vals,
            "price": [15.0] * 120,
            "is_holiday": [False] * 120,
            "abc_class": ["A"] * 120,
        }
    )

    res = train_forecast_pipeline(
        dataset_df=df,
        session=in_memory_session,
        models_to_run=["seasonal_naive", "naive"],
        n_folds=2,
        horizon=14,
        auto_promote=True,
        tracking_uri=temp_mlflow_uri,
    )

    assert res.champion_model in ("seasonal_naive", "naive")
    assert res.champion_version is not None

    # Check MLflow alias
    champ_ver = get_champion_version("forecast", tracking_uri=temp_mlflow_uri)
    assert champ_ver == res.champion_version

    # Check database ModelRun records
    runs = in_memory_session.execute(sa.select(ModelRun)).scalars().all()
    assert len(runs) >= 2
    for r in runs:
        assert r.task == "forecast"
        assert r.status == "succeeded"
        assert "wape" in r.metrics

    # Check database ForecastMetric records
    metrics = in_memory_session.execute(sa.select(ForecastMetric)).scalars().all()
    assert len(metrics) > 0


def test_cli_train_forecast_help() -> None:
    """Verify that retailpulse train forecast --help displays options."""
    res = runner.invoke(app, ["train", "forecast", "--help"])
    assert res.exit_code == 0
    assert "--data-as-of" in res.output
    assert "--n-folds" in res.output
    assert "--horizon" in res.output
    assert "--models" in res.output
    assert "--auto-promote" in res.output


def test_instantiate_model_variants() -> None:
    """Verify supported and unsupported model instantiation."""
    ma = _instantiate_model("moving_average")
    assert ma.name == "moving_average"
    csba = _instantiate_model("croston_sba")
    assert csba.name == "croston_sba"

    with pytest.raises(ValueError, match="Unsupported model name"):
        _instantiate_model("invalid_model")


def test_evaluate_champion_gate_baseline_fallback() -> None:
    """Verify fallback when designated baseline is absent but other baselines exist."""
    results = {
        "naive": _make_dummy_backtest_result("naive", wape=0.35, bias=0.02),
        "lightgbm": _make_dummy_backtest_result("lightgbm", wape=0.20, bias=0.01),
    }
    gate = evaluate_champion_gate(results, baseline_name="seasonal_naive")
    assert gate.passed_gate is True
    assert gate.selected_champion == "lightgbm"
    assert gate.baseline_wape == pytest.approx(0.35)


def test_train_forecast_pipeline_validation_errors(in_memory_session: Session) -> None:
    """Verify validation on missing parameters and empty datasets."""
    with pytest.raises(ValueError, match="Must provide either 'dataset_df' or 'session'"):
        train_forecast_pipeline()

    empty_df = pd.DataFrame(columns=["series_id", "date", "y"])
    with pytest.raises(ValueError, match="Training series DataFrame is empty"):
        train_forecast_pipeline(dataset_df=empty_df)


def test_cli_train_forecast_invalid_date() -> None:
    """Verify CLI error on invalid date argument."""
    res = runner.invoke(app, ["train", "forecast", "--data-as-of", "not-a-date"])
    assert res.exit_code != 0
    assert "Invalid date format" in res.output

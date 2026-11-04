"""Forecasting model training pipeline, backtest orchestration, and champion gate.

Implements P2-10, PRD FR-FC-05, and TRD §5.1 / §6:
- Multi-model rolling-origin backtest execution (baselines, LightGBM, Prophet, LSTM)
- Metrics aggregation across models and scopes
- Residual-quantile prediction intervals calibration
- Champion gate evaluation:
  - Candidate must have global WAPE strictly lower than seasonal_naive
  - Candidate must have |bias| <= 0.15
  - Coverage outside 0.65-0.95 raises warning alert, not a block
  - If nothing passes, best baseline becomes champion with alert 'forecast_no_model_beats_baseline'
- MLflow tracking, parameter/metric logging, artifact persistence, and model registration
- Champion promotion via promote_champion when auto_promote is enabled
"""

from __future__ import annotations

import datetime
import hashlib
import json
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import DimDate
from retailpulse.features.forecast_features import (
    build_zero_filled_series,
    select_top_products,
)
from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.backtest import BacktestResult, run_backtest
from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.baselines import (
    CrostonSBAForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from retailpulse.ml.forecasting.intervals import (
    ResidualQuantilesTable,
    apply_prediction_intervals,
    evaluate_interval_metrics,
    fit_residual_quantiles,
)
from retailpulse.ml.forecasting.lgbm_model import LightGBMForecaster
from retailpulse.ml.registry import (
    REGISTERED_MODEL_FORECAST,
    promote_champion,
    register_model,
)
from retailpulse.ml.runs import ModelRunTracker, track_model_run

logger = get_logger(__name__)

STANDARD_BASELINES = ("naive", "seasonal_naive", "moving_average", "croston_sba")
ALERT_NO_MODEL_BEATS_BASELINE = "forecast_no_model_beats_baseline"


@dataclass(frozen=True)
class GateEvaluationResult:
    """Outcome of evaluating candidate models against the champion quality gate."""

    candidate_model: str | None
    passed_gate: bool
    candidate_wape: float | None
    baseline_wape: float
    candidate_bias: float | None
    candidate_coverage: float | None
    wape_improved: bool
    bias_within_tolerance: bool
    coverage_warning: bool
    rejection_reasons: list[str]
    selected_champion: str
    alert: str | None


@dataclass(frozen=True)
class TrainingPipelineResult:
    """Consolidated outputs and metadata from the demand forecasting training pipeline."""

    data_as_of: datetime.date
    backtest_results: dict[str, BacktestResult]
    interval_evaluations: dict[str, dict[str, Any]]
    gate_result: GateEvaluationResult
    champion_model: str
    champion_version: str | None
    registered_versions: dict[str, str]
    intervals_table: ResidualQuantilesTable | None
    total_series: int
    n_folds: int
    horizon: int


def _instantiate_model(model_name: str, *, seed: int = 42, horizon: int = 28) -> Forecaster:
    """Instantiate a forecasting model by name with standard defaults."""
    clean_name = model_name.lower().strip()
    factories: dict[str, Callable[[], Forecaster]] = {
        "naive": NaiveForecaster,
        "seasonal_naive": lambda: SeasonalNaiveForecaster(seasonal_period=7),
        "moving_average": lambda: MovingAverageForecaster(window_size=28),
        "croston_sba": CrostonSBAForecaster,
        "lightgbm": lambda: LightGBMForecaster(random_state=seed, horizon=horizon),
    }

    if clean_name in factories:
        return factories[clean_name]()
    if clean_name == "prophet":
        from retailpulse.ml.forecasting.prophet_model import ProphetForecaster  # noqa: PLC0415

        return ProphetForecaster(n_jobs=1)
    if clean_name == "lstm":
        from retailpulse.ml.forecasting.lstm_model import LSTMForecaster  # noqa: PLC0415

        return LSTMForecaster(random_seed=seed, horizon=horizon)

    supported = (*STANDARD_BASELINES, "lightgbm", "prophet", "lstm")
    raise ValueError(f"Unsupported model name '{model_name}'. Supported models: {supported}")


def _check_candidate_gate(
    cand_name: str,
    cand_res: BacktestResult,
    *,
    baseline_name: str,
    baseline_wape: float,
    max_bias_threshold: float,
    min_cov: float,
    max_cov: float,
    interval_evals: dict[str, dict[str, Any]] | None,
) -> tuple[bool, list[str], float | None, bool]:
    """Check a single candidate against WAPE improvement, bias bound, and coverage range."""
    reasons: list[str] = []
    wape_improved = cand_res.global_wape < baseline_wape
    if not wape_improved:
        reasons.append(
            f"WAPE ({cand_res.global_wape:.4f}) >= {baseline_name} WAPE ({baseline_wape:.4f})"
        )

    bias_ok = abs(cand_res.global_bias) <= max_bias_threshold
    if not bias_ok:
        reasons.append(
            f"|bias| ({abs(cand_res.global_bias):.4f}) > threshold {max_bias_threshold:.2f}"
        )

    cand_cov: float | None = None
    cov_warning = False
    if interval_evals and cand_name in interval_evals:
        cov_val = interval_evals[cand_name].get("coverage")
        if cov_val is not None:
            cand_cov = float(cov_val)
            cov_warning = not (min_cov <= cand_cov <= max_cov)

    passed = wape_improved and bias_ok
    return passed, reasons, cand_cov, cov_warning


def evaluate_champion_gate(
    backtest_results: dict[str, BacktestResult],
    *,
    baseline_name: str = "seasonal_naive",
    candidate_names: Sequence[str] | None = None,
    max_bias_threshold: float = 0.15,
    min_coverage_threshold: float = 0.65,
    max_coverage_threshold: float = 0.95,
    interval_evaluations: dict[str, dict[str, Any]] | None = None,
) -> GateEvaluationResult:
    """Evaluate whether candidate models pass the promotion quality gate per TRD §5.1."""
    if baseline_name not in backtest_results:
        available_baselines = [b for b in STANDARD_BASELINES if b in backtest_results]
        if not available_baselines:
            raise ValueError(
                f"Benchmark baseline '{baseline_name}' not found in backtest results, "
                f"and no alternative baselines present. Available: {list(backtest_results.keys())}"
            )
        baseline_name = min(available_baselines, key=lambda b: backtest_results[b].global_wape)

    baseline_res = backtest_results[baseline_name]
    baseline_wape = baseline_res.global_wape

    eval_candidates = (
        [m for m in backtest_results if m not in STANDARD_BASELINES]
        if candidate_names is None
        else [m for m in candidate_names if m in backtest_results]
    )

    baselines_in_results = [b for b in STANDARD_BASELINES if b in backtest_results]
    best_baseline = (
        min(baselines_in_results, key=lambda b: backtest_results[b].global_wape)
        if baselines_in_results
        else baseline_name
    )

    if not eval_candidates:
        logger.info("no candidate models for champion gate; defaulting to best baseline")
        return GateEvaluationResult(
            candidate_model=None,
            passed_gate=False,
            candidate_wape=None,
            baseline_wape=baseline_wape,
            candidate_bias=None,
            candidate_coverage=None,
            wape_improved=False,
            bias_within_tolerance=False,
            coverage_warning=False,
            rejection_reasons=["No ML/DL candidate models evaluated"],
            selected_champion=best_baseline,
            alert=ALERT_NO_MODEL_BEATS_BASELINE,
        )

    passing: list[tuple[str, float, float, float | None, bool]] = []
    rejections: dict[str, list[str]] = {}

    for cand in eval_candidates:
        cand_res = backtest_results[cand]
        passed, reasons, cov, cov_warn = _check_candidate_gate(
            cand,
            cand_res,
            baseline_name=baseline_name,
            baseline_wape=baseline_wape,
            max_bias_threshold=max_bias_threshold,
            min_cov=min_coverage_threshold,
            max_cov=max_coverage_threshold,
            interval_evals=interval_evaluations,
        )
        if passed:
            passing.append((cand, cand_res.global_wape, cand_res.global_bias, cov, cov_warn))
        else:
            rejections[cand] = reasons

    if passing:
        passing.sort(key=lambda item: item[1])
        winner_name, win_wape, win_bias, win_cov, win_cov_warn = passing[0]
        return GateEvaluationResult(
            candidate_model=winner_name,
            passed_gate=True,
            candidate_wape=win_wape,
            baseline_wape=baseline_wape,
            candidate_bias=win_bias,
            candidate_coverage=win_cov,
            wape_improved=True,
            bias_within_tolerance=True,
            coverage_warning=win_cov_warn,
            rejection_reasons=[],
            selected_champion=winner_name,
            alert=None,
        )

    top_candidate = min(eval_candidates, key=lambda c: backtest_results[c].global_wape)
    top_cand_res = backtest_results[top_candidate]
    top_reasons = rejections.get(top_candidate, ["Failed quality gate"])

    return GateEvaluationResult(
        candidate_model=top_candidate,
        passed_gate=False,
        candidate_wape=top_cand_res.global_wape,
        baseline_wape=baseline_wape,
        candidate_bias=top_cand_res.global_bias,
        candidate_coverage=None,
        wape_improved=top_cand_res.global_wape < baseline_wape,
        bias_within_tolerance=abs(top_cand_res.global_bias) <= max_bias_threshold,
        coverage_warning=False,
        rejection_reasons=top_reasons,
        selected_champion=best_baseline,
        alert=ALERT_NO_MODEL_BEATS_BASELINE,
    )


def _compute_dataset_hash(df: pd.DataFrame) -> str:
    """Calculate deterministic SHA256 digest of input DataFrame values."""
    cols = [c for c in ("series_id", "date", "y") if c in df.columns]
    sample = df[cols].head(5000).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(sample).hexdigest()[:16]


def _save_and_log_model_artifact(
    model: Forecaster,
    tracker: ModelRunTracker,
    model_name: str,
) -> None:
    """Save model weights / config to a temporary directory and log as MLflow artifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        artifact_path = "model"

        if hasattr(model, "save"):
            model.save(str(tmp_path / "model_state"))
        else:
            joblib.dump(model, tmp_path / "model.joblib")

        meta = {
            "model_name": model_name,
            "class": model.__class__.__name__,
        }
        with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        tracker.log_artifact(str(tmp_path), artifact_path=artifact_path)


def _resolve_training_series(
    dataset_df: pd.DataFrame | None,
    session: Session | None,
    data_as_of: datetime.date | None,
    top_n: int,
) -> tuple[pd.DataFrame, datetime.date]:
    """Retrieve or build zero-filled training series and determine cutoff date."""
    if dataset_df is not None:
        series_df = dataset_df.copy()
    elif session is not None:
        if data_as_of is None:
            latest_date = session.execute(
                sa.select(sa.func.max(DimDate.date_key)).where(DimDate.is_complete.is_(True))
            ).scalar_one_or_none()
            if latest_date is None:
                raise ValueError("No complete dates found in dim_date to determine data_as_of")
            data_as_of = latest_date

        top_products = select_top_products(session, as_of=data_as_of, top_n=top_n)
        if not top_products:
            raise ValueError(f"No eligible products found up to {data_as_of}")

        p_ids = [int(p.product_id) for p in top_products]
        start_date = data_as_of - datetime.timedelta(days=730)
        series_df = build_zero_filled_series(
            session, product_ids=p_ids, start_date=start_date, as_of=data_as_of
        )
    else:
        raise ValueError("Must provide either 'dataset_df' or 'session' to train_forecast_pipeline")

    if series_df.empty:
        raise ValueError("Training series DataFrame is empty")

    if data_as_of is None:
        max_dt = pd.to_datetime(series_df["date"]).max()
        data_as_of = max_dt.date() if hasattr(max_dt, "date") else max_dt

    return series_df, data_as_of


def _log_and_register_run(
    model_name: str,
    forecaster: Forecaster,
    b_res: BacktestResult,
    int_eval: dict[str, Any],
    *,
    data_as_of: datetime.date,
    eff_hash: str,
    eff_seed: int,
    session: Session | None,
    tracking_uri: str | None,
    n_series: int,
    n_folds: int,
    horizon: int,
) -> str:
    """Record an individual model training run in MLflow and register model version."""
    with track_model_run(
        task="forecast",
        algorithm=model_name,
        data_as_of=data_as_of,
        dataset_hash=eff_hash,
        seed=eff_seed,
        session=session,
        tracking_uri=tracking_uri,
        tags={"n_series": n_series, "n_folds": n_folds, "horizon": horizon},
    ) as tracker:
        tracker.log_params(
            {
                "n_folds": n_folds,
                "horizon": horizon,
                "seed": eff_seed,
                "model_class": forecaster.__class__.__name__,
            }
        )

        scalar_metrics = {
            "wape": b_res.global_wape,
            "bias": b_res.global_bias,
            "smape": b_res.global_smape,
            "mae": b_res.global_mae,
            "rmse": b_res.global_rmse,
        }
        if "coverage" in int_eval:
            scalar_metrics["coverage"] = float(int_eval["coverage"])
        if "pinball_loss_q90" in int_eval:
            scalar_metrics["pinball_loss_q90"] = float(int_eval["pinball_loss_q90"])

        tracker.log_metrics(scalar_metrics)

        if session is not None and not b_res.metrics.empty:
            tracker.mirror_forecast_metrics(b_res.metrics, session=session)

        _save_and_log_model_artifact(forecaster, tracker, model_name)

        return register_model(
            mlflow_run_id=tracker.mlflow_run_id,
            task="forecast",
            artifact_path="model",
            tracking_uri=tracking_uri,
            tags={"algorithm": model_name, "wape": f"{b_res.global_wape:.4f}"},
        )


def train_forecast_pipeline(
    *,
    data_as_of: datetime.date | None = None,
    session: Session | None = None,
    dataset_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    models_to_run: Sequence[str] | None = None,
    candidate_names: Sequence[str] | None = None,
    n_folds: int | None = None,
    horizon: int | None = None,
    top_n: int | None = None,
    auto_promote: bool | None = None,
    seed: int | None = None,
    tracking_uri: str | None = None,
    dataset_hash: str | None = None,
    actor_id: uuid.UUID | None = None,
) -> TrainingPipelineResult:
    """Orchestrate demand forecasting training, backtesting, and champion gate."""
    settings = get_settings()
    eff_horizon = horizon or settings.forecast_horizon_days
    eff_n_folds = n_folds or settings.backtest_folds
    eff_seed = seed if seed is not None else settings.seed
    eff_top_n = top_n or settings.forecast_top_n
    eff_auto_promote = auto_promote if auto_promote is not None else settings.auto_promote

    series_df, data_as_of = _resolve_training_series(dataset_df, session, data_as_of, eff_top_n)
    eff_hash = dataset_hash or _compute_dataset_hash(series_df)
    n_unique_series = series_df["series_id"].nunique()
    default_models = ["seasonal_naive", "naive", "lightgbm"]
    models_list = list(models_to_run) if models_to_run is not None else default_models

    # 1. Execute backtests
    backtest_results: dict[str, BacktestResult] = {}
    fitted_models: dict[str, Forecaster] = {}

    for model_name in models_list:
        forecaster = _instantiate_model(model_name, seed=eff_seed, horizon=eff_horizon)
        b_res = run_backtest(
            model=forecaster,
            series_df=series_df,
            calendar_df=calendar_df,
            n_folds=eff_n_folds,
            horizon=eff_horizon,
        )
        backtest_results[model_name] = b_res
        fitted_models[model_name] = forecaster

    # 2. Fit residual quantiles and interval evaluations
    interval_evaluations: dict[str, dict[str, Any]] = {}
    fitted_quantiles: dict[str, ResidualQuantilesTable] = {}

    for model_name, b_res in backtest_results.items():
        q_table = fit_residual_quantiles(
            b_res.predictions,
            model_name=model_name,
            level=settings.interval_level,
        )
        fitted_quantiles[model_name] = q_table
        annotated_preds = apply_prediction_intervals(b_res.predictions, q_table)
        interval_evaluations[model_name] = evaluate_interval_metrics(annotated_preds)

    # 3. Champion Gate
    gate_result = evaluate_champion_gate(
        backtest_results,
        baseline_name="seasonal_naive",
        candidate_names=candidate_names,
        interval_evaluations=interval_evaluations,
    )

    # 4. Refit on complete dataset and register in MLflow
    registered_versions: dict[str, str] = {}
    history_up_to_as_of = series_df[pd.to_datetime(series_df["date"]).dt.date <= data_as_of]

    for model_name, forecaster in fitted_models.items():
        forecaster.fit(history_up_to_as_of)
        registered_versions[model_name] = _log_and_register_run(
            model_name,
            forecaster,
            backtest_results[model_name],
            interval_evaluations.get(model_name, {}),
            data_as_of=data_as_of,
            eff_hash=eff_hash,
            eff_seed=eff_seed,
            session=session,
            tracking_uri=tracking_uri,
            n_series=n_unique_series,
            n_folds=eff_n_folds,
            horizon=eff_horizon,
        )

    # 5. Promotion handling
    champion_model_name = gate_result.selected_champion
    champion_version = registered_versions.get(champion_model_name)

    if eff_auto_promote and champion_version is not None:
        promote_champion(
            task="forecast",
            version=champion_version,
            session=session,
            model_name=REGISTERED_MODEL_FORECAST,
            tracking_uri=tracking_uri,
            actor_id=actor_id,
        )

    return TrainingPipelineResult(
        data_as_of=data_as_of,
        backtest_results=backtest_results,
        interval_evaluations=interval_evaluations,
        gate_result=gate_result,
        champion_model=champion_model_name,
        champion_version=champion_version,
        registered_versions=registered_versions,
        intervals_table=fitted_quantiles.get(champion_model_name),
        total_series=n_unique_series,
        n_folds=eff_n_folds,
        horizon=eff_horizon,
    )

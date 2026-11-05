"""Batch forecasting inference, prediction intervals, and active output pointer management.

Implements P2-11 & TRD §5.1 / §6 / DATABASE §11:
- Automatic model selection: active champion (or fallback to best baseline)
- Fitting/predicting on historical series up to data_as_of (origin_date)
- Residual-quantile prediction intervals application (0 <= yhat_lower <= yhat <= yhat_upper)
- Atomic transactional persistence: writes ml.forecast rows and advances ml.active_output
  pointer in the SAME database transaction
- If any error occurs, rolls back transaction leaving previous ml.active_output unchanged
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import DimDate
from retailpulse.db.models.ml import ActiveOutput, Forecast, ModelRun
from retailpulse.features.forecast_features import (
    build_zero_filled_series,
    select_top_products,
)
from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.intervals import (
    ResidualQuantilesTable,
    apply_prediction_intervals,
)
from retailpulse.ml.forecasting.train import instantiate_model
from retailpulse.ml.registry import get_champion_run

logger = get_logger(__name__)

TASK_FORECAST = "forecast"
DEFAULT_BASELINE = "seasonal_naive"


@dataclass(frozen=True)
class BatchForecastResult:
    """Consolidated summary of a batch forecasting run."""

    model_run_id: uuid.UUID
    model_name: str
    origin_date: datetime.date
    horizon: int
    n_series: int
    n_forecast_rows: int
    interval_level: float
    active_output_updated: bool
    predictions_df: pd.DataFrame


def _resolve_model_and_run(
    session: Session,
    explicit_model_name: str | None,
    data_as_of: datetime.date,
    seed: int,
) -> tuple[str, uuid.UUID]:
    """Identify which model algorithm to execute and obtain a valid ModelRun ID."""
    if explicit_model_name:
        # Search for recent matching run
        stmt = (
            sa.select(ModelRun)
            .where(ModelRun.task == TASK_FORECAST, ModelRun.algorithm == explicit_model_name)
            .order_by(ModelRun.trained_at.desc())
        )
        existing_run = session.execute(stmt).scalars().first()
        if existing_run is not None:
            return explicit_model_name, existing_run.run_id

        # Create ad-hoc model run record
        new_run_id = uuid.uuid4()
        new_run = ModelRun(
            run_id=new_run_id,
            task=TASK_FORECAST,
            algorithm=explicit_model_name,
            mlflow_run_id=f"adhoc-{new_run_id.hex}",
            status="succeeded",
            is_champion=False,
            data_end=data_as_of,
            params={"source": "ad_hoc_batch_predict"},
            metrics={},
            seed=seed,
        )
        session.add(new_run)
        session.flush()
        return explicit_model_name, new_run_id

    # No model specified: look for current champion
    champ_run = get_champion_run(TASK_FORECAST, session)
    if champ_run is not None:
        return champ_run.algorithm, champ_run.run_id

    # Fallback to best baseline
    stmt = (
        sa.select(ModelRun)
        .where(ModelRun.task == TASK_FORECAST, ModelRun.algorithm == DEFAULT_BASELINE)
        .order_by(ModelRun.trained_at.desc())
    )
    fallback_run = session.execute(stmt).scalars().first()
    if fallback_run is not None:
        return DEFAULT_BASELINE, fallback_run.run_id

    # Create baseline run record
    new_run_id = uuid.uuid4()
    new_run = ModelRun(
        run_id=new_run_id,
        task=TASK_FORECAST,
        algorithm=DEFAULT_BASELINE,
        mlflow_run_id=f"fallback-{new_run_id.hex}",
        status="succeeded",
        is_champion=True,
        data_end=data_as_of,
        params={"source": "fallback_baseline_batch_predict"},
        metrics={},
        seed=seed,
    )
    session.add(new_run)
    session.flush()
    return DEFAULT_BASELINE, new_run_id


def _resolve_data_as_of(
    session: Session,
    dataset_df: pd.DataFrame | None,
    data_as_of: datetime.date | None,
) -> datetime.date:
    """Resolve data cutoff date from explicit parameter, dataset, or dim_date."""
    if data_as_of is not None:
        return data_as_of

    if dataset_df is not None and not dataset_df.empty:
        max_val = pd.to_datetime(dataset_df["date"]).max()
        if hasattr(max_val, "date"):
            return cast(datetime.date, max_val.date())
        return datetime.date.fromisoformat(str(max_val)[:10])

    latest_complete = session.execute(
        sa.select(sa.func.max(DimDate.date_key)).where(DimDate.is_complete.is_(True))
    ).scalar_one_or_none()
    if latest_complete is not None:
        return latest_complete

    latest_any = session.execute(sa.select(sa.func.max(DimDate.date_key))).scalar_one_or_none()
    if latest_any is not None:
        return latest_any

    raise ValueError("Cannot determine data_as_of: database has no complete dates")


def _resolve_scoring_series(
    session: Session,
    dataset_df: pd.DataFrame | None,
    data_as_of: datetime.date,
    top_n: int,
) -> pd.DataFrame:
    """Retrieve or prepare trading-day series up to data_as_of."""
    if dataset_df is not None:
        return dataset_df.copy()

    top_products = select_top_products(session, as_of=data_as_of, top_n=top_n)
    if not top_products:
        raise ValueError(f"No eligible products found up to {data_as_of}")

    p_ids = [int(p.product_id) for p in top_products]
    start_date = data_as_of - datetime.timedelta(days=730)
    return build_zero_filled_series(
        session, product_ids=p_ids, start_date=start_date, as_of=data_as_of
    )


def _apply_default_intervals(
    preds_df: pd.DataFrame,
    quantiles_table: ResidualQuantilesTable | None,
    interval_level: float,
) -> pd.DataFrame:
    """Apply residual quantiles or fallback heuristic bounds enforcing db constraints."""
    out = preds_df.copy()

    if quantiles_table is not None:
        out = apply_prediction_intervals(out, quantiles_table)
    # Default heuristic intervals if table not calibrated
    elif "yhat_lower" not in out.columns or "yhat_upper" not in out.columns:
        yhat = out["yhat"].astype(float).to_numpy()
        out["yhat_lower"] = np.maximum(0.0, yhat * 0.80)
        out["yhat_upper"] = np.maximum(yhat, yhat * 1.20)
        out["interval_level"] = interval_level

    # Strictly enforce: 0 <= yhat_lower <= yhat <= yhat_upper
    yhat_arr = np.maximum(0.0, out["yhat"].astype(float).to_numpy())
    yl_arr = np.maximum(0.0, np.minimum(yhat_arr, out["yhat_lower"].astype(float).to_numpy()))
    yu_arr = np.maximum(yhat_arr, out["yhat_upper"].astype(float).to_numpy())

    out["yhat"] = yhat_arr
    out["yhat_lower"] = yl_arr
    out["yhat_upper"] = yu_arr
    if "interval_level" not in out.columns:
        out["interval_level"] = interval_level

    return out


def _build_forecast_entities(
    scored_df: pd.DataFrame,
    model_run_id: uuid.UUID,
    origin_date: datetime.date,
    store_id: int,
    interval_level: float,
) -> list[Forecast]:
    """Convert scoring DataFrame into SQLAlchemy Forecast model entities."""
    entities: list[Forecast] = []

    for _, row in scored_df.iterrows():
        p_id = int(row.get("product_id", row.get("series_id")))
        t_date = row["date"]
        if isinstance(t_date, str):
            t_date = datetime.date.fromisoformat(t_date)
        elif hasattr(t_date, "date"):
            t_date = t_date.date()

        h_day = int(row.get("horizon_day", (t_date - origin_date).days))
        h_day = max(h_day, 1)

        yhat_dec = Decimal(f"{float(row['yhat']):.4f}")
        yl_dec = Decimal(f"{float(row['yhat_lower']):.4f}")
        yu_dec = Decimal(f"{float(row['yhat_upper']):.4f}")

        # Ensure Decimal bounds: yhat_lower <= yhat <= yhat_upper
        yl_dec = min(yl_dec, yhat_dec)
        yu_dec = max(yu_dec, yhat_dec)

        f_entity = Forecast(
            model_run_id=model_run_id,
            store_id=store_id,
            product_id=p_id,
            origin_date=origin_date,
            target_date=t_date,
            horizon_day=h_day,
            yhat=yhat_dec,
            yhat_lower=yl_dec,
            yhat_upper=yu_dec,
            interval_level=Decimal(f"{float(row.get('interval_level', interval_level)):.2f}"),
        )
        entities.append(f_entity)

    return entities


def predict_batch_forecast(
    session: Session,
    *,
    data_as_of: datetime.date | None = None,
    model_name: str | None = None,
    horizon: int | None = None,
    top_n: int | None = None,
    interval_level: float | None = None,
    store_id: int = 1,
    seed: int | None = None,
    dataset_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    quantiles_table: ResidualQuantilesTable | None = None,
) -> BatchForecastResult:
    """Generate demand forecasts and atomically update active output in one transaction.

    Per TRD §5.1, §6 & DATABASE §11:
    - Automatically picks current champion model (or falls back to best baseline).
    - Fits forecaster on all historical data up to data_as_of.
    - Generates 1..horizon day forecasts with residual-based intervals.
    - Inserts ml.forecast rows and advances ml.active_output pointer in the SAME transaction.
    - On failure, rolls back transaction leaving previous ml.active_output unchanged.

    Args:
        session: Active SQLAlchemy session.
        data_as_of: Forecasting origin cutoff date. If omitted, uses latest complete date.
        model_name: Optional explicit model algorithm to run. If None, uses champion.
        horizon: Forecast horizon length in days (defaults to config: 28).
        top_n: Number of top SKUs to forecast (defaults to config: 300).
        interval_level: Nominal interval coverage (defaults to config: 0.80).
        store_id: Target store identifier (defaults to 1).
        seed: Random seed for deterministic execution.
        dataset_df: Optional preloaded sales series DataFrame (series_id, date, y).
        calendar_df: Optional calendar features DataFrame.
        quantiles_table: Optional calibrated ResidualQuantilesTable for intervals.

    Returns:
        Structured BatchForecastResult.
    """
    settings = get_settings()
    eff_horizon = horizon or settings.forecast_horizon_days
    eff_top_n = top_n or settings.forecast_top_n
    eff_level = interval_level or settings.interval_level
    eff_seed = seed if seed is not None else settings.seed

    origin_date = _resolve_data_as_of(session, dataset_df, data_as_of)
    eff_model, run_id = _resolve_model_and_run(session, model_name, origin_date, eff_seed)

    logger.info(
        "starting batch forecasting run",
        model=eff_model,
        model_run_id=str(run_id),
        origin_date=str(origin_date),
        horizon=eff_horizon,
    )

    series_df = _resolve_scoring_series(session, dataset_df, origin_date, eff_top_n)
    if series_df.empty:
        raise ValueError("Cannot score: training series DataFrame is empty")

    n_series = series_df["series_id"].nunique()

    # Fit forecaster and generate point predictions
    forecaster: Forecaster = instantiate_model(eff_model, seed=eff_seed, horizon=eff_horizon)
    history_df = series_df[pd.to_datetime(series_df["date"]).dt.date <= origin_date]
    forecaster.fit(history_df)

    raw_preds = forecaster.predict(
        origin=origin_date,
        horizon=eff_horizon,
        calendar_df=calendar_df,
    )
    if raw_preds.empty:
        raise RuntimeError(f"Forecaster '{eff_model}' generated empty predictions")

    # Add horizon_day if missing
    if "horizon_day" not in raw_preds.columns:
        dt_series = pd.to_datetime(raw_preds["date"]).dt.date
        raw_preds["horizon_day"] = [(d - origin_date).days for d in dt_series]

    # Add product_id / series_id compatibility
    if "product_id" not in raw_preds.columns and "series_id" in raw_preds.columns:
        raw_preds["product_id"] = raw_preds["series_id"]

    # Calibrate prediction intervals
    scored_df = _apply_default_intervals(raw_preds, quantiles_table, eff_level)

    # Build DB forecast entities
    forecast_entities = _build_forecast_entities(
        scored_df,
        model_run_id=run_id,
        origin_date=origin_date,
        store_id=store_id,
        interval_level=eff_level,
    )

    # Atomic transaction: write ml.forecast rows and move ml.active_output pointer
    try:
        # 1. Clean previous rows for same (model_run_id, origin_date) for idempotency
        session.execute(
            sa.delete(Forecast).where(
                Forecast.model_run_id == run_id,
                Forecast.origin_date == origin_date,
            )
        )

        # 2. Insert new forecast rows
        session.add_all(forecast_entities)
        session.flush()

        # 3. Move ml.active_output pointer in the SAME transaction
        active = session.get(ActiveOutput, TASK_FORECAST)
        if active is None:
            active = ActiveOutput(
                task=TASK_FORECAST,
                model_run_id=run_id,
                data_as_of=origin_date,
            )
            session.add(active)
        else:
            active.model_run_id = run_id
            active.data_as_of = origin_date
            active.scored_at = sa.func.now()

        session.commit()
        logger.info(
            "finished batch forecasting; active_output pointer updated",
            model_run_id=str(run_id),
            origin_date=str(origin_date),
            rows_written=len(forecast_entities),
        )

    except Exception as exc:
        session.rollback()
        logger.error(
            "batch forecasting failed; active_output pointer preserved",
            error=str(exc),
            model_run_id=str(run_id),
        )
        raise

    return BatchForecastResult(
        model_run_id=run_id,
        model_name=eff_model,
        origin_date=origin_date,
        horizon=eff_horizon,
        n_series=n_series,
        n_forecast_rows=len(forecast_entities),
        interval_level=eff_level,
        active_output_updated=True,
        predictions_df=scored_df,
    )

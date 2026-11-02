"""Rolling-origin backtest harness for demand forecasting models.

Implements P2-04 & TRD §5.1:
- Rolling-origin cross-validation with configurable folds and horizon (default folds=5, horizon=28)
- Contiguous trading-day test windows without data leakage
- Strict assertion verifying training data does not exceed origin date (date <= origin)
- Training target clipping at 99.5th percentile per series (unclipped actuals for evaluation)
- Model-agnostic execution conforming to Forecaster protocol
- Full metrics calculation across scopes (global, abc, horizon, sku)
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from retailpulse.features.forecast_features import clip_training_targets
from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.base import Forecaster, resolve_target_dates
from retailpulse.ml.forecasting.metrics import compute_forecast_metrics_by_scope

logger = get_logger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """Consolidated outputs and metrics from a rolling-origin backtest run."""

    model_name: str
    n_folds: int
    horizon: int
    origins: list[datetime.date]
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    global_wape: float
    global_bias: float
    global_smape: float
    global_mae: float
    global_rmse: float


def compute_backtest_origins(
    available_dates: Sequence[datetime.date],
    *,
    n_folds: int = 5,
    horizon: int = 28,
    step_days: int | None = None,
    min_train_days: int = 28,
) -> list[datetime.date]:
    """Calculate rolling-origin cutoff dates stepping backwards from latest data.

    Args:
        available_dates: Sorted sequence of available calendar/trading dates.
        n_folds: Number of backtest evaluation folds.
        horizon: Forecast horizon length in trading days.
        step_days: Step between origins (defaults to horizon for contiguous evaluation).
        min_train_days: Minimum number of historical dates required before first origin.

    Returns:
        List of origin dates sorted chronologically (earliest to latest).

    Raises:
        ValueError: If available dates are insufficient to support requested folds and horizon.
    """
    step = step_days if step_days is not None else horizon
    unique_dates = sorted(set(available_dates))
    total_dates = len(unique_dates)

    required_dates = min_train_days + horizon + (n_folds - 1) * step
    if total_dates < required_dates:
        raise ValueError(
            f"Insufficient dates for {n_folds} folds with horizon {horizon} and "
            f"min_train_days {min_train_days}: available {total_dates}, required {required_dates}"
        )

    # Origin k has a test window spanning horizon dates after it.
    # The final fold (k = n_folds - 1) ends at unique_dates[-1].
    # Its origin index is total_dates - horizon - 1.
    last_origin_idx = total_dates - horizon - 1
    origins: list[datetime.date] = []

    for i in range(n_folds):
        # Stepping backwards from last origin: (n_folds - 1 - i) * step
        idx = last_origin_idx - (n_folds - 1 - i) * step
        if idx < min_train_days - 1:
            raise ValueError(
                f"Computed origin index {idx} violates min_train_days ({min_train_days})"
            )
        origins.append(unique_dates[idx])

    return origins


def assert_no_leakage(train_df: pd.DataFrame, origin: datetime.date, fold_idx: int = 0) -> None:
    """Assert strictly that no training observations exceed the origin cutoff date.

    Raises:
        AssertionError: If any training record has date > origin.
    """
    if not train_df.empty and "date" in train_df.columns:
        max_train_date = train_df["date"].max()
        if max_train_date > origin:
            raise AssertionError(
                f"Data leakage detected in backtest fold {fold_idx}: "
                f"max training date {max_train_date} > origin {origin}"
            )


def _validate_and_prepare_training_data(
    series_df: pd.DataFrame,
    origin: datetime.date,
    *,
    fold_idx: int,
    clip_targets: bool,
    clip_quantile: float,
) -> pd.DataFrame:
    """Filter and validate training data strictly at or before origin date."""
    train_mask = series_df["date"] <= origin
    train_df = series_df.loc[train_mask].copy()

    # Strict assertion: no data leakage beyond origin
    assert_no_leakage(train_df, origin, fold_idx=fold_idx)

    if clip_targets and not train_df.empty and "y" in train_df.columns:
        train_df = clip_training_targets(
            train_df,
            target_col="y",
            group_col="series_id",
            quantile=clip_quantile,
        )

    return train_df


def _evaluate_single_fold(
    model: Forecaster,
    series_df: pd.DataFrame,
    calendar_df: pd.DataFrame | None,
    *,
    origin: datetime.date,
    horizon: int,
    fold_idx: int,
    clip_targets: bool,
    clip_quantile: float,
) -> pd.DataFrame:
    """Fit model on data <= origin and generate predictions for horizon."""
    train_df = _validate_and_prepare_training_data(
        series_df,
        origin,
        fold_idx=fold_idx,
        clip_targets=clip_targets,
        clip_quantile=clip_quantile,
    )

    # Fit model on training slice
    model.fit(train_df)

    # Generate predictions
    preds = model.predict(origin=origin, horizon=horizon, calendar_df=calendar_df)
    if preds.empty:
        return pd.DataFrame()

    target_dates = resolve_target_dates(origin, horizon, calendar_df)
    date_to_horizon = {d: h for h, d in enumerate(target_dates, start=1)}

    # Extract unclipped test actuals from series_df (strictly > origin)
    test_mask = (series_df["date"] > origin) & (series_df["date"].isin(target_dates))
    test_actuals = series_df.loc[test_mask].copy()

    merge_cols = ["series_id", "date"]
    merged = preds.merge(test_actuals, on=merge_cols, suffixes=("", "_actual"))

    # Map product_id and abc_class if present
    prod_id_col = "product_id" if "product_id" in merged.columns else "series_id"
    abc_col = "abc_class" if "abc_class" in merged.columns else None

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        y_val = float(row.get("y", 0.0))
        yhat_val = max(0.0, float(row.get("yhat", 0.0)))
        t_date = row["date"]
        h_day = date_to_horizon.get(t_date, 1)

        record: dict[str, Any] = {
            "fold": fold_idx,
            "origin": origin,
            "series_id": row["series_id"],
            "product_id": row[prod_id_col],
            "date": t_date,
            "horizon_day": h_day,
            "y": y_val,
            "yhat": yhat_val,
            "residual": y_val - yhat_val,
        }
        if abc_col and abc_col in row:
            record["abc_class"] = str(row[abc_col])

        rows.append(record)

    return pd.DataFrame(rows)


def run_backtest(
    model: Forecaster | Callable[[], Forecaster],
    series_df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
    *,
    n_folds: int = 5,
    horizon: int = 28,
    step_days: int | None = None,
    min_train_days: int = 28,
    clip_targets: bool = True,
    clip_quantile: float = 0.995,
) -> BacktestResult:
    """Execute rolling-origin backtest across historical series.

    Args:
        model: Forecaster instance or factory callable generating fresh instances per fold.
        series_df: DataFrame with trading day series [series_id, date, y, ...].
        calendar_df: Optional dim_date DataFrame specifying trading days.
        n_folds: Number of rolling-origin folds.
        horizon: Forecast horizon length in trading days.
        step_days: Step between origins (defaults to horizon).
        min_train_days: Minimum training observations required before first origin.
        clip_targets: Whether to clip wholesale spikes in training data at clip_quantile.
        clip_quantile: Percentile for training spike clipping (default 0.995).

    Returns:
        BacktestResult with combined predictions and scoped evaluation metrics.
    """
    if series_df.empty:
        raise ValueError("Cannot run backtest on empty series_df")

    # Determine unique available dates
    trading_dates = sorted(series_df["date"].unique())
    origins = compute_backtest_origins(
        trading_dates,
        n_folds=n_folds,
        horizon=horizon,
        step_days=step_days,
        min_train_days=min_train_days,
    )

    logger.info(
        "starting rolling-origin backtest",
        n_folds=n_folds,
        horizon=horizon,
        origins=[str(d) for d in origins],
    )

    # Prepare in-sample historical actuals for MASE/RMSSE scaling per series
    in_sample_history: dict[Any, np.ndarray[Any, Any]] = {}
    for sid, group in series_df.groupby("series_id"):
        in_sample_history[sid] = group.sort_values("date")["y"].to_numpy(dtype=float)

    fold_predictions: list[pd.DataFrame] = []

    for fold_idx, origin in enumerate(origins):
        # Create fresh model instance if factory was provided, else reuse
        model_instance = model() if callable(model) else model

        fold_preds = _evaluate_single_fold(
            model=model_instance,
            series_df=series_df,
            calendar_df=calendar_df,
            origin=origin,
            horizon=horizon,
            fold_idx=fold_idx,
            clip_targets=clip_targets,
            clip_quantile=clip_quantile,
        )
        if not fold_preds.empty:
            fold_predictions.append(fold_preds)

    if not fold_predictions:
        raise RuntimeError("Backtest produced no predictions across all folds")

    all_preds = pd.concat(fold_predictions, ignore_index=True)

    # Compute standardized scoped metrics across all folds
    metrics_df = compute_forecast_metrics_by_scope(
        all_preds,
        in_sample_history=in_sample_history,
    )

    # Extract global headline metrics
    global_rows = metrics_df[
        (metrics_df["scope"] == "global") & (metrics_df["scope_key"] == "all")
    ].set_index("metric")["value"]

    model_name = getattr(model, "name", "forecaster") if not callable(model) else model().name

    return BacktestResult(
        model_name=model_name,
        n_folds=n_folds,
        horizon=horizon,
        origins=origins,
        predictions=all_preds,
        metrics=metrics_df,
        global_wape=float(global_rows.get("wape", float("nan"))),
        global_bias=float(global_rows.get("bias", float("nan"))),
        global_smape=float(global_rows.get("smape", float("nan"))),
        global_mae=float(global_rows.get("mae", float("nan"))),
        global_rmse=float(global_rows.get("rmse", float("nan"))),
    )

"""Demand forecasting accuracy metrics, interval evaluation, and champion gate.

Implements P2-02, TRD §5.1, and DATABASE.md §7:
- WAPE (Weighted Absolute Percentage Error) with zero-scale protection
- MASE (Mean Absolute Scaled Error) with in-sample seasonal naive baseline scale (m=7)
- RMSSE (Root Mean Squared Scaled Error)
- Signed percentage bias
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Prediction interval coverage (80% nominal level)
- Pinball / quantile loss for prediction intervals
- Multi-scope aggregation (global, abc class, horizon buckets 1-7/8-14/15-28, sku)
- Champion promotion gate logic
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from retailpulse.logging import get_logger

logger = get_logger(__name__)


def wape(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
) -> float:
    """Compute Weighted Absolute Percentage Error (WAPE).

    Formula: sum(|y - yhat|) / sum(y).

    Zero-scale handling:
    - If sum(y) == 0 and sum(|y - yhat|) == 0: returns 0.0 (perfect zero forecast).
    - If sum(y) == 0 and sum(|y - yhat|) > 0: returns NaN (non-zero forecast on zero demand).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    total_actual = float(np.sum(yt))
    total_abs_err = float(np.sum(np.abs(yt - yp)))

    if total_actual == 0.0:
        return 0.0 if total_abs_err == 0.0 else float("nan")

    return total_abs_err / total_actual


def bias(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
) -> float:
    """Compute signed percentage forecast bias.

    Formula: sum(yhat - y) / sum(y).
    A positive bias indicates over-forecasting; negative indicates under-forecasting.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    total_actual = float(np.sum(yt))
    total_diff = float(np.sum(yp - yt))

    if total_actual == 0.0:
        return 0.0 if total_diff == 0.0 else float("nan")

    return total_diff / total_actual


def mae(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
) -> float:
    """Compute Mean Absolute Error (MAE)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def compute_seasonal_naive_scale(
    y_train: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    seasonal_period: int = 7,
) -> float:
    """Compute in-sample seasonal naive MAE scale for MASE calculation.

    Scale = mean(|y_train[t] - y_train[t - m]|) for t in m..N.
    """
    yt = np.asarray(y_train, dtype=float)
    if len(yt) <= seasonal_period:
        return float("nan")

    diffs = np.abs(yt[seasonal_period:] - yt[:-seasonal_period])
    return float(np.mean(diffs))


def mase(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_train: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    *,
    seasonal_period: int = 7,
) -> float:
    """Compute Mean Absolute Scaled Error (MASE).

    Scale is computed on in-sample training actuals with the given seasonal period (default 7).
    Returns NaN if in-sample scale is 0 or insufficient history.
    """
    scale = compute_seasonal_naive_scale(y_train, seasonal_period=seasonal_period)
    if math.isnan(scale) or scale == 0.0:
        return float("nan")

    forecast_mae = mae(y_true, y_pred)
    if math.isnan(forecast_mae):
        return float("nan")

    return forecast_mae / scale


def rmsse(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_train: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    *,
    seasonal_period: int = 7,
) -> float:
    """Compute Root Mean Squared Scaled Error (RMSSE).

    Scale is computed from the mean squared seasonal differences of in-sample training actuals.
    """
    yt_train = np.asarray(y_train, dtype=float)
    if len(yt_train) <= seasonal_period:
        return float("nan")

    sq_diffs = (yt_train[seasonal_period:] - yt_train[:-seasonal_period]) ** 2
    mse_scale = float(np.mean(sq_diffs))
    if mse_scale == 0.0:
        return float("nan")

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")

    mse_forecast = float(np.mean((yt - yp) ** 2))
    return float(np.sqrt(mse_forecast / mse_scale))


def smape(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (sMAPE) in [0.0, 1.0].

    Formula: mean(2 * |y - yhat| / (|y| + |yhat|)).
    Where |y| + |yhat| == 0, error is treated as 0.0.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return float("nan")

    denom = np.abs(yt) + np.abs(yp)
    abs_err = np.abs(yt - yp)

    valid_mask = denom > 0
    scores = np.zeros_like(yt, dtype=float)
    scores[valid_mask] = (2.0 * abs_err[valid_mask]) / denom[valid_mask]

    return float(np.mean(scores))


def coverage(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_lower: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_upper: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
) -> float:
    """Compute empirical coverage rate of prediction intervals.

    Fraction of actuals y falling within [y_lower, y_upper] inclusive.
    """
    yt = np.asarray(y_true, dtype=float)
    yl = np.asarray(y_lower, dtype=float)
    yu = np.asarray(y_upper, dtype=float)
    if len(yt) == 0:
        return float("nan")

    in_bounds = (yt >= yl) & (yt <= yu)
    return float(np.mean(in_bounds.astype(float)))


def pinball_loss(
    y_true: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    y_pred_q: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    *,
    quantile: float,
) -> float:
    """Compute mean pinball / quantile loss for quantile q in (0, 1).

    Formula: mean(max(q * (y - yhat_q), (1 - q) * (yhat_q - y))).
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be strictly in (0, 1), got {quantile}")

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_q, dtype=float)
    if len(yt) == 0:
        return float("nan")

    err = yt - yp
    loss = np.maximum(quantile * err, (quantile - 1.0) * err)
    return float(np.mean(loss))


def check_champion_gate(
    candidate_wape: float,
    baseline_wape: float,
    candidate_bias: float,
    candidate_coverage: float | None = None,
    *,
    max_abs_bias: float = 0.15,
    min_coverage: float = 0.65,
    max_coverage: float = 0.95,
) -> tuple[bool, list[str], str]:
    """Evaluate candidate model against champion gate rules per TRD §5.1.

    Rules:
    - Candidate global WAPE must be strictly lower than baseline WAPE (seasonal naive).
    - Absolute candidate bias must be <= max_abs_bias (default 0.15 / 15%).
    - Coverage outside [0.65, 0.95] generates a warning alert, not a block.

    Returns:
        tuple of (gate_passed, warning_messages, reason_description).
    """
    warnings: list[str] = []

    if math.isnan(candidate_wape):
        return False, warnings, "Candidate WAPE is NaN."
    if math.isnan(baseline_wape):
        return False, warnings, "Baseline WAPE is NaN."

    if candidate_wape >= baseline_wape:
        reason = (
            f"Candidate WAPE ({candidate_wape:.4f}) does not beat baseline ({baseline_wape:.4f})."
        )
        return False, warnings, reason

    if abs(candidate_bias) > max_abs_bias:
        reason = (
            f"Candidate |bias| ({abs(candidate_bias):.4f}) exceeds threshold ({max_abs_bias:.2f})."
        )
        return False, warnings, reason

    if (
        candidate_coverage is not None
        and not math.isnan(candidate_coverage)
        and (candidate_coverage < min_coverage or candidate_coverage > max_coverage)
    ):
        msg = (
            f"Candidate interval coverage ({candidate_coverage:.2%}) is outside "
            f"nominal tolerance [{min_coverage:.0%}, {max_coverage:.0%}]."
        )
        warnings.append(msg)

    reason = (
        f"Candidate beats baseline WAPE ({candidate_wape:.4f} < {baseline_wape:.4f}) "
        f"with acceptable bias ({candidate_bias:.4f})."
    )
    return True, warnings, reason


def _compute_bucket_metrics(subset: pd.DataFrame) -> dict[str, float]:
    """Calculate core metrics for a single slice of predictions."""
    y = subset["y"].to_numpy(dtype=float)
    yhat = subset["yhat"].to_numpy(dtype=float)

    metrics = {
        "wape": wape(y, yhat),
        "smape": smape(y, yhat),
        "bias": bias(y, yhat),
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
    }

    if "yhat_lower" in subset.columns and "yhat_upper" in subset.columns:
        yl = subset["yhat_lower"].to_numpy(dtype=float)
        yu = subset["yhat_upper"].to_numpy(dtype=float)
        metrics["coverage"] = coverage(y, yl, yu)
        metrics["pinball"] = pinball_loss(y, yu, quantile=0.9)

    return metrics


def _append_scope_records(
    records: list[dict[str, Any]],
    subset: pd.DataFrame,
    scope: str,
    scope_key: str,
    n_series: int,
) -> None:
    """Compute and append valid non-NaN metric rows to records."""
    metrics_map = _compute_bucket_metrics(subset)
    for m_name, val in metrics_map.items():
        if not math.isnan(val):
            records.append(
                {
                    "scope": scope,
                    "scope_key": scope_key,
                    "metric": m_name,
                    "value": val,
                    "n_series": n_series,
                }
            )


def compute_forecast_metrics_by_scope(
    eval_df: pd.DataFrame,
    *,
    in_sample_history: Mapping[Any, np.ndarray[Any, Any] | Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Compute aggregated metrics across scopes matching ml.forecast_metric schema.

    Scopes produced:
    - 'global' (scope_key='all')
    - 'abc' (scope_key='A', 'B', 'C')
    - 'horizon' (scope_key='1-7', '8-14', '15-28')
    - 'sku' (scope_key=product_id)

    Returns DataFrame with columns:
    [scope, scope_key, metric, value, n_series]
    """
    required_cols = {"series_id", "product_id", "horizon_day", "y", "yhat"}
    missing = required_cols - set(eval_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in eval_df: {missing}")

    records: list[dict[str, Any]] = []

    # 1. Global scope
    n_global = int(eval_df["series_id"].nunique())
    _append_scope_records(records, eval_df, "global", "all", n_global)

    # 2. ABC class scope
    if "abc_class" in eval_df.columns:
        for abc_key, group in eval_df.groupby("abc_class"):
            _append_scope_records(
                records, group, "abc", str(abc_key), int(group["series_id"].nunique())
            )

    # 3. Horizon buckets scope (1-7, 8-14, 15-28)
    horizon_bins = [
        ("1-7", (1, 7)),
        ("8-14", (8, 14)),
        ("15-28", (15, 28)),
    ]
    for bucket_name, (h_min, h_max) in horizon_bins:
        mask = (eval_df["horizon_day"] >= h_min) & (eval_df["horizon_day"] <= h_max)
        subset = eval_df.loc[mask]
        if not subset.empty:
            _append_scope_records(
                records, subset, "horizon", bucket_name, int(subset["series_id"].nunique())
            )

    # 4. Per-SKU scope
    for pid, group in eval_df.groupby("product_id"):
        sku_key = str(pid)
        _append_scope_records(records, group, "sku", sku_key, 1)

        # In-sample MASE / RMSSE if training history provided
        if in_sample_history and pid in in_sample_history:
            ytrain = in_sample_history[pid]
            mase_val = mase(group["y"], group["yhat"], ytrain)
            if not math.isnan(mase_val):
                records.append(
                    {
                        "scope": "sku",
                        "scope_key": sku_key,
                        "metric": "mase",
                        "value": mase_val,
                        "n_series": 1,
                    }
                )
            rmsse_val = rmsse(group["y"], group["yhat"], ytrain)
            if not math.isnan(rmsse_val):
                records.append(
                    {
                        "scope": "sku",
                        "scope_key": sku_key,
                        "metric": "rmsse",
                        "value": rmsse_val,
                        "n_series": 1,
                    }
                )

    return pd.DataFrame(records)

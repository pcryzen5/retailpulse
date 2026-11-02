"""Residual-quantile prediction intervals for demand forecasts.

Implements P2-05 & TRD §5.1:
- Empirical residual quantiles from backtests per (model, ABC class, horizon bucket)
- Quantile resolution for nominal coverage (default 80% level: q10 and q90)
- Lower and upper bound projection:
    lower = min(yhat, max(0, yhat + q10))
    upper = max(yhat, yhat + q90)
- Enforces database constraint: 0 <= yhat_lower <= yhat <= yhat_upper
- Robust fallback hierarchy for small-sample buckets:
    (abc, horizon) -> horizon -> abc -> global
- Evaluation helpers for interval coverage and pinball score
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.metrics import coverage, pinball_loss

logger = get_logger(__name__)

# Standard horizon buckets matching DATABASE.md and TRD §5.1
HORIZON_BUCKETS = [
    ("1-7", 1, 7),
    ("8-14", 8, 14),
    ("15-28", 15, 28),
]


def map_horizon_to_bucket(horizon_day: int) -> str:
    """Map a 1-based horizon day integer to its standard horizon bucket string.

    Args:
        horizon_day: Horizon day (>= 1).

    Returns:
        Bucket string '1-7', '8-14', or '15-28'.
    """
    if horizon_day <= 7:
        return "1-7"
    if horizon_day <= 14:
        return "8-14"
    return "15-28"


@dataclass(frozen=True)
class ResidualQuantilesTable:
    """Calibrated empirical residual quantiles partitioned by ABC class and horizon bucket."""

    model_name: str
    level: float
    q_lower: float
    q_upper: float
    # Map (abc_class, horizon_bucket) -> (lower_delta, upper_delta)
    buckets: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    # Fallback by horizon bucket -> (lower_delta, upper_delta)
    horizon_fallback: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Fallback by ABC class -> (lower_delta, upper_delta)
    abc_fallback: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Global fallback -> (lower_delta, upper_delta)
    global_fallback: tuple[float, float] = (0.0, 0.0)
    # Sample counts per bucket
    sample_counts: dict[str, int] = field(default_factory=dict)

    def lookup(self, abc_class: str, horizon_bucket: str) -> tuple[float, float]:
        """Look up lower and upper residual deltas with tiered fallback."""
        key = (str(abc_class).upper(), str(horizon_bucket))
        if key in self.buckets:
            return self.buckets[key]

        if horizon_bucket in self.horizon_fallback:
            return self.horizon_fallback[horizon_bucket]

        if str(abc_class).upper() in self.abc_fallback:
            return self.abc_fallback[str(abc_class).upper()]

        return self.global_fallback

    def to_dict(self) -> dict[str, Any]:
        """Serialize quantiles table to JSON-friendly dictionary."""
        return {
            "model_name": self.model_name,
            "level": self.level,
            "q_lower": self.q_lower,
            "q_upper": self.q_upper,
            "buckets": {f"{k[0]}|{k[1]}": v for k, v in self.buckets.items()},
            "horizon_fallback": self.horizon_fallback,
            "abc_fallback": self.abc_fallback,
            "global_fallback": list(self.global_fallback),
            "sample_counts": self.sample_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualQuantilesTable:
        """Deserialize quantiles table from dictionary."""
        raw_buckets = data.get("buckets", {})
        buckets: dict[tuple[str, str], tuple[float, float]] = {}
        for k_str, val in raw_buckets.items():
            parts = k_str.split("|", 1)
            if len(parts) == 2:
                buckets[(parts[0], parts[1])] = (float(val[0]), float(val[1]))

        raw_global = data.get("global_fallback", [0.0, 0.0])
        global_fallback = (float(raw_global[0]), float(raw_global[1]))

        h_fb = {k: (float(v[0]), float(v[1])) for k, v in data.get("horizon_fallback", {}).items()}
        a_fb = {k: (float(v[0]), float(v[1])) for k, v in data.get("abc_fallback", {}).items()}

        return cls(
            model_name=str(data.get("model_name", "champion")),
            level=float(data.get("level", 0.80)),
            q_lower=float(data.get("q_lower", 0.10)),
            q_upper=float(data.get("q_upper", 0.90)),
            buckets=buckets,
            horizon_fallback=h_fb,
            abc_fallback=a_fb,
            global_fallback=global_fallback,
            sample_counts=dict(data.get("sample_counts", {})),
        )


def _compute_quantiles(
    residuals: np.ndarray[Any, Any] | Sequence[float] | pd.Series,
    q_lower: float,
    q_upper: float,
) -> tuple[float, float]:
    """Calculate pair of lower and upper quantiles for a collection of residuals."""
    arr = np.asarray(residuals, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    q_low = float(np.quantile(arr, q_lower))
    q_high = float(np.quantile(arr, q_upper))
    return q_low, q_high


def _prepare_backtest_df(backtest_predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize backtest predictions DataFrame."""
    if backtest_predictions.empty:
        raise ValueError("Cannot fit residual quantiles on empty backtest_predictions")

    df = backtest_predictions.copy()

    # Ensure residual column exists (e = y - yhat)
    if "residual" not in df.columns:
        if "y" in df.columns and "yhat" in df.columns:
            df["residual"] = df["y"].astype(float) - df["yhat"].astype(float)
        else:
            raise ValueError("backtest_predictions must contain 'residual' or ('y' and 'yhat')")

    if "abc_class" not in df.columns:
        df["abc_class"] = "C"
    else:
        df["abc_class"] = df["abc_class"].fillna("C").astype(str).str.upper()

    if "horizon_bucket" not in df.columns:
        if "horizon_day" in df.columns:
            df["horizon_bucket"] = df["horizon_day"].astype(int).map(map_horizon_to_bucket)
        else:
            df["horizon_bucket"] = "1-7"

    return df


def fit_residual_quantiles(
    backtest_predictions: pd.DataFrame,
    *,
    level: float = 0.80,
    min_samples: int = 10,
    model_name: str = "champion",
) -> ResidualQuantilesTable:
    """Calibrate residual quantiles from backtest prediction errors per (ABC, horizon bucket).

    Per TRD §5.1:
    - Residuals are defined as error = y - yhat.
    - Default level 0.80 yields q10 (lower) and q90 (upper).
    - Intervals per (model, ABC class, horizon bucket).

    Args:
        backtest_predictions: DataFrame containing backtest predictions with columns
            'y', 'yhat', and optionally 'residual', 'horizon_day', 'abc_class'.
        level: Nominal coverage probability (default 0.80).
        min_samples: Minimum samples required to compute a specific bucket quantile.
        model_name: Identifier of the forecasting model.

    Returns:
        Calibrated ResidualQuantilesTable.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"Coverage level must be in (0, 1), got {level}")

    df = _prepare_backtest_df(backtest_predictions)

    q_lower = (1.0 - level) / 2.0
    q_upper = 1.0 - q_lower

    # 1. Global fallback
    all_res = df["residual"].dropna().to_numpy(dtype=float)
    global_delta = _compute_quantiles(all_res, q_lower, q_upper)

    # 2. Horizon fallback
    horizon_fb: dict[str, tuple[float, float]] = {}
    for h_bkt, group in df.groupby("horizon_bucket"):
        res = group["residual"].dropna().to_numpy(dtype=float)
        if len(res) >= min_samples:
            horizon_fb[str(h_bkt)] = _compute_quantiles(res, q_lower, q_upper)

    # 3. ABC fallback
    abc_fb: dict[str, tuple[float, float]] = {}
    for abc, group in df.groupby("abc_class"):
        res = group["residual"].dropna().to_numpy(dtype=float)
        if len(res) >= min_samples:
            abc_fb[str(abc)] = _compute_quantiles(res, q_lower, q_upper)

    # 4. Specific (ABC, horizon) buckets
    buckets: dict[tuple[str, str], tuple[float, float]] = {}
    sample_counts: dict[str, int] = {}

    for (abc, h_bkt), group in df.groupby(["abc_class", "horizon_bucket"]):
        res = group["residual"].dropna().to_numpy(dtype=float)
        count = len(res)
        sample_counts[f"{abc}|{h_bkt}"] = count
        if count >= min_samples:
            buckets[(str(abc), str(h_bkt))] = _compute_quantiles(res, q_lower, q_upper)

    logger.info(
        "fitted residual quantiles table",
        model_name=model_name,
        level=level,
        buckets_calibrated=len(buckets),
        global_deltas=global_delta,
    )

    return ResidualQuantilesTable(
        model_name=model_name,
        level=level,
        q_lower=q_lower,
        q_upper=q_upper,
        buckets=buckets,
        horizon_fallback=horizon_fb,
        abc_fallback=abc_fb,
        global_fallback=global_delta,
        sample_counts=sample_counts,
    )


def apply_prediction_intervals(
    predictions: pd.DataFrame,
    quantiles_table: ResidualQuantilesTable,
) -> pd.DataFrame:
    """Project lower and upper prediction intervals onto point forecasts.

    Per TRD §5.1 and DATABASE.md §7:
    - lower = min(yhat, max(0.0, yhat + q_lower_delta))
    - upper = max(yhat, yhat + q_upper_delta)
    - Enforces constraint: 0 <= yhat_lower <= yhat <= yhat_upper.

    Args:
        predictions: DataFrame with point forecasts containing 'yhat', and optionally
            'horizon_day' / 'horizon_bucket' and 'abc_class'.
        quantiles_table: Calibrated ResidualQuantilesTable.

    Returns:
        DataFrame with attached 'yhat_lower', 'yhat_upper', and 'interval_level'.
    """
    if predictions.empty:
        out = predictions.copy()
        out["yhat_lower"] = pd.Series(dtype=float)
        out["yhat_upper"] = pd.Series(dtype=float)
        out["interval_level"] = quantiles_table.level
        return out

    out = predictions.copy()
    if "yhat" not in out.columns:
        raise ValueError("predictions DataFrame must contain 'yhat' column")

    abc_series = (
        out["abc_class"].fillna("C").astype(str).str.upper()
        if "abc_class" in out.columns
        else pd.Series(["C"] * len(out), index=out.index)
    )

    if "horizon_bucket" in out.columns:
        h_series = out["horizon_bucket"].astype(str)
    elif "horizon_day" in out.columns:
        h_series = out["horizon_day"].astype(int).map(map_horizon_to_bucket)
    else:
        h_series = pd.Series(["1-7"] * len(out), index=out.index)

    lower_deltas = np.zeros(len(out), dtype=float)
    upper_deltas = np.zeros(len(out), dtype=float)

    for i in range(len(out)):
        abc = abc_series.iloc[i]
        hb = h_series.iloc[i]
        q_low, q_high = quantiles_table.lookup(abc, hb)
        lower_deltas[i] = q_low
        upper_deltas[i] = q_high

    yhat_vals = out["yhat"].astype(float).to_numpy()

    # Apply bounds with constraints: 0 <= yhat_lower <= yhat <= yhat_upper
    lower_bounded = np.minimum(yhat_vals, np.maximum(0.0, yhat_vals + lower_deltas))
    upper_bounded = np.maximum(yhat_vals, yhat_vals + upper_deltas)

    out["yhat_lower"] = lower_bounded
    out["yhat_upper"] = upper_bounded
    out["interval_level"] = quantiles_table.level

    return out


def evaluate_interval_metrics(
    eval_df: pd.DataFrame,
    *,
    min_coverage_threshold: float = 0.65,
    max_coverage_threshold: float = 0.95,
) -> dict[str, Any]:
    """Calculate empirical coverage and pinball loss metrics on forecast intervals.

    Args:
        eval_df: DataFrame with 'y', 'yhat_lower', 'yhat_upper', and optionally 'yhat'.
        min_coverage_threshold: Alert trigger floor for coverage (default 0.65).
        max_coverage_threshold: Alert trigger ceiling for coverage (default 0.95).

    Returns:
        Summary dict containing coverage, pinball loss, and threshold status.
    """
    required = {"y", "yhat_lower", "yhat_upper"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in eval_df: {missing}")

    y = eval_df["y"].to_numpy(dtype=float)
    yl = eval_df["yhat_lower"].to_numpy(dtype=float)
    yu = eval_df["yhat_upper"].to_numpy(dtype=float)

    cov_rate = coverage(y, yl, yu)
    pinball_90 = pinball_loss(y, yu, quantile=0.9)
    pinball_10 = pinball_loss(y, yl, quantile=0.1)

    is_within_tolerance = min_coverage_threshold <= cov_rate <= max_coverage_threshold

    return {
        "coverage": cov_rate,
        "pinball_loss_q90": pinball_90,
        "pinball_loss_q10": pinball_10,
        "is_within_tolerance": is_within_tolerance,
        "min_coverage_threshold": min_coverage_threshold,
        "max_coverage_threshold": max_coverage_threshold,
    }

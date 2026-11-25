"""Realized error and performance monitoring for elapsed models.

Implements P7-05, ARCHITECTURE §7, and TRD §6:
- Computes realized WAPE for elapsed forecasts joined with actual ground-truth sales.
- Compares realized WAPE against backtest WAPE (alert if > 1.25x) and seasonal naive baseline.
- Computes realized PR-AUC for customer churn after the 90-day outcome window has elapsed.
- Persists realized snapshots to PostgreSQL `ops.model_performance`.
- Generates operational alerts in `ops.alert` on performance degradation.
"""

from __future__ import annotations

import dataclasses
import datetime
import uuid
from collections.abc import Sequence

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn import metrics as sk_metrics
from sqlalchemy.orm import Session

from retailpulse.db.models.core import FactDailySales, FactSalesLine
from retailpulse.db.models.ml import ChurnScore, Forecast, ModelRun
from retailpulse.db.models.ops import Alert, ModelPerformance
from retailpulse.errors import DomainValidationError
from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.metrics import wape
from retailpulse.timeutil import utc_now

logger = get_logger(__name__)

DEFAULT_CHURN_OUTCOME_WINDOW_DAYS: int = 90
FORECAST_DEGRADATION_MULTIPLIER: float = 1.25


@dataclasses.dataclass(frozen=True)
class ModelPerformanceSnapshot:
    """Summary of a realized model evaluation window."""

    task: str
    metric: str
    model_run_id: uuid.UUID
    window_start: datetime.date
    window_end: datetime.date
    value: float
    baseline_value: float | None
    backtest_value: float | None
    n_observations: int
    is_degraded: bool
    degradation_reason: str | None = None
    alert_created: bool = False


def _get_champion_run(session: Session, task: str) -> ModelRun | None:
    """Find the active champion ModelRun for a given task, falling back to latest succeeded run."""
    run = session.execute(
        sa.select(ModelRun)
        .where(
            ModelRun.task == task,
            ModelRun.is_champion.is_(True),
            ModelRun.status == "succeeded",
        )
        .order_by(ModelRun.trained_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    if run is None:
        run = session.execute(
            sa.select(ModelRun)
            .where(
                ModelRun.task == task,
                ModelRun.status == "succeeded",
            )
            .order_by(ModelRun.trained_at.desc())
            .limit(1)
        ).scalar_one_or_none()

    return run


def save_performance_snapshot(
    session: Session,
    snapshot: ModelPerformanceSnapshot,
) -> ModelPerformance:
    """Persist or update a model performance snapshot row in ops.model_performance.

    Enforces idempotency on UNIQUE (model_run_id, metric, window_start, window_end).
    """
    stmt = sa.select(ModelPerformance).where(
        ModelPerformance.model_run_id == snapshot.model_run_id,
        ModelPerformance.metric == snapshot.metric,
        ModelPerformance.window_start == snapshot.window_start,
        ModelPerformance.window_end == snapshot.window_end,
    )
    existing = session.execute(stmt).scalar_one_or_none()

    if existing is not None:
        existing.value = snapshot.value
        existing.baseline_value = snapshot.baseline_value
        existing.n_observations = snapshot.n_observations
        session.flush()
        return existing

    record = ModelPerformance(
        model_run_id=snapshot.model_run_id,
        task=snapshot.task,
        window_start=snapshot.window_start,
        window_end=snapshot.window_end,
        metric=snapshot.metric,
        value=snapshot.value,
        baseline_value=snapshot.baseline_value,
        n_observations=snapshot.n_observations,
    )
    session.add(record)
    session.flush()
    return record


def _load_forecast_df(
    session: Session,
    model_run_id: uuid.UUID,
    window_start: datetime.date | None,
    eff_window_end: datetime.date,
) -> pd.DataFrame:
    """Query elapsed forecast rows from ml.forecast."""
    q_fc = sa.select(
        Forecast.product_id,
        Forecast.store_id,
        Forecast.target_date,
        Forecast.yhat,
    ).where(
        Forecast.model_run_id == model_run_id,
        Forecast.target_date <= eff_window_end,
    )
    if window_start is not None:
        q_fc = q_fc.where(Forecast.target_date >= window_start)

    rows = session.execute(q_fc).all()
    if not rows:
        raise DomainValidationError(
            f"No elapsed forecasts found for model run {model_run_id} up to {eff_window_end}",
            details={"model_run_id": str(model_run_id), "window_end": str(eff_window_end)},
        )
    return pd.DataFrame(rows, columns=["product_id", "store_id", "target_date", "yhat"])


def _load_actuals_df(
    session: Session,
    min_date: datetime.date,
    max_date: datetime.date,
) -> pd.DataFrame:
    """Query actual sales quantities from core.fact_daily_sales."""
    q_act = sa.select(
        FactDailySales.product_id,
        FactDailySales.store_id,
        FactDailySales.date_key.label("target_date"),
        sa.func.coalesce(FactDailySales.net_units, FactDailySales.units_sold).label("actual_qty"),
    ).where(
        FactDailySales.date_key >= min_date,
        FactDailySales.date_key <= max_date,
    )
    act_rows = session.execute(q_act).all()
    if not act_rows:
        raise DomainValidationError(
            f"No actual sales records found for evaluation window [{min_date}, {max_date}]",
            details={"window_start": str(min_date), "window_end": str(max_date)},
        )
    return pd.DataFrame(act_rows, columns=["product_id", "store_id", "target_date", "actual_qty"])


def _join_and_calculate_wape(
    fc_df: pd.DataFrame,
    act_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    run_baseline_wape: float | None,
) -> tuple[float, float | None, pd.DataFrame]:
    """Join forecasts and actuals and compute candidate and baseline WAPE."""
    clean_act = act_df.copy()
    if "actual_qty" not in clean_act.columns:
        for col_cand in ("quantity", "net_units", "units_sold"):
            if col_cand in clean_act.columns:
                clean_act = clean_act.rename(columns={col_cand: "actual_qty"})
                break

    clean_act["target_date"] = pd.to_datetime(clean_act["target_date"]).dt.date
    clean_fc = fc_df.copy()
    clean_fc["target_date"] = pd.to_datetime(clean_fc["target_date"]).dt.date

    join_keys = ["product_id", "store_id", "target_date"]
    if all(k in clean_fc.columns and k in clean_act.columns for k in join_keys):
        merged = clean_fc.merge(clean_act, on=join_keys, how="inner")
    elif "target_date" in clean_fc.columns and "target_date" in clean_act.columns:
        merged = clean_fc.merge(clean_act, on=["target_date"], how="inner")
    else:
        merged = pd.DataFrame()

    if merged.empty:
        raise DomainValidationError(
            "Forecasts and actuals do not overlap on matching evaluation keys",
            details={"forecast_rows": len(clean_fc), "actual_rows": len(clean_act)},
        )

    y_true = pd.to_numeric(merged["actual_qty"], errors="coerce").fillna(0.0).to_numpy()
    y_pred = pd.to_numeric(merged["yhat"], errors="coerce").fillna(0.0).to_numpy()
    realized_val = float(wape(y_true, y_pred))

    baseline_val = run_baseline_wape
    if baseline_df is not None:
        b_df = baseline_df.copy()
        b_df["target_date"] = pd.to_datetime(b_df["target_date"]).dt.date
        if "yhat" in b_df.columns:
            b_df = b_df.rename(columns={"yhat": "baseline_yhat"})
        b_col = "baseline_yhat"
        b_merged = merged.merge(b_df, on=["product_id", "store_id", "target_date"], how="inner")
        if not b_merged.empty and b_col in b_merged.columns:
            b_pred = pd.to_numeric(b_merged[b_col], errors="coerce").fillna(0.0).to_numpy()
            b_true = pd.to_numeric(b_merged["actual_qty"], errors="coerce").fillna(0.0).to_numpy()
            baseline_val = float(wape(b_true, b_pred))

    return realized_val, baseline_val, merged


def _check_forecast_degradation(
    realized_val: float,
    backtest_wape: float,
    baseline_val: float | None,
) -> tuple[bool, str | None]:
    """Check whether realized forecast error triggers degradation thresholds."""
    reasons: list[str] = []
    if backtest_wape > 0.0 and realized_val > (backtest_wape * FORECAST_DEGRADATION_MULTIPLIER):
        reasons.append(
            f"Realized WAPE {realized_val:.4f} exceeded 1.25x backtest WAPE {backtest_wape:.4f}"
        )
    if baseline_val is not None and realized_val > baseline_val:
        reasons.append(
            f"Realized WAPE {realized_val:.4f} was worse than seasonal naive baseline "
            f"{baseline_val:.4f}"
        )
    is_degraded = len(reasons) > 0
    reason_str = "; ".join(reasons) if reasons else None
    return is_degraded, reason_str


def compute_forecast_realized_wape(
    session: Session,
    *,
    model_run_id: uuid.UUID | None = None,
    window_start: datetime.date | None = None,
    window_end: datetime.date | None = None,
    actuals_df: pd.DataFrame | None = None,
    forecasts_df: pd.DataFrame | None = None,
    baseline_df: pd.DataFrame | None = None,
    as_of_date: datetime.date | None = None,
    persist: bool = True,
    alert_on_degradation: bool = True,
) -> ModelPerformanceSnapshot:
    """Compute realized WAPE for elapsed forecasts joined against actual observed demand.

    Per TRD §6:
    - Weekly/scheduled job computes realized WAPE for forecasts whose target dates have passed.
    - Alerts with 'forecast_degraded' if realized WAPE > 1.25x backtest WAPE or is worse than
      seasonal naive over the same window.
    """
    today = as_of_date or utc_now().date()
    eff_window_end = window_end or today

    if model_run_id is not None:
        run = session.execute(
            sa.select(ModelRun).where(ModelRun.run_id == model_run_id)
        ).scalar_one_or_none()
        if run is None:
            raise DomainValidationError(
                f"Model run '{model_run_id}' not found",
                details={"model_run_id": str(model_run_id)},
            )
    else:
        run = _get_champion_run(session, "forecast")
        if run is None:
            raise DomainValidationError("No succeeded forecast model run found to evaluate")
        model_run_id = run.run_id

    backtest_metrics = run.metrics or {}
    backtest_wape = float(backtest_metrics.get("wape", 0.0))
    run_baseline_wape = (
        float(backtest_metrics["baseline_wape"])
        if "baseline_wape" in backtest_metrics and backtest_metrics["baseline_wape"] is not None
        else None
    )

    fc_df = (
        forecasts_df.copy()
        if forecasts_df is not None
        else _load_forecast_df(session, model_run_id, window_start, eff_window_end)
    )

    if actuals_df is not None:
        act_df = actuals_df.copy()
    else:
        min_date = window_start or fc_df["target_date"].min()
        act_df = _load_actuals_df(session, min_date, eff_window_end)

    realized_val, baseline_val, merged = _join_and_calculate_wape(
        fc_df=fc_df,
        act_df=act_df,
        baseline_df=baseline_df,
        run_baseline_wape=run_baseline_wape,
    )
    n_obs = len(merged)

    is_degraded, degradation_reason = _check_forecast_degradation(
        realized_val=realized_val,
        backtest_wape=backtest_wape,
        baseline_val=baseline_val,
    )

    eff_window_start = window_start or merged["target_date"].min()
    eff_window_end = window_end or merged["target_date"].max()

    alert_created = False
    if is_degraded and alert_on_degradation:
        dedupe_key = f"forecast_degraded:{model_run_id}:{eff_window_start}:{eff_window_end}"
        existing_alert = session.execute(
            sa.select(Alert).where(Alert.dedupe_key == dedupe_key, Alert.acknowledged_at.is_(None))
        ).scalar_one_or_none()

        if existing_alert is None:
            alert = Alert(
                severity="warning",
                kind="forecast_degraded",
                message=degradation_reason or f"Forecast degraded for model run {model_run_id}",
                payload={
                    "model_run_id": str(model_run_id),
                    "realized_wape": round(realized_val, 4),
                    "backtest_wape": round(backtest_wape, 4),
                    "baseline_wape": round(baseline_val, 4) if baseline_val is not None else None,
                    "n_observations": n_obs,
                    "window_start": str(eff_window_start),
                    "window_end": str(eff_window_end),
                },
                dedupe_key=dedupe_key,
            )
            session.add(alert)
            session.flush()
            alert_created = True

    snapshot = ModelPerformanceSnapshot(
        task="forecast",
        metric="realized_wape",
        model_run_id=model_run_id,
        window_start=eff_window_start,
        window_end=eff_window_end,
        value=round(realized_val, 4),
        baseline_value=round(baseline_val, 4) if baseline_val is not None else None,
        backtest_value=round(backtest_wape, 4) if backtest_wape > 0.0 else None,
        n_observations=n_obs,
        is_degraded=is_degraded,
        degradation_reason=degradation_reason,
        alert_created=alert_created,
    )

    if persist:
        save_performance_snapshot(session, snapshot)
        session.commit()

    logger.info(
        "Forecast realized performance calculated",
        model_run_id=str(model_run_id),
        realized_wape=snapshot.value,
        baseline_wape=snapshot.baseline_value,
        backtest_wape=snapshot.backtest_value,
        is_degraded=is_degraded,
        alert_created=alert_created,
    )

    return snapshot


def _load_churn_scores(
    session: Session,
    model_run_id: uuid.UUID,
    cutoff_date: datetime.date,
    outcome_window_days: int,
) -> tuple[pd.DataFrame, datetime.date]:
    """Query churn scores from ml.churn_score up to the outcome cutoff date."""
    q_scores = (
        sa.select(
            ChurnScore.customer_id,
            ChurnScore.as_of_date,
            ChurnScore.churn_probability,
        )
        .where(
            ChurnScore.model_run_id == model_run_id,
            ChurnScore.as_of_date <= cutoff_date,
        )
        .order_by(ChurnScore.as_of_date.desc())
    )
    rows = session.execute(q_scores).all()
    if not rows:
        raise DomainValidationError(
            f"No elapsed churn predictions found for model run {model_run_id} with outcome "
            f"window {outcome_window_days} days",
            details={"model_run_id": str(model_run_id), "cutoff_date": str(cutoff_date)},
        )
    sc_df = pd.DataFrame(rows, columns=["customer_id", "as_of_date", "churn_probability"])
    score_date = sc_df["as_of_date"].max()
    return sc_df[sc_df["as_of_date"] == score_date], score_date


def _load_churn_actuals(
    session: Session,
    customer_ids: Sequence[int],
    window_start: datetime.date,
    window_end: datetime.date,
) -> pd.DataFrame:
    """Query customer transactions in outcome window to determine actual churn status."""
    purchased_custs = session.execute(
        sa.select(FactSalesLine.customer_id).where(
            FactSalesLine.date_key > window_start,
            FactSalesLine.date_key <= window_end,
            FactSalesLine.customer_id.is_not(None),
        ).distinct()
    ).scalars().all()
    active_set = set(purchased_custs)
    return pd.DataFrame({
        "customer_id": list(customer_ids),
        "churned": [1 if c not in active_set else 0 for c in customer_ids],
    })


def _evaluate_churn_pr_auc(
    sc_df: pd.DataFrame,
    act_df: pd.DataFrame,
    run_baseline_pr_auc: float | None,
) -> tuple[float, float, int]:
    """Join scores and actuals to compute realized PR-AUC and empirical baseline."""
    clean_act = act_df.copy()
    if "churned" not in clean_act.columns:
        if "label" in clean_act.columns:
            clean_act = clean_act.rename(columns={"label": "churned"})
        elif "is_churned" in clean_act.columns:
            clean_act = clean_act.rename(columns={"is_churned": "churned"})

    merged = sc_df.merge(clean_act, on=["customer_id"], how="inner")
    if merged.empty:
        raise DomainValidationError(
            "Churn predictions and actual outcomes do not overlap on customer_id",
            details={"scores_count": len(sc_df), "actuals_count": len(clean_act)},
        )

    y_true = pd.to_numeric(merged["churned"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob = (
        pd.to_numeric(merged["churn_probability"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )

    realized_val = (
        float(sk_metrics.average_precision_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float(np.mean(y_true))
    )

    empirical_base_rate = float(np.mean(y_true))
    baseline_val = run_baseline_pr_auc if run_baseline_pr_auc is not None else empirical_base_rate
    return realized_val, baseline_val, len(merged)


def _check_churn_degradation(
    realized_val: float,
    backtest_pr_auc: float,
    baseline_val: float,
) -> tuple[bool, str | None]:
    """Check whether churn prediction quality dropped below backtest or baseline."""
    reasons: list[str] = []
    if backtest_pr_auc > 0.0 and realized_val < (0.80 * backtest_pr_auc):
        reasons.append(
            f"Realized PR-AUC {realized_val:.4f} fell below 80% of backtest PR-AUC "
            f"{backtest_pr_auc:.4f}"
        )
    if realized_val < baseline_val:
        reasons.append(
            f"Realized PR-AUC {realized_val:.4f} was worse than baseline {baseline_val:.4f}"
        )
    is_degraded = len(reasons) > 0
    reason_str = "; ".join(reasons) if reasons else None
    return is_degraded, reason_str


def compute_churn_realized_pr_auc(
    session: Session,
    *,
    model_run_id: uuid.UUID | None = None,
    as_of_date: datetime.date | None = None,
    outcome_window_days: int = DEFAULT_CHURN_OUTCOME_WINDOW_DAYS,
    actuals_df: pd.DataFrame | None = None,
    scores_df: pd.DataFrame | None = None,
    persist: bool = True,
    alert_on_degradation: bool = True,
) -> ModelPerformanceSnapshot:
    """Compute realized PR-AUC for churn predictions once the 90-day outcome window has elapsed.

    Per TRD §6:
    - Churn: compute realized PR-AUC once the 90-day outcome window has elapsed.
    - Compares predicted churn probabilities against actual customer retention/churn outcomes.
    """
    today = as_of_date or utc_now().date()

    if model_run_id is not None:
        run = session.execute(
            sa.select(ModelRun).where(ModelRun.run_id == model_run_id)
        ).scalar_one_or_none()
        if run is None:
            raise DomainValidationError(
                f"Model run '{model_run_id}' not found",
                details={"model_run_id": str(model_run_id)},
            )
    else:
        run = _get_champion_run(session, "churn")
        if run is None:
            raise DomainValidationError("No succeeded churn model run found to evaluate")
        model_run_id = run.run_id

    backtest_metrics = run.metrics or {}
    backtest_pr_auc = float(backtest_metrics.get("pr_auc", 0.0))
    run_baseline_pr_auc = (
        float(backtest_metrics["baseline_pr_auc"])
        if "baseline_pr_auc" in backtest_metrics and backtest_metrics["baseline_pr_auc"] is not None
        else None
    )

    if scores_df is not None:
        sc_df = scores_df.copy()
        score_date = today - datetime.timedelta(days=outcome_window_days)
    else:
        cutoff = today - datetime.timedelta(days=outcome_window_days)
        sc_df, score_date = _load_churn_scores(session, model_run_id, cutoff, outcome_window_days)

    window_start = score_date
    window_end = score_date + datetime.timedelta(days=outcome_window_days)

    act_df = (
        actuals_df.copy()
        if actuals_df is not None
        else _load_churn_actuals(session, sc_df["customer_id"].unique(), window_start, window_end)
    )

    realized_val, baseline_val, n_obs = _evaluate_churn_pr_auc(
        sc_df=sc_df,
        act_df=act_df,
        run_baseline_pr_auc=run_baseline_pr_auc,
    )

    is_degraded, degradation_reason = _check_churn_degradation(
        realized_val=realized_val,
        backtest_pr_auc=backtest_pr_auc,
        baseline_val=baseline_val,
    )

    alert_created = False
    if is_degraded and alert_on_degradation:
        dedupe_key = f"churn_degraded:{model_run_id}:{window_start}:{window_end}"
        existing_alert = session.execute(
            sa.select(Alert).where(Alert.dedupe_key == dedupe_key, Alert.acknowledged_at.is_(None))
        ).scalar_one_or_none()

        if existing_alert is None:
            alert = Alert(
                severity="warning",
                kind="churn_degraded",
                message=(
                    degradation_reason
                    or f"Churn prediction degraded for model run {model_run_id}"
                ),
                payload={
                    "model_run_id": str(model_run_id),
                    "realized_pr_auc": round(realized_val, 4),
                    "backtest_pr_auc": round(backtest_pr_auc, 4),
                    "baseline_pr_auc": round(baseline_val, 4),
                    "n_observations": n_obs,
                    "window_start": str(window_start),
                    "window_end": str(window_end),
                },
                dedupe_key=dedupe_key,
            )
            session.add(alert)
            session.flush()
            alert_created = True

    snapshot = ModelPerformanceSnapshot(
        task="churn",
        metric="realized_pr_auc",
        model_run_id=model_run_id,
        window_start=window_start,
        window_end=window_end,
        value=round(realized_val, 4),
        baseline_value=round(baseline_val, 4),
        backtest_value=round(backtest_pr_auc, 4) if backtest_pr_auc > 0.0 else None,
        n_observations=n_obs,
        is_degraded=is_degraded,
        degradation_reason=degradation_reason,
        alert_created=alert_created,
    )

    if persist:
        save_performance_snapshot(session, snapshot)
        session.commit()

    logger.info(
        "Churn realized performance calculated",
        model_run_id=str(model_run_id),
        realized_pr_auc=snapshot.value,
        baseline_pr_auc=snapshot.baseline_value,
        backtest_pr_auc=snapshot.backtest_value,
        is_degraded=is_degraded,
        alert_created=alert_created,
    )

    return snapshot


def run_performance_monitoring_job(
    session: Session,
    *,
    as_of_date: datetime.date | None = None,
) -> list[ModelPerformanceSnapshot]:
    """Execute performance monitoring evaluations across active champion models."""
    snapshots: list[ModelPerformanceSnapshot] = []

    # 1. Forecast realized performance
    try:
        fc_snapshot = compute_forecast_realized_wape(session=session, as_of_date=as_of_date)
        snapshots.append(fc_snapshot)
    except Exception as exc:
        logger.warning("Forecast realized performance monitoring skipped or failed", error=str(exc))

    # 2. Churn realized performance
    try:
        churn_snapshot = compute_churn_realized_pr_auc(session=session, as_of_date=as_of_date)
        snapshots.append(churn_snapshot)
    except Exception as exc:
        logger.warning("Churn realized performance monitoring skipped or failed", error=str(exc))

    return snapshots

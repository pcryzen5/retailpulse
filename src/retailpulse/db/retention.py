"""Database and file artifact retention policies per DATABASE §8 and SEC-DATA-04.

Implements P7-08:
- Retention rules:
  * Prediction tables: Keep latest 14 scoring dates per task in `ml.forecast`,
    `ml.customer_segment`, `ml.segment_profile`, `ml.churn_score`, and
    `ml.inventory_recommendation`, PLUS whatever `ml.active_output` points to.
  * Audit logs: Keep `ops.audit_log` rows for at least 365 days.
  * Drift reports: Keep `ops.drift_report` rows for 180 days and unlink HTML files on disk.
  * Raw uploads: Delete source upload files 30 days after a successful load, preserving
    ingestion batch rows and SHA256 hashes.
  * Indefinite tables: `ml.model_run`, `ml.forecast_metric`, `ops.model_performance`.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    CustomerSegment,
    Forecast,
    InventoryRecommendation,
    SegmentProfile,
)
from retailpulse.db.models.ops import AuditLog, DriftReport, IngestionBatch
from retailpulse.logging import get_logger
from retailpulse.timeutil import utc_now

logger = get_logger(__name__)

DEFAULT_PREDICTION_KEEP_DATES = 14
DEFAULT_AUDIT_LOG_DAYS = 365
DEFAULT_DRIFT_REPORT_DAYS = 180
DEFAULT_UPLOAD_FILE_DAYS = 30


@dataclass(frozen=True)
class RetentionSummary:
    """Summary metrics of executed retention cleanup job."""

    forecast_rows_pruned: int
    customer_segment_rows_pruned: int
    segment_profile_rows_pruned: int
    churn_score_rows_pruned: int
    inventory_rec_rows_pruned: int
    audit_log_rows_pruned: int
    drift_reports_pruned: int
    drift_files_deleted: int
    upload_files_deleted: int
    executed_at: str

    @property
    def total_rows_pruned(self) -> int:
        """Sum of database rows pruned across all tables."""
        return (
            self.forecast_rows_pruned
            + self.customer_segment_rows_pruned
            + self.segment_profile_rows_pruned
            + self.churn_score_rows_pruned
            + self.inventory_rec_rows_pruned
            + self.audit_log_rows_pruned
            + self.drift_reports_pruned
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        d = asdict(self)
        d["total_rows_pruned"] = self.total_rows_pruned
        return d


def _prune_forecast(session: Session, keep_dates: int) -> int:
    """Prune ml.forecast keeping the latest N dates plus active output."""
    active_fc = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "forecast")
    ).scalar_one_or_none()

    fc_dates_res = session.execute(
        sa.select(Forecast.origin_date)
        .distinct()
        .order_by(Forecast.origin_date.desc())
        .limit(keep_dates)
    ).scalars().all()
    kept_fc_dates = set(fc_dates_res)

    if not kept_fc_dates:
        return 0

    cond_fc = Forecast.origin_date.not_in(kept_fc_dates)
    if active_fc is not None:
        cond_fc = sa.and_(
            cond_fc,
            sa.not_(
                sa.and_(
                    Forecast.model_run_id == active_fc.model_run_id,
                    Forecast.origin_date == active_fc.data_as_of,
                )
            ),
        )
    fc_delete_stmt = (
        sa.delete(Forecast)
        .where(cond_fc)
        .execution_options(synchronize_session=False)
    )
    res_fc = session.execute(fc_delete_stmt)
    return int(res_fc.rowcount or 0)


def _prune_segmentation(session: Session, keep_dates: int) -> tuple[int, int]:
    """Prune customer_segment and segment_profile keeping latest N dates plus active output."""
    active_seg = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "segmentation")
    ).scalar_one_or_none()

    seg_dates_res = session.execute(
        sa.select(SegmentProfile.as_of_date)
        .distinct()
        .order_by(SegmentProfile.as_of_date.desc())
        .limit(keep_dates)
    ).scalars().all()
    kept_seg_dates = set(seg_dates_res)

    if not kept_seg_dates:
        return 0, 0

    cond_cust = CustomerSegment.as_of_date.not_in(kept_seg_dates)
    cond_prof = SegmentProfile.as_of_date.not_in(kept_seg_dates)
    if active_seg is not None:
        active_seg_match = sa.and_(
            SegmentProfile.model_run_id == active_seg.model_run_id,
            SegmentProfile.as_of_date == active_seg.data_as_of,
        )
        cond_prof = sa.and_(cond_prof, sa.not_(active_seg_match))

        active_cust_match = sa.and_(
            CustomerSegment.model_run_id == active_seg.model_run_id,
            CustomerSegment.as_of_date == active_seg.data_as_of,
        )
        cond_cust = sa.and_(cond_cust, sa.not_(active_cust_match))

    cust_del_stmt = (
        sa.delete(CustomerSegment)
        .where(cond_cust)
        .execution_options(synchronize_session=False)
    )
    res_cust = session.execute(cust_del_stmt)

    prof_del_stmt = (
        sa.delete(SegmentProfile)
        .where(cond_prof)
        .execution_options(synchronize_session=False)
    )
    res_prof = session.execute(prof_del_stmt)

    return int(res_cust.rowcount or 0), int(res_prof.rowcount or 0)


def _prune_churn(session: Session, keep_dates: int) -> int:
    """Prune ml.churn_score keeping latest N dates plus active output."""
    active_churn = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "churn")
    ).scalar_one_or_none()

    churn_dates_res = session.execute(
        sa.select(ChurnScore.as_of_date)
        .distinct()
        .order_by(ChurnScore.as_of_date.desc())
        .limit(keep_dates)
    ).scalars().all()
    kept_churn_dates = set(churn_dates_res)

    if not kept_churn_dates:
        return 0

    cond_churn = ChurnScore.as_of_date.not_in(kept_churn_dates)
    if active_churn is not None:
        active_churn_match = sa.and_(
            ChurnScore.model_run_id == active_churn.model_run_id,
            ChurnScore.as_of_date == active_churn.data_as_of,
        )
        cond_churn = sa.and_(cond_churn, sa.not_(active_churn_match))

    churn_del_stmt = (
        sa.delete(ChurnScore)
        .where(cond_churn)
        .execution_options(synchronize_session=False)
    )
    res_churn = session.execute(churn_del_stmt)
    return int(res_churn.rowcount or 0)


def _prune_inventory(session: Session, keep_dates: int) -> int:
    """Prune ml.inventory_recommendation keeping latest N dates plus active output."""
    active_inv = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "inventory")
    ).scalar_one_or_none()

    inv_dates_res = session.execute(
        sa.select(InventoryRecommendation.as_of_date)
        .distinct()
        .order_by(InventoryRecommendation.as_of_date.desc())
        .limit(keep_dates)
    ).scalars().all()
    kept_inv_dates = set(inv_dates_res)

    if not kept_inv_dates:
        return 0

    cond_inv = InventoryRecommendation.as_of_date.not_in(kept_inv_dates)
    if active_inv is not None:
        active_inv_match = sa.and_(
            InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
            InventoryRecommendation.as_of_date == active_inv.data_as_of,
        )
        cond_inv = sa.and_(cond_inv, sa.not_(active_inv_match))

    inv_del_stmt = (
        sa.delete(InventoryRecommendation)
        .where(cond_inv)
        .execution_options(synchronize_session=False)
    )
    res_inv = session.execute(inv_del_stmt)
    return int(res_inv.rowcount or 0)


def prune_prediction_retention(
    session: Session,
    keep_dates: int = DEFAULT_PREDICTION_KEEP_DATES,
) -> dict[str, int]:
    """Prune prediction tables keeping the latest N scoring dates plus active output pointers.

    Per DATABASE §8:
    Keeps the latest 14 scoring dates per task in `ml.forecast`, `ml.customer_segment`,
    `ml.segment_profile`, `ml.churn_score`, and `ml.inventory_recommendation`,
    PLUS whatever `ml.active_output` points to.
    """
    if keep_dates < 1:
        raise ValueError(f"keep_dates must be >= 1, received: {keep_dates}")

    fc_count = _prune_forecast(session, keep_dates)
    cust_count, prof_count = _prune_segmentation(session, keep_dates)
    churn_count = _prune_churn(session, keep_dates)
    inv_count = _prune_inventory(session, keep_dates)

    pruned_counts: dict[str, int] = {
        "forecast": fc_count,
        "customer_segment": cust_count,
        "segment_profile": prof_count,
        "churn_score": churn_count,
        "inventory_recommendation": inv_count,
    }

    logger.info("Pruned prediction retention", pruned=pruned_counts, keep_dates=keep_dates)
    return pruned_counts

    logger.info("Pruned prediction retention", pruned=pruned_counts, keep_dates=keep_dates)
    return pruned_counts


def prune_ops_retention(
    session: Session,
    audit_retention_days: int = DEFAULT_AUDIT_LOG_DAYS,
    drift_retention_days: int = DEFAULT_DRIFT_REPORT_DAYS,
    as_of: datetime.datetime | None = None,
) -> dict[str, int]:
    """Prune audit logs and drift reports beyond their retention horizons.

    Per DATABASE §8:
    - Keep `ops.audit_log` for at least 365 days.
    - Keep `ops.drift_report` rows for 180 days and delete their HTML files on disk.
    """
    now = as_of or utc_now()
    results = {"audit_log": 0, "drift_report": 0, "drift_files": 0}

    # 1. Prune audit logs
    audit_cutoff = now - datetime.timedelta(days=audit_retention_days)
    try:
        res_audit = session.execute(
            sa.delete(AuditLog)
            .where(AuditLog.ts < audit_cutoff)
            .execution_options(synchronize_session=False)
        )
        results["audit_log"] = int(res_audit.rowcount or 0)
    except Exception as exc:
        # In case the database user has REVOKE DELETE on audit_log per DATABASE §10
        logger.warning(
            "Could not prune ops.audit_log (append-only restriction or permissions)",
            error=str(exc),
        )

    # 2. Prune drift reports and their HTML reports
    drift_cutoff = now - datetime.timedelta(days=drift_retention_days)
    old_drift_reports = session.execute(
        sa.select(DriftReport).where(DriftReport.created_at < drift_cutoff)
    ).scalars().all()

    drift_files_deleted = 0
    for rep in old_drift_reports:
        if rep.report_path:
            p = Path(rep.report_path)
            if p.is_file():
                try:
                    p.unlink()
                    drift_files_deleted += 1
                except OSError as exc:
                    logger.warning(
                        "Failed to remove drift HTML report file",
                        path=str(p),
                        error=str(exc),
                    )

    results["drift_files"] = drift_files_deleted

    res_drift = session.execute(
        sa.delete(DriftReport)
        .where(DriftReport.created_at < drift_cutoff)
        .execution_options(synchronize_session=False)
    )
    results["drift_report"] = int(res_drift.rowcount or 0)

    logger.info("Pruned ops retention", results=results)
    return results


def prune_raw_uploads_retention(
    session: Session,
    upload_retention_days: int = DEFAULT_UPLOAD_FILE_DAYS,
    uploads_dir: Path | str | None = None,
    as_of: datetime.datetime | None = None,
) -> int:
    """Delete raw uploaded files 30 days after successful load while preserving batch metadata.

    Per DATABASE §8 and SEC-DATA-04:
    Raw uploaded files: keep 30 days after a successful load, then delete
    (hash and batch row remain).
    """
    now = as_of or utc_now()
    upload_cutoff = now - datetime.timedelta(days=upload_retention_days)

    old_batches = session.execute(
        sa.select(IngestionBatch).where(
            IngestionBatch.status == "succeeded",
            IngestionBatch.created_at < upload_cutoff,
        )
    ).scalars().all()

    deleted_count = 0
    for batch in old_batches:
        if batch.source_file:
            p = Path(batch.source_file)
            if p.is_file():
                try:
                    p.unlink()
                    deleted_count += 1
                except OSError as exc:
                    logger.warning(
                        "Failed to delete aged raw upload file",
                        file=str(p),
                        error=str(exc),
                    )

    # If uploads directory is explicitly specified, clean orphan files older than cutoff
    if uploads_dir is not None:
        target_dir = Path(uploads_dir)
        if target_dir.is_dir():
            cutoff_ts = upload_cutoff.timestamp()
            for item in target_dir.glob("*"):
                if item.is_file() and item.stat().st_mtime < cutoff_ts:
                    try:
                        item.unlink()
                        deleted_count += 1
                    except OSError:
                        pass

    logger.info("Pruned raw uploads retention", files_deleted=deleted_count)
    return deleted_count


def run_retention_job(  # noqa: PLR0917
    session: Session,
    keep_prediction_dates: int = DEFAULT_PREDICTION_KEEP_DATES,
    audit_retention_days: int = DEFAULT_AUDIT_LOG_DAYS,
    drift_retention_days: int = DEFAULT_DRIFT_REPORT_DAYS,
    upload_retention_days: int = DEFAULT_UPLOAD_FILE_DAYS,
    as_of: datetime.datetime | None = None,
) -> RetentionSummary:
    """Execute all database and filesystem retention policies per DATABASE §8."""
    now = as_of or utc_now()
    logger.info("Starting retention cleanup job", as_of=now.isoformat())

    # 1. Predictions
    pred_res = prune_prediction_retention(
        session=session,
        keep_dates=keep_prediction_dates,
    )

    # 2. Ops audit and drift
    ops_res = prune_ops_retention(
        session=session,
        audit_retention_days=audit_retention_days,
        drift_retention_days=drift_retention_days,
        as_of=now,
    )

    # 3. Raw uploads
    uploads_deleted = prune_raw_uploads_retention(
        session=session,
        upload_retention_days=upload_retention_days,
        as_of=now,
    )

    summary = RetentionSummary(
        forecast_rows_pruned=pred_res["forecast"],
        customer_segment_rows_pruned=pred_res["customer_segment"],
        segment_profile_rows_pruned=pred_res["segment_profile"],
        churn_score_rows_pruned=pred_res["churn_score"],
        inventory_rec_rows_pruned=pred_res["inventory_recommendation"],
        audit_log_rows_pruned=ops_res["audit_log"],
        drift_reports_pruned=ops_res["drift_report"],
        drift_files_deleted=ops_res["drift_files"],
        upload_files_deleted=uploads_deleted,
        executed_at=now.isoformat(),
    )

    session.commit()
    logger.info("Retention cleanup completed successfully", summary=summary.to_dict())
    return summary

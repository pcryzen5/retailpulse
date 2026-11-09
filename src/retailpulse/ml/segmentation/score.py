"""Batch customer segmentation scoring and active output management.

Implements P3-02, TRD §5.2, §6 & DATABASE §11:
- Resolves current champion model run (or falls back to latest succeeded run).
- Computes customer RFM features as of data_as_of.
- Assigns customers to clusters and business labels.
- Atomically writes ml.customer_segment, ml.segment_profile, and advances
  ml.active_output pointer in the SAME database transaction.
- On failure, rolls back transaction leaving previous ml.active_output unchanged.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import FactSalesLine
from retailpulse.db.models.ml import ActiveOutput, CustomerSegment, ModelRun, SegmentProfile
from retailpulse.features.rfm import compute_rfm_from_db
from retailpulse.logging import get_logger
from retailpulse.ml.registry import get_champion_run
from retailpulse.ml.segmentation.kmeans import RFMSegmentationModel

logger = get_logger(__name__)

TASK_SEGMENTATION = "segmentation"


class SegmentationScoreResult:
    """Strongly-typed summary of segmentation scoring execution."""

    def __init__(
        self,
        *,
        model_run_id: uuid.UUID,
        data_as_of: datetime.date,
        n_customers: int,
        active_output_updated: bool,
        profiles_df: pd.DataFrame,
        customer_segments_df: pd.DataFrame,
    ) -> None:
        self.model_run_id = model_run_id
        self.data_as_of = data_as_of
        self.n_customers = n_customers
        self.active_output_updated = active_output_updated
        self.profiles_df = profiles_df
        self.customer_segments_df = customer_segments_df


def _resolve_effective_as_of(
    session: Session, candidate_date: datetime.date | None
) -> datetime.date:
    """Resolve data cutoff date from explicit parameter or max sales date in DB."""
    if candidate_date is not None:
        return candidate_date

    stmt = sa.select(sa.func.max(FactSalesLine.date_key))
    max_date = session.execute(stmt).scalar()
    if max_date is not None:
        return max_date

    return datetime.date(2011, 12, 9)


def _resolve_model_run(
    session: Session,
    explicit_run_id: uuid.UUID | None,
    data_as_of: datetime.date,
    seed: int,
) -> tuple[uuid.UUID, int]:
    """Identify which ModelRun to score and its chosen k."""
    if explicit_run_id is not None:
        db_run = session.get(ModelRun, explicit_run_id)
        if db_run is None:
            raise ValueError(f"Specified ModelRun '{explicit_run_id}' not found")
        k_val = int(db_run.params.get("k", 4)) if db_run.params else 4
        return explicit_run_id, k_val

    # Look for active champion
    champ_run = get_champion_run(TASK_SEGMENTATION, session)
    if champ_run is not None:
        k_val = int(champ_run.params.get("k", 4)) if champ_run.params else 4
        return champ_run.run_id, k_val

    # Fallback to latest succeeded segmentation run
    stmt = (
        sa.select(ModelRun)
        .where(ModelRun.task == TASK_SEGMENTATION, ModelRun.status == "succeeded")
        .order_by(ModelRun.trained_at.desc())
    )
    fallback_run = session.execute(stmt).scalars().first()
    if fallback_run is not None:
        k_val = int(fallback_run.params.get("k", 4)) if fallback_run.params else 4
        return fallback_run.run_id, k_val

    # Create ad-hoc model run record if none exists
    new_run_id = uuid.uuid4()
    ad_hoc_run = ModelRun(
        run_id=new_run_id,
        task=TASK_SEGMENTATION,
        algorithm="kmeans",
        mlflow_run_id=f"adhoc-{new_run_id.hex}",
        status="succeeded",
        is_champion=False,
        data_end=data_as_of,
        params={"k": 4, "source": "ad_hoc_batch_score"},
        metrics={},
        seed=seed,
    )
    session.add(ad_hoc_run)
    session.flush()
    return new_run_id, 4


def score_segmentation_pipeline(
    session: Session,
    *,
    data_as_of: datetime.date | None = None,
    model_run_id: uuid.UUID | None = None,
    rfm_window_days: int | None = None,
) -> SegmentationScoreResult:
    """Score all eligible customers into segments and atomically update active output.

    Per TRD §5.2, §6 & DATABASE §11:
    - Resolves model run (champion or specified).
    - Computes RFM features for eligible population up to data_as_of.
    - Fits/predicts cluster assignments and canonical business labels.
    - Atomically writes ml.customer_segment and ml.segment_profile.
    - Advances ml.active_output pointer in the SAME transaction.
    - On failure, rolls back transaction leaving previous ml.active_output intact.

    Args:
        session: Active SQLAlchemy session.
        data_as_of: Reference cutoff date (defaults to latest sales date).
        model_run_id: Optional explicit ModelRun UUID.
        rfm_window_days: Lookback window in days (defaults to setting).

    Returns:
        SegmentationScoreResult summary.
    """
    settings = get_settings()
    effective_window = rfm_window_days if rfm_window_days is not None else settings.rfm_window_days
    effective_as_of = _resolve_effective_as_of(session, data_as_of)

    # 1. Resolve ModelRun ID and chosen k
    run_id, chosen_k = _resolve_model_run(
        session,
        explicit_run_id=model_run_id,
        data_as_of=effective_as_of,
        seed=settings.seed,
    )

    # 2. Compute customer RFM features
    rfm_df = compute_rfm_from_db(session, as_of=effective_as_of, rfm_window_days=effective_window)
    if rfm_df.empty:
        raise ValueError(f"No eligible customers found up to {effective_as_of}")

    # 3. Fit/score model on population
    model = RFMSegmentationModel(k=chosen_k, seed=settings.seed)
    model.fit(rfm_df)

    rfm_scored = rfm_df.copy()
    rfm_scored["segment_id"] = model.predict(rfm_scored)
    rfm_scored["segment_label"] = model.predict_labels(rfm_scored)

    # 4. Build segment profiles
    profiles_df = model.build_profiles(rfm_scored)

    # 5. Atomic transaction: write profiles, customer segments, and advance pointer
    try:
        # Idempotency: Clean previous rows for same (model_run_id, as_of_date)
        session.execute(
            sa.delete(CustomerSegment).where(
                CustomerSegment.model_run_id == run_id,
                CustomerSegment.as_of_date == effective_as_of,
            )
        )
        session.execute(
            sa.delete(SegmentProfile).where(
                SegmentProfile.model_run_id == run_id,
                SegmentProfile.as_of_date == effective_as_of,
            )
        )

        # Insert ml.segment_profile records
        profile_entities = []
        for _, row in profiles_df.iterrows():
            profile_entities.append(
                SegmentProfile(
                    model_run_id=run_id,
                    as_of_date=effective_as_of,
                    segment_id=int(row["segment_id"]),
                    segment_label=str(row["segment_label"]),
                    n_customers=int(row["n_customers"]),
                    avg_recency_days=Decimal(str(row["avg_recency_days"])),
                    avg_frequency=Decimal(str(row["avg_frequency"])),
                    avg_monetary=Decimal(str(row["avg_monetary"])),
                    revenue_share=Decimal(str(row["revenue_share"])),
                )
            )
        session.add_all(profile_entities)

        # Insert ml.customer_segment records
        segment_entities = []
        for _, row in rfm_scored.iterrows():
            segment_entities.append(
                CustomerSegment(
                    model_run_id=run_id,
                    as_of_date=effective_as_of,
                    customer_id=int(row["customer_id"]),
                    segment_id=int(row["segment_id"]),
                    recency_days=int(row["recency_days"]),
                    frequency=int(row["frequency"]),
                    monetary=Decimal(str(row["monetary"])),
                    r_score=int(row["r_score"]) if pd.notna(row.get("r_score")) else None,
                    f_score=int(row["f_score"]) if pd.notna(row.get("f_score")) else None,
                    m_score=int(row["m_score"]) if pd.notna(row.get("m_score")) else None,
                )
            )
        session.add_all(segment_entities)

        # Advance ml.active_output pointer in the SAME transaction
        active = session.get(ActiveOutput, TASK_SEGMENTATION)
        if active is None:
            active = ActiveOutput(
                task=TASK_SEGMENTATION,
                model_run_id=run_id,
                data_as_of=effective_as_of,
            )
            session.add(active)
        else:
            active.model_run_id = run_id
            active.data_as_of = effective_as_of

        session.commit()
        logger.info(
            "finished batch customer segmentation; active_output pointer updated",
            model_run_id=str(run_id),
            as_of=str(effective_as_of),
            customer_count=len(segment_entities),
            profile_count=len(profile_entities),
        )
    except Exception as exc:
        session.rollback()
        logger.error(
            "segmentation scoring failed; active_output pointer preserved",
            error=str(exc),
            model_run_id=str(run_id),
        )
        raise

    return SegmentationScoreResult(
        model_run_id=run_id,
        data_as_of=effective_as_of,
        n_customers=len(rfm_scored),
        active_output_updated=True,
        profiles_df=profiles_df,
        customer_segments_df=rfm_scored,
    )

"""Training pipeline for RFM customer segmentation.

Implements P3-02, TRD §5.2, §6, and PRD §7.3 (FR-CUS-02, FR-CUS-03):
- Queries RFM customer features as of explicit data_as_of date.
- Evaluates k in [3, 8] using silhouette score with min cluster ratio >= 2%.
- Logs candidate silhouette and Davies-Bouldin metrics.
- Fits RFMSegmentationModel with deterministic business labels.
- Evaluates stability ARI across 5 random seeds.
- Builds segment profiles covering 100% of customers.
- Tracks run in MLflow 'retailpulse/segmentation' and mirrors to ml.model_run.
- Registers model version in MLflow registry as 'retailpulse-segmentation'.
- Optionally promotes to champion alias and updates database mirrors.
- Persists ml.segment_profile records to the database.
"""

from __future__ import annotations

import datetime
import tempfile
import uuid
from decimal import Decimal
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import FactSalesLine
from retailpulse.db.models.ml import SegmentProfile
from retailpulse.features.rfm import compute_rfm_from_db, transform_rfm_for_clustering
from retailpulse.logging import get_logger
from retailpulse.ml.registry import promote_champion, register_model
from retailpulse.ml.runs import track_model_run
from retailpulse.ml.segmentation.kmeans import (
    RFMSegmentationModel,
    evaluate_clustering_stability,
    select_optimal_k,
)

logger = get_logger(__name__)

TASK_SEGMENTATION = "segmentation"
MODEL_NAME_SEGMENTATION = "retailpulse-segmentation"


class SegmentationTrainResult:
    """Strongly-typed summary of segmentation training pipeline execution."""

    def __init__(
        self,
        *,
        model_run_id: uuid.UUID,
        data_as_of: datetime.date,
        chosen_k: int,
        silhouette: float,
        davies_bouldin: float,
        mean_ari_stability: float,
        n_customers: int,
        is_champion: bool,
        version: str | None,
        profiles_df: pd.DataFrame,
        candidate_metrics: dict[int, dict[str, float]],
    ) -> None:
        self.model_run_id = model_run_id
        self.data_as_of = data_as_of
        self.chosen_k = chosen_k
        self.silhouette = silhouette
        self.davies_bouldin = davies_bouldin
        self.mean_ari_stability = mean_ari_stability
        self.n_customers = n_customers
        self.is_champion = is_champion
        self.version = version
        self.profiles_df = profiles_df
        self.candidate_metrics = candidate_metrics


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

    # Default fallback
    return datetime.date(2011, 12, 9)


def train_segmentation_pipeline(  # noqa: PLR0915
    session: Session,
    *,
    data_as_of: datetime.date | None = None,
    k: int | None = None,
    auto_promote: bool | None = None,
    seed: int | None = None,
    rfm_window_days: int | None = None,
    min_cluster_ratio: float = 0.02,
    tracking_uri: str | None = None,
) -> SegmentationTrainResult:
    """Execute customer segmentation training pipeline.

    Args:
        session: Active SQLAlchemy database session.
        data_as_of: Reference cutoff date (defaults to latest sales date).
        k: Optional explicit cluster count. If None, selects optimal k in [3, 8].
        auto_promote: Whether to promote new model to champion (defaults to setting).
        seed: Random seed for clustering initialization (defaults to RP_SEED).
        rfm_window_days: Historical lookback window in days (defaults to setting).
        min_cluster_ratio: Minimum cluster customer ratio required (0.02).
        tracking_uri: Optional explicit MLflow tracking URI for test isolation.

    Returns:
        SegmentationTrainResult with trained model run metrics, profiles, and champion status.
    """
    settings = get_settings()
    effective_seed = seed if seed is not None else settings.seed
    effective_window = rfm_window_days if rfm_window_days is not None else settings.rfm_window_days
    effective_promote = auto_promote if auto_promote is not None else settings.auto_promote

    effective_as_of = _resolve_effective_as_of(session, data_as_of)
    logger.info(
        "starting segmentation training pipeline",
        as_of=str(effective_as_of),
        seed=effective_seed,
        rfm_window_days=effective_window,
        explicit_k=k,
    )

    # 1. Compute customer RFM features
    rfm_df = compute_rfm_from_db(session, as_of=effective_as_of, rfm_window_days=effective_window)
    n_customers = len(rfm_df)
    if n_customers < 10:
        raise ValueError(
            f"Insufficient customer data ({n_customers} customers) to train segmentation model"
        )

    # 2. Optimal k selection or explicit k evaluation
    if k is None:
        chosen_k, candidate_metrics = select_optimal_k(
            rfm_df,
            k_range=(3, 8),
            min_cluster_ratio=min_cluster_ratio,
            seed=effective_seed,
        )
    else:
        chosen_k = k
        x_scaled, _ = transform_rfm_for_clustering(rfm_df)
        _, candidate_metrics = select_optimal_k(
            rfm_df,
            k_range=(k, k),
            min_cluster_ratio=min_cluster_ratio,
            seed=effective_seed,
        )

    # 3. Fit champion model on optimal k
    model = RFMSegmentationModel(k=chosen_k, seed=effective_seed)
    model.fit(rfm_df)

    # 4. Evaluate clustering stability ARI across 5 seeds
    x_scaled, _ = transform_rfm_for_clustering(rfm_df)
    mean_ari, _seed_aris = evaluate_clustering_stability(
        x_scaled,
        chosen_k,
        base_seed=effective_seed,
        n_seeds=5,
    )
    model.stability_ari_ = mean_ari
    model.candidate_metrics_ = candidate_metrics

    # 5. Build segment profiles
    profiles_df = model.build_profiles(rfm_df)

    # Extract primary evaluation metrics
    best_metrics = candidate_metrics.get(chosen_k, {})
    sil_score = float(best_metrics.get("silhouette", 0.0))
    db_score = float(best_metrics.get("davies_bouldin", 0.0))

    # 6. MLflow experiment tracking and database mirroring via track_model_run
    with track_model_run(
        task=TASK_SEGMENTATION,
        algorithm="kmeans",
        data_as_of=effective_as_of,
        seed=effective_seed,
        session=session,
        tracking_uri=tracking_uri,
    ) as tracker:
        # Log parameters
        tracker.log_params(
            {
                "k": chosen_k,
                "k_specified": k is not None,
                "k_search_min": 3,
                "k_search_max": 8,
                "n_init": 20,
                "seed": effective_seed,
                "min_cluster_ratio": min_cluster_ratio,
                "n_customers": n_customers,
                "rfm_window_days": effective_window,
            }
        )

        # Log final metrics
        metrics_dict: dict[str, float] = {
            "chosen_k": float(chosen_k),
            "silhouette": sil_score,
            "davies_bouldin": db_score,
            "mean_ari_stability": mean_ari,
        }
        # Add candidate metrics
        for cand_k, cand_m in candidate_metrics.items():
            metrics_dict[f"k{cand_k}_silhouette"] = cand_m["silhouette"]
            metrics_dict[f"k{cand_k}_davies_bouldin"] = cand_m["davies_bouldin"]
            metrics_dict[f"k{cand_k}_min_cluster_share"] = cand_m["min_cluster_share"]

        tracker.log_metrics(metrics_dict)

        # Log segment profiles JSON artifact
        tracker.log_dict(
            profiles_df.to_dict(orient="records"),
            "segment_profiles.json",
        )

        # Save and log model artifact
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.joblib"
            model.save(model_file)
            tracker.log_artifact(str(model_file), artifact_path="model")

        run_id = tracker.run_id
        mlflow_run_id = tracker.mlflow_run_id

    # 7. Model registration in MLflow registry
    version: str | None = None
    is_champion = False
    try:
        version = register_model(
            mlflow_run_id=mlflow_run_id,
            task=TASK_SEGMENTATION,
            model_name=MODEL_NAME_SEGMENTATION,
            tracking_uri=tracking_uri,
        )
        if effective_promote:
            promote_champion(
                task=TASK_SEGMENTATION,
                version=version,
                session=session,
                model_name=MODEL_NAME_SEGMENTATION,
                tracking_uri=tracking_uri,
            )
            is_champion = True
    except Exception as exc:
        logger.warning("model registration/promotion failed", error=str(exc))

    # 8. Persist ml.segment_profile records to database
    try:
        session.execute(
            sa.delete(SegmentProfile).where(
                SegmentProfile.model_run_id == run_id,
                SegmentProfile.as_of_date == effective_as_of,
            )
        )
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
        session.commit()
        logger.info(
            "persisted segment profiles to database",
            model_run_id=str(run_id),
            as_of=str(effective_as_of),
            profile_count=len(profile_entities),
        )
    except Exception as exc:
        session.rollback()
        logger.error("failed persisting segment profiles", error=str(exc))
        raise

    return SegmentationTrainResult(
        model_run_id=run_id,
        data_as_of=effective_as_of,
        chosen_k=chosen_k,
        silhouette=sil_score,
        davies_bouldin=db_score,
        mean_ari_stability=mean_ari,
        n_customers=n_customers,
        is_champion=is_champion,
        version=version,
        profiles_df=profiles_df,
        candidate_metrics=candidate_metrics,
    )

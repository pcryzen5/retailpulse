"""K-Means customer segmentation, k selection, and deterministic label mapping.

Implements P3-02, TRD §5.2, and PRD §7.3 (FR-CUS-02, FR-CUS-03):
- Feature transformation: log1p + StandardScaler via retailpulse.features.rfm.
- K-Means clustering with n_init=20, random_state=RP_SEED.
- Optimal k selection: evaluates k in [3, 8] using silhouette score subject to
  every cluster having >= 2% of total customers; logs candidate silhouette and
  Davies-Bouldin indices.
- Deterministic centroid rank label mapping to canonical business labels:
  Champions, Loyal, New / Promising, Big Spenders, At Risk, Hibernating.
- Stability assessment: refits optimal k across 5 seeds, reporting mean ARI.
- Segment profile generation: customer counts, average R/F/M, and revenue shares.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from retailpulse.features.rfm import (
    SEGMENT_AT_RISK,
    SEGMENT_BIG_SPENDERS,
    SEGMENT_CHAMPIONS,
    SEGMENT_HIBERNATING,
    SEGMENT_LOYAL,
    SEGMENT_NEW_PROMISING,
    transform_rfm_for_clustering,
)
from retailpulse.logging import get_logger

logger = get_logger(__name__)


def assign_cluster_labels(centroids_df: pd.DataFrame) -> dict[int, str]:
    """Map cluster centroids to canonical business labels using deterministic ranking.

    Per TRD §5.2 and DESIGN.md §41:
    Pure, deterministic function mapping cluster centroids in RFM space to the
    canonical business segments:
    - Champions: Highest overall RFM engagement (high spend, high frequency, recent).
    - Hibernating: Longest elapsed recency days with lowest spend.
    - At Risk: Lapsed shoppers (high recency days) with historical frequency/spend.
    - New / Promising: Active recent shoppers with low order frequency.
    - Big Spenders: High monetary spenders who are not Champions.
    - Loyal: High frequency regular shoppers.

    Args:
        centroids_df: DataFrame with columns:
            `['cluster', 'avg_recency_days', 'avg_frequency', 'avg_monetary']`.

    Returns:
        Dictionary mapping cluster integer ID to canonical segment label string.
    """
    df = centroids_df.copy()
    k = len(df)
    if k == 0:
        return {}

    # Rank 1 is best along each dimension:
    # Recency: lowest recency_days = rank 1 (most recent)
    df["r_rank"] = df["avg_recency_days"].rank(ascending=True, method="first").astype(int)
    # Frequency: highest frequency = rank 1 (most frequent)
    df["f_rank"] = df["avg_frequency"].rank(ascending=False, method="first").astype(int)
    # Monetary: highest monetary = rank 1 (highest spend)
    df["m_rank"] = df["avg_monetary"].rank(ascending=False, method="first").astype(int)

    # Composite rank: lower sum = superior overall engagement
    df["composite"] = df["r_rank"] + df["f_rank"] + df["m_rank"]

    assigned: dict[int, str] = {}
    remaining_clusters = set(df["cluster"].astype(int))

    # 1. Champions: Lowest composite rank and highest spend among active
    champ_candidates = df.loc[df["cluster"].isin(remaining_clusters)].sort_values(
        by=["m_rank", "composite", "r_rank"], ascending=[True, True, True]
    )
    champ_id = int(champ_candidates.iloc[0]["cluster"])
    assigned[champ_id] = SEGMENT_CHAMPIONS
    remaining_clusters.remove(champ_id)

    # 2. Hibernating: Worst recency (highest days) and low spend
    if remaining_clusters:
        hib_candidates = df.loc[df["cluster"].isin(remaining_clusters)].sort_values(
            by=["r_rank", "m_rank"], ascending=[False, False]
        )
        hib_id = int(hib_candidates.iloc[0]["cluster"])
        assigned[hib_id] = SEGMENT_HIBERNATING
        remaining_clusters.remove(hib_id)

    # 3. At Risk: High recency days (inactive), but moderate/high historical engagement
    for cid in list(remaining_clusters):
        row = df.loc[df["cluster"] == cid].iloc[0]
        if row["r_rank"] > k / 2:
            assigned[int(cid)] = SEGMENT_AT_RISK
            remaining_clusters.remove(cid)
            break

    # 4. New / Promising: Very recent purchase (low recency days) with low frequency
    for cid in list(remaining_clusters):
        row = df.loc[df["cluster"] == cid].iloc[0]
        if row["r_rank"] <= max(2, k // 2) and row["f_rank"] > k / 2:
            assigned[int(cid)] = SEGMENT_NEW_PROMISING
            remaining_clusters.remove(cid)
            break

    # 5. Big Spenders: High spend among remaining active clusters
    for cid in list(remaining_clusters):
        row = df.loc[df["cluster"] == cid].iloc[0]
        if row["m_rank"] <= max(2, k // 2):
            assigned[int(cid)] = SEGMENT_BIG_SPENDERS
            remaining_clusters.remove(cid)
            break

    # 6. Remaining clusters get Loyal or cycle through canonical labels
    available_labels = [
        SEGMENT_LOYAL,
        SEGMENT_BIG_SPENDERS,
        SEGMENT_NEW_PROMISING,
        SEGMENT_AT_RISK,
    ]
    for cid in sorted(remaining_clusters):
        # Pick first label not yet assigned, or fall back to Loyal
        chosen_label = next(
            (label for label in available_labels if label not in assigned.values()),
            SEGMENT_LOYAL,
        )
        assigned[int(cid)] = chosen_label
        remaining_clusters.remove(cid)

    return assigned


def build_segment_profiles(
    rfm_df: pd.DataFrame,
    cluster_labels: dict[int, str],
) -> pd.DataFrame:
    """Build summary profiles for each cluster verifying 100% coverage.

    Args:
        rfm_df: DataFrame containing `cluster`, `recency_days`, `frequency`, `monetary`.
        cluster_labels: Mapping from cluster ID to business label.

    Returns:
        DataFrame with columns:
        `['segment_id', 'segment_label', 'n_customers', 'avg_recency_days',
          'avg_frequency', 'avg_monetary', 'revenue_share']`.
    """
    if "cluster" not in rfm_df.columns:
        raise ValueError("DataFrame must contain 'cluster' column to build profiles")

    total_customers = len(rfm_df)
    total_revenue = float(rfm_df["monetary"].sum())

    profiles = (
        rfm_df.groupby("cluster")
        .agg(
            n_customers=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            tot_monetary=("monetary", "sum"),
        )
        .reset_index()
    )

    profiles["segment_id"] = profiles["cluster"].astype(int)
    profiles["segment_label"] = [
        cluster_labels.get(cid, f"Segment {cid}") for cid in profiles["segment_id"]
    ]

    if total_revenue > 0:
        profiles["revenue_share"] = (profiles["tot_monetary"] / total_revenue).round(5)
    else:
        profiles["revenue_share"] = 0.0

    profiles["avg_recency_days"] = profiles["avg_recency_days"].round(2)
    profiles["avg_frequency"] = profiles["avg_frequency"].round(2)
    profiles["avg_monetary"] = profiles["avg_monetary"].round(2)

    cols = [
        "segment_id",
        "segment_label",
        "n_customers",
        "avg_recency_days",
        "avg_frequency",
        "avg_monetary",
        "revenue_share",
    ]
    result = profiles[cols].sort_values("segment_id").reset_index(drop=True)

    # Verification: Profiles must cover 100% of customers
    covered_customers = int(result["n_customers"].sum())
    if covered_customers != total_customers:
        raise ValueError(
            f"Segment profiles cover {covered_customers} customers but total is {total_customers}"
        )

    return result


def select_optimal_k(
    rfm_df: pd.DataFrame,
    *,
    k_range: tuple[int, int] = (3, 8),
    min_cluster_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[int, dict[int, dict[str, float]]]:
    """Select optimal k in range [3, 8] by silhouette score with min cluster size.

    Implements TRD §5.2:
    - k in 3..8 chosen by highest silhouette with every cluster >= 2% of customers.
    - Logs all candidates' silhouette and Davies-Bouldin index.

    Args:
        rfm_df: Customer RFM feature DataFrame.
        k_range: Tuple of (min_k, max_k), defaults to (3, 8).
        min_cluster_ratio: Minimum fraction of customers required per cluster (0.02).
        seed: Random seed for K-Means initialization.

    Returns:
        Tuple of (optimal_k: int, candidate_metrics: dict).
    """
    x_scaled, _ = transform_rfm_for_clustering(rfm_df)
    n_samples = len(x_scaled)

    min_k, max_k = k_range
    # Ensure k does not exceed sample count
    max_k = min(max_k, max(3, n_samples - 1))
    min_k = min(min_k, max_k)

    candidate_metrics: dict[int, dict[str, float]] = {}

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(x_scaled)

        sil = float(silhouette_score(x_scaled, labels))
        db = float(davies_bouldin_score(x_scaled, labels))

        cluster_counts = pd.Series(labels).value_counts(normalize=True)
        min_share = float(cluster_counts.min())
        is_valid = bool(min_share >= min_cluster_ratio)

        candidate_metrics[k] = {
            "silhouette": sil,
            "davies_bouldin": db,
            "min_cluster_share": min_share,
            "is_valid": 1.0 if is_valid else 0.0,
        }
        logger.info(
            "evaluated k candidate",
            k=k,
            silhouette=round(sil, 4),
            davies_bouldin=round(db, 4),
            min_cluster_share=round(min_share, 4),
            is_valid=is_valid,
        )

    # Filter to valid candidates meeting the min_cluster_ratio constraint
    valid_ks = [k for k, m in candidate_metrics.items() if m["is_valid"] == 1.0]
    if valid_ks:
        best_k = max(valid_ks, key=lambda k: candidate_metrics[k]["silhouette"])
    else:
        # Fall back to highest silhouette overall if none met strict cluster size
        best_k = max(candidate_metrics.keys(), key=lambda k: candidate_metrics[k]["silhouette"])
        logger.warning(
            "no k met min_cluster_ratio threshold; selected highest overall silhouette",
            chosen_k=best_k,
        )

    logger.info(
        "selected optimal k",
        chosen_k=best_k,
        silhouette=candidate_metrics[best_k]["silhouette"],
        davies_bouldin=candidate_metrics[best_k]["davies_bouldin"],
    )
    return best_k, candidate_metrics


def evaluate_clustering_stability(
    x_scaled: npt.NDArray[np.float64],
    k: int,
    *,
    base_seed: int = 42,
    n_seeds: int = 5,
) -> tuple[float, list[float]]:
    """Assess clustering stability using Adjusted Rand Index (ARI) across seeds.

    Implements TRD §5.2:
    - Refits model with 5 seeds and reports mean adjusted Rand index.

    Args:
        x_scaled: Preprocessed feature array.
        k: Number of clusters to evaluate.
        base_seed: Base random seed.
        n_seeds: Number of distinct seeds to compare against (defaults to 5).

    Returns:
        Tuple of (mean_ari: float, seed_aris: list[float]).
    """
    km_base = KMeans(n_clusters=k, n_init=20, random_state=base_seed).fit(x_scaled)
    base_labels = km_base.labels_

    seed_aris: list[float] = []
    for s in range(1, n_seeds):
        km_seed = KMeans(n_clusters=k, n_init=20, random_state=base_seed + s).fit(x_scaled)
        ari = float(adjusted_rand_score(base_labels, km_seed.labels_))
        seed_aris.append(ari)

    mean_ari = float(np.mean(seed_aris)) if seed_aris else 1.0
    logger.info("evaluated clustering stability", k=k, mean_ari=round(mean_ari, 4), seeds=n_seeds)
    return mean_ari, seed_aris


class RFMSegmentationModel:
    """K-Means customer segmentation model with deterministic business labels."""

    def __init__(
        self,
        *,
        k: int = 4,
        seed: int = 42,
        n_init: int = 20,
    ) -> None:
        """Initialize RFM segmentation model.

        Args:
            k: Number of clusters.
            seed: Random seed for reproducibility.
            n_init: Number of random initializations for K-Means (default 20).
        """
        self.k = k
        self.seed = seed
        self.n_init = n_init
        self.scaler: StandardScaler | None = None
        self.kmeans: KMeans | None = None
        self.cluster_labels_: dict[int, str] = {}
        self.centroids_unscaled_: pd.DataFrame | None = None
        self.candidate_metrics_: dict[int, dict[str, float]] | None = None
        self.stability_ari_: float | None = None
        self.is_fitted_: bool = False

    def fit(self, rfm_df: pd.DataFrame) -> RFMSegmentationModel:
        """Fit scaler, K-Means clustering, and assign deterministic business labels.

        Args:
            rfm_df: Customer RFM DataFrame with columns:
                `recency_days`, `frequency`, `monetary`.

        Returns:
            Fitted RFMSegmentationModel instance.
        """
        required_cols = ["recency_days", "frequency", "monetary"]
        missing = [c for c in required_cols if c not in rfm_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for segmentation: {missing}")

        x_scaled, self.scaler = transform_rfm_for_clustering(rfm_df)

        self.kmeans = KMeans(n_clusters=self.k, n_init=self.n_init, random_state=self.seed)
        cluster_assignments = self.kmeans.fit_predict(x_scaled)

        # Calculate centroids in original unscaled RFM space
        clustered_df = rfm_df.copy()
        clustered_df["cluster"] = cluster_assignments

        centroids = (
            clustered_df.groupby("cluster")
            .agg(
                avg_recency_days=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
            )
            .reset_index()
        )
        self.centroids_unscaled_ = centroids

        # Assign deterministic business labels
        self.cluster_labels_ = assign_cluster_labels(centroids)
        self.is_fitted_ = True

        logger.info(
            "fitted rfm segmentation model",
            k=self.k,
            labels=self.cluster_labels_,
            n_customers=len(rfm_df),
        )
        return self

    def predict(self, rfm_df: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Predict cluster integer ID (0..k-1) for customers.

        Args:
            rfm_df: DataFrame with `recency_days`, `frequency`, `monetary`.

        Returns:
            Numpy array of integer cluster assignments.
        """
        if not self.is_fitted_ or self.scaler is None or self.kmeans is None:
            raise ValueError("RFMSegmentationModel is not fitted yet; call fit() first")

        x_scaled, _ = transform_rfm_for_clustering(rfm_df, scaler=self.scaler)
        preds: npt.NDArray[np.int64] = np.asarray(self.kmeans.predict(x_scaled), dtype=np.int64)
        return preds

    def predict_labels(self, rfm_df: pd.DataFrame) -> list[str]:
        """Predict canonical segment business labels for customers.

        Args:
            rfm_df: Customer RFM DataFrame.

        Returns:
            List of business label strings corresponding to rows in `rfm_df`.
        """
        cluster_ids = self.predict(rfm_df)
        return [self.cluster_labels_.get(int(cid), f"Segment {cid}") for cid in cluster_ids]

    def build_profiles(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Score rfm_df and build segment profiles covering 100% of customers.

        Args:
            rfm_df: Customer RFM DataFrame.

        Returns:
            DataFrame of segment profiles.
        """
        data = rfm_df.copy()
        data["cluster"] = self.predict(data)
        return build_segment_profiles(data, self.cluster_labels_)

    def save(self, path: str | Path) -> None:
        """Persist model state to disk via joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("saved segmentation model", path=str(save_path))

    @classmethod
    def load(cls, path: str | Path) -> RFMSegmentationModel:
        """Load persisted model state from disk via joblib."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        model = joblib.load(load_path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is {type(model)}, expected {cls}")
        return model

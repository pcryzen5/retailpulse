"""Unit tests for RFM customer segmentation, k selection, and scoring (P3-02)."""

from __future__ import annotations

import datetime
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from retailpulse.db.models.core import (
    DimCustomer,
    DimDate,
    DimProduct,
    DimStore,
    FactSalesLine,
)
from retailpulse.db.models.ml import ActiveOutput, CustomerSegment, ModelRun, SegmentProfile
from retailpulse.db.models.ops import IngestionBatch
from retailpulse.features.rfm import (
    CANONICAL_SEGMENTS,
    SEGMENT_AT_RISK,
    SEGMENT_CHAMPIONS,
    SEGMENT_HIBERNATING,
    SEGMENT_NEW_PROMISING,
    transform_rfm_for_clustering,
)
from retailpulse.ml.segmentation.kmeans import (
    RFMSegmentationModel,
    assign_cluster_labels,
    evaluate_clustering_stability,
    select_optimal_k,
)
from retailpulse.ml.segmentation.score import score_segmentation_pipeline
from retailpulse.ml.segmentation.train import train_segmentation_pipeline


@pytest.fixture
def synthetic_rfm_df() -> pd.DataFrame:
    """Generate reproducible customer RFM features for clustering tests."""
    rng = np.random.default_rng(42)

    # Cluster 1: Champions (low recency days, high frequency, high spend)
    c1 = pd.DataFrame(
        {
            "customer_id": range(1, 51),
            "recency_days": rng.integers(1, 30, 50),
            "frequency": rng.integers(10, 30, 50),
            "monetary": rng.uniform(2000.0, 10000.0, 50),
        }
    )

    # Cluster 2: New / Promising (low recency days, low frequency, moderate spend)
    c2 = pd.DataFrame(
        {
            "customer_id": range(51, 101),
            "recency_days": rng.integers(1, 20, 50),
            "frequency": rng.integers(1, 3, 50),
            "monetary": rng.uniform(50.0, 300.0, 50),
        }
    )

    # Cluster 3: At Risk (moderate/high recency days, good past frequency/spend)
    c3 = pd.DataFrame(
        {
            "customer_id": range(101, 151),
            "recency_days": rng.integers(120, 250, 50),
            "frequency": rng.integers(4, 12, 50),
            "monetary": rng.uniform(500.0, 2500.0, 50),
        }
    )

    # Cluster 4: Hibernating (very high recency days, low frequency, low spend)
    c4 = pd.DataFrame(
        {
            "customer_id": range(151, 201),
            "recency_days": rng.integers(300, 600, 50),
            "frequency": rng.integers(1, 2, 50),
            "monetary": rng.uniform(10.0, 100.0, 50),
        }
    )

    df = pd.concat([c1, c2, c3, c4], ignore_index=True)
    df["r_score"] = 3
    df["f_score"] = 3
    df["m_score"] = 3
    df["rfm_score"] = "333"
    df["rule_segment"] = "Loyal"
    return df


def test_assign_cluster_labels_deterministic() -> None:
    """VERIFY REQUIREMENT: Centroid label mapping is pure, deterministic across runs."""
    centroids = pd.DataFrame(
        [
            {
                "cluster": 0,
                "avg_recency_days": 226.4,
                "avg_frequency": 3.1,
                "avg_monetary": 1758.0,
            },
            {
                "cluster": 1,
                "avg_recency_days": 27.0,
                "avg_frequency": 11.2,
                "avg_monetary": 10196.7,
            },
            {
                "cluster": 2,
                "avg_recency_days": 27.1,
                "avg_frequency": 1.8,
                "avg_monetary": 851.9,
            },
            {
                "cluster": 3,
                "avg_recency_days": 392.8,
                "avg_frequency": 1.3,
                "avg_monetary": 295.0,
            },
        ]
    )

    # Run mapping multiple times
    labels_run1 = assign_cluster_labels(centroids)
    labels_run2 = assign_cluster_labels(centroids)

    # Must be 100% identical across runs
    assert labels_run1 == labels_run2

    # Verify expected business assignments
    assert labels_run1[1] == SEGMENT_CHAMPIONS
    assert labels_run1[3] == SEGMENT_HIBERNATING
    assert labels_run1[0] == SEGMENT_AT_RISK
    assert labels_run1[2] == SEGMENT_NEW_PROMISING

    # All assigned labels belong to CANONICAL_SEGMENTS
    for label in labels_run1.values():
        assert label in CANONICAL_SEGMENTS


def test_select_optimal_k(synthetic_rfm_df: pd.DataFrame) -> None:
    """Test k selection in [3, 8] with silhouette and min_cluster_ratio constraint."""
    best_k, candidate_metrics = select_optimal_k(
        synthetic_rfm_df,
        k_range=(3, 6),
        min_cluster_ratio=0.02,
        seed=42,
    )

    assert best_k in range(3, 7)
    assert len(candidate_metrics) == 4  # k=3, 4, 5, 6

    for _k_val, metrics in candidate_metrics.items():
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert "min_cluster_share" in metrics
        assert "is_valid" in metrics
        assert metrics["min_cluster_share"] >= 0.0


def test_evaluate_clustering_stability_ari(synthetic_rfm_df: pd.DataFrame) -> None:
    """Test ARI stability calculation across 5 seeds per TRD §5.2."""
    x_scaled, _ = transform_rfm_for_clustering(synthetic_rfm_df)
    mean_ari, seed_aris = evaluate_clustering_stability(
        x_scaled,
        k=4,
        base_seed=42,
        n_seeds=5,
    )

    assert len(seed_aris) == 4
    assert 0.0 <= mean_ari <= 1.0
    # For well-separated clusters, ARI should be high (> 0.70)
    assert mean_ari > 0.70


def test_rfm_segmentation_model_lifecycle(synthetic_rfm_df: pd.DataFrame) -> None:
    """Test RFMSegmentationModel fitting, predictions, persistence, and loading."""
    model = RFMSegmentationModel(k=4, seed=42)
    model.fit(synthetic_rfm_df)

    assert model.is_fitted_
    assert len(model.cluster_labels_) == 4

    # Predictions
    preds = model.predict(synthetic_rfm_df)
    labels = model.predict_labels(synthetic_rfm_df)

    assert len(preds) == len(synthetic_rfm_df)
    assert set(preds).issubset({0, 1, 2, 3})
    assert len(labels) == len(synthetic_rfm_df)
    for lbl in labels:
        assert lbl in CANONICAL_SEGMENTS

    # Persistence roundtrip
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "model.joblib"
        model.save(tmp_path)
        loaded = RFMSegmentationModel.load(tmp_path)
        assert loaded.is_fitted_
        assert loaded.k == 4
        assert loaded.cluster_labels_ == model.cluster_labels_

        loaded_preds = loaded.predict(synthetic_rfm_df)
        np.testing.assert_array_equal(preds, loaded_preds)


def test_segment_profiles_coverage_100_percent(synthetic_rfm_df: pd.DataFrame) -> None:
    """VERIFY REQUIREMENT: Profiles cover 100% of customers and revenue share sums to 1.0."""
    model = RFMSegmentationModel(k=4, seed=42)
    model.fit(synthetic_rfm_df)

    profiles = model.build_profiles(synthetic_rfm_df)

    # 1. Total customers in profiles must equal cohort customer count (100% coverage)
    total_customers = len(synthetic_rfm_df)
    assert int(profiles["n_customers"].sum()) == total_customers

    # 2. Revenue share must sum to approximately 1.0
    assert pytest.approx(float(profiles["revenue_share"].sum()), abs=1e-3) == 1.0

    # 3. Every row has valid metrics
    for _, row in profiles.iterrows():
        assert row["n_customers"] > 0
        assert row["avg_recency_days"] > 0
        assert row["avg_frequency"] > 0
        assert row["avg_monetary"] > 0
        assert row["segment_label"] in CANONICAL_SEGMENTS


@pytest.fixture
def mock_db_session() -> sessionmaker:
    """Create in-memory SQLite database session factory with schema translate map."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        execution_options={
            "schema_translate_map": {"core": None, "ops": None, "ml": None, "app": None}
        },
    )

    tables = [
        IngestionBatch,
        DimStore,
        DimProduct,
        DimCustomer,
        DimDate,
        FactSalesLine,
        ModelRun,
        SegmentProfile,
        CustomerSegment,
        ActiveOutput,
    ]
    for table_cls in tables:
        table_cls.__table__.create(engine)

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        batch_id = uuid.uuid4()
        # Seed 20 test customers
        for cid in range(1, 21):
            session.add(
                FactSalesLine(
                    batch_id=batch_id,
                    source_row_hash=f"hash-{cid}",
                    invoice_no=f"INV-{cid}",
                    invoice_ts=datetime.datetime(
                        2025, 12, 1 + (cid % 25), 10, 0, tzinfo=datetime.UTC
                    ),
                    date_key=datetime.date(2025, 12, 1 + (cid % 25)),
                    store_id=1,
                    product_id=1,
                    customer_id=cid,
                    quantity=cid * 2,
                    unit_price=25.0,
                    is_return=False,
                )
            )
        session.commit()

    return session_factory


@pytest.fixture
def temp_mlflow_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create an isolated tracking URI for testing."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    return mlflow_dir.as_uri()


def test_train_and_score_segmentation_mock_db(
    mock_db_session: sessionmaker,
    temp_mlflow_uri: str,
) -> None:
    """Test full training and scoring cycle with transactional active_output advancement."""
    as_of = datetime.date(2025, 12, 31)

    with mock_db_session() as session:
        # 1. Train segmentation pipeline
        train_res = train_segmentation_pipeline(
            session=session,
            data_as_of=as_of,
            k=3,
            auto_promote=True,
            seed=42,
            tracking_uri=temp_mlflow_uri,
        )

        assert train_res.chosen_k == 3
        assert train_res.n_customers == 20
        assert len(train_res.profiles_df) == 3

        # 2. Score segmentation pipeline
        score_res = score_segmentation_pipeline(
            session=session,
            data_as_of=as_of,
            model_run_id=train_res.model_run_id,
        )

        assert score_res.n_customers == 20
        assert score_res.active_output_updated

        # Verify database entities: ml.customer_segment
        cust_rows = session.execute(sa.select(CustomerSegment)).scalars().all()
        assert len(cust_rows) == 20

        # Verify ml.segment_profile
        prof_rows = session.execute(sa.select(SegmentProfile)).scalars().all()
        assert len(prof_rows) == 3

        # Verify ml.active_output
        active_row = session.get(ActiveOutput, "segmentation")
        assert active_row is not None
        assert str(active_row.model_run_id) == str(train_res.model_run_id)

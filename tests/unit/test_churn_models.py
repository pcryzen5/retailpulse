"""Unit tests for churn metrics, models, calibration, and training pipeline (P3-04).

Verifies TRD §5.3, PRD §7.4 (FR-CH-03, FR-CH-04) & §8.2:
- Churn evaluation metrics: PR-AUC, ROC-AUC, Brier score, Log Loss, Top-Decile Lift/Precision.
- Risk band assignment logic and thresholds.
- Logistic Regression baseline with median imputation and missing indicators.
- LightGBM classifier with early stopping.
- Probability calibration (Isotonic vs Platt/Sigmoid selection).
- CalibratedChurnModel serialization and lifecycle.
- Champion gate evaluation.
- End-to-end churn training pipeline with SQLite and temporary MLflow store.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal
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
from retailpulse.db.models.ml import ModelRun
from retailpulse.db.models.ops import IngestionBatch
from retailpulse.ml.churn.metrics import (
    RISK_BAND_HIGH,
    RISK_BAND_LOW,
    RISK_BAND_MEDIUM,
    assign_risk_band,
    assign_risk_bands,
    compute_churn_metrics,
    plot_calibration_curve,
)
from retailpulse.ml.churn.models import (
    ALL_FEATURES,
    CalibratedChurnModel,
    calibrate_model,
    create_logistic_baseline_pipeline,
    fit_lightgbm_with_early_stopping,
)
from retailpulse.ml.churn.train import train_churn_pipeline


@pytest.fixture
def temp_mlflow_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Provide isolated local file-backed MLflow tracking URI."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    return mlflow_dir.as_uri()


def test_compute_churn_metrics_exactness() -> None:
    """Verify exact calculation of ranking, calibration, and decile metrics."""
    # 10 samples, 4 positives -> base rate 0.40
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=np.int64)
    # Perfect probabilities ranking
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.65, 0.05, 0.15])

    metrics = compute_churn_metrics(y_true, y_prob)

    assert metrics["base_rate"] == 0.40
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier_score"] < metrics["brier_score_base_rate"]

    # Top decile: 10% of 10 samples = top 1 customer (y_prob=0.9, y_true=1)
    assert metrics["precision_top_10"] == 1.0
    assert pytest.approx(metrics["lift_top_10"], abs=1e-3) == 1.0 / 0.40  # 2.5x lift

    # Top quintile: 20% of 10 samples = top 2 customers (both positive) -> recall = 2 / 4 = 0.50
    assert metrics["recall_top_20"] == 0.50

    # Empty inputs edge case
    empty_m = compute_churn_metrics([], [])
    assert empty_m["pr_auc"] == 0.0
    assert empty_m["roc_auc"] == 0.5


def test_assign_risk_bands() -> None:
    """Verify churn risk band thresholds: low (<0.30), medium (0.30-0.60), high (>=0.60)."""
    assert assign_risk_band(0.10) == RISK_BAND_LOW
    assert assign_risk_band(0.299) == RISK_BAND_LOW
    assert assign_risk_band(0.30) == RISK_BAND_MEDIUM
    assert assign_risk_band(0.599) == RISK_BAND_MEDIUM
    assert assign_risk_band(0.60) == RISK_BAND_HIGH
    assert assign_risk_band(0.95) == RISK_BAND_HIGH

    # Vectorized
    probs = np.array([0.05, 0.35, 0.75, 0.30, 0.60])
    bands = assign_risk_bands(probs, medium_threshold=0.30, high_threshold=0.60)
    expected = np.array(
        [RISK_BAND_LOW, RISK_BAND_MEDIUM, RISK_BAND_HIGH, RISK_BAND_MEDIUM, RISK_BAND_HIGH]
    )
    np.testing.assert_array_equal(bands, expected)


def test_plot_calibration_curve(tmp_path: Path) -> None:
    """Verify calibration curve generation and artifact saving."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0] * 5)
    probs_a = np.linspace(0.1, 0.9, 50)
    probs_b = np.linspace(0.2, 0.8, 50)

    plot_path = tmp_path / "calibration.png"
    plot_calibration_curve(
        y_true,
        {"Model A": probs_a, "Model B": probs_b},
        plot_path,
        n_bins=5,
    )

    assert plot_path.is_file()
    assert plot_path.stat().st_size > 500


def test_logistic_baseline_pipeline() -> None:
    """Verify Logistic Regression baseline handles missing numeric values and unseen categories."""
    pipe = create_logistic_baseline_pipeline(seed=42)

    # Synthetic training data with missing gap_mean
    countries = ["United Kingdom", "France", "Germany", "United Kingdom", "Spain", "EIRE"]
    train_data = pd.DataFrame(
        {
            "recency_days": [10.0, 45.0, 120.0, 300.0, 20.0, 80.0] * 10,
            "tenure_days": [100.0, 200.0, 120.0, 350.0, 20.0, 150.0] * 10,
            "orders_30d": [1, 0, 0, 0, 1, 0] * 10,
            "orders_90d": [2, 1, 0, 0, 1, 1] * 10,
            "orders_180d": [3, 2, 1, 0, 1, 1] * 10,
            "orders_365d": [5, 3, 1, 1, 2, 2] * 10,
            "orders_prior_90d": [1, 1, 1, 0, 0, 0] * 10,
            "order_count_trend": [0.33, 0.0, -1.0, 0.0, 1.0, 1.0] * 10,
            "net_revenue_90d": [100.0, 50.0, 0.0, 0.0, 80.0, 40.0] * 10,
            "net_revenue_365d": [250.0, 150.0, 30.0, 25.0, 120.0, 90.0] * 10,
            "avg_order_value": [50.0, 50.0, 30.0, 25.0, 60.0, 45.0] * 10,
            "items_per_order": [10.0, 8.0, 3.0, 2.0, 5.0, 4.0] * 10,
            "distinct_skus": [8, 5, 2, 1, 3, 3] * 10,
            "return_rate": [0.05, 0.0, 0.0, 0.0, 0.1, 0.0] * 10,
            "gap_mean": [20.0, 50.0, np.nan, np.nan, 10.0, 70.0] * 10,  # missing values
            "gap_std": [5.0, 10.0, np.nan, np.nan, 0.0, 15.0] * 10,
            "overdue_ratio": [0.5, 0.9, np.nan, np.nan, 2.0, 1.1] * 10,
            "has_single_order": [0, 0, 1, 1, 0, 0] * 10,
            "country": countries * 10,
            "month_of_cutoff": [6, 6, 6, 6, 6, 6] * 10,
        }
    )
    y_train = np.array([0, 0, 1, 1, 0, 0] * 10, dtype=np.int64)

    pipe.fit(train_data[ALL_FEATURES], y_train)

    # Test inference on test data with an unseen category
    test_data = train_data.iloc[:5].copy()
    test_data.loc[0, "country"] = "Antarctica"  # Unseen category

    probs = pipe.predict_proba(test_data[ALL_FEATURES])
    assert probs.shape == (5, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_lightgbm_and_calibration() -> None:
    """Verify LightGBM fit with early stopping and calibration selection."""
    rng = np.random.default_rng(42)
    has_gap = rng.random(200) > 0.3
    x_tr = pd.DataFrame(
        {
            "recency_days": rng.uniform(1, 365, 200),
            "tenure_days": rng.uniform(10, 700, 200),
            "orders_30d": rng.integers(0, 5, 200),
            "orders_90d": rng.integers(0, 10, 200),
            "orders_180d": rng.integers(0, 15, 200),
            "orders_365d": rng.integers(1, 20, 200),
            "orders_prior_90d": rng.integers(0, 5, 200),
            "order_count_trend": rng.uniform(-1, 1, 200),
            "net_revenue_90d": rng.uniform(0, 500, 200),
            "net_revenue_365d": rng.uniform(50, 2000, 200),
            "avg_order_value": rng.uniform(10, 200, 200),
            "items_per_order": rng.uniform(1, 20, 200),
            "distinct_skus": rng.integers(1, 30, 200),
            "return_rate": rng.uniform(0, 0.2, 200),
            "gap_mean": np.where(has_gap, rng.uniform(5, 100, 200), np.nan),
            "gap_std": np.where(has_gap, rng.uniform(0, 30, 200), np.nan),
            "overdue_ratio": np.where(has_gap, rng.uniform(0.1, 5.0, 200), np.nan),
            "has_single_order": rng.choice([0, 1], 200),
            "country": rng.choice(["United Kingdom", "France", "Germany"], 200),
            "month_of_cutoff": rng.choice([3, 6, 9], 200),
        }
    )
    y_tr = (x_tr["recency_days"] > 180).astype(np.int64)

    x_val = x_tr.iloc[:50].copy()
    y_val = y_tr.iloc[:50].copy()

    # 1. Fit LightGBM with early stopping
    lgb_pipe = fit_lightgbm_with_early_stopping(
        x_tr, y_tr, x_val, y_val, seed=42, n_estimators=50, early_stopping_rounds=10
    )

    # 2. Calibrate model
    calibrator, method, b_iso, b_sig = calibrate_model(lgb_pipe, x_val, y_val)
    assert method in ("isotonic", "sigmoid")
    assert b_iso >= 0.0
    assert b_sig >= 0.0

    # 3. Wrap in CalibratedChurnModel
    model = CalibratedChurnModel(calibrator, algorithm="lightgbm", calibration_method=method)

    probs = model.predict_proba(x_val)
    assert len(probs) == len(x_val)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = model.predict(x_val, threshold=0.5)
    assert set(np.unique(preds)).issubset({0, 1})

    bands = model.predict_bands(x_val)
    assert set(np.unique(bands)).issubset({RISK_BAND_LOW, RISK_BAND_MEDIUM, RISK_BAND_HIGH})


def test_calibrated_churn_model_container_lifecycle(tmp_path: Path) -> None:
    """Verify CalibratedChurnModel save and load serialization with identical predictions."""
    pipe = create_logistic_baseline_pipeline(seed=42)
    x = pd.DataFrame(
        {
            "recency_days": [10.0, 200.0] * 5,
            "tenure_days": [100.0, 250.0] * 5,
            "orders_30d": [1, 0] * 5,
            "orders_90d": [2, 0] * 5,
            "orders_180d": [3, 0] * 5,
            "orders_365d": [4, 1] * 5,
            "orders_prior_90d": [1, 0] * 5,
            "order_count_trend": [0.5, 0.0] * 5,
            "net_revenue_90d": [100.0, 0.0] * 5,
            "net_revenue_365d": [200.0, 30.0] * 5,
            "avg_order_value": [50.0, 30.0] * 5,
            "items_per_order": [5.0, 2.0] * 5,
            "distinct_skus": [4, 1] * 5,
            "return_rate": [0.0, 0.0] * 5,
            "gap_mean": [25.0, np.nan] * 5,
            "gap_std": [5.0, np.nan] * 5,
            "overdue_ratio": [0.4, np.nan] * 5,
            "has_single_order": [0, 1] * 5,
            "country": ["United Kingdom", "France"] * 5,
            "month_of_cutoff": [6, 6] * 5,
        }
    )
    y = np.array([0, 1] * 5, dtype=np.int64)
    pipe.fit(x[ALL_FEATURES], y)

    model = CalibratedChurnModel(
        pipe,
        algorithm="logistic_regression",
        calibration_method=None,
    )

    p1 = model.predict_proba(x)
    save_path = tmp_path / "churn_model.joblib"
    model.save(save_path)

    loaded = CalibratedChurnModel.load(save_path)
    p2 = loaded.predict_proba(x)

    np.testing.assert_allclose(p1, p2)
    assert loaded.algorithm == "logistic_regression"


def test_train_churn_pipeline_mock_db(temp_mlflow_uri: str) -> None:
    """Verify end-to-end churn training, MLflow logging, registration, and DB tracking."""
    engine = sa.create_engine("sqlite:///:memory:")
    engine = engine.execution_options(
        schema_translate_map={"core": None, "ops": None, "ml": None, "app": None}
    )

    for tbl in [
        DimStore.__table__,
        DimProduct.__table__,
        DimCustomer.__table__,
        DimDate.__table__,
        IngestionBatch.__table__,
        FactSalesLine.__table__,
        ModelRun.__table__,
    ]:
        tbl.create(engine, checkfirst=True)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    batch_id = uuid.uuid4()
    session.add(
        IngestionBatch(
            batch_id=batch_id,
            source_name="test",
            source_file="test.xlsx",
            file_sha256="b" * 64,
            status="succeeded",
        )
    )
    session.add(DimStore(store_id=1, store_code="ONLINE", store_name="Online Store"))
    session.add(DimProduct(product_id=1, sku="SKU-1", description="Test SKU 1", category="General"))

    # Seed 30 customers with purchases spanning from 2009-12-01 to 2011-12-09
    # This provides the necessary timeline for generating monthly cutoffs
    for cid in range(1, 31):
        session.add(
            DimCustomer(
                customer_id=cid,
                customer_key=f"CUST-{cid}",
                country="United Kingdom" if cid % 4 != 0 else "Germany",
                first_order_date=datetime.date(2009, 12, 1),
            )
        )

    # Create monthly purchase events
    dates = [
        datetime.date(2009, 12, 15),
        datetime.date(2010, 3, 15),
        datetime.date(2010, 6, 15),
        datetime.date(2010, 9, 15),
        datetime.date(2010, 12, 15),
        datetime.date(2011, 3, 15),
        datetime.date(2011, 6, 15),
        datetime.date(2011, 9, 15),
        datetime.date(2011, 11, 15),
    ]

    lines: list[FactSalesLine] = []
    line_counter = 1
    for cid in range(1, 31):
        for d in dates:
            # Low ID (1-10): continuous buyers across entire timeline (retained)
            # Mid ID (11-20): buy up to 2011-03-15, then churn (active at val/test cutoffs)
            # High ID (21-30): buy up to 2010-09-15, then churn
            if 11 <= cid <= 20 and d > datetime.date(2011, 3, 15):
                continue
            if cid > 20 and d > datetime.date(2010, 9, 15):
                continue
            lines.append(
                FactSalesLine(
                    batch_id=batch_id,
                    source_row_hash=f"hash-{line_counter}",
                    invoice_no=f"INV-{cid}-{line_counter}",
                    invoice_ts=datetime.datetime(
                        d.year, d.month, d.day, 10, 0, tzinfo=datetime.UTC
                    ),
                    date_key=d,
                    store_id=1,
                    product_id=1,
                    customer_id=cid,
                    quantity=cid * 2,
                    unit_price=Decimal("15.00"),
                    is_return=False,
                )
            )
            line_counter += 1

    session.add_all(lines)
    session.commit()

    # Run churn training pipeline
    result = train_churn_pipeline(
        session=session,
        data_as_of=datetime.date(2011, 12, 9),
        churn_window_days=90,
        auto_promote=False,
        seed=42,
        val_size_cutoffs=2,
        test_size_cutoffs=1,
        tracking_uri=temp_mlflow_uri,
    )

    assert result is not None
    assert result.model_run_id is not None
    assert result.selected_algorithm in ("lightgbm", "logistic_regression")
    assert result.candidate_pr_auc >= 0.0
    assert result.baseline_pr_auc >= 0.0
    assert "pr_auc" in result.candidate_metrics
    assert "brier_score" in result.candidate_metrics
    assert len(result.train_cutoffs) >= 1
    assert len(result.val_cutoffs) == 2
    assert len(result.test_cutoffs) == 1

    # Verify model_run recorded in database
    run_record = session.get(ModelRun, result.model_run_id)
    assert run_record is not None
    assert run_record.task == "churn"
    assert run_record.status == "succeeded"

    session.close()

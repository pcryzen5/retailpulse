"""Training pipeline, calibration, and champion gating for customer churn models.

Implements P3-04, TRD §5.3, §6, and PRD §7.4 (FR-CH-03, FR-CH-04) & §8.2:
- Generates monthly cutoffs and enforces purged temporal train/val/test splits.
- Builds behavioral features (<= T) and forward labels (T, T + 90].
- Fits Logistic Regression (standardized, class-weighted, median imputed) baseline.
- Fits LightGBM with early stopping on validation cutoff.
- Evaluates Isotonic vs Platt (Sigmoid) calibration on validation data.
- Evaluates Champion Gate: candidate LightGBM must beat baseline PR-AUC on holdout test cutoff.
- Logs parameters, metrics, calibration curves, and model artifacts to MLflow 'retailpulse/churn'.
- Registers model in MLflow model registry as 'retailpulse-churn'.
- Promotes champion if auto_promote is enabled and gate passes.
"""

from __future__ import annotations

import datetime
import json
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import FactSalesLine
from retailpulse.logging import get_logger
from retailpulse.ml.churn.explain import (
    compute_global_feature_importance,
    plot_global_feature_importance,
)
from retailpulse.ml.churn.labels import (
    build_churn_dataset,
    create_purged_temporal_split,
    generate_monthly_cutoffs,
)
from retailpulse.ml.churn.metrics import (
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
from retailpulse.ml.registry import (
    REGISTERED_MODEL_CHURN,
    promote_champion,
    register_model,
)
from retailpulse.ml.runs import track_model_run

logger = get_logger(__name__)

TASK_CHURN = "churn"


class ChurnTrainResult:
    """Strongly-typed summary of churn training pipeline execution."""

    def __init__(
        self,
        *,
        model_run_id: uuid.UUID,
        data_as_of: datetime.date,
        selected_algorithm: str,
        calibration_method: str | None,
        passed_gate: bool,
        candidate_pr_auc: float,
        baseline_pr_auc: float,
        candidate_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        is_champion: bool,
        version: str | None,
        train_cutoffs: list[datetime.date],
        val_cutoffs: list[datetime.date],
        test_cutoffs: list[datetime.date],
    ) -> None:
        self.model_run_id = model_run_id
        self.data_as_of = data_as_of
        self.selected_algorithm = selected_algorithm
        self.calibration_method = calibration_method
        self.passed_gate = passed_gate
        self.candidate_pr_auc = candidate_pr_auc
        self.baseline_pr_auc = baseline_pr_auc
        self.candidate_metrics = candidate_metrics
        self.baseline_metrics = baseline_metrics
        self.is_champion = is_champion
        self.version = version
        self.train_cutoffs = train_cutoffs
        self.val_cutoffs = val_cutoffs
        self.test_cutoffs = test_cutoffs


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


def train_churn_pipeline(  # noqa: PLR0915
    session: Session,
    *,
    data_as_of: datetime.date | None = None,
    churn_window_days: int | None = None,
    auto_promote: bool | None = None,
    seed: int | None = None,
    val_size_cutoffs: int = 2,
    test_size_cutoffs: int = 1,
    tracking_uri: str | None = None,
) -> ChurnTrainResult:
    """Execute end-to-end churn model training, calibration, and champion gating.

    Args:
        session: Active SQLAlchemy database session.
        data_as_of: Effective cutoff date (defaults to max date in DB).
        churn_window_days: Churn label observation window in days (default 90).
        auto_promote: Whether to promote gate-passing model to active champion.
        seed: Random seed for model training and splitting.
        val_size_cutoffs: Number of monthly cutoffs in the validation set.
        test_size_cutoffs: Number of monthly cutoffs in the holdout test set.
        tracking_uri: Optional override for MLflow tracking URI.

    Returns:
        ChurnTrainResult object with full evaluation metrics and gate verdict.
    """
    settings = get_settings()
    effective_as_of = _resolve_effective_as_of(session, data_as_of)
    effective_window = (
        churn_window_days if churn_window_days is not None else settings.churn_window_days
    )
    effective_seed = seed if seed is not None else settings.seed
    effective_auto_promote = auto_promote if auto_promote is not None else settings.auto_promote

    logger.info(
        "starting churn training pipeline",
        data_as_of=str(effective_as_of),
        churn_window_days=effective_window,
        seed=effective_seed,
    )

    # 1. Determine min date in DB to generate cutoffs
    stmt_min = sa.select(sa.func.min(FactSalesLine.date_key))
    min_date = session.execute(stmt_min).scalar()
    if min_date is None:
        min_date = datetime.date(2009, 12, 1)

    all_cutoffs = generate_monthly_cutoffs(
        min_date,
        effective_as_of,
        min_history_days=180,
        churn_window_days=effective_window,
    )
    if not all_cutoffs:
        raise ValueError(
            f"No eligible cutoffs found between {min_date} and {effective_as_of} "
            f"with churn window {effective_window} days"
        )

    train_cutoffs, val_cutoffs, test_cutoffs = create_purged_temporal_split(
        all_cutoffs,
        churn_window_days=effective_window,
        val_size_cutoffs=val_size_cutoffs,
        test_size_cutoffs=test_size_cutoffs,
    )

    logger.info(
        "purged temporal split generated",
        n_train=len(train_cutoffs),
        n_val=len(val_cutoffs),
        n_test=len(test_cutoffs),
        train_start=str(train_cutoffs[0]),
        train_end=str(train_cutoffs[-1]),
        val_start=str(val_cutoffs[0]),
        val_end=str(val_cutoffs[-1]),
        test_cutoff=str(test_cutoffs[0]),
    )

    # 2. Extract transaction data up to test outcome boundary
    outcome_max_date = test_cutoffs[-1] + datetime.timedelta(days=effective_window)
    stmt_sales = sa.select(
        FactSalesLine.customer_id,
        FactSalesLine.date_key,
        FactSalesLine.quantity,
        FactSalesLine.unit_price,
        FactSalesLine.invoice_no,
        FactSalesLine.product_id,
    ).where(
        FactSalesLine.date_key <= outcome_max_date,
        FactSalesLine.customer_id.isnot(None),
    )
    rows = session.execute(stmt_sales).all()
    if not rows:
        raise ValueError("No sales transaction records found in database to train churn models")

    sales_df = pd.DataFrame(
        rows,
        columns=["customer_id", "date_key", "quantity", "unit_price", "invoice_no", "product_id"],
    )

    # 3. Build churn feature & label datasets for train, val, and test
    train_df = build_churn_dataset(
        sales_df,
        train_cutoffs,
        churn_window_days=effective_window,
    )
    val_df = build_churn_dataset(
        sales_df,
        val_cutoffs,
        churn_window_days=effective_window,
    )
    test_df = build_churn_dataset(
        sales_df,
        test_cutoffs,
        churn_window_days=effective_window,
    )

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Insufficient customer observations across splits: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    # 4. Train Logistic Regression baseline
    logger.info("training logistic regression baseline", n_samples=len(train_df))
    lr_pipeline = create_logistic_baseline_pipeline(seed=effective_seed)
    lr_pipeline.fit(train_df[ALL_FEATURES], train_df["churn_label"])

    lr_test_probs = lr_pipeline.predict_proba(test_df[ALL_FEATURES])[:, 1]
    baseline_metrics = compute_churn_metrics(test_df["churn_label"], lr_test_probs)

    # 5. Train LightGBM with early stopping on validation cutoff
    logger.info("training lightgbm classifier with early stopping", n_samples=len(train_df))
    lgb_pipeline = fit_lightgbm_with_early_stopping(
        train_df,
        train_df["churn_label"],
        val_df,
        val_df["churn_label"],
        seed=effective_seed,
    )

    # 6. Probability Calibration on validation data
    logger.info("evaluating probability calibration on validation data", n_val=len(val_df))
    calibrated_estimator, cal_method, brier_iso, brier_sig = calibrate_model(
        lgb_pipeline,
        val_df,
        val_df["churn_label"],
    )

    candidate_model = CalibratedChurnModel(
        calibrated_estimator,
        algorithm="lightgbm",
        calibration_method=cal_method,
        feature_names=list(ALL_FEATURES),
    )

    candidate_test_probs = candidate_model.predict_proba(test_df)
    candidate_metrics = compute_churn_metrics(test_df["churn_label"], candidate_test_probs)

    # 7. Champion Gate Evaluation (PRD §8.2)
    # Candidate must beat baseline PR-AUC on holdout test cutoff
    passed_pr_auc = candidate_metrics["pr_auc"] > baseline_metrics["pr_auc"]
    passed_brier = candidate_metrics["brier_score"] < candidate_metrics["brier_score_base_rate"]
    passed_gate = bool(passed_pr_auc and passed_brier)

    if passed_gate:
        selected_model = candidate_model
        selected_algorithm = "lightgbm"
        selected_cal_method = cal_method
        logger.info(
            "champion gate PASSED",
            candidate_pr_auc=round(candidate_metrics["pr_auc"], 4),
            baseline_pr_auc=round(baseline_metrics["pr_auc"], 4),
        )
    else:
        # Fallback to baseline
        selected_model = CalibratedChurnModel(
            lr_pipeline,
            algorithm="logistic_regression",
            calibration_method=None,
            feature_names=list(ALL_FEATURES),
        )
        selected_algorithm = "logistic_regression"
        selected_cal_method = None
        logger.warning(
            "champion gate FAILED: candidate did not beat baseline; falling back to baseline",
            candidate_pr_auc=round(candidate_metrics["pr_auc"], 4),
            baseline_pr_auc=round(baseline_metrics["pr_auc"], 4),
        )

    # 8. MLflow Experiment Tracking and Model Registration
    model_run_uuid = uuid.uuid4()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Artifact 1: Model Joblib Serialization
        model_artifact_path = temp_path / "model.joblib"
        selected_model.save(model_artifact_path)

        # Artifact 2: Calibration curve plot
        curve_path = temp_path / "calibration_curve.png"
        plot_calibration_curve(
            test_df["churn_label"],
            {
                "Baseline (Logistic)": lr_test_probs,
                f"Candidate (LightGBM + {cal_method})": candidate_test_probs,
            },
            curve_path,
            title=f"Churn Calibration on Test Cutoff ({test_cutoffs[0]})",
        )

        # Artifact 3: Full metrics JSON
        metrics_summary = {
            "task": TASK_CHURN,
            "data_as_of": str(effective_as_of),
            "test_cutoff": str(test_cutoffs[0]),
            "passed_gate": passed_gate,
            "selected_algorithm": selected_algorithm,
            "calibration_method": selected_cal_method,
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
            "brier_isotonic_val": brier_iso,
            "brier_sigmoid_val": brier_sig,
        }
        metrics_json_path = temp_path / "metrics.json"
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        # Artifact 4: Global feature importance plot and JSON
        importance_df = compute_global_feature_importance(selected_model, test_df)
        feat_imp_plot_path = temp_path / "feature_importance.png"
        plot_global_feature_importance(
            importance_df,
            feat_imp_plot_path,
            title=f"Global Feature Importance on Test Cutoff ({test_cutoffs[0]})",
        )
        feat_imp_json_path = temp_path / "feature_importance.json"
        with feat_imp_json_path.open("w", encoding="utf-8") as f:
            json.dump(importance_df.to_dict(orient="records"), f, indent=2)

        # Parameters to log
        run_params = {
            "task": TASK_CHURN,
            "algorithm": selected_algorithm,
            "seed": effective_seed,
            "churn_window_days": effective_window,
            "train_cutoffs_count": len(train_cutoffs),
            "val_cutoffs_count": len(val_cutoffs),
            "test_cutoffs_count": len(test_cutoffs),
            "test_cutoff": str(test_cutoffs[0]),
            "calibration_method": str(selected_cal_method),
            "passed_gate": str(passed_gate),
        }

        # Metrics to log in MLflow
        run_metrics = {
            "pr_auc": candidate_metrics["pr_auc"],
            "roc_auc": candidate_metrics["roc_auc"],
            "brier_score": candidate_metrics["brier_score"],
            "brier_score_base_rate": candidate_metrics["brier_score_base_rate"],
            "log_loss": candidate_metrics["log_loss"],
            "precision_top_10": candidate_metrics["precision_top_10"],
            "lift_top_10": candidate_metrics["lift_top_10"],
            "recall_top_20": candidate_metrics["recall_top_20"],
            "baseline_pr_auc": baseline_metrics["pr_auc"],
            "baseline_roc_auc": baseline_metrics["roc_auc"],
            "baseline_brier_score": baseline_metrics["brier_score"],
        }

        run_tags = {
            "task": TASK_CHURN,
            "data_as_of": str(effective_as_of),
            "selected_algorithm": selected_algorithm,
            "passed_gate": str(passed_gate),
        }

        with track_model_run(
            session=session,
            task=TASK_CHURN,
            algorithm=selected_algorithm,
            data_as_of=effective_as_of,
            seed=effective_seed,
            tags=run_tags,
            tracking_uri=tracking_uri,
        ) as tracker:
            tracker.log_params(run_params)
            tracker.log_metrics(run_metrics)
            tracker.log_artifact(str(model_artifact_path), artifact_path="model")
            tracker.log_artifact(str(curve_path), artifact_path="artifacts")
            tracker.log_artifact(str(metrics_json_path), artifact_path="artifacts")
            tracker.log_artifact(str(feat_imp_plot_path), artifact_path="artifacts")
            tracker.log_artifact(str(feat_imp_json_path), artifact_path="artifacts")

            model_run_uuid = tracker.run_id
            mlflow_run_id = tracker.mlflow_run_id

        # 9. Register model in MLflow model registry
        version_str: str | None = None
        is_champion = False
        try:
            version_str = register_model(
                mlflow_run_id=mlflow_run_id,
                task=TASK_CHURN,
                model_name=REGISTERED_MODEL_CHURN,
                tracking_uri=tracking_uri,
            )
            if effective_auto_promote and passed_gate and version_str:
                logger.info("auto-promoting passing churn model to champion", version=version_str)
                promote_champion(
                    task=TASK_CHURN,
                    version=version_str,
                    session=session,
                    tracking_uri=tracking_uri,
                )
                is_champion = True
        except Exception as exc:
            logger.warning("model registration/promotion failed", error=str(exc))

    return ChurnTrainResult(
        model_run_id=model_run_uuid,
        data_as_of=effective_as_of,
        selected_algorithm=selected_algorithm,
        calibration_method=selected_cal_method,
        passed_gate=passed_gate,
        candidate_pr_auc=candidate_metrics["pr_auc"],
        baseline_pr_auc=baseline_metrics["pr_auc"],
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        is_champion=is_champion,
        version=version_str,
        train_cutoffs=train_cutoffs,
        val_cutoffs=val_cutoffs,
        test_cutoffs=test_cutoffs,
    )

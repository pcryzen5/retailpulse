"""Churn modeling, cutoff generation, feature extraction, calibration, and training."""

from __future__ import annotations

from retailpulse.ml.churn.labels import (
    build_churn_dataset,
    compute_churn_labels,
    compute_churn_labels_from_db,
    create_purged_temporal_split,
    generate_monthly_cutoffs,
)
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
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    CalibratedChurnModel,
    calibrate_model,
    create_logistic_baseline_pipeline,
    fit_lightgbm_with_early_stopping,
)
from retailpulse.ml.churn.train import (
    TASK_CHURN,
    ChurnTrainResult,
    train_churn_pipeline,
)

__all__ = [
    "ALL_FEATURES",
    "CATEGORICAL_FEATURES",
    "NUMERIC_FEATURES",
    "RISK_BAND_HIGH",
    "RISK_BAND_LOW",
    "RISK_BAND_MEDIUM",
    "TASK_CHURN",
    "CalibratedChurnModel",
    "ChurnTrainResult",
    "assign_risk_band",
    "assign_risk_bands",
    "build_churn_dataset",
    "calibrate_model",
    "compute_churn_labels",
    "compute_churn_labels_from_db",
    "compute_churn_metrics",
    "create_logistic_baseline_pipeline",
    "create_purged_temporal_split",
    "fit_lightgbm_with_early_stopping",
    "generate_monthly_cutoffs",
    "plot_calibration_curve",
    "train_churn_pipeline",
]

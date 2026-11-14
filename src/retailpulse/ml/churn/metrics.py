"""Evaluation metrics, ranking, and calibration curves for customer churn models.

Implements P3-04, TRD §5.3, PRD §7.4 (FR-CH-03, FR-CH-04) & §8.2:
- Binary ranking metrics: PR-AUC, ROC-AUC
- Probability calibration metrics: Brier score, Brier score of base rate, Log Loss
- Top-decile and quintile metrics: Precision@10%, Lift@10%, Recall@20%
- Risk band assignment: low (<0.30), medium (0.30-0.60), high (>=0.60)
- Calibration curve generation and artifact plotting
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sklearn.metrics as sk_metrics
from sklearn.calibration import calibration_curve

from retailpulse.logging import get_logger

matplotlib.use("Agg")  # Non-interactive backend for server environments

logger = get_logger(__name__)

RISK_BAND_LOW = "low"
RISK_BAND_MEDIUM = "medium"
RISK_BAND_HIGH = "high"


def compute_churn_metrics(
    y_true: Sequence[int] | npt.NDArray[Any],
    y_prob: Sequence[float] | npt.NDArray[Any],
    *,
    base_rate: float | None = None,
) -> dict[str, float]:
    """Compute comprehensive churn evaluation metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted churn probabilities in [0, 1].
        base_rate: Optional empirical base rate. If None, derived from y_true.

    Returns:
        Dictionary mapping metric names to computed floating point values:
          - `pr_auc`: Area under the Precision-Recall curve
          - `roc_auc`: Area under the Receiver Operating Characteristic curve
          - `brier_score`: Mean squared error between probabilities and outcomes
          - `brier_score_base_rate`: Brier score of a naive base-rate predictor
          - `log_loss`: Cross-entropy loss
          - `base_rate`: Fraction of positive instances
          - `precision_top_10`: Precision in the top 10% highest scored customers
          - `lift_top_10`: Precision@10% divided by the base rate
          - `recall_top_20`: Recall in the top 20% highest scored customers
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)

    if len(y_t) == 0:
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.5,
            "brier_score": 0.0,
            "brier_score_base_rate": 0.0,
            "log_loss": 0.0,
            "base_rate": 0.0,
            "precision_top_10": 0.0,
            "lift_top_10": 0.0,
            "recall_top_20": 0.0,
        }

    empirical_base_rate = float(np.mean(y_t))
    effective_base_rate = base_rate if base_rate is not None else empirical_base_rate

    # Clip probabilities for stable log-loss
    clipped_probs = np.clip(y_p, 1e-7, 1.0 - 1e-7)

    # PR-AUC
    pr_auc = float(sk_metrics.average_precision_score(y_t, y_p))

    # ROC-AUC (fallback to 0.5 if only one class exists in y_t)
    roc_auc = float(sk_metrics.roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5

    # Calibration & Error metrics
    brier = float(sk_metrics.brier_score_loss(y_t, y_p))
    naive_preds = np.full_like(y_p, effective_base_rate)
    brier_base = float(sk_metrics.brier_score_loss(y_t, naive_preds))
    loss = float(sk_metrics.log_loss(y_t, clipped_probs, labels=[0, 1]))

    # Top Decile (top 10%)
    n = len(y_t)
    sort_idx = np.argsort(y_p)[::-1]
    k_10 = max(1, int(np.ceil(0.10 * n)))
    top_10_labels = y_t[sort_idx[:k_10]]
    precision_top_10 = float(np.mean(top_10_labels))
    lift_top_10 = float(precision_top_10 / effective_base_rate) if effective_base_rate > 0 else 0.0

    # Top Quintile (top 20%)
    k_20 = max(1, int(np.ceil(0.20 * n)))
    top_20_labels = y_t[sort_idx[:k_20]]
    total_positives = int(np.sum(y_t))
    recall_top_20 = float(np.sum(top_20_labels) / total_positives) if total_positives > 0 else 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "brier_score_base_rate": brier_base,
        "log_loss": loss,
        "base_rate": effective_base_rate,
        "precision_top_10": precision_top_10,
        "lift_top_10": lift_top_10,
        "recall_top_20": recall_top_20,
    }


def assign_risk_band(
    prob: float,
    *,
    medium_threshold: float = 0.30,
    high_threshold: float = 0.60,
) -> str:
    """Classify a single churn probability into canonical risk band ('low', 'medium', 'high')."""
    if prob >= high_threshold:
        return RISK_BAND_HIGH
    if prob >= medium_threshold:
        return RISK_BAND_MEDIUM
    return RISK_BAND_LOW


def assign_risk_bands(
    probs: Sequence[float] | npt.NDArray[Any],
    *,
    medium_threshold: float = 0.30,
    high_threshold: float = 0.60,
) -> npt.NDArray[Any]:
    """Vectorized classification of churn probabilities into canonical risk bands."""
    p_arr = np.asarray(probs, dtype=np.float64)
    conditions = [
        p_arr >= high_threshold,
        p_arr >= medium_threshold,
    ]
    choices = [RISK_BAND_HIGH, RISK_BAND_MEDIUM]
    return np.select(conditions, choices, default=RISK_BAND_LOW)


def plot_calibration_curve(
    y_true: Sequence[int] | npt.NDArray[Any],
    model_probabilities: dict[str, Sequence[float] | npt.NDArray[Any]],
    output_path: Path | str,
    *,
    n_bins: int = 10,
    title: str = "Reliability Diagram (Calibration Curve)",
) -> None:
    """Generate and save a reliability diagram comparing model calibration.

    Args:
        y_true: True binary target labels.
        model_probabilities: Mapping from model name to predicted probabilities.
        output_path: File system destination path for the saved plot image.
        n_bins: Number of probability bins (default 10).
        title: Plot title.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonal perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for model_name, probs in model_probabilities.items():
        p_arr = np.asarray(probs, dtype=np.float64)
        prob_true, prob_pred = calibration_curve(y_t, p_arr, n_bins=n_bins, strategy="uniform")
        brier = sk_metrics.brier_score_loss(y_t, p_arr)
        ax.plot(
            prob_pred,
            prob_true,
            marker="s",
            label=f"{model_name} (Brier: {brier:.4f})",
        )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Churn)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

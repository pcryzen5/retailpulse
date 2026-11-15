"""Explainability and feature attribution for customer churn predictions.

Implements P3-05, TRD §5.3, PRD §7.4 (FR-CH-05), API.md §3.5:
- LightGBM: SHAP TreeExplainer for local and global driver attribution.
- Logistic Regression: standardized feature value * coefficient linear attribution.
- Top-3 reason formatting: {feature, value, impact, direction} per customer.
- Global feature importance computation and horizontal bar chart artifact generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from retailpulse.logging import get_logger
from retailpulse.ml.churn.models import (
    ALL_FEATURES,
    CalibratedChurnModel,
)

matplotlib.use("Agg")  # Non-interactive backend for server environments

logger = get_logger(__name__)


class ChurnDriverReason(TypedDict):
    """Structured representation of a single churn driver reason."""

    feature: str
    value: Any
    impact: float
    direction: str


def extract_base_pipeline(model: Any) -> Pipeline:
    """Extract underlying Pipeline from CalibratedChurnModel or CalibratedClassifierCV."""
    curr = model
    while hasattr(curr, "estimator") and not isinstance(curr, Pipeline):
        curr = curr.estimator
    if isinstance(curr, Pipeline):
        return curr
    raise TypeError(f"Could not extract Pipeline from {type(model).__name__}")


def compute_shap_matrix_lgbm(
    pipeline: Pipeline,
    x: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Compute sample-level SHAP values for a LightGBM pipeline.

    Returns:
        (shap_matrix, feature_names) where shap_matrix has shape (N, len(ALL_FEATURES)).
    """
    if x.empty:
        return np.empty((0, len(ALL_FEATURES)), dtype=np.float64), list(ALL_FEATURES)

    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    x_prep = preprocessor.transform(x[ALL_FEATURES])
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer(x_prep)

    values = np.asarray(shap_vals.values, dtype=np.float64)
    # If binary classification returns 3D (samples, features, classes), slice positive class (1)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]

    return values, list(ALL_FEATURES)


def compute_linear_matrix_logistic(
    pipeline: Pipeline,
    x: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Compute sample-level standardized linear contributions aggregated to original features.

    Returns:
        (contrib_matrix, feature_names) where contrib_matrix has shape (N, len(ALL_FEATURES)).
    """
    if x.empty:
        return np.empty((0, len(ALL_FEATURES)), dtype=np.float64), list(ALL_FEATURES)

    preprocessor = pipeline.named_steps["prep"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    x_prep = preprocessor.transform(x[ALL_FEATURES])
    coefs = clf.coef_[0]

    raw_contribs = x_prep * coefs

    # Map transformed column names back to original feature names
    feature_names_out = preprocessor.get_feature_names_out()
    col_to_orig: list[str] = []
    for name in feature_names_out:
        if name.startswith("num__"):
            clean = name[len("num__") :]
            if clean.startswith("missingindicator_"):
                clean = clean[len("missingindicator_") :]
            col_to_orig.append(clean)
        elif name.startswith("cat__"):
            col_to_orig.append("country")
        else:
            col_to_orig.append(name)

    n_samples = len(x)
    n_features = len(ALL_FEATURES)
    agg_matrix = np.zeros((n_samples, n_features), dtype=np.float64)
    feat_to_idx = {f: i for i, f in enumerate(ALL_FEATURES)}

    for col_idx, orig_feat in enumerate(col_to_orig):
        if orig_feat in feat_to_idx:
            agg_matrix[:, feat_to_idx[orig_feat]] += raw_contribs[:, col_idx]

    return agg_matrix, list(ALL_FEATURES)


def explain_churn_predictions(
    model: CalibratedChurnModel | Pipeline | Any,
    x: pd.DataFrame,
    *,
    top_n: int = 3,
) -> list[list[dict[str, Any]]]:
    """Generate top-N plain-language driver reasons for customer churn predictions.

    Args:
        model: Fitted churn model container or Pipeline.
        x: Customer feature DataFrame containing ALL_FEATURES.
        top_n: Number of top drivers to extract per customer (default 3).

    Returns:
        List of driver lists per customer, each containing dictionaries:
          - feature: string name of feature
          - value: customer's raw feature value (JSON serializable)
          - impact: signed contribution to churn risk (rounded to 4 decimal places)
          - direction: 'increases_risk' (if impact >= 0) or 'decreases_risk' (if impact < 0)
    """
    if x.empty:
        return []

    pipeline = extract_base_pipeline(model)
    clf = pipeline.named_steps["clf"]

    if isinstance(clf, LogisticRegression):
        contrib_matrix, feature_names = compute_linear_matrix_logistic(pipeline, x)
    else:
        contrib_matrix, feature_names = compute_shap_matrix_lgbm(pipeline, x)

    results: list[list[dict[str, Any]]] = []
    x_features = x[feature_names].reset_index(drop=True)

    for i in range(len(x)):
        row_impacts = contrib_matrix[i]
        scored_features: list[tuple[float, str]] = []
        for j, feat in enumerate(feature_names):
            imp = float(row_impacts[j])
            scored_features.append((imp, feat))

        # Sort: primary key -abs(impact), secondary key feature name for deterministic stability
        scored_features.sort(key=lambda item: (-abs(item[0]), item[1]))

        top_drivers: list[dict[str, Any]] = []
        for imp, feat in scored_features[:top_n]:
            raw_val = x_features.loc[i, feat]
            if pd.isna(raw_val):
                clean_val: Any = None
            elif isinstance(raw_val, (np.floating, float)):
                clean_val = round(float(raw_val), 4)
            elif isinstance(raw_val, (np.integer, int)):
                clean_val = int(raw_val)
            else:
                clean_val = str(raw_val)

            direction = "increases_risk" if imp >= 0.0 else "decreases_risk"
            top_drivers.append(
                {
                    "feature": feat,
                    "value": clean_val,
                    "impact": round(imp, 4),
                    "direction": direction,
                }
            )
        results.append(top_drivers)

    return results


def compute_global_feature_importance(
    model: CalibratedChurnModel | Pipeline | Any,
    x: pd.DataFrame,
) -> pd.DataFrame:
    """Compute global feature importance as mean absolute impact across the dataset.

    Returns:
        pd.DataFrame with columns ['feature', 'mean_abs_impact'], sorted descending.
    """
    if x.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_impact"])

    pipeline = extract_base_pipeline(model)
    clf = pipeline.named_steps["clf"]

    if isinstance(clf, LogisticRegression):
        contrib_matrix, feature_names = compute_linear_matrix_logistic(pipeline, x)
    else:
        contrib_matrix, feature_names = compute_shap_matrix_lgbm(pipeline, x)

    mean_abs = np.mean(np.abs(contrib_matrix), axis=0)
    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_impact": [float(m) for m in mean_abs],
        }
    )
    df_imp = df_imp.sort_values(by=["mean_abs_impact", "feature"], ascending=[False, True])
    return df_imp.reset_index(drop=True)


def plot_global_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path | str,
    *,
    top_n: int = 15,
    title: str = "Global Feature Importance (Customer Churn)",
) -> None:
    """Render and save a horizontal bar chart of global feature importance."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if importance_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No feature importance data", ha="center", va="center")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return

    top_df = importance_df.head(top_n).iloc[::-1]  # Invert so highest is at the top

    fig, ax = plt.subplots(figsize=(8, max(4, int(0.35 * len(top_df)))))
    y_pos = np.arange(len(top_df))
    ax.barh(y_pos, top_df["mean_abs_impact"], color="#2b5c8f", alpha=0.85, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"])
    ax.set_xlabel("Mean Absolute Attribution (SHAP / Standardized Impact)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

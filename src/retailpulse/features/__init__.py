"""Feature engineering for forecasting, RFM, and churn."""

from __future__ import annotations

from retailpulse.features.churn_features import (
    CHURN_FEATURE_COLUMNS,
    build_churn_features,
    build_churn_features_from_db,
)
from retailpulse.features.rfm import (
    CANONICAL_SEGMENTS,
    SEGMENT_AT_RISK,
    SEGMENT_BIG_SPENDERS,
    SEGMENT_CHAMPIONS,
    SEGMENT_HIBERNATING,
    SEGMENT_LOYAL,
    SEGMENT_NEW_PROMISING,
    CustomerRFM,
    assign_rule_based_segment,
    compute_rfm_from_db,
    compute_rfm_from_df,
    rfm_df_to_records,
    transform_rfm_for_clustering,
)

__all__ = [
    "CANONICAL_SEGMENTS",
    "CHURN_FEATURE_COLUMNS",
    "SEGMENT_AT_RISK",
    "SEGMENT_BIG_SPENDERS",
    "SEGMENT_CHAMPIONS",
    "SEGMENT_HIBERNATING",
    "SEGMENT_LOYAL",
    "SEGMENT_NEW_PROMISING",
    "CustomerRFM",
    "assign_rule_based_segment",
    "build_churn_features",
    "build_churn_features_from_db",
    "compute_rfm_from_db",
    "compute_rfm_from_df",
    "rfm_df_to_records",
    "transform_rfm_for_clustering",
]

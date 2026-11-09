"""Customer RFM segmentation, optimal k selection, and scoring."""

from __future__ import annotations

from retailpulse.ml.segmentation.kmeans import (
    RFMSegmentationModel,
    assign_cluster_labels,
    build_segment_profiles,
    evaluate_clustering_stability,
    select_optimal_k,
)
from retailpulse.ml.segmentation.score import (
    SegmentationScoreResult,
    score_segmentation_pipeline,
)
from retailpulse.ml.segmentation.train import (
    SegmentationTrainResult,
    train_segmentation_pipeline,
)

__all__ = [
    "RFMSegmentationModel",
    "SegmentationScoreResult",
    "SegmentationTrainResult",
    "assign_cluster_labels",
    "build_segment_profiles",
    "evaluate_clustering_stability",
    "score_segmentation_pipeline",
    "select_optimal_k",
    "train_segmentation_pipeline",
]

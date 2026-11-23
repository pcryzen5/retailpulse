"""Database entity models across all schemas."""

from __future__ import annotations

from retailpulse.db.models.app import AppUser
from retailpulse.db.models.core import (
    DimCustomer,
    DimDate,
    DimProduct,
    DimStore,
    FactDailySales,
    FactSalesLine,
    InventoryParam,
    InventorySnapshot,
)
from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    CustomerSegment,
    Forecast,
    ForecastMetric,
    InventoryRecommendation,
    ModelRun,
    PolicySimulation,
    SegmentProfile,
)
from retailpulse.db.models.ops import (
    Alert,
    AuditLog,
    DataQualityResult,
    DriftReport,
    IngestionBatch,
    ModelPerformance,
    PipelineRun,
)

__all__ = [
    "ActiveOutput",
    "Alert",
    "AppUser",
    "AuditLog",
    "ChurnScore",
    "CustomerSegment",
    "DataQualityResult",
    "DimCustomer",
    "DimDate",
    "DimProduct",
    "DimStore",
    "DriftReport",
    "FactDailySales",
    "FactSalesLine",
    "Forecast",
    "ForecastMetric",
    "IngestionBatch",
    "InventoryParam",
    "InventoryRecommendation",
    "InventorySnapshot",
    "ModelPerformance",
    "ModelRun",
    "PipelineRun",
    "PolicySimulation",
    "SegmentProfile",
]

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
)
from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    CustomerSegment,
    Forecast,
    ForecastMetric,
    InventoryRecommendation,
    ModelRun,
    SegmentProfile,
)
from retailpulse.db.models.ops import DataQualityResult, IngestionBatch, PipelineRun

__all__ = [
    "ActiveOutput",
    "AppUser",
    "ChurnScore",
    "CustomerSegment",
    "DataQualityResult",
    "DimCustomer",
    "DimDate",
    "DimProduct",
    "DimStore",
    "FactDailySales",
    "FactSalesLine",
    "Forecast",
    "ForecastMetric",
    "IngestionBatch",
    "InventoryParam",
    "InventoryRecommendation",
    "ModelRun",
    "PipelineRun",
    "SegmentProfile",
]

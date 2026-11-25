"""RetailPulse Prefect Deployment Worker.

Configures and serves deployments for all 6 pipelines with their respective
schedules per ARCHITECTURE §6:
  - etl: daily at 01:00 UTC ('0 1 * * *')
  - score_all: chained (no periodic schedule)
  - train_forecast: Sunday at 02:00 UTC ('0 2 * * 0')
  - train_segmentation: Sunday at 03:00 UTC ('0 3 * * 0')
  - train_churn: Sunday at 03:30 UTC ('30 3 * * 0')
  - drift_check: daily at 04:00 UTC ('0 4 * * *')
"""

from __future__ import annotations

from typing import Any

from prefect import serve
from prefect.schedules import Cron

from retailpulse.logging import get_logger
from retailpulse.pipelines.flows import (
    drift_check_flow,
    etl_flow,
    score_all_flow,
    train_churn_flow,
    train_forecast_flow,
    train_segmentation_flow,
)

logger = get_logger(__name__)


def get_worker_deployments() -> list[Any]:
    """Build RunnerDeployments for all 6 RetailPulse pipeline workflows."""
    etl_dep = etl_flow.to_deployment(
        name="etl-deployment",
        schedule=Cron("0 1 * * *", timezone="UTC"),
        description="Daily sales data ETL ingestion and aggregation (01:00 UTC)",
    )

    score_all_dep = score_all_flow.to_deployment(
        name="score_all-deployment",
        description="End-to-end model scoring and inventory recommendations (chained/manual)",
    )

    train_forecast_dep = train_forecast_flow.to_deployment(
        name="train_forecast-deployment",
        schedule=Cron("0 2 * * 0", timezone="UTC"),
        description="Weekly demand forecast retraining (Sun 02:00 UTC)",
    )

    train_segmentation_dep = train_segmentation_flow.to_deployment(
        name="train_segmentation-deployment",
        schedule=Cron("0 3 * * 0", timezone="UTC"),
        description="Weekly customer segmentation clustering retraining (Sun 03:00 UTC)",
    )

    train_churn_dep = train_churn_flow.to_deployment(
        name="train_churn-deployment",
        schedule=Cron("30 3 * * 0", timezone="UTC"),
        description="Weekly churn prediction model retraining (Sun 03:30 UTC)",
    )

    drift_check_dep = drift_check_flow.to_deployment(
        name="drift_check-deployment",
        schedule=Cron("0 4 * * *", timezone="UTC"),
        description="Daily feature and prediction drift monitoring (04:00 UTC)",
    )

    return [
        etl_dep,
        score_all_dep,
        train_forecast_dep,
        train_segmentation_dep,
        train_churn_dep,
        drift_check_dep,
    ]


def serve_worker() -> None:
    """Serve all pipeline deployments in a long-running Prefect worker process."""
    logger.info("Initializing RetailPulse Prefect deployment worker...")
    deployments = get_worker_deployments()
    logger.info(
        "Serving 6 pipeline deployments with configured schedules",
        deployments=[getattr(d, "name", str(d)) for d in deployments],
    )
    serve(*deployments)


if __name__ == "__main__":
    serve_worker()

"""Pipeline trigger abstraction and execution adapters.

Provides a unified trigger interface for dispatching data ingestion, scoring,
training, and drift monitoring pipelines. Supports Noop (for testing/worker mode)
and Inline (for local synchronous execution) implementations.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from fastapi import Request

from retailpulse.logging import get_logger

logger = get_logger(__name__)


class PipelineTrigger(ABC):
    """Abstract interface for pipeline execution triggers."""

    @abstractmethod
    async def trigger(
        self,
        flow_name: str,
        run_id: uuid.UUID,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Trigger an asynchronous or background pipeline run.

        Args:
            flow_name: Name of the workflow ('etl', 'score_all', 'train_*', etc.).
            run_id: UUID of the ops.pipeline_run record.
            params: Optional execution parameters for the workflow.

        Returns:
            True if the trigger was accepted/dispatched successfully, False otherwise.
        """
        ...


class NoopPipelineTrigger(PipelineTrigger):
    """No-op trigger implementation for testing and worker-polled mode."""

    async def trigger(
        self,
        flow_name: str,
        run_id: uuid.UUID,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Record the trigger call and return True without executing external flows."""
        del params
        logger.info(
            "NoopPipelineTrigger: trigger recorded",
            flow_name=flow_name,
            run_id=str(run_id),
        )
        return True


class InlinePipelineTrigger(PipelineTrigger):
    """Inline trigger implementation executing synchronous callbacks or recording runs."""

    def __init__(self, callback: Any = None) -> None:
        self.callback = callback
        self.dispatched_runs: list[tuple[str, uuid.UUID, dict[str, Any]]] = []

    async def trigger(
        self,
        flow_name: str,
        run_id: uuid.UUID,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Record and optionally invoke an inline execution callback."""
        logger.info(
            "InlinePipelineTrigger: dispatching inline pipeline run",
            flow_name=flow_name,
            run_id=str(run_id),
        )
        p = params or {}
        self.dispatched_runs.append((flow_name, run_id, p))
        if self.callback is not None and callable(self.callback):
            res = self.callback(flow_name, run_id, p)
            if isinstance(res, bool):
                return res
        return True


def get_pipeline_trigger(request: Request) -> PipelineTrigger:
    """FastAPI dependency to retrieve the configured PipelineTrigger."""
    trigger = getattr(request.app.state, "pipeline_trigger", None)
    if trigger is None or not isinstance(trigger, PipelineTrigger):
        return NoopPipelineTrigger()
    return trigger

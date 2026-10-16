"""Structured logging configuration for RetailPulse."""

from __future__ import annotations

import contextvars
import logging
import sys
from typing import cast

import structlog
from structlog.types import EventDict, WrappedLogger

# Context variable for request ID tracking across async and sync call chains
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def bind_request_id(request_id: str | None) -> contextvars.Token[str | None]:
    """Bind a request ID to the current context.

    Args:
        request_id: The request ID to bind.

    Returns:
        The context token for resetting the contextvar.
    """
    return request_id_ctx.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    """Reset the request ID contextvar.

    Args:
        token: Token returned by bind_request_id.
    """
    request_id_ctx.reset(token)


def add_request_id(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject request_id from contextvar into the log event if present.

    Args:
        logger: Underlying standard library logger.
        method_name: Logger method name.
        event_dict: Current logging event dictionary.

    Returns:
        Modified event dictionary with request_id attached if present.
    """
    req_id = request_id_ctx.get()
    if req_id is not None:
        event_dict["request_id"] = req_id
    return event_dict


def configure_logging(level: str = "INFO", *, json_format: bool = True) -> None:
    """Configure structlog and standard library logging.

    Args:
        level: Minimum log level string (e.g. 'DEBUG', 'INFO', 'WARNING').
        json_format: Whether to output JSON formatted lines (default: True).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.types.Processor
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    for handler in logging.root.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Name of the logger, typically __name__.

    Returns:
        A structlog bound logger configured with contextual processors.
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))

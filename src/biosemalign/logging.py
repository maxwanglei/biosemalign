"""Structured logging setup.

Every log line carries the identifiers needed to trace a mapping — ``sau_id``,
``run_id``, provider, cache outcome — because "why did this one string map
differently last Tuesday" is the question this project will be asked most.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]

_configured = False


def configure_logging(*, level: str | None = None, fmt: str | None = None) -> None:
    """Configure structlog once per process.

    Safe to call repeatedly; only the first call takes effect unless arguments
    are supplied explicitly.
    """
    global _configured
    if _configured and level is None and fmt is None:
        return

    from biosemalign.settings import get_settings

    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    resolved_fmt = fmt or settings.log_format

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if resolved_fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(resolved_level, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    configure_logging()
    return structlog.get_logger(name)

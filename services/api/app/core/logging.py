"""Structured JSON logging with request correlation.

Human-readable logs are useless on a free-tier host where you cannot attach a
debugger and the container may be recycled at any moment. Every line emitted
here is a single JSON object carrying ``request_id``, so a user-reported failure
can be traced end to end.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import structlog


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure ``structlog`` and route stdlib logs through it.

    Args:
        level: Root log level name, e.g. ``"INFO"``.
        json_output: Emit JSON (production) or colored key/value (local dev).

    Why both: JSON is machine-parseable for hosted log drains, while a human
    reading a terminal wants aligned, colored output. Same event names either way.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Bridge uvicorn/httpx logs into the same pipeline so nothing escapes as
    # unstructured text that a log drain would index badly.
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for a module.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A structlog logger that accepts keyword context (``logger.info("x", n=1)``).
    """
    return structlog.get_logger(name)


class Timer:
    """Context manager that records wall-clock duration in milliseconds.

    Usage:
        with Timer() as t:
            do_work()
        logger.info("done", duration_ms=t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)

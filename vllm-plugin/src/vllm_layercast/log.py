"""Structured logging for the layercast vLLM plugin.

Uses structlog with JSON rendering to stderr for guaranteed visibility
in container logs and easy parsing by benchmark/orchestration scripts.
"""

from __future__ import annotations

import sys

import structlog


def setup() -> None:
    """Configure structured JSON logging to stderr.

    Safe to call multiple times (idempotent). Should be called once
    during plugin registration before any loggers are used.
    """
    if structlog.is_configured():
        return
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def get_logger(name: str = "vllm_layercast") -> structlog.stdlib.BoundLogger:
    """Get a bound structlog logger.

    Always returns a usable logger even if setup() hasn't been called yet
    (structlog falls back to sensible defaults).
    """
    return structlog.get_logger(name)

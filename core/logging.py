"""
Structured logging module for SpeechShield.

Provides:
  • Structured JSON/text logging via structlog
  • Correlation IDs for request tracing
  • Lazy initialization compatible with config system
  • Per-module loggers with context

Usage:
    from core.logging import configure_logging, get_logger

    configure_logging(level="DEBUG", format="json")
    logger = get_logger("MyModule")
    logger.info("event", user="alice", action="login")
"""

import logging
import sys
import os
from typing import Literal, Optional
import structlog

_logger_configured = False
_log_format: Literal["text", "json"] = "text"


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    format: Literal["text", "json"] = "text",
    log_dir: str = "logs/",
    enable_file_logging: bool = True,
) -> None:
    """
    Configure structured logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format (text for development, json for production)
        log_dir: Directory for log files
        enable_file_logging: Write logs to files

    Note:
        Call this once at application startup, before creating loggers.
    """
    global _logger_configured, _log_format

    if _logger_configured:
        return  # Already configured

    _log_format = format

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Configure Python's standard logging
    stdlib_handlers = [logging.StreamHandler(sys.stdout)]

    if enable_file_logging:
        stdlib_handlers.append(
            logging.FileHandler(
                os.path.join(log_dir, "speechshield.log"),
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=getattr(logging, level),
        handlers=stdlib_handlers,
        format="%(message)s",
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if format == "json":
        processors.extend([
            structlog.processors.JSONRenderer(),
        ])
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(),
        ])

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _logger_configured = True


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Bound structlog logger with module name context
    """
    return structlog.get_logger(name)


# Context management for correlation IDs
_correlation_id: Optional[str] = None


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID for the current context.

    All subsequent logs will include this ID for request tracing.

    Args:
        correlation_id: Unique ID for this request/trace
    """
    global _correlation_id
    _correlation_id = correlation_id
    logger = structlog.get_logger()
    logger.bind(correlation_id=correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID."""
    global _correlation_id
    _correlation_id = None


class CorrelationIDFilter(logging.Filter):
    """
    Logging filter that adds correlation ID to all log records.

    Use with Python's standard logging to include correlation IDs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to the log record."""
        record.correlation_id = get_correlation_id() or "no-trace"
        return True

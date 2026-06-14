"""
Structured logging configuration.
Uses Python's standard logging module with JSON-friendly formatting for production.
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from app.core.config import get_settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured log lines."""

    def format(self, record: logging.LogRecord) -> str:
        record.service = get_settings().app_name
        return super().format(record)


def setup_logging() -> None:
    """Configure application-wide logging."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = StructuredFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(service)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # 1. Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 2. File handler (rotated daily, keeping 7 days of history)
    # Determine project root dynamically relative to this file
    project_root = Path(__file__).resolve().parent.parent.parent
    logs_dir = project_root / settings.log_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    file_path = logs_dir / settings.log_file
    file_handler = TimedRotatingFileHandler(
        filename=file_path,
        when="midnight",
        interval=1,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance."""
    return logging.getLogger(name)

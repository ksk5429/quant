"""Structured logging setup using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure loguru with file rotation and structured format."""
    if config is None:
        config = LoggingConfig()

    # Remove default handler
    logger.remove()

    # Console handler with rich formatting
    logger.add(
        sys.stderr,
        level=config.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # File handler with rotation
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=config.level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation=config.rotation,
        retention=config.retention,
        encoding="utf-8",
    )


def get_logger(name: str) -> logger:
    """Get a named logger instance."""
    return logger.bind(name=name)

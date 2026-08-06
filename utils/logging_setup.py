"""Logging configuration: console + rotating file logger."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path,
    log_file: str = "run.log",
    level: str = "INFO",
    to_console: bool = True,
) -> logging.Logger:
    """Configure the root logger writing to console and a file.

    Args:
        log_dir: Directory where the log file is stored (created if needed).
        log_file: Name of the log file.
        level: Logging level string, e.g. "INFO" or "DEBUG".
        to_console: Whether to also emit logs to stdout.

    Returns:
        The configured root logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)

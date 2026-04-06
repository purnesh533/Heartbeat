"""Logging: file heartbeat.log + optional console; business events use pipe format."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_BUSINESS_FORMAT = "%(asctime)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: Path,
    *,
    console: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure root logger: file handler for production, optional stderr.
    Business events should log via log_event() for consistent 'event | details' body.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_BUSINESS_FORMAT, datefmt=_DATEFMT))
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)s | " + _BUSINESS_FORMAT, datefmt=_DATEFMT))
        root.addHandler(ch)

    return logging.getLogger("heartbeat")


def get_logger(name: str = "heartbeat") -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, details: str = "") -> None:
    """Log a business event: timestamp | event | details (details in message after first pipe)."""

    if details:
        logger.info("%s | %s", event, details)
    else:
        logger.info("%s |", event)

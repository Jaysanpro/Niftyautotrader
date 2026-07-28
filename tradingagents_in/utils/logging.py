"""Project logging. Rich handler when available, plain stderr otherwise."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | int = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich is a hard dep but be safe
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt, handlers=[handler], force=True)
    # These libraries are chatty at INFO and drown out the agent trace.
    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging(os.getenv("ITA_LOG_LEVEL", "INFO"))
    return logging.getLogger(name)

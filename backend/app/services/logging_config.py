"""Centralized logging configuration for the backend.

Provides:
  - setup_logging(): install root logger handlers (console + rotating file).
  - attach_run_log(): per-run file handler context manager.
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

_DEFAULT_LOG_FILE = Path("logs/backend.log")
_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(
    *,
    log_file: Path | str = _DEFAULT_LOG_FILE,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Install console + rotating-file handlers on the root logger.

    Idempotent: repeat calls are no-ops once configured.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    _CONFIGURED = True


# ---------------------------------------------------------------------------
# Per-run log file
# ---------------------------------------------------------------------------

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "png_shader_current_run_id", default=None
)


class _RunIdFilter(logging.Filter):
    """Accept a record only when the current contextvar matches ``run_id``."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        return _current_run_id.get() == self._run_id


@contextmanager
def attach_run_log(*, run_id: str, log_file: Path | str) -> Iterator[logging.Handler]:
    """Attach a per-run FileHandler that captures only the current run's logs.

    Uses a ContextVar to scope which records the handler accepts, so
    concurrent runs writing to different files do not cross-contaminate.
    Must be entered inside the worker thread that produces the run's logs,
    because ContextVar values are per-context and ``threading.Thread``
    does not inherit the parent context by default.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler.addFilter(_RunIdFilter(run_id))

    token = _current_run_id.set(run_id)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
        _current_run_id.reset(token)

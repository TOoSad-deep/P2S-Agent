"""Centralized logging configuration for the backend.

Provides:
  - setup_logging(): install console, text file, and JSONL file handlers.
  - logging_context(): bind request_id/run_id to all log records.
  - log_event(): emit structured events that are easy to grep or parse.
  - attach_run_log(): per-run file handler context manager.
"""
from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

_DEFAULT_LOG_FILE = Path("logs/backend.log")
_DEFAULT_JSON_LOG_FILE = Path("logs/backend.jsonl")
_FORMAT = "%(asctime)s %(levelname)s [%(name)s] [request=%(request_id)s run=%(run_id)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False

_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "png_shader_current_run_id", default=None
)
_current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "png_shader_current_request_id", default=None
)


class _ContextFilter(logging.Filter):
    """Attach request/run context so every formatter can rely on it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = _current_request_id.get() or "-"
        if not getattr(record, "run_id", None):
            record.run_id = _current_run_id.get() or "-"
        if not hasattr(record, "event"):
            record.event = None
        if not hasattr(record, "event_data"):
            record.event_data = None
        return True


_RESERVED_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {
    "asctime",
    "message",
    "request_id",
    "run_id",
    "event",
    "event_data",
}


class _JsonLogFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "run_id": getattr(record, "run_id", "-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, Mapping):
            payload["data"] = dict(event_data)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    *,
    log_file: Path | str = _DEFAULT_LOG_FILE,
    json_log_file: Path | str = _DEFAULT_JSON_LOG_FILE,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Install console + rotating text/JSONL handlers on the root logger.

    Idempotent: repeat calls are no-ops once configured.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_file)
    json_log_path = Path(json_log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    context_filter = _ContextFilter()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(context_filter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)

    json_handler = RotatingFileHandler(
        json_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    json_handler.setLevel(level)
    json_handler.setFormatter(_JsonLogFormatter())
    json_handler.addFilter(context_filter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
    root.addHandler(json_handler)

    # Keep chatty dependencies from burying pipeline events.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


# ---------------------------------------------------------------------------
# Context + structured events
# ---------------------------------------------------------------------------


def current_run_id() -> str | None:
    return _current_run_id.get()


def current_request_id() -> str | None:
    return _current_request_id.get()


@contextmanager
def logging_context(
    *,
    request_id: str | None = None,
    run_id: str | None = None,
) -> Iterator[None]:
    """Bind request/run identifiers for logs emitted in this context."""
    request_token = _current_request_id.set(request_id) if request_id is not None else None
    run_token = _current_run_id.set(run_id) if run_id is not None else None
    try:
        yield
    finally:
        if run_token is not None:
            _current_run_id.reset(run_token)
        if request_token is not None:
            _current_request_id.reset(request_token)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    message: str | None = None,
    **fields: Any,
) -> None:
    """Emit a structured event to both text and JSONL handlers."""
    logger.log(
        level,
        message or event,
        extra={"event": event, "event_data": fields},
    )


# ---------------------------------------------------------------------------
# Per-run log file
# ---------------------------------------------------------------------------


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

    context_filter = _ContextFilter()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler.addFilter(context_filter)
    handler.addFilter(_RunIdFilter(run_id))

    json_handler = logging.FileHandler(log_path.with_suffix(".jsonl"), encoding="utf-8")
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(_JsonLogFormatter())
    json_handler.addFilter(context_filter)
    json_handler.addFilter(_RunIdFilter(run_id))

    token = _current_run_id.set(run_id)
    root = logging.getLogger()
    root.addHandler(handler)
    root.addHandler(json_handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.removeHandler(json_handler)
        try:
            handler.close()
        except Exception:
            pass
        try:
            json_handler.close()
        except Exception:
            pass
        _current_run_id.reset(token)

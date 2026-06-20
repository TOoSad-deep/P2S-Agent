"""Tests for backend logging context and structured events."""

from __future__ import annotations

import io
import json
import logging

from p2s_agent.core.logging_config import (
    _ContextFilter,
    _JsonLogFormatter,
    log_event,
    logging_context,
)


def test_log_event_includes_request_and_run_context():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonLogFormatter())
    handler.addFilter(_ContextFilter())

    logger = logging.getLogger("tests.logging_config.context")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    with logging_context(request_id="req_test", run_id="run_test"):
        log_event(logger, "unit_event", answer=42)

    payload = json.loads(stream.getvalue())
    assert payload["request_id"] == "req_test"
    assert payload["run_id"] == "run_test"
    assert payload["event"] == "unit_event"
    assert payload["data"] == {"answer": 42}

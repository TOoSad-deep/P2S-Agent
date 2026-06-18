"""Unit tests for the in-memory `_run_store` LRU/eviction policy and the
`_run_models` lifecycle alignment (Bugs 1 & 2)."""

from __future__ import annotations

import pytest

import app.routers.png_shader as ps
from app.routers.png_shader import (
    _MAX_STORE_SIZE,
    _get_run_model,
    _store_run,
    _store_run_model,
    _touch_run,
)


@pytest.fixture(autouse=True)
def _clean_stores():
    """Reset both in-memory stores around each test."""
    ps._run_store.clear()
    with ps._run_models_lock:
        ps._run_models.clear()
    yield
    ps._run_store.clear()
    with ps._run_models_lock:
        ps._run_models.clear()


# ---------------------------------------------------------------------------
# Bug 1: LRU + never-evict-live policy
# ---------------------------------------------------------------------------


def test_running_run_is_never_evicted_when_capacity_exceeded():
    """An early-inserted 'running' run survives even when >cap terminal runs
    are stored afterward."""
    _store_run("live", {"run_id": "live", "status": "running"})
    # Insert cap-many terminal runs, forcing eviction pressure.
    for i in range(_MAX_STORE_SIZE + 10):
        rid = f"term_{i}"
        _store_run(rid, {"run_id": rid, "status": "completed"})

    assert "live" in ps._run_store, "running run must never be evicted"
    assert len(ps._run_store) <= _MAX_STORE_SIZE


def test_lru_recently_accessed_terminal_survives_over_older_untouched():
    """A recently-accessed terminal entry survives over an older untouched
    terminal entry (true LRU, not FIFO-by-insertion)."""
    # Fill to exactly cap with terminal entries.
    for i in range(_MAX_STORE_SIZE):
        rid = f"t_{i}"
        _store_run(rid, {"run_id": rid, "status": "completed"})

    # Touch the oldest-inserted entry so it becomes most-recently-used.
    assert _touch_run("t_0") is not None

    # Insert one more terminal entry -> something must be evicted.
    _store_run("newcomer", {"run_id": "newcomer", "status": "completed"})

    assert len(ps._run_store) <= _MAX_STORE_SIZE
    # t_0 was touched, so it must survive; the next-oldest (t_1) is evicted.
    assert "t_0" in ps._run_store, "recently-accessed terminal must survive (LRU)"
    assert "t_1" not in ps._run_store, "oldest untouched terminal must be evicted"
    assert "newcomer" in ps._run_store


def test_direct_terminal_write_respects_cap():
    """Every write path (including direct terminal/queued writes) goes through
    the capped setter, so the cap always holds."""
    for i in range(_MAX_STORE_SIZE + 25):
        rid = f"w_{i}"
        _store_run(rid, {"run_id": rid, "status": "completed"})

    assert len(ps._run_store) <= _MAX_STORE_SIZE


def test_update_in_place_moves_to_end_so_terminal_run_survives():
    """Re-storing (updating) a terminal run re-orders it to most-recent so it
    is not the next eviction victim."""
    for i in range(_MAX_STORE_SIZE):
        rid = f"u_{i}"
        _store_run(rid, {"run_id": rid, "status": "completed"})

    # Update the oldest entry in place -> should move to end.
    _store_run("u_0", {"run_id": "u_0", "status": "completed", "extra": 1})

    _store_run("after", {"run_id": "after", "status": "completed"})

    assert "u_0" in ps._run_store, "updated terminal must move-to-end and survive"
    assert "u_1" not in ps._run_store


# ---------------------------------------------------------------------------
# Bug 2: _run_models lifecycle aligned with _run_store liveness
# ---------------------------------------------------------------------------


def test_run_model_access_keeps_it_alive():
    """Reading a model via _get_run_model marks it most-recently-used so it is
    not the next eviction victim."""
    sentinel = object()
    _store_run_model("m_keep", sentinel)
    # Fill models near cap with throwaway entries.
    for i in range(_MAX_STORE_SIZE - 1):
        _store_run_model(f"m_{i}", object())

    # Access the original so it moves to end.
    assert _get_run_model("m_keep") is sentinel

    # Push one more model in -> the oldest *untouched* model is evicted, not m_keep.
    _store_run_model("m_new", object())

    assert _get_run_model("m_keep") is sentinel, "accessed model must survive eviction"


def test_run_model_for_live_run_is_not_evicted():
    """A model whose run is still live (running/queued in _run_store) must not
    be evicted even under capacity pressure."""
    sentinel = object()
    _store_run("live_run", {"run_id": "live_run", "status": "running"})
    _store_run_model("live_run", sentinel)

    for i in range(_MAX_STORE_SIZE + 5):
        _store_run_model(f"mm_{i}", object())

    assert _get_run_model("live_run") is sentinel, "live run's model must not be evicted"

"""In-memory run / model stores for the PNG-to-Shader pipeline.

Framework-free home for the two bounded, LRU-aware in-memory maps and the
best-effort run-index glue. Extracted from ``app.routers.png_shader`` so the
web layer holds no domain state of its own.

Secret-isolation design (preserved from the router):
  * ``_run_store`` holds client-facing run state and is the ONLY map read by
    ``_snapshot_run`` — so the selected model and any secret api_key are never
    serialized into the /status response.
  * ``_run_models`` holds per-run ``ModelConfig`` objects (which may carry an
    api_key) and is kept SEPARATE from ``_run_store``.

Lock-order invariant (do not reorder): both maps live in this one module so the
cross-map eviction path is correct. ``_evict_one_model_locked`` runs while
holding ``_run_models_lock`` and calls ``_run_is_live``, which acquires
``_run_store_lock``. The order is therefore ``_run_models_lock`` → then
``_run_store_lock``; no code path acquires them in the opposite order.

Dependency direction: store → orchestration (run_index) is allowed;
orchestration never imports store, so there is no cycle. This module imports no
``app.*`` / ``fastapi`` symbols (enforced by test_agent_web_boundary).
"""

from __future__ import annotations

import copy
import logging
import threading
from collections import OrderedDict
from typing import Optional

from p2s_agent.config import ModelConfig
from p2s_agent.orchestration import run_index
from p2s_agent.orchestration.run_index import (
    RunLineageRecord,
    append_run_created,
    append_run_updated,
)

logger = logging.getLogger(__name__)

# Insertion-ordered so that LRU eviction can prefer the least-recently-USED
# entry. Every read/update calls ``move_to_end`` to keep ordering = recency.
_run_store: "OrderedDict[str, dict]" = OrderedDict()
_run_store_lock = threading.Lock()
_MAX_STORE_SIZE = 100

# Run states that are still "live" — these must never be evicted from either
# the run store or the per-run model store, even under capacity pressure.
_LIVE_STATUSES: frozenset[str] = frozenset({"running", "queued"})

# Tests override this to isolate from the real backend/test_results/run_index.jsonl.
_RUN_INDEX_PATH: Optional[str] = None

# Resolved per-run model configs (may hold api_keys). Kept SEPARATE from
# _run_store so the selected model — and any secret key — is never returned by
# the client-facing /status endpoint. Insertion-ordered for LRU eviction whose
# lifecycle is aligned with ``_run_store`` liveness (Bug 2).
_run_models: "OrderedDict[str, ModelConfig]" = OrderedDict()
_run_models_lock = threading.Lock()

# Metadata fields that are allowed to be patched and mirrored into the in-memory store.
_METADATA_MIRROR_KEYS: frozenset[str] = frozenset({"title", "favorite", "tags"})

# Terminal run statuses — the natural low-frequency checkpoint for opportunistic
# run-index compaction.
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


# ---------------------------------------------------------------------------
# Best-effort run-index helpers — I/O errors only log; they never 500 a request
# or kill a worker thread.
# ---------------------------------------------------------------------------

def _index_created(record: RunLineageRecord) -> None:
    try:
        append_run_created(record, path=_RUN_INDEX_PATH)
    except Exception:
        logger.warning("run index append_created failed", exc_info=True)


def _index_updated(run_id: str, fields: dict) -> None:
    try:
        append_run_updated(run_id, fields, path=_RUN_INDEX_PATH)
    except Exception:
        logger.warning("run index append_updated failed", exc_info=True)
        return
    # Opportunistic, threshold-gated compaction at a run's terminal transition —
    # the natural low-frequency checkpoint to keep run_index.jsonl from growing
    # unbounded. Best-effort: maybe_compact never raises, but guard anyway so a
    # compaction hiccup can never affect the worker. Imported as a module
    # attribute (run_index.maybe_compact_run_index) so tests can patch
    # ``p2s_agent.orchestration.run_index.maybe_compact_run_index``.
    if fields.get("status") in _TERMINAL_RUN_STATUSES:
        try:
            run_index.maybe_compact_run_index(path=_RUN_INDEX_PATH)
        except Exception:
            logger.warning("run index opportunistic compaction failed", exc_info=True)


def _run_is_live(run_id: str) -> bool:
    """True when ``run_id`` exists in the run store with a live status.

    Used to anchor the model store's eviction policy to run liveness so a
    still-running child never falls back to the .env default model (Bug 2).
    """
    with _run_store_lock:
        stored = _run_store.get(run_id)
        return bool(stored) and stored.get("status") in _LIVE_STATUSES


def _store_run_model(run_id: str, model_config: ModelConfig) -> None:
    """Store a per-run model with LRU eviction that never drops a live run's model."""
    with _run_models_lock:
        if run_id in _run_models:
            _run_models[run_id] = model_config
            _run_models.move_to_end(run_id)
            return
        if len(_run_models) >= _MAX_STORE_SIZE:
            _evict_one_model_locked()
        _run_models[run_id] = model_config


def _evict_one_model_locked() -> None:
    """Evict the least-recently-used model whose run is no longer live.

    Caller must hold ``_run_models_lock``. Falls back to the LRU entry only if
    every model belongs to a live run (avoids unbounded growth in the
    pathological all-live case)."""
    for candidate in list(_run_models.keys()):
        if not _run_is_live(candidate):
            del _run_models[candidate]
            return
    # Every model belongs to a live run — drop the LRU one to respect the cap.
    _run_models.popitem(last=False)


def _get_run_model(run_id: str) -> Optional[ModelConfig]:
    with _run_models_lock:
        model = _run_models.get(run_id)
        if model is not None:
            _run_models.move_to_end(run_id)
        return model


def _store_run(run_id: str, payload: dict) -> None:
    """Store a PNG shader run state with bounded, LRU-aware retention.

    Single capped setter for ALL write paths (including direct terminal/queued
    writes). On update, moves the entry to most-recently-used. On insert past
    the cap, evicts the least-recently-used TERMINAL entry and NEVER evicts a
    run whose status is still live (running/queued) — so a still-running run can
    never be evicted out from under /status, /stop, or child-lookup (Bug 1).
    """
    with _run_store_lock:
        _store_run_locked(run_id, payload)


def _store_run_locked(run_id: str, payload: dict) -> None:
    """``_store_run`` body; caller must hold ``_run_store_lock``."""
    if run_id in _run_store:
        _run_store[run_id] = payload
        _run_store.move_to_end(run_id)
        return
    if len(_run_store) >= _MAX_STORE_SIZE:
        _evict_one_run_locked()
    _run_store[run_id] = payload


def _drop_run(run_id: str) -> None:
    """Remove a run entry (and its model config) from the in-memory stores.

    Used to roll back a run that was registered but never admitted — e.g. when
    the global worker pool is saturated and the submission is rejected with 429.
    Best-effort: missing keys are ignored.
    """
    with _run_store_lock:
        _run_store.pop(run_id, None)
    with _run_models_lock:
        _run_models.pop(run_id, None)


def _evict_one_run_locked() -> None:
    """Evict the least-recently-used TERMINAL run.

    Caller must hold ``_run_store_lock``. Scans from oldest→newest and removes
    the first entry whose status is not live. If every entry is live, no
    eviction happens (we tolerate transient overflow rather than drop a live
    run)."""
    for candidate, stored in list(_run_store.items()):
        if (stored or {}).get("status") not in _LIVE_STATUSES:
            del _run_store[candidate]
            return
    # All live — do not evict; the cap is best-effort under all-live pressure.


def _touch_run(run_id: str) -> Optional[dict]:
    """Read a run entry and mark it most-recently-used (LRU read path).

    Returns the live store dict (NOT a copy) or ``None``. Callers that hold the
    lock and only need a snapshot should use ``_snapshot_run`` instead.
    """
    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is not None:
            _run_store.move_to_end(run_id)
        return stored


def _snapshot_run(run_id: str) -> Optional[dict]:
    """Return a deep copy of a run entry (or None), marking it MRU.

    Safe to return to clients / iterate without risking concurrent worker
    mutation (Bug 4)."""
    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is None:
            return None
        _run_store.move_to_end(run_id)
        return copy.deepcopy(stored)


def _publish_partial_to_store(run_id: str, partial: dict) -> None:
    """Merge a partial pipeline result into a still-running run's store entry.

    Best-effort: silently no-ops when the run was evicted or already reached a
    terminal state, so a late partial can't resurrect a finished run. Only data
    fields are merged; control fields (strategy / stop_requested /
    strategy_revision / status / ...) are preserved.
    """
    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is None or stored.get("status") != "running":
            return
        stored.update(partial)

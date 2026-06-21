"""JSONL-backed run-lineage index for the PNG-to-Shader pipeline (M3-1).

Records the parent/child relationships between pipeline runs so branch
lineage survives the in-memory ``_run_store`` being evicted (capped at 100
entries) or the service restarting.

Design
------
* Append-only JSONL: each call to ``append_run_created`` / ``append_run_updated``
  writes exactly one JSON line terminated by ``\\n``.
* ``load_run_index`` folds the whole file; later events override earlier ones.
* The module depends only on the standard library and ``app.pipeline.artifacts``
  — no FastAPI, no langgraph.

Thread-safety
-------------
A module-level ``threading.Lock`` serialises all append operations so
multiple pipeline workers writing concurrently do not interleave partial
lines.

Default path
------------
``backend/test_results/run_index.jsonl``

Every public function accepts ``path=None``; tests pass a ``tmp_path``
override so the real index is never touched during testing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------
# Stamped onto every written JSONL line under the "v" key. The fold tolerates
# legacy lines without "v" (treated as v1) and best-effort-folds lines whose
# version is in the future (with a WARNING).
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_PATH = DEFAULT_RESULTS_ROOT / "run_index.jsonl"

# ---------------------------------------------------------------------------
# Module-level lock for concurrent append safety
# ---------------------------------------------------------------------------

_APPEND_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# load_run_index cache (perf): fold result keyed by resolved path, validated
# against the file's (st_mtime_ns, st_size). A separate lock guards the cache
# because workers append concurrently and may trigger re-folds.
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
# path-str -> ((st_mtime_ns, st_size), folded dict[str, RunLineageRecord])
_INDEX_CACHE: dict[str, tuple[tuple[int, int], dict[str, "RunLineageRecord"]]] = {}

# Statuses that, once reached, must not be regressed by a later stale/out-of-order
# ``updated`` event flipping the run back to a non-terminal status.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# ---------------------------------------------------------------------------
# Custom error
# ---------------------------------------------------------------------------


class RunIndexError(ValueError):
    """Raised when a run_id is unknown or an index operation is invalid."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunLineageRecord:
    run_id: str
    root_run_id: str
    parent_run_id: str | None
    source_checkpoint_id: str | None
    source_checkpoint_label: str | None
    mode: str | None
    feedback: str | None
    title: str | None
    status: str
    run_dir: str | None
    created_at: float
    completed_at: float | None = None
    final_score: float | None = None
    favorite: bool = False
    tags: list[str] = field(default_factory=list)
    variant_group_id: str | None = None
    variant_index: int | None = None
    variant_label: str | None = None
    draw_session_id: str | None = None
    draw_card_index: int | None = None
    replacement_of_run_id: str | None = None
    fusion_id: str | None = None
    base_run_id: str | None = None
    source_run_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: Path | str | None) -> Path:
    return Path(path) if path is not None else _DEFAULT_INDEX_PATH


def _append_line(data: dict[str, Any], path: Path) -> None:
    """Append one JSONL line to *path*, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _record_to_dict(record: RunLineageRecord) -> dict[str, Any]:
    return asdict(record)


def _dict_to_record(d: dict[str, Any]) -> RunLineageRecord:
    """Build a RunLineageRecord from a dict, tolerating extra / missing keys."""
    run_id = d.get("run_id", "")
    # M-1: distinguish None (missing) from an explicit 0; the `or 0.0` idiom
    # would silently coerce an explicit 0 to 0.0 too, but also masks None.
    _created = d.get("created_at")
    created_at = float(_created) if _created is not None else 0.0
    return RunLineageRecord(
        run_id=run_id,
        root_run_id=d.get("root_run_id", run_id),
        parent_run_id=d.get("parent_run_id"),
        source_checkpoint_id=d.get("source_checkpoint_id"),
        source_checkpoint_label=d.get("source_checkpoint_label"),
        mode=d.get("mode"),
        feedback=d.get("feedback"),
        title=d.get("title"),
        status=d.get("status", "unknown"),
        run_dir=d.get("run_dir"),
        created_at=created_at,
        completed_at=d.get("completed_at"),
        final_score=d.get("final_score"),
        favorite=bool(d.get("favorite", False)),
        tags=list(d.get("tags") or []),
        variant_group_id=d.get("variant_group_id"),
        variant_index=int(d["variant_index"]) if d.get("variant_index") is not None else None,
        variant_label=d.get("variant_label"),
        draw_session_id=d.get("draw_session_id"),
        draw_card_index=int(d["draw_card_index"]) if d.get("draw_card_index") is not None else None,
        replacement_of_run_id=d.get("replacement_of_run_id"),
        fusion_id=d.get("fusion_id"),
        base_run_id=d.get("base_run_id"),
        source_run_ids=list(d.get("source_run_ids") or []),
    )


def _merge_fields(record: RunLineageRecord, fields: dict[str, Any]) -> RunLineageRecord:
    """Return a new RunLineageRecord with *fields* overlaid on *record*."""
    base = _record_to_dict(record)
    base.update(fields)
    return _dict_to_record(base)


# ---------------------------------------------------------------------------
# Shadow SQLite mirror (additive; JSONL remains authoritative).
# Best-effort: a DB failure NEVER breaks the JSONL write path or reads. Reads
# still fold JSONL; this only populates the `runs` table so the eventual
# read-cutover has live data. Known shadow-mode divergences (resolved at
# read-cutover): update-before-create is dropped (no row to update),
# terminal-status stickiness is not enforced DB-side, and compact/prune are
# not mirrored.
# ---------------------------------------------------------------------------
_SHADOW_DB_ENABLED = True
_SHADOW_INIT_LOCK = threading.Lock()
_SHADOW_INITED: set[str] = set()


def _shadow_engine(path: Path | str | None):
    """Engine for the shadow DB sitting BESIDE the JSONL (never the same file).

    path=None → canonical app DB (backend/data/p2s.db);
    path=<jsonl file> → <its parent dir>/p2s.db (per-test isolation).
    Lazily ensures the schema exists (idempotent).
    """
    from p2s_agent.core.db.engine import get_engine, init_db
    eng = get_engine(None) if path is None else get_engine(Path(path).parent)
    key = str(eng.url)
    if key not in _SHADOW_INITED:
        with _SHADOW_INIT_LOCK:
            if key not in _SHADOW_INITED:
                init_db(eng)  # create_all: only adds missing tables
                _SHADOW_INITED.add(key)
    return eng


def _shadow_upsert_created(record: RunLineageRecord, path: Path | str | None) -> None:
    if not _SHADOW_DB_ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import runs as runs_repo
        runs_repo.upsert_run(_shadow_engine(path), asdict(record))
    except Exception:
        logger.debug("run_index shadow upsert (created) failed", exc_info=True)


def _shadow_update(run_id: str, fields: dict[str, Any], path: Path | str | None) -> None:
    if not _SHADOW_DB_ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import runs as runs_repo
        runs_repo.update_run(_shadow_engine(path), run_id, fields)
    except Exception:
        logger.debug("run_index shadow update failed", exc_info=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_run_created(record: RunLineageRecord, *, path: Path | str | None = None) -> None:
    """Append a ``created`` event for *record* to the JSONL index.

    Each call writes exactly one line:
    ``{"event": "created", <all record fields>}``.
    """
    resolved = _resolve_path(path)
    data: dict[str, Any] = {"event": "created", "v": SCHEMA_VERSION}
    data.update(_record_to_dict(record))
    _append_line(data, resolved)
    _shadow_upsert_created(record, path)


def append_run_updated(
    run_id: str,
    fields: dict[str, Any],
    *,
    path: Path | str | None = None,
) -> None:
    """Append an ``updated`` event for *run_id* to the JSONL index.

    Each call writes exactly one line:
    ``{"event": "updated", "run_id": run_id, **fields}``.
    """
    resolved = _resolve_path(path)
    data: dict[str, Any] = {"event": "updated", "v": SCHEMA_VERSION, "run_id": run_id}
    data.update(fields)
    _append_line(data, resolved)
    _shadow_update(run_id, fields, path)


def _fold_index_file(resolved: Path) -> dict[str, RunLineageRecord]:
    """Read *resolved* and fold all events into a dict of RunLineageRecord.

    See :func:`load_run_index` for the fold rules. This helper does the actual
    line-by-line read; callers handle existence checks and caching.
    """
    records: dict[str, RunLineageRecord] = {}

    with resolved.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                # Observability: surface corruption instead of dropping it
                # silently. Truncate the snippet so logs stay readable.
                snippet = line[:120] + ("…" if len(line) > 120 else "")
                logger.warning(
                    "run_index: skipping malformed JSONL line %d in %s: %s (snippet=%r)",
                    lineno,
                    resolved,
                    exc,
                    snippet,
                )
                continue

            if not isinstance(data, dict):
                logger.warning(
                    "run_index: skipping non-object JSONL line %d in %s (type=%s)",
                    lineno,
                    resolved,
                    type(data).__name__,
                )
                continue

            # Schema version: legacy lines (no "v") are treated as v1. A line
            # from a *future* schema is folded best-effort but flagged so the
            # operator knows the reader may not understand every field.
            ver = data.get("v", 1)
            if isinstance(ver, int) and ver > SCHEMA_VERSION:
                logger.warning(
                    "run_index: line %d has future schema version %s (current %s); "
                    "folding best-effort",
                    lineno,
                    ver,
                    SCHEMA_VERSION,
                )

            event = data.get("event")
            run_id = data.get("run_id")
            if not run_id:
                continue

            if event == "created":
                # Exclude the 'event' / 'v' keys before building the record.
                payload = {k: v for k, v in data.items() if k not in ("event", "v")}
                records[run_id] = _dict_to_record(payload)

            elif event == "updated":
                fields = {k: v for k, v in data.items() if k not in ("event", "v", "run_id")}
                if run_id in records:
                    existing = records[run_id]
                    # Terminal-status stickiness: once a run reaches a terminal
                    # status, a later out-of-order / stale ``updated`` must not
                    # regress it to a non-terminal status. Non-status fields in
                    # that event are still allowed to merge.
                    if (
                        existing.status in _TERMINAL_STATUSES
                        and "status" in fields
                        and fields["status"] not in _TERMINAL_STATUSES
                    ):
                        fields = {k: v for k, v in fields.items() if k != "status"}
                    records[run_id] = _merge_fields(existing, fields)
                else:
                    # Best-effort: create from available fields.
                    synthetic: dict[str, Any] = {"run_id": run_id}
                    synthetic.update(fields)
                    records[run_id] = _dict_to_record(synthetic)

    return records


def load_run_index(*, path: Path | str | None = None) -> dict[str, RunLineageRecord]:
    """Read the JSONL file and fold all events into a dict of RunLineageRecord.

    Rules:
    * ``created`` events initialise a record.
    * ``updated`` events overlay their fields onto the existing record.
    * Later events always override earlier ones, EXCEPT a terminal status
      (completed/failed/cancelled) is never regressed to a non-terminal
      status by a later stale/out-of-order ``updated``.
    * Blank lines are skipped; malformed / non-object lines are logged at
      WARNING (with line number + snippet) and skipped.
    * An ``updated`` that arrives before any ``created`` for that run_id
      creates a best-effort record (root_run_id defaults to run_id,
      status defaults to "unknown").

    Performance: the folded result is cached per resolved path and validated
    against the file's ``(st_mtime_ns, st_size)``. Repeated calls with no file
    change reuse the cache instead of re-reading O(total_runs) lines; an append
    (which changes mtime/size) invalidates the entry and triggers a re-fold.

    Returns an empty dict if the file does not exist. The returned dict is a
    fresh shallow copy on every call, so callers may mutate it without
    corrupting the cache.
    """
    resolved = _resolve_path(path)

    try:
        stat = resolved.stat()
    except (FileNotFoundError, NotADirectoryError):
        return {}
    except OSError as exc:
        # A uvicorn --reload worker on macOS can lose TCC access to the data
        # root; stat() then raises PermissionError. Degrade to an empty index
        # rather than propagating into an HTTP 500.
        logger.warning(
            "run_index: cannot stat %s (%s); returning empty index", resolved, exc
        )
        return {}

    key = str(resolved)
    signature = (stat.st_mtime_ns, stat.st_size)

    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached[0] == signature:
            # Hand back a shallow copy so caller mutations can't corrupt the
            # cached mapping. RunLineageRecord values are treated as immutable.
            return dict(cached[1])

    try:
        folded = _fold_index_file(resolved)
    except OSError as exc:
        # PermissionError on the underlying open() (lost TCC access) degrades
        # to an empty index; do not poison the cache with the error path.
        logger.warning(
            "run_index: cannot read %s (%s); returning empty index", resolved, exc
        )
        return {}

    with _CACHE_LOCK:
        _INDEX_CACHE[key] = (signature, folded)

    return dict(folded)


# ---------------------------------------------------------------------------
# Compaction / rotation
# ---------------------------------------------------------------------------


def compact_run_index(*, path: Path | str | None = None) -> int:
    """Atomically rewrite the index as one 'created' line per run (folded state).

    Collapses N append events into 1 line per run. Fold-equivalent: load_run_index
    returns the same records before and after. Writes a single '.bak' of the prior
    file, replaces atomically, and invalidates the cache for this path.
    Returns the number of records written. No-op (returns 0) if the file is absent.
    """
    resolved = _resolve_path(path)

    if not resolved.exists():
        return 0

    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    bak = resolved.with_suffix(resolved.suffix + ".bak")

    # Hold the append lock for the whole read+write so no append interleaves
    # with the snapshot, which would otherwise be silently dropped.
    with _APPEND_LOCK:
        # Fold the file directly (not via the cache) to get authoritative state.
        records = _fold_index_file(resolved)

        with tmp.open("w", encoding="utf-8") as fh:
            for rec in records.values():
                line_data: dict[str, Any] = {"event": "created", "v": SCHEMA_VERSION}
                line_data.update(_record_to_dict(rec))
                fh.write(json.dumps(line_data, ensure_ascii=False) + "\n")

        # Back up the original by COPY (so `resolved` is never missing), then
        # atomically swap the compacted temp file into place. A concurrent
        # load_run_index (which does not hold _APPEND_LOCK) therefore always
        # stats a complete file — the pre- or post-compaction one, never a gap.
        shutil.copy2(resolved, bak)
        os.replace(tmp, resolved)

    # Invalidate the cache so the next load re-folds the compacted file.
    with _CACHE_LOCK:
        _INDEX_CACHE.pop(str(resolved), None)

    return len(records)


def prune_run_index(run_ids: set[str], *, path: Path | str | None = None) -> int:
    """Atomically rewrite the index, dropping every record in *run_ids*.

    Used by retention cleanup: when a run directory is deleted from disk, its
    lineage record must also leave the index so the branch tree never points at
    a missing directory. Like :func:`compact_run_index` this folds the file,
    writes a single ``.bak`` of the prior file, replaces atomically, and
    invalidates the cache for this path.

    Returns the number of records actually removed (run_ids absent from the
    index contribute 0). No-op (returns 0) if the file is absent or *run_ids*
    is empty.
    """
    resolved = _resolve_path(path)

    if not run_ids or not resolved.exists():
        return 0

    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    bak = resolved.with_suffix(resolved.suffix + ".bak")

    # Hold the append lock across read+write so no append interleaves with the
    # snapshot (mirrors compact_run_index).
    with _APPEND_LOCK:
        records = _fold_index_file(resolved)
        removed = sum(1 for rid in run_ids if rid in records)
        if removed == 0:
            return 0

        with tmp.open("w", encoding="utf-8") as fh:
            for rid, rec in records.items():
                if rid in run_ids:
                    continue
                line_data: dict[str, Any] = {"event": "created", "v": SCHEMA_VERSION}
                line_data.update(_record_to_dict(rec))
                fh.write(json.dumps(line_data, ensure_ascii=False) + "\n")

        shutil.copy2(resolved, bak)
        os.replace(tmp, resolved)

    with _CACHE_LOCK:
        _INDEX_CACHE.pop(str(resolved), None)

    return removed


def maybe_compact_run_index(
    *, path: Path | str | None = None, min_lines: int = 2000, bloat_ratio: float = 3.0
) -> bool:
    """Compact only when the file is bloated: >= min_lines AND lines >= bloat_ratio*runs.

    Best-effort: returns True if it compacted, False otherwise. Never raises
    (wrap the work so a failure here never breaks the caller).
    """
    resolved = _resolve_path(path)

    if not resolved.exists():
        return False

    try:
        records = load_run_index(path=resolved)
        num_runs = len(records)

        with resolved.open("r", encoding="utf-8") as fh:
            num_lines = sum(1 for _ in fh)

        if (
            num_lines >= min_lines
            and num_runs > 0
            and num_lines >= bloat_ratio * num_runs
        ):
            compact_run_index(path=resolved)
            return True
        return False
    except Exception as exc:  # noqa: BLE001 — best-effort, must never break caller
        logger.warning(
            "run_index: maybe_compact_run_index failed for %s: %s", resolved, exc
        )
        return False


def build_branch_tree(
    records: dict[str, RunLineageRecord],
    run_id: str,
) -> dict[str, Any]:
    """Build a nested JSON-serialisable tree rooted at *run_id*'s root.

    Resolves *run_id* to its ``root_run_id``, then collects all records
    sharing that root and assembles them into a nested tree.

    Children are sorted by ``created_at`` ascending.

    A node whose ``parent_run_id`` is ``None`` (or whose parent is not in
    the same root-set) attaches directly under the root.

    Raises:
        RunIndexError: if *run_id* is not found in *records*, or if the
            root record is missing from the index.
    """
    if run_id not in records:
        raise RunIndexError(f"run_id not found in index: {run_id!r}")

    root_run_id = records[run_id].root_run_id

    # Collect all records that share this root.
    family = {rid: rec for rid, rec in records.items() if rec.root_run_id == root_run_id}

    if root_run_id not in family:
        # The root itself may have been evicted; build a stub so the tree
        # has a stable anchor.
        raise RunIndexError(f"root run_id not found in index: {root_run_id!r}")

    def _node(rec: RunLineageRecord, children: list[dict[str, Any]]) -> dict[str, Any]:
        # `run_dir` and `tags` are intentionally omitted — they are not part of
        # the branch-tree API contract surfaced to callers.
        return {
            "run_id": rec.run_id,
            "root_run_id": rec.root_run_id,
            "parent_run_id": rec.parent_run_id,
            "source_checkpoint_id": rec.source_checkpoint_id,
            "source_checkpoint_label": rec.source_checkpoint_label,
            "title": rec.title,
            "mode": rec.mode,
            "feedback": rec.feedback,
            "status": rec.status,
            "final_score": rec.final_score,
            "created_at": rec.created_at,
            "completed_at": rec.completed_at,
            "favorite": rec.favorite,
            "variant_group_id": rec.variant_group_id,
            "variant_index": rec.variant_index,
            "variant_label": rec.variant_label,
            "draw_session_id": rec.draw_session_id,
            "draw_card_index": rec.draw_card_index,
            "replacement_of_run_id": rec.replacement_of_run_id,
            "fusion_id": rec.fusion_id,
            "base_run_id": rec.base_run_id,
            "source_run_ids": rec.source_run_ids,
            "children": children,
        }

    # Build adjacency: parent_run_id -> list of child RunLineageRecord.
    children_of: dict[str | None, list[RunLineageRecord]] = {}
    for rec in family.values():
        # If a record's parent is outside the family (or None), it attaches
        # under the root.
        parent = rec.parent_run_id
        if parent not in family or parent == rec.run_id:
            parent = None if rec.run_id == root_run_id else root_run_id

        # The root itself has parent=None.
        if rec.run_id == root_run_id:
            parent = None

        children_of.setdefault(parent, []).append(rec)

    # Sort every child list by created_at ascending.
    for lst in children_of.values():
        lst.sort(key=lambda r: r.created_at)

    def _build(rid: str) -> dict[str, Any]:
        rec = family[rid]
        child_nodes = [
            _build(child.run_id)
            for child in children_of.get(rid, [])
            if child.run_id != rid  # guard against self-reference
        ]
        return _node(rec, child_nodes)

    return _build(root_run_id)


# ---------------------------------------------------------------------------
# Allowed metadata-only patch keys
# ---------------------------------------------------------------------------

_METADATA_ALLOWED = frozenset({"title", "favorite", "tags"})

def update_run_metadata(
    run_id: str,
    patch: dict[str, Any],
    *,
    path: Path | str | None = None,
) -> RunLineageRecord:
    """Patch metadata-only fields (``title``, ``favorite``, ``tags``) for a run.

    Any key not in {title, favorite, tags} is rejected, including lineage
    fields such as parent_run_id, root_run_id, status, run_dir.

    Raises:
        ValueError: if *patch* contains a disallowed key.
        RunIndexError: if *run_id* is not found in the index.

    Returns the merged RunLineageRecord after writing the update to the file.
    Empty patches write nothing to disk and simply return the current record.
    """
    resolved = _resolve_path(path)

    if not patch:
        # I-2: empty patch — validate existence but write nothing to disk.
        index = load_run_index(path=resolved)
        if run_id not in index:
            raise RunIndexError(f"update_run_metadata: unknown run_id: {run_id!r}")
        return index[run_id]

    disallowed = set(patch.keys()) - _METADATA_ALLOWED
    if disallowed:
        raise ValueError(
            f"update_run_metadata: disallowed patch keys: {sorted(disallowed)}. "
            f"Only {sorted(_METADATA_ALLOWED)} may be patched."
        )

    index = load_run_index(path=resolved)

    if run_id not in index:
        raise RunIndexError(f"update_run_metadata: unknown run_id: {run_id!r}")

    # Append the allowed fields as an updated event.
    append_run_updated(run_id, patch, path=resolved)

    # Return the merged record.
    return _merge_fields(index[run_id], patch)


# ---------------------------------------------------------------------------
# Read-cutover step 1: backfill the runs table from the JSONL + reconcile.
# Additive — does NOT change any read path. Idempotent (upsert by run_id).
# ---------------------------------------------------------------------------


def backfill_runs_to_db(*, path: Path | str | None = None, engine=None) -> int:
    """Fold the JSONL and upsert every record into the runs table. Idempotent.
    Returns the number of records upserted; 0 if the JSONL is absent."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        return 0
    records = _fold_index_file(resolved)
    eng = engine if engine is not None else _shadow_engine(path)
    from p2s_agent.core.db.repositories import runs as runs_repo
    for rec in records.values():
        runs_repo.upsert_run(eng, asdict(rec))
    return len(records)


def reconcile_runs_with_db(*, path: Path | str | None = None, engine=None) -> list[str]:
    """Return run_ids that differ between the JSONL fold and the DB (empty = parity)."""
    resolved = _resolve_path(path)
    folded = _fold_index_file(resolved) if resolved.exists() else {}
    eng = engine if engine is not None else _shadow_engine(path)
    from p2s_agent.core.db.repositories import runs as runs_repo
    db_rows = runs_repo.get_all_runs(eng)
    mismatches: list[str] = []
    for rid, rec in folded.items():
        db = db_rows.get(rid)
        if db is None or _dict_to_record(db) != rec:
            mismatches.append(rid)
    mismatches.extend(rid for rid in db_rows if rid not in folded)
    return mismatches

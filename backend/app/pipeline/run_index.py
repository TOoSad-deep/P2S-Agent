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
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.pipeline.artifacts import DEFAULT_RESULTS_ROOT

# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------

_DEFAULT_INDEX_PATH = DEFAULT_RESULTS_ROOT / "run_index.jsonl"

# ---------------------------------------------------------------------------
# Module-level lock for concurrent append safety
# ---------------------------------------------------------------------------

_APPEND_LOCK = threading.Lock()

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
    )


def _merge_fields(record: RunLineageRecord, fields: dict[str, Any]) -> RunLineageRecord:
    """Return a new RunLineageRecord with *fields* overlaid on *record*."""
    base = _record_to_dict(record)
    base.update(fields)
    return _dict_to_record(base)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_run_created(record: RunLineageRecord, *, path: Path | str | None = None) -> None:
    """Append a ``created`` event for *record* to the JSONL index.

    Each call writes exactly one line:
    ``{"event": "created", <all record fields>}``.
    """
    resolved = _resolve_path(path)
    data: dict[str, Any] = {"event": "created"}
    data.update(_record_to_dict(record))
    _append_line(data, resolved)


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
    data: dict[str, Any] = {"event": "updated", "run_id": run_id}
    data.update(fields)
    _append_line(data, resolved)


def load_run_index(*, path: Path | str | None = None) -> dict[str, RunLineageRecord]:
    """Read the JSONL file and fold all events into a dict of RunLineageRecord.

    Rules:
    * ``created`` events initialise a record.
    * ``updated`` events overlay their fields onto the existing record.
    * Later events always override earlier ones.
    * Blank lines and non-JSON lines are silently skipped.
    * An ``updated`` that arrives before any ``created`` for that run_id
      creates a best-effort record (root_run_id defaults to run_id,
      status defaults to "unknown").

    Returns an empty dict if the file does not exist.
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {}

    records: dict[str, RunLineageRecord] = {}

    with resolved.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed line

            if not isinstance(data, dict):
                continue

            event = data.get("event")
            run_id = data.get("run_id")
            if not run_id:
                continue

            if event == "created":
                # Exclude the 'event' key before building the record.
                payload = {k: v for k, v in data.items() if k != "event"}
                records[run_id] = _dict_to_record(payload)

            elif event == "updated":
                fields = {k: v for k, v in data.items() if k not in ("event", "run_id")}
                if run_id in records:
                    records[run_id] = _merge_fields(records[run_id], fields)
                else:
                    # Best-effort: create from available fields.
                    synthetic: dict[str, Any] = {"run_id": run_id}
                    synthetic.update(fields)
                    records[run_id] = _dict_to_record(synthetic)

    return records


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

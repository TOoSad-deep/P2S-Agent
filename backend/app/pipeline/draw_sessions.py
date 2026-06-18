"""Draw-session helpers for the V3.5 Human-in-the-loop Batch Draw (抽卡式多结果生成) phase.

A *draw session* represents a gacha-style multi-result generation request where
N cards are drawn across one or more variant groups, each with a different
exploration strategy.

Design constraints (mirror variant_groups.py):
- Depends only on stdlib + ``app.pipeline.artifacts`` + ``app.pipeline.variant_groups``.
- Module-level ``threading.Lock`` serialises all JSONL append operations.
- Caller supplies any timestamps; this module never calls ``time.time()`` so
  functions are deterministic (safe for caching / testing).

Layout::

    backend/test_results/draw_sessions/
        <draw_id>.json
        <draw_id>_events.jsonl
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any

from app.pipeline.artifacts import DEFAULT_RESULTS_ROOT, save_json
from app.pipeline.variant_groups import aggregate_group_status

# ---------------------------------------------------------------------------
# Module-level lock for concurrent JSONL append safety (mirrors variant_groups.py)
# ---------------------------------------------------------------------------

_EVENTS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Default draw sessions directory
# ---------------------------------------------------------------------------

_DEFAULT_SESSIONS_DIR = DEFAULT_RESULTS_ROOT / "draw_sessions"


def _resolve_sessions_dir(root: "Path | str | None") -> Path:
    if root is not None:
        return Path(root) / "draw_sessions"
    return _DEFAULT_SESSIONS_DIR


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DrawSessionRecord:
    draw_id: str
    root_run_id: str
    parent_run_id: str
    source_checkpoint_id: str
    feedback: str
    status: str  # queued|running|completed|partial_failed|failed|cancelled
    requested_count: int
    diversity: str
    mode: str = "batch_draw"
    group_ids: list[str] = field(default_factory=list)
    card_run_ids: list[str] = field(default_factory=list)
    winner_run_id: str | None = None
    created_at: float = 0.0
    updated_at: float | None = None
    completed_at: float | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batch planning (pure, deterministic)
# ---------------------------------------------------------------------------


def plan_draw_batches(count: int) -> list[int]:
    """Split *count* cards into per-group batch sizes.

    Each VariantGroup accepts 2..6 cards (V3 ``build_variant_strategies``
    constraint).  This function distributes *count* as evenly as possible
    across ``ceil(count / 6)`` groups so every batch stays within [2, 6].

    Args:
        count: Total number of cards requested (must be in [2, 12]).

    Returns:
        List of batch sizes (each in [2, 6]) that sum to *count*.
        First ``count % groups`` batches get one extra card for even spread.

    Raises:
        ValueError: if *count* < 2 or *count* > 12.

    Examples::

        >>> plan_draw_batches(2)
        [2]
        >>> plan_draw_batches(7)
        [4, 3]
        >>> plan_draw_batches(12)
        [6, 6]
    """
    if count < 2:
        raise ValueError(f"plan_draw_batches: count must be >= 2, got {count}")
    if count > 12:
        raise ValueError(f"plan_draw_batches: count must be <= 12, got {count}")

    groups = ceil(count / 6)
    base, remainder = divmod(count, groups)
    # First `remainder` batches get base+1; the rest get base
    batches = [base + 1] * remainder + [base] * (groups - remainder)
    return batches


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_session(
    record: DrawSessionRecord,
    *,
    root: "Path | str | None" = None,
) -> Path:
    """Write *record* as ``<draw_id>.json`` under the draw_sessions dir.

    Uses ``save_json`` (atomic write via a temp file + os.replace).
    """
    sessions_dir = _resolve_sessions_dir(root)
    target = sessions_dir / f"{record.draw_id}.json"
    save_json(target, dataclasses.asdict(record))
    return target


def load_session(
    draw_id: str,
    *,
    root: "Path | str | None" = None,
) -> DrawSessionRecord | None:
    """Read and deserialise ``<draw_id>.json``.

    Returns ``None`` if the file is missing or JSON is malformed.
    Tolerantly reconstructs ALL fields with sane defaults, including
    ``int(...)`` coercion for requested_count and ``float(...)`` for
    created_at (mirrors ``load_group``'s field-by-field parsing).
    """
    sessions_dir = _resolve_sessions_dir(root)
    path = sessions_dir / f"{draw_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        _created = data.get("created_at")
        created_at_val = float(_created) if _created is not None else 0.0

        _updated = data.get("updated_at")
        updated_at_val = float(_updated) if _updated is not None else None

        _completed = data.get("completed_at")
        completed_at_val = float(_completed) if _completed is not None else None

        return DrawSessionRecord(
            draw_id=data.get("draw_id", draw_id),
            root_run_id=data.get("root_run_id", ""),
            parent_run_id=data.get("parent_run_id", ""),
            source_checkpoint_id=data.get("source_checkpoint_id", ""),
            feedback=data.get("feedback", ""),
            status=data.get("status", "queued"),
            requested_count=int(data.get("requested_count", 0)),
            diversity=data.get("diversity", "medium"),
            mode=data.get("mode", "batch_draw"),
            group_ids=list(data.get("group_ids") or []),
            card_run_ids=list(data.get("card_run_ids") or []),
            winner_run_id=data.get("winner_run_id"),
            created_at=created_at_val,
            updated_at=updated_at_val,
            completed_at=completed_at_val,
            metadata=dict(data.get("metadata") or {}),
        )
    except (TypeError, ValueError):
        return None


def append_session_event(
    draw_id: str,
    event: dict[str, Any],
    *,
    root: "Path | str | None" = None,
) -> None:
    """Append one JSON line to ``<draw_id>_events.jsonl``.

    A module-level lock serialises concurrent appends so partial lines are
    never written. The caller is responsible for supplying any timestamps
    inside *event* — this function does not inject ``time()`` calls.
    """
    sessions_dir = _resolve_sessions_dir(root)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{draw_id}_events.jsonl"
    line = json.dumps(event, ensure_ascii=False) + "\n"
    # Module-level lock: serializes event appends across all sessions (events files are per-session; contention is low given 2-12 cards per draw).
    with _EVENTS_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def load_session_events(
    draw_id: str,
    *,
    root: "Path | str | None" = None,
) -> list[dict[str, Any]]:
    """Read and parse ``<draw_id>_events.jsonl``.

    Skips blank lines, non-JSON lines, and non-dict JSON values silently.
    Returns an empty list if the file does not exist.
    """
    sessions_dir = _resolve_sessions_dir(root)
    path = sessions_dir / f"{draw_id}_events.jsonl"
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    events.append(obj)
    except OSError:
        return []
    return events


# ---------------------------------------------------------------------------
# Status aggregation (pure) — delegates to variant_groups.aggregate_group_status
# ---------------------------------------------------------------------------


def aggregate_draw_status(card_statuses: list[str]) -> str:
    """Derive a draw session's overall status from its cards' individual statuses.

    Delegates to ``aggregate_group_status`` — the aggregation rules are
    identical. There is no import cycle (variant_groups does not import
    draw_sessions).

    Rules (mirrored from aggregate_group_status):
    - empty list → ``"queued"``
    - all ``"queued"`` → ``"queued"``
    - any non-terminal status present → ``"running"``
    - all terminal:
      - all ``"completed"`` → ``"completed"``
      - some ``"completed"`` + some non-completed → ``"partial_failed"``
      - no ``"completed"``, any ``"cancelled"`` → ``"cancelled"``
      - else (all ``"failed"``) → ``"failed"``
    """
    return aggregate_group_status(card_statuses)

"""Variant-group helpers for the V3 Human-in-the-loop Variant Exploration phase.

A *variant group* represents one human-feedback event at a checkpoint that
spawns N child runs, each guided by a different exploration strategy.

Design constraints (mirror run_index.py):
- Depends only on stdlib + ``app.pipeline.artifacts`` (no FastAPI, no langgraph).
- Module-level ``threading.Lock`` serialises all JSONL append operations.
- Caller supplies any timestamps; this module never calls ``time.time()`` so
  strategy-building functions are deterministic (safe for caching / testing).

Layout::

    backend/test_results/variant_groups/
        <group_id>.json
        <group_id>_events.jsonl
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p2s_agent.core.db import shadow
from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT, save_json

# ---------------------------------------------------------------------------
# Module-level lock for concurrent JSONL append safety (mirrors run_index.py)
# ---------------------------------------------------------------------------

_EVENTS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Default group directory
# ---------------------------------------------------------------------------

_DEFAULT_GROUPS_DIR = DEFAULT_RESULTS_ROOT / "variant_groups"


def _resolve_groups_dir(root: "Path | str | None") -> Path:
    if root is not None:
        return Path(root) / "variant_groups"
    return _DEFAULT_GROUPS_DIR


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class VariantGroupRecord:
    group_id: str
    root_run_id: str
    parent_run_id: str
    source_checkpoint_id: str
    feedback: str
    mode: str
    variant_count: int
    diversity: str
    status: str  # queued|running|completed|partial_failed|failed|cancelled
    child_run_ids: list[str] = field(default_factory=list)
    winner_run_id: str | None = None
    created_at: float = 0.0
    completed_at: float | None = None
    draw_session_id: str | None = None


# ---------------------------------------------------------------------------
# Strategy templates (ordered; deterministic; pure)
# ---------------------------------------------------------------------------

_STRATEGY_TEMPLATES: list[dict[str, Any]] = [
    {
        "label": "conservative",
        "prompt_focus": "保持构图和调色，只做小幅改动",
        "score_drop_tolerance": 0.005,
    },
    {
        "label": "semantic",
        "prompt_focus": "更强烈满足用户反馈，可适度改变局部表现",
        "score_drop_tolerance": 0.03,
    },
    {
        "label": "lighting_color",
        "prompt_focus": "优先调整亮度、对比、色彩、反射/阴影",
        "score_drop_tolerance": 0.02,
    },
    {
        "label": "detail_texture",
        "prompt_focus": "优先增强纹理、边缘、局部细节",
        "score_drop_tolerance": 0.02,
    },
    {
        "label": "structure_form",
        "prompt_focus": "优先调整结构与形态比例",
        "score_drop_tolerance": 0.02,
    },
    {
        "label": "alt_technique",
        "prompt_focus": "尝试不同的渲染思路/技术",
        "score_drop_tolerance": 0.03,
    },
]


def build_variant_strategies(
    *,
    feedback: str,
    count: int,
    diversity: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Return *count* strategy dicts (deterministic, pure — no Date/random).

    Args:
        feedback: The human feedback text (included in caller context; not
            used to vary strategy content here — kept for API symmetry).
        count: Number of strategies to return (2..6 inclusive).
        diversity: One of ``low`` | ``medium`` | ``high``; unknown values
            are treated as ``medium``.
        mode: Pipeline mode string (passed through to caller; not used to
            alter strategy shapes here).

    Returns:
        List of *count* strategy dicts, each containing:
        ``label``, ``prompt_focus``, ``score_drop_tolerance``,
        ``diversity``, ``locks``, ``notes``.

    Raises:
        ValueError: if *count* < 2 or *count* > 6.
    """
    if count < 2:
        raise ValueError(f"build_variant_strategies: count must be >= 2, got {count}")
    if count > 6:
        raise ValueError(f"build_variant_strategies: count must be <= 6, got {count}")

    # Normalise diversity
    if diversity not in ("low", "medium", "high"):
        diversity = "medium"

    templates = _STRATEGY_TEMPLATES[:count]
    result: list[dict[str, Any]] = []

    for tmpl in templates:
        label: str = tmpl["label"]
        prompt_focus: str = tmpl["prompt_focus"]
        base_tolerance: float = tmpl["score_drop_tolerance"]

        locks: dict[str, Any] = {}
        notes: list[str] = [f"[VARIANT] {prompt_focus}"]

        if diversity == "low":
            locks["small_edits_only"] = True
            tolerance = min(base_tolerance, 0.01)
        elif diversity == "high":
            notes.append("[VARIANT] try a different rendering technique")
            # bump by +0.02 then clamp at 0.05
            tolerance = min(base_tolerance + 0.02, 0.05)
        else:
            # medium (default)
            tolerance = base_tolerance

        result.append(
            {
                "label": label,
                "prompt_focus": prompt_focus,
                "score_drop_tolerance": tolerance,
                "diversity": diversity,
                "locks": locks,
                "notes": notes,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_group(
    record: VariantGroupRecord,
    *,
    root: "Path | str | None" = None,
) -> Path:
    """Write *record* as ``<group_id>.json`` under the variant_groups dir.

    Uses ``save_json`` (atomic write via a temp file + os.replace).
    """
    groups_dir = _resolve_groups_dir(root)
    target = groups_dir / f"{record.group_id}.json"
    save_json(target, dataclasses.asdict(record))
    shadow.mirror_group(root, record)
    return target


def load_group(
    group_id: str,
    *,
    root: "Path | str | None" = None,
) -> VariantGroupRecord | None:
    """Read and deserialise ``<group_id>.json``.

    Returns ``None`` if the file is missing or JSON is malformed.
    """
    # File-first: <group_id>.json is the authoritative snapshot (written first);
    # the best-effort DB mirror is read only when the file is absent.
    groups_dir = _resolve_groups_dir(root)
    path = groups_dir / f"{group_id}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
    else:
        data = shadow.read_group(root, group_id)
        if data is None:
            return None
    try:
        _created = data.get("created_at")
        created_at_val = float(_created) if _created is not None else 0.0
        return VariantGroupRecord(
            group_id=data.get("group_id", group_id),
            root_run_id=data.get("root_run_id", ""),
            parent_run_id=data.get("parent_run_id", ""),
            source_checkpoint_id=data.get("source_checkpoint_id", ""),
            feedback=data.get("feedback", ""),
            mode=data.get("mode", ""),
            variant_count=int(data.get("variant_count", 0)),
            diversity=data.get("diversity", "medium"),
            status=data.get("status", "queued"),
            child_run_ids=list(data.get("child_run_ids") or []),
            winner_run_id=data.get("winner_run_id"),
            created_at=created_at_val,
            completed_at=data.get("completed_at"),
            draw_session_id=data.get("draw_session_id"),
        )
    except (TypeError, ValueError):
        return None


def append_group_event(
    group_id: str,
    event: dict[str, Any],
    *,
    root: "Path | str | None" = None,
) -> None:
    """Append one JSON line to ``<group_id>_events.jsonl``.

    A module-level lock serialises concurrent appends so partial lines are
    never written. The caller is responsible for supplying any timestamps
    inside *event* — this function does not inject ``time()`` calls.
    """
    groups_dir = _resolve_groups_dir(root)
    groups_dir.mkdir(parents=True, exist_ok=True)
    path = groups_dir / f"{group_id}_events.jsonl"
    line = json.dumps(event, ensure_ascii=False) + "\n"
    # Module-level lock: serializes event appends across all groups (events files are per-group; contention is low given 2-6 variants).
    with _EVENTS_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    shadow.mirror_group_event(root, group_id, event)


def load_group_events(
    group_id: str,
    *,
    root: "Path | str | None" = None,
) -> list[dict[str, Any]]:
    """Read and parse ``<group_id>_events.jsonl``.

    Skips blank lines and non-JSON lines silently. Returns an empty list
    if the file does not exist.
    """
    # File-first: the *_events.jsonl is the complete append-only log; the DB
    # mirror is best-effort and can't re-sync a swallowed event, so it is read
    # only when the file is absent (e.g. after the file is retired).
    groups_dir = _resolve_groups_dir(root)
    path = groups_dir / f"{group_id}_events.jsonl"
    if not path.exists():
        return shadow.read_events(root, "variant_group", group_id)

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
# Status aggregation (pure)
# ---------------------------------------------------------------------------

_TERMINAL = frozenset({"completed", "failed", "cancelled"})


def aggregate_group_status(child_statuses: list[str]) -> str:
    """Derive a group's overall status from its children's individual statuses.

    Rules:
    - empty list → ``"queued"``
    - all ``"queued"`` → ``"queued"``
    - any non-terminal status present → ``"running"``
    - all terminal:
      - all ``"completed"`` → ``"completed"``
      - some ``"completed"`` + some non-completed → ``"partial_failed"``
      - no ``"completed"``, any ``"cancelled"`` → ``"cancelled"``
      - else (all ``"failed"``) → ``"failed"``
    """
    if not child_statuses:
        return "queued"

    statuses = list(child_statuses)

    # All queued → queued
    if all(s == "queued" for s in statuses):
        return "queued"

    # Any non-terminal → running
    if any(s not in _TERMINAL for s in statuses):
        return "running"

    # All terminal from here
    completed_count = sum(1 for s in statuses if s == "completed")

    if completed_count == len(statuses):
        return "completed"

    if completed_count > 0:
        # Some completed, some non-completed terminal
        return "partial_failed"

    # No completed; check for cancelled
    if any(s == "cancelled" for s in statuses):
        return "cancelled"

    # All failed
    return "failed"

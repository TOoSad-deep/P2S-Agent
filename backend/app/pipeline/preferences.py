"""Preference event persistence and profile management for V4.3 Human-in-Loop.

Stores auditable user-preference events and a deterministic, editable
preference profile; turns the profile into LLM prompt notes.

Design constraints (mirror variant_groups.py / run_index.py):
- Depends only on stdlib + ``app.pipeline.artifacts`` (no FastAPI, no langgraph).
- Module-level ``threading.Lock`` serialises all JSONL append operations.
- Caller supplies any timestamps; this module never calls ``time.time()`` or
  ``random`` so all functions are deterministic and safe for caching / testing.

Layout::

    <root>/preferences/
        events.jsonl
        profile.json
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.pipeline.artifacts import DEFAULT_RESULTS_ROOT, save_json

# ---------------------------------------------------------------------------
# Module-level lock for concurrent JSONL append safety
# ---------------------------------------------------------------------------

_EVENTS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Default preferences directory
# ---------------------------------------------------------------------------

_DEFAULT_PREFS_DIR = DEFAULT_RESULTS_ROOT / "preferences"


def _resolve_prefs_dir(root: "Path | str | None") -> Path:
    if root is not None:
        return Path(root) / "preferences"
    return _DEFAULT_PREFS_DIR


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PreferenceEvent:
    event_id: str
    event_type: str  # "winner_selected" | "variant_rated" | "branch_accepted" | "manual_note"
    timestamp: float
    run_id: str | None = None
    group_id: str | None = None
    feedback: str | None = None
    winner_run_id: str | None = None
    loser_run_ids: list[str] = field(default_factory=list)
    rating: int | None = None
    reason: str | None = None
    tags: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Profile schema
# ---------------------------------------------------------------------------

_PROFILE_EDITABLE_KEYS = frozenset(
    {
        "enabled",
        "default_locks",
        "positive_preferences",
        "negative_preferences",
        "score_drop_tolerance_hint",
    }
)


def default_profile() -> dict:
    """Return a fresh default preference profile dict."""
    return {
        "schema_version": 1,
        "updated_at": 0.0,
        "enabled": True,
        "default_locks": {},
        "positive_preferences": [],
        "negative_preferences": [],
        "preferred_variant_labels": [],
        "score_drop_tolerance_hint": 0.02,
        "summary_source_event_count": 0,
    }


# ---------------------------------------------------------------------------
# Event persistence
# ---------------------------------------------------------------------------


def append_preference_event(
    event: PreferenceEvent,
    *,
    root: "Path | str | None" = None,
) -> None:
    """Append one JSON line to events.jsonl under a module-level lock.

    The caller supplies any timestamps; this function does not inject
    ``time()`` calls.
    """
    prefs_dir = _resolve_prefs_dir(root)
    prefs_dir.mkdir(parents=True, exist_ok=True)
    path = prefs_dir / "events.jsonl"
    line = json.dumps(dataclasses.asdict(event), ensure_ascii=False) + "\n"
    with _EVENTS_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def load_preference_events(
    *,
    limit: int | None = None,
    root: "Path | str | None" = None,
) -> list[PreferenceEvent]:
    """Parse events.jsonl; skip blank/non-JSON/non-dict lines.

    Args:
        limit: If given, return only the LAST ``limit`` events (most recent).
        root: Override root directory (for testing).

    Returns:
        List of ``PreferenceEvent`` instances; empty list if file missing.
    """
    prefs_dir = _resolve_prefs_dir(root)
    path = prefs_dir / "events.jsonl"
    if not path.exists():
        return []

    events: list[PreferenceEvent] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            # Tolerant reconstruction — missing keys get defaults.
            try:
                ev = PreferenceEvent(
                    event_id=obj.get("event_id", ""),
                    event_type=obj.get("event_type", ""),
                    timestamp=float(obj.get("timestamp", 0.0)),
                    run_id=obj.get("run_id"),
                    group_id=obj.get("group_id"),
                    feedback=obj.get("feedback"),
                    winner_run_id=obj.get("winner_run_id"),
                    loser_run_ids=list(obj.get("loser_run_ids") or []),
                    rating=obj.get("rating"),
                    reason=obj.get("reason"),
                    tags=list(obj.get("tags") or []),
                    context=dict(obj.get("context") or {}),
                )
                events.append(ev)
            except (TypeError, ValueError):
                continue

    if limit is not None:
        events = events[-limit:]
    return events


# ---------------------------------------------------------------------------
# Profile persistence
# ---------------------------------------------------------------------------


def load_profile(*, root: "Path | str | None" = None) -> dict:
    """Read profile.json; return ``default_profile()`` if missing or malformed."""
    prefs_dir = _resolve_prefs_dir(root)
    path = prefs_dir / "profile.json"
    if not path.exists():
        return default_profile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_profile()
    if not isinstance(data, dict):
        return default_profile()
    return data


def save_profile(profile: dict, *, root: "Path | str | None" = None) -> Path:
    """Write profile.json via ``save_json`` (atomic write)."""
    prefs_dir = _resolve_prefs_dir(root)
    target = prefs_dir / "profile.json"
    save_json(target, profile)
    return target


def patch_profile(
    patch: dict,
    updated_at: float,
    *,
    root: "Path | str | None" = None,
) -> dict:
    """Load current profile, apply only editable keys from *patch*, save.

    Editable keys: ``enabled``, ``default_locks``, ``positive_preferences``,
    ``negative_preferences``, ``score_drop_tolerance_hint``.

    Any OTHER key in *patch* raises ``ValueError`` (mirrors run_index
    ``update_run_metadata`` allowed-keys style).

    Sets ``updated_at`` from the supplied arg (NOT ``time()``).

    Returns the merged profile dict.
    """
    disallowed = set(patch.keys()) - _PROFILE_EDITABLE_KEYS
    if disallowed:
        raise ValueError(
            f"patch_profile: disallowed patch keys: {sorted(disallowed)}. "
            f"Only {sorted(_PROFILE_EDITABLE_KEYS)} may be patched."
        )

    profile = load_profile(root=root)
    for key, value in patch.items():
        profile[key] = value
    profile["updated_at"] = updated_at
    save_profile(profile, root=root)
    return profile


def clear_preferences(*, root: "Path | str | None" = None) -> None:
    """Remove/empty events.jsonl AND reset profile.json to default_profile().

    Best-effort; does not crash if files are absent.
    """
    prefs_dir = _resolve_prefs_dir(root)
    events_path = prefs_dir / "events.jsonl"
    profile_path = prefs_dir / "profile.json"

    try:
        if events_path.exists():
            events_path.unlink()
    except OSError:
        pass

    try:
        prefs_dir.mkdir(parents=True, exist_ok=True)
        save_profile(default_profile(), root=root)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Deterministic profile rebuild from events
# ---------------------------------------------------------------------------


def rebuild_profile(
    events: list[PreferenceEvent],
    *,
    updated_at: float,
    base_profile: dict | None = None,
) -> dict:
    """Deterministically rebuild a preference profile from a list of events.

    Args:
        events: All preference events to fold in.
        updated_at: Timestamp to record as ``updated_at`` (caller-supplied; no
            ``time()`` inside).
        base_profile: Optional starting profile dict; ``default_profile()`` is
            used if None. The user-editable ``enabled`` flag is preserved.

    Returns:
        A new JSON-serializable profile dict.

    Rules:
    - ``winner_selected`` and ``rating == 1`` events: add ``reason`` (non-empty)
      and each tag to ``positive_preferences`` (dedupe, preserve first-seen order).
    - ``variant_rated`` with ``rating == -1``: reason/tags → ``negative_preferences``.
    - ``preferred_variant_labels``: variant_label (from context) or tags of winner
      events; top ≤ 4 labels by descending frequency then alpha tie-break.
    - ``default_locks``: a lock key is set in ``default_locks`` if it is True in
      ≥ 50 % of the events that carry a ``locks`` dict in their context.
    - ``summary_source_event_count``: len(events).
    """
    profile = dict(base_profile) if base_profile is not None else default_profile()

    # Preserve user-editable enabled flag from base_profile
    enabled = profile.get("enabled", True)

    # --- positive / negative preferences (dedupe, first-seen order) ----------
    positive: list[str] = []
    negative: list[str] = []
    pos_seen: set[str] = set()
    neg_seen: set[str] = set()

    def _add_to(lst: list[str], seen: set[str], items: list[str]) -> None:
        for item in items:
            if item and item not in seen:
                seen.add(item)
                lst.append(item)

    # --- preferred_variant_labels frequency counter --------------------------
    label_freq: dict[str, int] = {}

    # --- lock accumulation --------------------------------------------------
    # We need to track: per lock key, how many events had it True, how many
    # events carried a locks dict at all.
    lock_true_count: dict[str, int] = {}
    lock_event_count: dict[str, int] = {}  # events that contained this key

    for ev in events:
        is_winner = ev.event_type == "winner_selected"
        is_positive_rating = ev.event_type == "variant_rated" and ev.rating == 1
        is_negative_rating = ev.event_type == "variant_rated" and ev.rating == -1

        # Positive signals
        if is_winner or is_positive_rating:
            candidates: list[str] = []
            if ev.reason:
                candidates.append(ev.reason)
            candidates.extend(ev.tags)
            _add_to(positive, pos_seen, candidates)

        # Negative signals
        if is_negative_rating:
            candidates = []
            if ev.reason:
                candidates.append(ev.reason)
            candidates.extend(ev.tags)
            _add_to(negative, neg_seen, candidates)

        # preferred_variant_labels: count variant_label or tags on winner events
        if is_winner:
            variant_label = ev.context.get("variant_label") if ev.context else None
            if variant_label:
                label_freq[variant_label] = label_freq.get(variant_label, 0) + 1
            else:
                for tag in ev.tags:
                    if tag:
                        label_freq[tag] = label_freq.get(tag, 0) + 1

        # locks accumulation
        locks_dict = ev.context.get("locks") if ev.context else None
        if isinstance(locks_dict, dict):
            for lock_key, lock_val in locks_dict.items():
                lock_event_count[lock_key] = lock_event_count.get(lock_key, 0) + 1
                if lock_val is True:
                    lock_true_count[lock_key] = lock_true_count.get(lock_key, 0) + 1

    # preferred_variant_labels: top ≤ 4 by descending frequency, alpha tie-break
    preferred_labels: list[str] = sorted(
        label_freq.keys(),
        key=lambda lbl: (-label_freq[lbl], lbl),
    )[:4]

    # default_locks: lock is True if True in ≥ 50 % of events that carry it
    default_locks: dict[str, bool] = {}
    for lock_key, total in lock_event_count.items():
        true_count = lock_true_count.get(lock_key, 0)
        if total > 0 and true_count >= total / 2:
            default_locks[lock_key] = True

    return {
        "schema_version": profile.get("schema_version", 1),
        "updated_at": updated_at,
        "enabled": enabled,
        "default_locks": default_locks,
        "positive_preferences": positive,
        "negative_preferences": negative,
        "preferred_variant_labels": preferred_labels,
        "score_drop_tolerance_hint": profile.get("score_drop_tolerance_hint", 0.02),
        "summary_source_event_count": len(events),
    }


# ---------------------------------------------------------------------------
# LLM prompt notes
# ---------------------------------------------------------------------------


def build_preference_notes(profile: dict) -> list[str]:
    """Turn a preference profile into a list of LLM prompt note strings.

    If ``profile.get("enabled")`` is falsy, returns ``[]`` immediately —
    preferences are not injected when disabled.

    Emits deterministic notes:
    - ``[PREFERENCE+] <p>`` for each positive preference.
    - ``[PREFERENCE-] <n>`` for each negative preference.
    - ``[PREFERENCE LOCK] <k>`` for each ``default_locks`` key that is True.
    - Optionally a ``[PREFERENCE LABELS] prefer: a, b`` note when labels exist.
    """
    if not profile.get("enabled"):
        return []

    notes: list[str] = []

    for p in profile.get("positive_preferences") or []:
        notes.append(f"[PREFERENCE+] {p}")

    for n in profile.get("negative_preferences") or []:
        notes.append(f"[PREFERENCE-] {n}")

    for k, v in (profile.get("default_locks") or {}).items():
        if v is True:
            notes.append(f"[PREFERENCE LOCK] {k}")

    labels = profile.get("preferred_variant_labels") or []
    if labels:
        notes.append(f"[PREFERENCE LABELS] prefer: {', '.join(labels)}")

    return notes

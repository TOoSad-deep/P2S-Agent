"""Fusion-plan helpers for the V4.5 Local Fusion phase.

A *fusion plan* describes how to blend selected local qualities from one or
more source renders into a base render, producing a single unified shader.

Design constraints (mirror draw_sessions.py):
- Depends only on stdlib + ``app.pipeline.artifacts``.
- Module-level ``threading.Lock`` serialises all JSONL append operations.
- Caller supplies any timestamps; this module never calls ``time.time()`` so
  functions are deterministic (safe for caching / testing).
- No FastAPI, numpy, or random imports.

Layout::

    backend/test_results/fusions/
        <fusion_id>.json
        <fusion_id>_events.jsonl
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
from p2s_agent.core.pipeline.region_types import FusionRegion  # re-exported for back-compat

# ---------------------------------------------------------------------------
# Module-level lock for concurrent JSONL append safety (mirrors draw_sessions.py)
# ---------------------------------------------------------------------------

_EVENTS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Default fusions directory
# ---------------------------------------------------------------------------

_DEFAULT_FUSIONS_DIR = DEFAULT_RESULTS_ROOT / "fusions"


def _resolve_fusions_dir(root: "Path | str | None") -> Path:
    if root is not None:
        return Path(root) / "fusions"
    return _DEFAULT_FUSIONS_DIR


@dataclass
class FusionPlanRecord:
    fusion_id: str
    root_run_id: str
    parent_run_id: str
    base_run_id: str
    source_run_ids: list[str]
    draw_session_id: str | None
    feedback: str
    status: str                 # "draft" | "target_ready" | "running" | "completed" | "failed"
    regions: list[FusionRegion] = field(default_factory=list)
    composite_target_artifact_id: str | None = None
    output_run_id: str | None = None
    created_at: float = 0.0
    updated_at: float | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_plan(
    record: FusionPlanRecord,
    *,
    root: "Path | str | None" = None,
) -> Path:
    """Write *record* as ``<fusion_id>.json`` under the fusions dir.

    Uses ``save_json`` (atomic write via a temp file + os.replace).
    """
    fusions_dir = _resolve_fusions_dir(root)
    target = fusions_dir / f"{record.fusion_id}.json"
    save_json(target, dataclasses.asdict(record))
    shadow.mirror_fusion(root, record)
    return target


def update_plan_status(
    fusion_id: str,
    status: str,
    *,
    updated_at: "float | None" = None,
    root: "Path | str | None" = None,
) -> "FusionPlanRecord | None":
    """Load ``<fusion_id>.json``, set its ``status`` (and ``updated_at``), and
    re-save it atomically.

    Returns the updated record, or ``None`` if the plan does not exist (or could
    not be loaded). Intended for best-effort terminal transitions from the
    pipeline worker, so callers should treat a ``None`` result as a no-op rather
    than an error.
    """
    record = load_plan(fusion_id, root=root)
    if record is None:
        return None
    record.status = status
    if updated_at is not None:
        record.updated_at = updated_at
    save_plan(record, root=root)
    return record


def load_plan(
    fusion_id: str,
    *,
    root: "Path | str | None" = None,
) -> "FusionPlanRecord | None":
    """Read and deserialise ``<fusion_id>.json``.

    Returns ``None`` if the file is missing or JSON is malformed.
    Tolerantly reconstructs ALL fields with sane defaults, including
    nested regions (parse each region dict → FusionRegion),
    int/float coercion, and lists defaulted.
    """
    data = shadow.read_fusion(root, fusion_id)  # read-cutover: DB first
    if data is None:
        fusions_dir = _resolve_fusions_dir(root)
        path = fusions_dir / f"{fusion_id}.json"
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

        # Parse nested regions
        raw_regions = data.get("regions") or []
        regions: list[FusionRegion] = []
        for r in raw_regions:
            if not isinstance(r, dict):
                continue
            regions.append(FusionRegion(
                id=r.get("id", ""),
                label=r.get("label", ""),
                source_run_id=r.get("source_run_id", ""),
                instruction=r.get("instruction", ""),
                geometry_type=r.get("geometry_type", "rect"),
                geometry=dict(r.get("geometry") or {}),
                strength=float(r.get("strength", 0.5)),
                blend_mode=r.get("blend_mode", "soft"),
                feather=float(r.get("feather", 0.08)),
            ))

        return FusionPlanRecord(
            fusion_id=data.get("fusion_id", fusion_id),
            root_run_id=data.get("root_run_id", ""),
            parent_run_id=data.get("parent_run_id", ""),
            base_run_id=data.get("base_run_id", ""),
            source_run_ids=list(data.get("source_run_ids") or []),
            draw_session_id=data.get("draw_session_id"),
            feedback=data.get("feedback", ""),
            status=data.get("status", "draft"),
            regions=regions,
            composite_target_artifact_id=data.get("composite_target_artifact_id"),
            output_run_id=data.get("output_run_id"),
            created_at=created_at_val,
            updated_at=updated_at_val,
            metadata=dict(data.get("metadata") or {}),
        )
    except (TypeError, ValueError):
        return None


def append_plan_event(
    fusion_id: str,
    event: dict[str, Any],
    *,
    root: "Path | str | None" = None,
) -> None:
    """Append one JSON line to ``<fusion_id>_events.jsonl``.

    A module-level lock serialises concurrent appends so partial lines are
    never written. The caller is responsible for supplying any timestamps
    inside *event* — this function does not inject ``time()`` calls.
    """
    fusions_dir = _resolve_fusions_dir(root)
    fusions_dir.mkdir(parents=True, exist_ok=True)
    path = fusions_dir / f"{fusion_id}_events.jsonl"
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _EVENTS_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    shadow.mirror_fusion_event(root, fusion_id, event)


def load_plan_events(
    fusion_id: str,
    *,
    root: "Path | str | None" = None,
) -> list[dict[str, Any]]:
    """Read and parse ``<fusion_id>_events.jsonl``.

    Skips blank lines, non-JSON lines, and non-dict JSON values silently.
    Returns an empty list if the file does not exist.
    """
    # File-first: the *_events.jsonl is the complete append-only log; the DB
    # mirror is best-effort and can't re-sync a swallowed event, so it is read
    # only when the file is absent (e.g. after the file is retired).
    fusions_dir = _resolve_fusions_dir(root)
    path = fusions_dir / f"{fusion_id}_events.jsonl"
    if not path.exists():
        return shadow.read_events(root, "fusion", fusion_id)

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
# Parsing (tolerant)
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float) -> float:
    """Coerce *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_fusion_plan(
    payload: dict,
    *,
    fusion_id: str,
    root_run_id: str,
    parent_run_id: str,
    created_at: float,
) -> FusionPlanRecord:
    """Parse a raw payload dict into a FusionPlanRecord.

    Tolerant parsing:
    - ``base_run_id``: str, defaults to "".
    - ``source_run_ids``: list[str]; if absent/empty, derived from the
      regions' ``source_run_id`` set (dedup, stable insertion order).
    - ``draw_session_id``: optional str.
    - ``feedback``: str, defaults to "".
    - ``regions``: list of dicts → FusionRegion with defaults applied.
    - ``metadata``: dict, defaults to {}.
    - ``status``: always "draft" on creation.
    - Bad strength/feather coerced to defaults via ``_safe_float``.
    """
    # Guard: if payload is not a dict, treat as empty
    payload = payload if isinstance(payload, dict) else {}

    base_run_id = str(payload.get("base_run_id") or "")
    draw_session_id = payload.get("draw_session_id")
    feedback = str(payload.get("feedback") or "")
    _meta = payload.get("metadata")
    metadata = _meta if isinstance(_meta, dict) else {}

    # Parse regions first so we can derive source_run_ids if needed
    _raw_regions = payload.get("regions")
    raw_regions = _raw_regions if isinstance(_raw_regions, list) else []
    regions: list[FusionRegion] = []
    for r in raw_regions:
        if not isinstance(r, dict):
            continue
        strength = _safe_float(r.get("strength"), 0.5)
        feather = _safe_float(r.get("feather"), 0.08)
        _geo = r.get("geometry")
        geometry = _geo if isinstance(_geo, dict) else {}
        regions.append(FusionRegion(
            id=str(r.get("id") or ""),
            label=str(r.get("label") or ""),
            source_run_id=str(r.get("source_run_id") or ""),
            instruction=str(r.get("instruction") or ""),
            geometry_type=str(r.get("geometry_type") or "rect"),
            geometry=geometry,
            strength=strength,
            blend_mode=str(r.get("blend_mode") or "soft"),
            feather=feather,
        ))

    # source_run_ids: use payload value if present, else derive from regions
    raw_src = payload.get("source_run_ids")
    if raw_src is not None and isinstance(raw_src, list) and len(raw_src) > 0:
        source_run_ids = [str(s) for s in raw_src]
    else:
        # Derive: dedup, stable insertion order from regions
        seen: dict[str, None] = {}
        for region in regions:
            if region.source_run_id:
                seen[region.source_run_id] = None
        source_run_ids = list(seen.keys())

    return FusionPlanRecord(
        fusion_id=fusion_id,
        root_run_id=root_run_id,
        parent_run_id=parent_run_id,
        base_run_id=base_run_id,
        source_run_ids=source_run_ids,
        draw_session_id=draw_session_id,
        feedback=feedback,
        status="draft",
        regions=regions,
        composite_target_artifact_id=None,
        output_run_id=None,
        created_at=created_at,
        updated_at=None,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Validation (pure)
# ---------------------------------------------------------------------------

_VALID_BLEND_MODES = {"soft", "replace_target", "protect_base"}
_VALID_GEOMETRY_TYPES = {"rect", "polygon", "mask"}


def validate_fusion_plan(record: FusionPlanRecord) -> list[str]:
    """Validate a FusionPlanRecord and return a list of error strings.

    Returns [] if the plan is valid.  Does NOT check whether run IDs or
    artifact IDs actually exist on disk — that belongs in the router.
    """
    errors: list[str] = []

    # base_run_id must be non-empty
    if not record.base_run_id:
        errors.append("base_run_id is required")

    # region id uniqueness
    seen_ids: set[str] = set()
    for region in record.regions:
        if not region.id:
            errors.append("region id must be non-empty")
        elif region.id in seen_ids:
            errors.append(f"duplicate region id: {region.id!r}")
        else:
            seen_ids.add(region.id)

    # Each region's source_run_id must be non-empty AND in source_run_ids
    src_set = set(record.source_run_ids)
    for region in record.regions:
        if not region.source_run_id:
            errors.append(f"region {region.id!r}: source_run_id is required")
        elif region.source_run_id not in src_set:
            errors.append(
                f"region {region.id!r}: source_run_id {region.source_run_id!r}"
                f" not in source_run_ids"
            )

    # Per-region field validation
    for region in record.regions:
        # strength in [0, 1]
        if not (0.0 <= region.strength <= 1.0):
            errors.append(
                f"region {region.id!r}: strength {region.strength} not in [0, 1]"
            )
        # feather in [0, 1]
        if not (0.0 <= region.feather <= 1.0):
            errors.append(
                f"region {region.id!r}: feather {region.feather} not in [0, 1]"
            )
        # blend_mode
        if region.blend_mode not in _VALID_BLEND_MODES:
            errors.append(
                f"region {region.id!r}: blend_mode {region.blend_mode!r} is invalid"
            )
        # geometry_type
        if region.geometry_type not in _VALID_GEOMETRY_TYPES:
            errors.append(
                f"region {region.id!r}: geometry_type {region.geometry_type!r} is invalid"
            )
        # rect bounds check
        if region.geometry_type == "rect":
            g = region.geometry
            try:
                x = float(g["x"])
                y = float(g["y"])
                w = float(g["w"])
                h = float(g["h"])
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"region {region.id!r}: rect geometry must have numeric x, y, w, h"
                )
            else:
                if not (x >= 0 and y >= 0 and w > 0 and h > 0
                        and x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9):
                    errors.append(
                        f"region {region.id!r}: rect geometry out of bounds"
                        f" (x={x}, y={y}, w={w}, h={h})"
                    )

    return errors


# ---------------------------------------------------------------------------
# Notes builder (deterministic)
# ---------------------------------------------------------------------------


def build_fusion_notes(record: FusionPlanRecord) -> list[str]:
    """Build deterministic LLM prompt notes for a fusion plan.

    Returns a list of note strings in a fixed order.
    """
    notes: list[str] = []

    notes.append(
        "[FUSION GOAL] Create one coherent shader that preserves the base"
        " composition while borrowing selected local qualities from source renders."
    )
    notes.append(
        f"[BASE] Use {record.base_run_id} as the global structure and shader starting point."
    )

    for region in record.regions:
        if region.geometry_type == "rect":
            g = region.geometry
            x = g.get("x", 0)
            y = g.get("y", 0)
            w = g.get("w", 0)
            h = g.get("h", 0)
            notes.append(
                f"[REGION SOURCE {region.id}] In rect x={x} y={y} w={w} h={h},"
                f" borrow {region.instruction} from {region.source_run_id}."
                f" Blend {region.blend_mode}, strength {region.strength}."
            )
        else:
            notes.append(
                f"[REGION SOURCE {region.id}] borrow {region.instruction}"
                f" from {region.source_run_id}."
                f" Blend {region.blend_mode}, strength {region.strength}."
            )

    notes.append(
        "[IMPORTANT] Do not create a pasted collage."
        " Produce a single unified shader."
    )

    return notes


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def plan_to_dict(record: FusionPlanRecord) -> dict:
    """Return a JSON-serializable dict representation of *record*.

    Uses ``dataclasses.asdict`` which recursively converts nested dataclasses.
    """
    return dataclasses.asdict(record)

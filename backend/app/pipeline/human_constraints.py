"""Structured human-control constraints for V4.1 Human-in-the-loop.

Defines the HumanConstraintSpec / RegionConstraint data model, validation,
LLM prompt note generation, and artifact persistence.

Design constraints (mirrors sibling pure modules):
- stdlib + ``app.pipeline.artifacts`` only.
- No FastAPI, no numpy, no ``time()``/random.
- Fully deterministic (safe for caching / testing).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.pipeline.artifacts import save_json

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_VALID_MODES = frozenset({"modify", "protect"})
_VALID_GEOMETRY_TYPES = frozenset({"rect", "polygon", "mask"})
_VALID_DIRECTIONS = frozenset({"keep", "increase", "decrease"})


@dataclass
class RegionConstraint:
    """A spatial region that the user wants to modify or protect."""

    id: str
    label: str
    mode: str           # "modify" | "protect"
    instruction: str
    geometry_type: str  # "rect" (V4.2 adds "polygon"|"mask")
    geometry: dict      # rect: {"x","y","w","h"} normalised 0..1
    strength: float = 0.5


@dataclass
class HumanConstraintSpec:
    """Top-level structured constraint specification provided by the human."""

    locks: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)   # {attr: "keep"|"increase"|"decrease"}
    edit_strength: float = 0.5
    regions: list[RegionConstraint] = field(default_factory=list)
    use_preferences: bool = True


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_region(raw: Any) -> RegionConstraint:
    """Convert a raw dict to a RegionConstraint with sane defaults."""
    if not isinstance(raw, dict):
        raw = {}
    return RegionConstraint(
        id=str(raw.get("id", "")),
        label=str(raw.get("label", "")),
        mode=str(raw.get("mode", "modify")),
        instruction=str(raw.get("instruction", "")),
        geometry_type=str(raw.get("geometry_type", "rect")),
        geometry=raw.get("geometry", {}) if isinstance(raw.get("geometry"), dict) else {},
        strength=float(raw["strength"]) if "strength" in raw else 0.5,
    )


def parse_constraint_spec(payload: dict | None) -> HumanConstraintSpec:
    """Parse a raw payload dict into a ``HumanConstraintSpec``.

    ``None`` or ``{}`` → default ``HumanConstraintSpec()``.
    Tolerant: ignores unknown keys, coerces types defensively.
    """
    if not payload:
        return HumanConstraintSpec()

    raw_locks = payload.get("locks", {})
    locks = raw_locks if isinstance(raw_locks, dict) else {}

    raw_targets = payload.get("targets", {})
    targets = raw_targets if isinstance(raw_targets, dict) else {}

    edit_strength = float(payload["edit_strength"]) if "edit_strength" in payload else 0.5

    raw_regions = payload.get("regions", [])
    regions: list[RegionConstraint] = []
    if isinstance(raw_regions, list):
        for item in raw_regions:
            regions.append(_parse_region(item))

    use_preferences = bool(payload.get("use_preferences", True))

    return HumanConstraintSpec(
        locks=locks,
        targets=targets,
        edit_strength=edit_strength,
        regions=regions,
        use_preferences=use_preferences,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_constraint_spec(
    spec: HumanConstraintSpec,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[str]:
    """Return a list of human-readable error strings (empty = valid).

    Does NOT silently clamp values — reports them as errors so the caller
    can decide how to handle them.
    """
    errors: list[str] = []

    # edit_strength must be in [0, 1]
    if not (0.0 <= spec.edit_strength <= 1.0):
        errors.append(
            f"edit_strength={spec.edit_strength} is out of range [0, 1]."
        )

    # locks values must all be bool
    for key, val in spec.locks.items():
        if not isinstance(val, bool):
            errors.append(
                f"locks[{key!r}] must be a bool, got {type(val).__name__!r}."
            )

    # targets values must each be a valid direction
    for attr, direction in spec.targets.items():
        if direction not in _VALID_DIRECTIONS:
            errors.append(
                f"targets[{attr!r}] has invalid direction {direction!r}; "
                f"must be one of {sorted(_VALID_DIRECTIONS)}."
            )

    # Region validation
    seen_ids: set[str] = set()
    for region in spec.regions:
        rid = region.id

        # Non-empty id
        if not rid:
            errors.append("A region has an empty id; all region ids must be non-empty.")
            continue  # can't use rid in messages below

        # Duplicate id check
        if rid in seen_ids:
            errors.append(f"Duplicate region id {rid!r}.")
        seen_ids.add(rid)

        # mode
        if region.mode not in _VALID_MODES:
            errors.append(
                f"Region {rid!r}: mode={region.mode!r} is invalid; "
                f"must be one of {sorted(_VALID_MODES)}."
            )

        # geometry_type
        if region.geometry_type not in _VALID_GEOMETRY_TYPES:
            errors.append(
                f"Region {rid!r}: geometry_type={region.geometry_type!r} is invalid; "
                f"must be one of {sorted(_VALID_GEOMETRY_TYPES)}."
            )

        # strength in [0, 1]
        if not (0.0 <= region.strength <= 1.0):
            errors.append(
                f"Region {rid!r}: strength={region.strength} is out of range [0, 1]."
            )

        # Rect geometry bounds check (only for rect)
        if region.geometry_type == "rect":
            geo = region.geometry
            try:
                x = float(geo["x"])
                y = float(geo["y"])
                w = float(geo["w"])
                h = float(geo["h"])
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"Region {rid!r}: rect geometry must have numeric x, y, w, h keys."
                )
                continue

            if x < 0:
                errors.append(
                    f"Region {rid!r}: rect x={x} must be >= 0 (normalised)."
                )
            if y < 0:
                errors.append(
                    f"Region {rid!r}: rect y={y} must be >= 0 (normalised)."
                )
            if w <= 0:
                errors.append(
                    f"Region {rid!r}: rect w={w} must be > 0."
                )
            if h <= 0:
                errors.append(
                    f"Region {rid!r}: rect h={h} must be > 0."
                )
            if x + w > 1.0:
                errors.append(
                    f"Region {rid!r}: rect x+w={x + w:.4g} exceeds 1.0 (normalised bounds)."
                )
            if y + h > 1.0:
                errors.append(
                    f"Region {rid!r}: rect y+h={y + h:.4g} exceeds 1.0 (normalised bounds)."
                )

    return errors


# ---------------------------------------------------------------------------
# Prompt note generation
# ---------------------------------------------------------------------------

_LOCK_MESSAGES: dict[str, str] = {
    "preserve_layout": "Preserve layout and major object positions.",
    "preserve_palette": "Preserve the current color palette.",
    "preserve_background": "Preserve background and large-scale lighting.",
    "small_edits_only": "Make small, targeted edits; avoid rewriting the shader.",
}


def build_constraint_notes(spec: HumanConstraintSpec) -> list[str]:
    """Build ordered, deterministic prompt notes from a ``HumanConstraintSpec``.

    Note style mirrors ``human_feedback.build_human_feedback_notes``:
    bracketed tag prefix followed by natural-English description.
    """
    notes: list[str] = []

    # 1. Global lock notes (in a stable key order for determinism)
    for key in ("preserve_layout", "preserve_palette", "preserve_background", "small_edits_only"):
        if spec.locks.get(key):
            msg = _LOCK_MESSAGES.get(key, f"Preserve {key}.")
            notes.append(f"[GLOBAL LOCK] {msg}")

    # 2. Any extra lock keys not in the canonical list
    for key, val in spec.locks.items():
        if key not in _LOCK_MESSAGES and val:
            notes.append(f"[GLOBAL LOCK] Preserve {key}.")

    # 3. Target notes (only non-"keep" directions)
    for attr, direction in spec.targets.items():
        if direction == "increase":
            notes.append(f"[TARGET] Increase {attr}.")
        elif direction == "decrease":
            notes.append(f"[TARGET] Decrease {attr}.")
        # "keep" → omitted

    # 4. Edit-strength note (always emitted — useful LLM signal)
    notes.append(
        f"[EDIT STRENGTH] {spec.edit_strength}: "
        "make targeted changes, avoid rewriting the shader."
    )

    # 5. Region notes
    for region in spec.regions:
        if region.geometry_type == "rect":
            geo = region.geometry
            try:
                x = geo["x"]
                y = geo["y"]
                w = geo["w"]
                h = geo["h"]
                coord_str = f" in normalized rect x={x} y={y} w={w} h={h}"
            except KeyError:
                coord_str = ""
        else:
            coord_str = ""

        if region.mode == "modify":
            notes.append(
                f"[REGION MODIFY {region.id}] {region.instruction}{coord_str}."
            )
        else:  # protect
            notes.append(
                f"[REGION PROTECT {region.id}] {region.instruction}{coord_str}."
            )

    return notes


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def spec_to_dict(spec: HumanConstraintSpec) -> dict[str, Any]:
    """Convert a ``HumanConstraintSpec`` to a plain JSON-serialisable dict.

    The shape mirrors the JSON payload consumed by ``parse_constraint_spec``.
    """
    return {
        "locks": dict(spec.locks),
        "targets": dict(spec.targets),
        "edit_strength": spec.edit_strength,
        "regions": [dataclasses.asdict(r) for r in spec.regions],
        "use_preferences": spec.use_preferences,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def constraint_to_artifacts(
    run_dir: "str | Path | None",
    spec: HumanConstraintSpec,
) -> "Path | None":
    """Write ``constraints.json`` into *run_dir* and return the path.

    Returns ``None`` if *run_dir* is falsy/None (no write performed).
    Uses ``app.pipeline.artifacts.save_json`` (atomic temp-file write).
    """
    if not run_dir:
        return None

    target = Path(run_dir) / "constraints.json"
    save_json(target, spec_to_dict(spec))
    return target

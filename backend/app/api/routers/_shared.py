"""Shared input-validation helpers + length caps for the PNG-shader routers.

These are web-layer (HTTP-coupled) helpers used by more than one of the split
domain routers in this package. They live here so each domain module can import
them without re-creating them and without a module-to-module cycle.

The text-length caps and the safe-id helpers are also re-exported from
``app.routers.png_shader`` for backward compatibility with callers/tests that
import them from the historical router location.
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import HTTPException

from app.api.guards import _env_int

# Per-request variant-COUNT cap is a route input-validation limit (max variants a
# caller may request); the variant-CONCURRENCY cap that sizes the worker pool
# lives with the worker layer in p2s_agent.workers.
_MAX_VARIANT_COUNT = 6

# V4.2: allowlist regex for region_id — mirrors _CANDIDATE_ID_RE in checkpoints.py.
# Rejects path-traversal characters (/, ..) and leading dots.
_REGION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

# Item 3 — length caps (chars) for free-text / code inputs. Over-cap → 422.
_MAX_SEED_GLSL_CHARS = _env_int("MAX_SEED_GLSL_CHARS", 256 * 1024)
_MAX_INPUT_SPEC_CHARS = _env_int("MAX_INPUT_SPEC_CHARS", 256 * 1024)
_MAX_FEEDBACK_CHARS = _env_int("MAX_FEEDBACK_CHARS", 8 * 1024)
_MAX_MODIFIED_DSL_CHARS = _env_int("MAX_MODIFIED_DSL_CHARS", 256 * 1024)

# Item 2 — allowlist regex for client-supplied ids used in filesystem paths.
# Allows only [A-Za-z0-9_-]; rejects empty, "..", "/", and any other char.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_safe_id(value: object, *, field: str = "id") -> str:
    """Return *value* if it is a path-safe id, else raise HTTPException(422).

    A safe id contains only ``[A-Za-z0-9_-]`` and is non-empty. This blocks
    path-traversal payloads (``../``, ``/``, ``..``) before any id is joined
    into a filesystem path.
    """
    if not isinstance(value, str) or not _SAFE_ID_RE.match(value):
        raise HTTPException(
            status_code=422,
            detail=f"{field} contains disallowed characters",
        )
    return value


def _enforce_text_cap(value: Optional[str], cap: int, *, field: str) -> None:
    """Reject (422) a free-text/code input whose length exceeds *cap* chars."""
    if value is not None and len(value) > cap:
        raise HTTPException(
            status_code=422,
            detail=f"{field} exceeds maximum length of {cap} characters",
        )


def _coerce_int(value, field_name: str, default: int, lo: int, hi: int) -> int:
    """Coerce a JSON value to an int in ``[lo, hi]`` or raise HTTPException(422).

    Replaces bare ``int(...)`` coercions on request fields so a non-numeric or
    out-of-range value yields a clean 422 instead of an uncaught ``ValueError``
    bubbling up as a 500 (Bug 3). ``None`` / falsy → ``default``. Bools are
    rejected (a JSON ``true`` is not a valid count)."""
    if value is None or value == "":
        value = default
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{field_name} must be an integer")
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail=f"{field_name} must be an integer, got {value!r}"
        )
    if coerced < lo or coerced > hi:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be between {lo} and {hi}, got {coerced}",
        )
    return coerced

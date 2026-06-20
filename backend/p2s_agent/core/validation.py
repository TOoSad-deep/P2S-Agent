"""Agent-side input validation helpers.

These mirror the semantics of the router guards in app/routers/png_shader.py
(_coerce_int, validate_safe_id, _enforce_text_cap) but raise AgentInputError
instead of HTTPException.  The agent package never imports fastapi/starlette.

Semantics faithfully matched from the router source (2026-06-20):
  - coerce_int: None or "" → default; bool → error (before int()); int(value)
    coercion (accepts "3"); out-of-range → error.
  - validate_safe_id: must be str matching ^[A-Za-z0-9_-]+$; rejects empty, "..",
    "/", any other char.
  - enforce_text_cap: value is not None and len(value) > cap → error.
"""
from __future__ import annotations

import re
from typing import Optional

from p2s_agent.core.errors import AgentInputError

# Allowlist regex for client-supplied ids used in filesystem paths.
# Matches the router's _SAFE_ID_RE exactly.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def coerce_int(
    value: object,
    *,
    field: str,
    default: int,
    lo: int,
    hi: int,
) -> int:
    """Coerce *value* to an int in ``[lo, hi]`` or raise AgentInputError.

    Mirrors ``_coerce_int`` in app/routers/png_shader.py exactly:
    - ``None`` or ``""`` → *default*
    - ``bool`` → rejected (a JSON ``true`` is not a valid count)
    - ``int(value)`` coercion (accepts string digits like ``"3"``)
    - out-of-range → rejected
    """
    if value is None or value == "":
        value = default
    if isinstance(value, bool):
        raise AgentInputError(f"{field} must be an integer", field=field)
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise AgentInputError(
            f"{field} must be an integer, got {value!r}", field=field
        )
    if coerced < lo or coerced > hi:
        raise AgentInputError(
            f"{field} must be between {lo} and {hi}, got {coerced}", field=field
        )
    return coerced


def validate_safe_id(value: object, *, field: str = "id") -> str:
    """Return *value* if it is a path-safe id, else raise AgentInputError.

    Mirrors ``validate_safe_id`` in app/routers/png_shader.py exactly:
    a safe id contains only ``[A-Za-z0-9_-]`` and is non-empty.
    """
    if not isinstance(value, str) or not _SAFE_ID_RE.match(value):
        raise AgentInputError(f"{field} contains disallowed characters", field=field)
    return value


def enforce_text_cap(
    value: Optional[str],
    cap: int,
    *,
    field: str,
) -> None:
    """Reject a free-text/code input whose length exceeds *cap* chars.

    Mirrors ``_enforce_text_cap`` in app/routers/png_shader.py exactly.
    """
    if value is not None and len(value) > cap:
        raise AgentInputError(
            f"{field} exceeds maximum length of {cap} characters", field=field
        )

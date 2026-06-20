"""FastAPI router for the PNG-to-Shader pipeline (thin aggregator).

The route handlers now live in domain-split modules under
``app.api.routers.{core,branch,variant,draw,fusion,preferences}``. This module
keeps the historical ``router`` symbol working — it aggregates the six domain
routers into a single ``APIRouter`` that produces the identical public HTTP
surface (every ``/png-shader/...`` path, method, response model and status code
is unchanged).

Each domain sub-router already carries the ``/png-shader`` prefix and the
``png-shader`` tag, so this aggregator is a plain prefix-less ``APIRouter`` that
just includes them — adding a prefix here would double it up.

A handful of input-validation helpers and length caps are re-exported here for
backward compatibility with callers/tests that import them from this module's
historical location (``app.routers.png_shader``).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routers import branch, core, draw, fusion, preferences, variant

# Backward-compatible re-exports. These web-layer input-validation helpers and
# length caps moved to app.api.routers._shared; keep them importable here so
# existing callers/tests (`from app.routers.png_shader import validate_safe_id`,
# `_MAX_SEED_GLSL_CHARS`, ...) continue to resolve.
from app.api.routers._shared import (  # noqa: F401  re-export
    validate_safe_id,
    _coerce_int,
    _enforce_text_cap,
    _MAX_FEEDBACK_CHARS,
    _MAX_INPUT_SPEC_CHARS,
    _MAX_MODIFIED_DSL_CHARS,
    _MAX_SEED_GLSL_CHARS,
    _MAX_VARIANT_COUNT,
    _REGION_ID_RE,
    _SAFE_ID_RE,
)

# Aggregator router: no prefix here (the sub-routers carry "/png-shader"), so the
# combined surface is byte-identical to the pre-split single router.
router = APIRouter()
router.include_router(core.router)
router.include_router(branch.router)
router.include_router(variant.router)
router.include_router(draw.router)
router.include_router(fusion.router)
router.include_router(preferences.router)

"""Seed-GLSL adaptation and candidate construction.

Turns an externally-supplied GLSL shader (possibly not in Shadertoy form)
into a renderable Shadertoy ``mainImage`` shader and wraps it as a single
``CandidateRecord``, so the existing post-pipeline closed loop can refine it
without running candidate generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from p2s_agent.core.pipeline.pool import CandidateRecord
from p2s_agent.core.render.shader_validator import validate_shader_static
from p2s_agent.core.utils.glsl_postprocess import normalize_shadertoy_glsl

logger = logging.getLogger(__name__)


@dataclass
class SeedAdaptResult:
    """Outcome of adapting a seed shader to renderable Shadertoy GLSL."""

    glsl: str
    valid: bool
    adapted_by: str  # "normalized" | "wrapped" | "llm_ported" | "failed"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# `void main()` / `void main(void)` signature of a legacy fragment shader.
_MAIN_SIG_RE = re.compile(r"\bvoid\s+main\s*\(\s*(?:void)?\s*\)")


def _wrap_legacy_main(glsl: str) -> str | None:
    """Best-effort rewrite of a ``void main(){...gl_FragColor...}`` fragment
    shader into a Shadertoy ``mainImage``.

    Returns the rewritten shader, or ``None`` when *glsl* is not a recognizable
    legacy fragment shader (already has ``mainImage``, no ``main()``, or never
    writes ``gl_FragColor``). The transform is textual and best-effort: when it
    produces something that still fails validation, the caller falls back to the
    LLM port stage.  ``gl_FragCoord`` is rewritten to ``vec4(fragCoord, 0.0,
    1.0)``, so ``.xy`` stays correct but ``.z``/``.w`` become hardcoded ``0.0``/
    ``1.0`` (depth/clip channels are lost); such shaders still compile, so this
    is not caught by re-validation. Only ``gl_FragData[0]`` is rewritten; MRT
    indices (``gl_FragData[1]``+) force fallback to the LLM port stage.
    """
    if "void mainImage" in glsl:
        return None
    if not _MAIN_SIG_RE.search(glsl):
        return None
    if "gl_FragColor" not in glsl and "gl_FragData" not in glsl:
        return None
    wrapped = _MAIN_SIG_RE.sub(
        "void mainImage(out vec4 fragColor, in vec2 fragCoord)", glsl, count=1
    )
    wrapped = re.sub(r"\bgl_FragColor\b", "fragColor", wrapped)
    wrapped = re.sub(r"\bgl_FragData\s*\[\s*0\s*\]", "fragColor", wrapped)
    if re.search(r"\bgl_FragData\s*\[", wrapped):
        return None
    # gl_FragCoord (vec4) -> vec4(fragCoord, 0.0, 1.0); a trailing `.xy` swizzle
    # on the vec4 literal stays valid GLSL.
    wrapped = re.sub(r"\bgl_FragCoord\b", "vec4(fragCoord, 0.0, 1.0)", wrapped)
    return wrapped


def _llm_port_to_shadertoy(
    source: str, *, llm_client: "Callable | None" = None
) -> str | None:
    """Ask the LLM to rewrite arbitrary GLSL as a Shadertoy ``mainImage`` shader.

    Reuses ``generate_llm_glsl_refinement`` with a port instruction so we do not
    duplicate the GLSL parse/normalize chain. Returns the normalized GLSL string
    or ``None`` on any failure.
    """
    from p2s_agent.core.candidates.llm_scene import generate_llm_glsl_refinement

    try:
        result = generate_llm_glsl_refinement(
            current_glsl=source,
            metrics={},
            quality_router={"final_score": 0.0},
            extra_feedback=[
                "[PORT] The shader above is NOT in Shadertoy format. Rewrite it "
                "as a Shadertoy shader with a `void mainImage(out vec4 fragColor, "
                "in vec2 fragCoord)` entry point, preserving the original visual "
                "intent. Map gl_FragColor->fragColor and gl_FragCoord.xy->fragCoord."
            ],
            fresh_start=False,
            llm_client=llm_client,
        )
    except Exception:
        logger.warning("seed LLM port failed", exc_info=True)
        return None
    if not result:
        return None
    result.pop("_io", None)
    return result.get("glsl") or None


def adapt_seed_glsl(
    source: str,
    *,
    llm_client: "Callable | None" = None,
) -> SeedAdaptResult:
    """Adapt an arbitrary GLSL string into renderable Shadertoy GLSL.

    Strategy (design decision 5): deterministic normalize -> deterministic wrap
    -> LLM port fallback. Each stage's output is re-checked with
    ``validate_shader_static``; the first that passes wins.
    """
    if not source or not source.strip():
        return SeedAdaptResult(
            glsl="", valid=False, adapted_by="failed", errors=["seed GLSL is empty"]
        )

    # Stage 1: normalize (strip markdown / conflicting uniforms / #version, etc.)
    normalized = normalize_shadertoy_glsl(source)
    static = validate_shader_static(normalized.glsl)
    if static["valid"]:
        return SeedAdaptResult(
            glsl=normalized.glsl,
            valid=True,
            adapted_by="normalized",
            warnings=list(normalized.warnings),
        )

    # Stage 2: deterministic wrap of a legacy main() shader, then re-normalize.
    wrapped = _wrap_legacy_main(normalized.glsl)
    if wrapped is not None:
        renorm = normalize_shadertoy_glsl(wrapped)
        if validate_shader_static(renorm.glsl)["valid"]:
            return SeedAdaptResult(
                glsl=renorm.glsl,
                valid=True,
                adapted_by="wrapped",
                warnings=list(normalized.warnings)
                + list(renorm.warnings)
                + ["wrapped_legacy_main"],
            )

    # Stage 3: LLM port fallback.
    ported = _llm_port_to_shadertoy(source, llm_client=llm_client)
    if ported is not None and validate_shader_static(ported)["valid"]:
        return SeedAdaptResult(
            glsl=ported, valid=True, adapted_by="llm_ported", warnings=["llm_ported"]
        )

    return SeedAdaptResult(
        glsl=normalized.glsl,
        valid=False,
        adapted_by="failed",
        warnings=list(normalized.warnings),
        errors=list(static["errors"]),
    )


def build_seed_candidate(
    glsl: str,
    *,
    adapted_by: str = "normalized",
    warnings: "list[str] | None" = None,
) -> CandidateRecord:
    """Wrap adapted seed GLSL as the single selected GLSL candidate.

    The seed path never runs ``select_best_candidate``; ``priority`` is purely
    cosmetic (scoreboard ordering / display).
    """
    return CandidateRecord(
        id="seed_0",
        source="seed",
        enabled=True,
        priority=100,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl=glsl,
        compile_errors=[],
        final_score=0.0,
        selected=True,
        reason=[f"seed shader (adapted_by={adapted_by})"],
        glsl_metadata={"adapted_by": adapted_by, "warnings": list(warnings or [])},
    )

"""Candidate pool for PNG-to-Shader: generation, validation, compilation,
selection, and scoreboard assembly. Split out of graph.py (2026-06-11);
behavior is unchanged."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from app.candidates.baseline import generate_baseline_candidate
from app.candidates.decompose import generate_decompose_candidate
from app.candidates.fallback import generate_fallback_candidate
from app.candidates.llm_scene import Implementation, generate_llm_scene_candidate
from app.candidates.rule import generate_rule_candidate
from app.dsl.compiler import compile_dsl
from app.dsl.validator import validate_dsl
from app.services.shader_validator import validate_shader

# CV candidate — may not be available during test runs
try:
    from app.candidates.cv import generate_cv_candidate
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """Record for a single candidate in the pool."""

    id: str                      # e.g. "baseline_0", "rule_0", "cv_0", "fallback_0"
    source: str                  # "baseline", "rule", "cv", "fallback", "llm"
    enabled: bool
    priority: int
    dsl: dict | None
    output_kind: str
    validation_valid: bool
    validation_errors: list[str]
    compile_success: bool
    compile_glsl: str
    compile_errors: list[str]
    final_score: float           # 0.0 initially, updated by scorer
    selected: bool
    objective_metrics: dict = field(default_factory=dict)
    quality_router: dict | None = None
    render_path: str | None = None
    reason: list[str] = field(default_factory=list)
    llm_io: dict | None = None
    glsl_metadata: dict = field(default_factory=dict)
    # Structured per-source generation error, surfaced on the scoreboard so a
    # source that raised is visible (with reason) instead of silently dropped.
    # Shape: {"source": str, "error_type": str, "message": str}.
    error: dict | None = None


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def run_candidate_pool(
    preprocess: dict,
    input_spec: dict,
    *,
    image_path: "Path | None" = None,
    llm_image_path: "Path | None" = None,
    llm_enabled: bool = False,
    llm_implementation: Implementation = "auto",
    cv_enabled: bool = True,
    canvas_width: int = 512,
    canvas_height: int = 512,
) -> list[CandidateRecord]:
    """Generate, validate, and compile all enabled candidates.

    Steps:
    1. Generate candidates from all enabled sources.
    2. Validate each DSL with validate_dsl().
    3. Compile valid DSLs with compile_dsl().
    4. Return all records (failures are included, not silently dropped).

    Args:
        preprocess: Dict of preprocessed image features.
        input_spec: Input specification dict.
        image_path: Optional path to the source image; passed to the CV
            candidate so it can do real shape detection from the actual
            pixels rather than falling back to preprocess heuristics.
        llm_image_path: Optional model-facing PNG; when omitted, the source
            image is sent to the LLM candidate generator.
        llm_enabled: Whether to call the LLM candidate generator.
        llm_implementation: Which LLM implementation to request. ``auto``
            chooses DSL for shape/icon-like inputs and Shadertoy GLSL for
            texture/photo-like inputs.
        cv_enabled: Whether to attempt the CV candidate.
        canvas_width: Target canvas width.
        canvas_height: Target canvas height.

    Returns:
        List of CandidateRecord, one per candidate attempted.
    """
    candidates_raw: list[tuple[str, str, int, dict | None, str]] = []

    # Structured per-source errors. When a source raises we capture a compact
    # record (source name + exception type + short message — NOT the full
    # traceback) and tag it onto the matching CandidateRecord so the scoreboard
    # can show *why* a candidate is missing/failed, instead of the failure being
    # silently dropped to a log line only. The full traceback is still logged
    # via logger.warning(exc_info=True). Maps cand_id -> structured error dict.
    raw_errors: dict[str, dict] = {}

    def _record_error(source: str, exc: Exception) -> dict:
        return {
            "source": source,
            "error_type": type(exc).__name__,
            "message": str(exc) or type(exc).__name__,
        }

    # 1a. Baseline — always first
    try:
        baseline_dsl = generate_baseline_candidate(preprocess, canvas_width, canvas_height)
        priority = baseline_dsl.get("_meta", {}).get("priority", 0)
        logger.info(
            "candidate generated: source=baseline layers=%d priority=%d",
            len(baseline_dsl.get("layers", []) or []),
            priority,
        )
    except Exception as exc:
        logger.warning("baseline candidate failed", exc_info=True)
        raw_errors["baseline_0"] = _record_error("baseline", exc)
        baseline_dsl = None
        priority = 0
    candidates_raw.append(("baseline_0", "baseline", priority, baseline_dsl, "dsl"))

    # 1b. Rule — always run
    try:
        rule_dsl = generate_rule_candidate(preprocess, canvas_width, canvas_height)
        rule_dsl["_meta"] = {"source": "rule", "priority": 1}
        logger.info(
            "candidate generated: source=rule layers=%d",
            len(rule_dsl.get("layers", []) or []),
        )
    except Exception as exc:
        logger.warning("rule candidate failed", exc_info=True)
        raw_errors["rule_0"] = _record_error("rule", exc)
        rule_dsl = None
    candidates_raw.append(("rule_0", "rule", 1, rule_dsl, "dsl"))

    # 1b2. Decompose — measured-geometry candidate (needs opencv)
    if image_path is not None:
        try:
            dec_dsl = generate_decompose_candidate(
                preprocess, image_path, canvas_width, canvas_height
            )
            if dec_dsl is not None:
                candidates_raw.append(("decompose_0", "decompose", 1, dec_dsl, "dsl"))
                logger.info(
                    "candidate generated: source=decompose layers=%d",
                    len(dec_dsl.get("layers", []) or []),
                )
        except Exception as exc:
            logger.warning("decompose candidate failed", exc_info=True)
            # decompose appends no placeholder on success-with-None; on a raise
            # we add a visible failed-candidate record so it is not dropped.
            raw_errors["decompose_0"] = _record_error("decompose", exc)
            candidates_raw.append(("decompose_0", "decompose", 1, None, "dsl"))

    # 1c. CV — optional
    if cv_enabled and _CV_AVAILABLE:
        try:
            cv_dsl = generate_cv_candidate(preprocess, image_path=image_path, canvas_width=canvas_width, canvas_height=canvas_height)  # type: ignore[name-defined]
            if cv_dsl is not None:
                raw_cv_priority = cv_dsl.get("_meta", {}).get("priority", 2) if isinstance(cv_dsl, dict) else 2
                # CV candidate may use string priorities ("high"/"low"/"disabled") — normalise to int
                if isinstance(raw_cv_priority, str):
                    _cv_priority_map = {"high": 2, "low": 3, "disabled": 99}
                    cv_priority = _cv_priority_map.get(raw_cv_priority, 2)
                else:
                    cv_priority = int(raw_cv_priority)
                candidates_raw.append(("cv_0", "cv", cv_priority, cv_dsl, "dsl"))
                logger.info(
                    "candidate generated: source=cv layers=%d priority=%d",
                    len((cv_dsl or {}).get("layers", []) or []),
                    cv_priority,
                )
        except Exception as exc:
            logger.warning("CV candidate failed", exc_info=True)
            # CV appends no placeholder on success-with-None; on a raise we add a
            # visible failed-candidate record so it is not silently dropped.
            raw_errors["cv_0"] = _record_error("cv", exc)
            candidates_raw.append(("cv_0", "cv", 2, None, "dsl"))

    # 1d. LLM — optional
    _llm_io: dict | None = None
    _llm_attempted = bool(llm_enabled)
    _llm_empty_kind = "glsl" if llm_implementation == "shadertoy_glsl" else "dsl"
    try:
        llm_candidate = generate_llm_scene_candidate(
            preprocess,
            canvas_width,
            canvas_height,
            image_path=llm_image_path or image_path,
            llm_enabled=llm_enabled,
            implementation=llm_implementation,
        )
        if llm_candidate is not None:
            llm_meta = llm_candidate.get("_meta", {}) if isinstance(llm_candidate, dict) else {}
            llm_priority = llm_meta.get("priority", 3) if isinstance(llm_meta, dict) else 3
            output_kind = llm_meta.get("output_kind", "dsl") if isinstance(llm_meta, dict) else "dsl"
            _llm_io = llm_candidate.pop("_io", None) if isinstance(llm_candidate, dict) else None
            candidates_raw.append(("llm_0", "llm", int(llm_priority), llm_candidate, str(output_kind)))
            logger.info(
                "candidate generated: source=llm output_kind=%s priority=%d",
                str(output_kind),
                int(llm_priority),
            )
        elif llm_enabled:
            candidates_raw.append(("llm_0", "llm", 3, None, _llm_empty_kind))
            logger.info("candidate generated: source=llm returned none")
    except Exception as exc:
        logger.warning("LLM candidate failed", exc_info=True)
        raw_errors["llm_0"] = _record_error("llm", exc)
        if _llm_attempted:
            candidates_raw.append(("llm_0", "llm", 3, None, _llm_empty_kind))

    # 1e. Fallback — always last
    try:
        fallback_dsl = generate_fallback_candidate(preprocess, canvas_width, canvas_height)
        fb_priority = fallback_dsl.get("_meta", {}).get("priority", 99)
        logger.info(
            "candidate generated: source=fallback layers=%d",
            len(fallback_dsl.get("layers", []) or []),
        )
    except Exception as exc:
        logger.warning("fallback candidate failed", exc_info=True)
        raw_errors["fallback_0"] = _record_error("fallback", exc)
        fallback_dsl = None
        fb_priority = 99
    candidates_raw.append(("fallback_0", "fallback", fb_priority, fallback_dsl, "dsl"))

    # 2 & 3. Validate and compile each candidate
    records: list[CandidateRecord] = []
    for cand_id, source, priority, dsl, output_kind in candidates_raw:
        if dsl is None:
            record = CandidateRecord(
                id=cand_id,
                source=source,
                enabled=False,
                priority=priority,
                dsl=None,
                output_kind=output_kind,
                validation_valid=False,
                validation_errors=["Candidate generator returned None"],
                compile_success=False,
                compile_glsl="",
                compile_errors=[],
                final_score=0.0,
                selected=False,
                reason=["generator returned None"],
                error=raw_errors.get(cand_id),
            )
            records.append(record)
            continue

        if output_kind == "glsl":
            glsl = dsl.get("glsl", "") if isinstance(dsl, dict) else ""
            validation_result = validate_shader(glsl) if glsl else {
                "valid": False,
                "errors": ["LLM GLSL candidate missing glsl"],
                "warnings": [],
            }
            validation_valid = bool(validation_result.get("valid"))
            validation_errors = list(validation_result.get("errors", []))
            compile_success = validation_valid
            compile_glsl = glsl
            compile_errors = validation_errors if not validation_valid else []
            dsl_payload = None
            glsl_metadata = dsl.get("glsl_metadata", {}) if isinstance(dsl, dict) else {}
            if validation_result.get("warnings"):
                glsl_metadata = {
                    **(glsl_metadata if isinstance(glsl_metadata, dict) else {}),
                    "validation_warnings": validation_result.get("warnings", []),
                }
        else:
            # Validate
            val_result = validate_dsl(dsl)
            validation_valid = val_result.valid
            validation_errors = val_result.errors

            # Compile (only if valid)
            if validation_valid:
                compile_result = compile_dsl(dsl)
                compile_success = compile_result.success
                compile_glsl = compile_result.glsl
                compile_errors = compile_result.errors
            else:
                compile_success = False
                compile_glsl = ""
                compile_errors = []
            dsl_payload = dsl
            glsl_metadata = {}

        # Snapshot the initial compile result onto the LLM I/O record so the
        # "Initial Call" tab in the frontend can preview exactly what the first
        # LLM call produced, independent of any later refinement that mutates
        # the candidate's compile_glsl in place.
        if cand_id == "llm_0" and isinstance(_llm_io, dict):
            _llm_io.setdefault("compile_glsl", compile_glsl)

        record = CandidateRecord(
            id=cand_id,
            source=source,
            enabled=True,
            priority=priority,
            dsl=dsl_payload,
            output_kind=output_kind,
            validation_valid=validation_valid,
            validation_errors=validation_errors,
            compile_success=compile_success,
            compile_glsl=compile_glsl,
            compile_errors=compile_errors,
            final_score=0.0,
            selected=False,
            reason=[],
            llm_io=_llm_io if cand_id == "llm_0" else None,
            glsl_metadata=glsl_metadata if isinstance(glsl_metadata, dict) else {},
            error=raw_errors.get(cand_id),
        )
        logger.info(
            "candidate compiled: id=%s valid=%s compile_ok=%s chars=%d",
            cand_id,
            validation_valid,
            compile_success,
            len(compile_glsl or ""),
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_best_candidate(
    candidates: list[CandidateRecord],
    *,
    prefer_output_kind: str | None = None,
) -> CandidateRecord | None:
    """Select the best compilable candidate from the pool.

    Selection rules (in order):
    1. Among candidates where compile_success=True, prefer higher final_score
       once candidates have been scored.
    2. Tie-break: prefer by priority (lower = better).
    3. Tie-break: prefer more-detailed candidate (longer GLSL = more features).
    4. If no compilable candidate: return the fallback if present, else None.
    5. Mark selected candidate with selected=True, add reason.

    Args:
        candidates: List of CandidateRecord from run_candidate_pool().

    Returns:
        The selected CandidateRecord, or None if pool is empty.
    """
    compilable = [c for c in candidates if c.compile_success]

    if compilable:
        if prefer_output_kind:
            preferred = [c for c in compilable if c.output_kind == prefer_output_kind]
            if preferred:
                preferred.sort(key=lambda c: (-c.final_score, c.priority, -len(c.compile_glsl)))
                selected = preferred[0]
                selected.selected = True
                selected.reason.append(
                    f"selected preferred {prefer_output_kind}: score={selected.final_score:.4f}, priority={selected.priority}, glsl_len={len(selected.compile_glsl)}"
                )
                logger.info(
                    "candidate selected: id=%s source=%s score=%.4f priority=%d (preferred=%s)",
                    selected.id,
                    selected.source,
                    float(selected.final_score),
                    selected.priority,
                    prefer_output_kind,
                )
                return selected

        has_scores = any(c.final_score > 0.0 for c in compilable)
        if has_scores:
            compilable.sort(key=lambda c: (-c.final_score, c.priority, -len(c.compile_glsl)))
        else:
            compilable.sort(key=lambda c: (c.priority, -len(c.compile_glsl)))
        selected = compilable[0]
        selected.selected = True
        selected.reason.append(
            f"selected: score={selected.final_score:.4f}, priority={selected.priority}, glsl_len={len(selected.compile_glsl)}"
        )
        logger.info(
            "candidate selected: id=%s source=%s score=%.4f priority=%d",
            selected.id,
            selected.source,
            float(selected.final_score),
            selected.priority,
        )
        return selected

    # No compilable candidate — try fallback
    fallback_candidates = [c for c in candidates if c.source == "fallback"]
    if fallback_candidates:
        selected = fallback_candidates[0]
        selected.selected = True
        selected.reason = ["selected as fallback: no compilable candidate available"]
        logger.info(
            "candidate selected: id=%s source=fallback (no compilable candidate)",
            selected.id,
        )
        return selected

    return None


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

def build_scoreboard(candidates: list[CandidateRecord]) -> dict:
    """Build a scoreboard summary dict from the candidate pool.

    Args:
        candidates: List of CandidateRecord from run_candidate_pool().

    Returns:
        Dict with summary statistics and per-candidate entries.
    """
    total = len(candidates)
    enabled = sum(1 for c in candidates if c.enabled)
    compiled = sum(1 for c in candidates if c.compile_success)
    selected_record = next((c for c in candidates if c.selected), None)
    selected_id = selected_record.id if selected_record else None

    candidate_entries = [
        {
            "id": c.id,
            "source": c.source,
            "output_kind": c.output_kind,
            "enabled": c.enabled,
            "priority": c.priority,
            "validation_valid": c.validation_valid,
            "compile_success": c.compile_success,
            "previewable": bool(c.compile_success and c.compile_glsl.strip()),
            "score_status": _score_status(c),
            "final_score": c.final_score,
            "objective_metrics": c.objective_metrics,
            "quality_router": c.quality_router,
            "quality_status": c.quality_router.get("status") if c.quality_router else None,
            "quality_band": c.quality_router.get("quality_band") if c.quality_router else None,
            "selected": c.selected,
            "reason": c.reason,
            "validation_errors": c.validation_errors,
            "compile_errors": c.compile_errors,
            "compile_glsl": c.compile_glsl,
            "llm_io": c.llm_io,
            "glsl_metadata": c.glsl_metadata,
            "error": c.error,
        }
        for c in candidates
    ]

    # Aggregate structured per-source generation errors so the scoreboard can
    # show *why* a candidate source is missing/failed (source + exception type
    # + short message), instead of the failure being a silent log-only drop.
    source_errors = [c.error for c in candidates if c.error]

    return {
        "total": total,
        "enabled": enabled,
        "compiled": compiled,
        "selected_id": selected_id,
        "candidates": candidate_entries,
        "source_errors": source_errors,
    }


def _candidate_detail(candidate: CandidateRecord) -> dict:
    return {
        "id": candidate.id,
        "source": candidate.source,
        "output_kind": candidate.output_kind,
        "enabled": candidate.enabled,
        "priority": candidate.priority,
        "dsl": candidate.dsl,
        "validation_valid": candidate.validation_valid,
        "validation_errors": candidate.validation_errors,
        "compile_success": candidate.compile_success,
        "compile_errors": candidate.compile_errors,
        "compile_glsl": candidate.compile_glsl,
        "final_score": candidate.final_score,
        "selected": candidate.selected,
        "objective_metrics": candidate.objective_metrics,
        "quality_router": candidate.quality_router,
        "render_path": candidate.render_path,
        "reason": candidate.reason,
        "shader_chars": len(candidate.compile_glsl),
        "llm_io": candidate.llm_io,
        "glsl_metadata": candidate.glsl_metadata,
        "score_status": _score_status(candidate),
    }


def _score_status(candidate: CandidateRecord) -> str:
    if not candidate.enabled:
        return "disabled"
    if not candidate.validation_valid:
        return "validation_failed"
    if not candidate.compile_success:
        return "compile_failed"
    if candidate.quality_router is not None:
        if candidate.objective_metrics.get("backend_rasterized") is False:
            return "preview_only"
        return "scored"
    return "pending"

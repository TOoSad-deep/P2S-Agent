"""GLSL `#define` parameter optimizer.

Mirrors the DSL optimizer (``optimizer.py``) but operates on Shadertoy-style
GLSL: parses scalar and vec ``#define`` lines, perturbs their numeric values
with coordinate descent, re-renders via the WebGL backend, and keeps changes
that improve the objective score.

Only numeric ``#define`` values are touched — uniforms, code lines, loop
bounds, and animation parameters are intentionally left alone.
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from app.metrics.compute import compute_objective_metrics
from app.metrics.quality_router import compute_final_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

GlslType = Literal["float", "vec2", "vec3", "vec4"]

# Names whose values are loop counts / animation control: the optimizer must
# leave these alone. Changing loop bounds risks compile/runtime blow-ups, and
# animation params don't have a meaningful gradient against a static
# reference image.
_DENYLIST_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(^|_)iter(s|ations)?(_|$)",
        r"(^|_)sample(s)?(_|$)",
        r"(^|_)count(_|$)",
        r"(^|_)side(s)?(_|$)",
        r"(^|_)loop(_|$)",
        r"(^|_)time(_|$)",
        r"(^|_)speed(_|$)",
        r"(^|_)freq(uency)?(_|$)",
    ]
]

_COLOR_RE = re.compile(
    r"(^|_)(color|col|tint|rgb|hue)(_|$)|(_r|_g|_b)$|^bg_|_color$",
    re.IGNORECASE,
)

_DEFINE_RE = re.compile(r"^(\s*#define\s+)(\w+)(\s+)(.+?)(\s*)$")
_VEC_RE = re.compile(r"^vec([234])\s*\(\s*([^)]*)\s*\)\s*$")


@dataclass
class GlslParam:
    name: str
    glsl_type: GlslType
    value: float | list[float]  # scalar for "float", list for vec*
    line_idx: int               # index into glsl.split("\n")


def parse_glsl_defines(glsl: str) -> list[GlslParam]:
    """Extract optimizable ``#define`` parameters from a Shadertoy GLSL string.

    Skips:
      * bare integer literals (e.g. ``#define ITERATIONS 8``) — these are loop
        counts per the generator's float-literal rule.
      * names matching loop/animation patterns (``ITER``, ``SAMPLES``,
        ``COUNT``, ``SIDES``, ``LOOP``, ``TIME``, ``SPEED``, ``FREQ``).
      * malformed or non-numeric values.

    Returns:
        List of ``GlslParam``. Empty list when the shader has no perturbable
        ``#define``.
    """
    params: list[GlslParam] = []
    for idx, line in enumerate(glsl.split("\n")):
        m = _DEFINE_RE.match(line)
        if not m:
            continue
        name = m.group(2)
        value_str = m.group(4).strip()

        if any(p.search(name) for p in _DENYLIST_PATTERNS):
            continue

        vm = _VEC_RE.match(value_str)
        if vm:
            expected = int(vm.group(1))
            try:
                components = [
                    float(s.strip()) for s in vm.group(2).split(",") if s.strip()
                ]
            except ValueError:
                continue
            if len(components) != expected or any(
                not math.isfinite(c) for c in components
            ):
                continue
            params.append(
                GlslParam(
                    name=name,
                    glsl_type=f"vec{expected}",  # type: ignore[arg-type]
                    value=components,
                    line_idx=idx,
                )
            )
            continue

        # Scalar: must contain a decimal point. The generator's prompt rule
        # treats bare ints as loop counts (``#define POWER 2`` is forbidden;
        # ``#define POWER 2.0`` is required for floats), so this acts as a
        # reliable float-vs-int discriminator.
        if "." not in value_str:
            continue
        try:
            scalar = float(value_str)
        except ValueError:
            continue
        if not math.isfinite(scalar):
            continue
        params.append(
            GlslParam(
                name=name,
                glsl_type="float",
                value=scalar,
                line_idx=idx,
            )
        )

    return params


def update_glsl_define(
    glsl: str,
    name: str,
    new_value: float | list[float],
    glsl_type: GlslType,
) -> str:
    """Rewrite a single ``#define NAME ...`` line. No-op if ``name`` not found."""
    lines = glsl.split("\n")
    name_re = re.compile(rf"^(\s*#define\s+{re.escape(name)}\s+).+?(\s*)$")
    for i, line in enumerate(lines):
        m = name_re.match(line)
        if not m:
            continue
        prefix = m.group(1)
        if glsl_type == "float":
            v = float(new_value)  # type: ignore[arg-type]
            lines[i] = f"{prefix}{_fmt_float(v)}"
        else:
            comps = list(new_value)  # type: ignore[arg-type]
            inner = ", ".join(_fmt_float(float(v)) for v in comps)
            lines[i] = f"{prefix}{glsl_type}({inner})"
        return "\n".join(lines)
    return glsl


def _fmt_float(v: float) -> str:
    """Format a float so GLSL always sees a float literal (decimal point)."""
    if v == int(v):
        return f"{float(int(v)):.1f}"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


def _is_color_param(name: str) -> bool:
    return bool(_COLOR_RE.search(name))


def _clamp_value(candidate: float, anchor: float, name: str) -> float:
    """Clamp ``candidate`` using bounds inferred from ``anchor`` and ``name``.

    Color components and values already in [0, 1] stay in [0, 1]. Anything else
    is clamped to [0, max(anchor*2, 0.01)] so the optimizer cannot drift wildly
    above the original magnitude.
    """
    if _is_color_param(name) or 0.0 <= anchor <= 1.0:
        return max(0.0, min(1.0, candidate))
    upper = max(anchor * 2.0, 0.01)
    return max(0.0, min(upper, candidate))


def _step_for(value: float, scale: float) -> float:
    """Return an absolute step for normalized values, relative elsewhere."""
    if 0.0 <= value <= 1.0:
        return scale
    return scale * max(abs(value), 1.0)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GlslOptimizeStep:
    iteration: int
    param_name: str
    old_value: float | list[float]
    new_value: float | list[float]
    score_before: float
    score_after: float
    accepted: bool


@dataclass
class GlslOptimizeResult:
    best_glsl: str
    initial_score: float
    best_score: float
    improved: bool
    iterations_run: int
    loss_curve: list[float]
    optimizer_log: list[GlslOptimizeStep]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_glsl(
    glsl: str,
    ref_path: "str | Path",
    render_glsl_fn: "Callable[[str], Path | None]",
    *,
    max_shader_chars: int = 12000,
) -> float:
    """Render ``glsl`` via ``render_glsl_fn`` and compute the final score."""
    render_path = render_glsl_fn(glsl)
    if render_path is None:
        return 0.0
    metrics = compute_objective_metrics(
        ref_path,
        render_path,
        shader_chars=len(glsl),
        max_shader_chars=max_shader_chars,
    )
    return compute_final_score(metrics)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def optimize_glsl_candidate(
    glsl: str,
    ref_path: "str | Path",
    render_glsl_fn: "Callable[[str], Path | None]",
    *,
    max_iterations: int = 5,
    scale: float = 0.05,
    max_shader_chars: int = 12000,
    seed: "int | None" = None,
) -> GlslOptimizeResult:
    """Coordinate-descent optimize numeric ``#define`` values in ``glsl``.

    Round-robin through each parsed parameter, try ``+scale`` and ``-scale``
    (vec params get a uniform shift on all components — same pattern as the
    DSL optimizer), keep whichever direction improves the score. Stops after
    ``max_iterations`` total renders.
    """
    if seed is not None:
        random.seed(seed)

    best_glsl = glsl
    initial_score = score_glsl(
        best_glsl, ref_path, render_glsl_fn, max_shader_chars=max_shader_chars
    )
    best_score = initial_score
    loss_curve: list[float] = [initial_score]
    log: list[GlslOptimizeStep] = []

    collected = parse_glsl_defines(best_glsl)
    if not collected:
        return GlslOptimizeResult(
            best_glsl=best_glsl,
            initial_score=initial_score,
            best_score=best_score,
            improved=False,
            iterations_run=0,
            loss_curve=loss_curve,
            optimizer_log=log,
        )

    step_count = 0
    param_idx = 0
    params_since_improvement = 0

    while step_count < max_iterations and collected:
        param = collected[param_idx % len(collected)]
        param_idx += 1

        if param.glsl_type == "float":
            original = float(param.value)  # type: ignore[arg-type]
            step = _step_for(original, scale)
            plus = _clamp_value(original + step, original, param.name)
            minus = _clamp_value(original - step, original, param.name)
            trials: list[float | list[float]] = [plus, minus]
        else:
            original_components = list(param.value)  # type: ignore[arg-type]
            plus = [
                _clamp_value(v + _step_for(v, scale), v, param.name)
                for v in original_components
            ]
            minus = [
                _clamp_value(v - _step_for(v, scale), v, param.name)
                for v in original_components
            ]
            trials = [plus, minus]

        best_trial_value: float | list[float] | None = None
        best_trial_score = best_score

        for trial_value in trials:
            if step_count >= max_iterations:
                break
            if trial_value == param.value:
                continue
            trial_glsl = update_glsl_define(
                best_glsl, param.name, trial_value, param.glsl_type
            )
            if trial_glsl == best_glsl:
                continue
            trial_score = score_glsl(
                trial_glsl,
                ref_path,
                render_glsl_fn,
                max_shader_chars=max_shader_chars,
            )
            step_count += 1
            accepted = trial_score > best_trial_score
            log.append(
                GlslOptimizeStep(
                    iteration=step_count - 1,
                    param_name=param.name,
                    old_value=param.value,
                    new_value=trial_value,
                    score_before=best_score,
                    score_after=trial_score,
                    accepted=accepted,
                )
            )
            if accepted:
                best_trial_value = trial_value
                best_trial_score = trial_score

        if best_trial_value is not None:
            best_glsl = update_glsl_define(
                best_glsl, param.name, best_trial_value, param.glsl_type
            )
            best_score = best_trial_score
            loss_curve.append(best_score)
            collected = parse_glsl_defines(best_glsl)
            params_since_improvement = 0
        else:
            params_since_improvement += 1

        # Stop once a full pass over every param produced no accepted trial
        # since the last improvement.
        if collected and params_since_improvement >= len(collected) and step_count > 0:
            break

    return GlslOptimizeResult(
        best_glsl=best_glsl,
        initial_score=initial_score,
        best_score=best_score,
        improved=best_score > initial_score,
        iterations_run=len(log),
        loss_curve=loss_curve,
        optimizer_log=log,
    )


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_glsl_optimization_artifacts(result: GlslOptimizeResult) -> dict:
    """Return a JSON-serializable summary of a ``GlslOptimizeResult``."""
    steps_accepted = sum(1 for s in result.optimizer_log if s.accepted)
    steps_rejected = len(result.optimizer_log) - steps_accepted
    return {
        "mode": "glsl_defines",
        "initial_score": result.initial_score,
        "best_score": result.best_score,
        "improved": result.improved,
        "iterations_run": result.iterations_run,
        "loss_curve": result.loss_curve,
        "steps_accepted": steps_accepted,
        "steps_rejected": steps_rejected,
        "steps": [
            {
                "iteration": s.iteration,
                "param_name": s.param_name,
                "old_value": s.old_value,
                "new_value": s.new_value,
                "score_before": s.score_before,
                "score_after": s.score_after,
                "accepted": s.accepted,
            }
            for s in result.optimizer_log
        ],
    }

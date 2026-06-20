"""Parameter optimizer for PNG-to-Shader DSL.

Adjusts numeric parameters (center, radius, size, opacity, color stops,
glow settings, gradient settings) within the compiled DSL to minimize
the visual distance between the render and the reference image.

Operates only on parameter values — never changes primitive types,
layer count, or layer order.
"""

from __future__ import annotations

import copy
import math
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from p2s_agent.core.dsl.compiler import compile_dsl
from p2s_agent.core.metrics.compute import compute_objective_metrics
from p2s_agent.core.metrics.quality_router import compute_final_score

# ---------------------------------------------------------------------------
# Allowed optimizable parameter sets
# ---------------------------------------------------------------------------

OPTIMIZABLE_LAYER_PARAMS = {
    "center", "radius", "size", "opacity", "edge_softness"
}

# Canonical DSL fill accessors (see p2s_agent.core.dsl.validator):
#   * radialGradient uses 'center' (a vec2 in [0, 1])
#   * linearGradient uses 'direction' (a vec2 that may be negative, e.g.
#     [-1.0, 0.0] for a right-to-left gradient)
OPTIMIZABLE_FILL_PARAMS = {
    "center",      # radialGradient center
    "direction",   # linearGradient direction (sign-bearing axis)
}

# Fill params whose components are a signed axis/direction and must NOT be
# clamped to [0, 1] (that would degenerate a negative axis to 0).
DIRECTION_PARAM_KEYS = {"direction"}

OPTIMIZABLE_EFFECT_PARAMS = {
    "radius", "intensity",  # glow
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OptimizeStep:
    iteration: int
    param_path: str    # e.g. "layer[circle_01].params.radius"
    old_value: Any
    new_value: Any
    score_before: float
    score_after: float
    accepted: bool


@dataclass
class OptimizeResult:
    best_dsl: dict
    initial_score: float
    best_score: float
    improved: bool
    iterations_run: int
    loss_curve: list[float]          # score at each accepted step
    optimizer_log: list[OptimizeStep]
    protected_aspects_violations: list[str]
    # Why the optimizer halted: "max_iterations" (hit the cap),
    # "converged_no_improvement" (a full sweep produced no accepted step),
    # or "no_params" (nothing optimizable to perturb).
    stop_reason: str = "max_iterations"


# Smallest score gain that counts as a real improvement. Perturbations below
# this threshold are treated as noise and rejected, so negligible/floating-point
# "improvements" don't churn the optimizer or block early-stop.
DEFAULT_ACCEPT_EPSILON = 1e-4


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def _perturb_scalar(
    value: float, scale: float = 0.05, *, rng: "random.Random | None" = None
) -> float:
    """Add Gaussian noise to a scalar value.

    Clamps to [0.0, 1.0] for values already in that range;
    otherwise clamps to [0.0, max(value * 2, 0.01)].
    """
    new_val = value + (rng or random).gauss(0, scale)
    if 0.0 <= value <= 1.0:
        return max(0.0, min(1.0, new_val))
    else:
        upper = max(value * 2, 0.01)
        return max(0.0, min(upper, new_val))


def _perturb_vec2(
    value: list[float],
    scale: float = 0.05,
    *,
    is_direction: bool = False,
    rng: "random.Random | None" = None,
) -> list[float]:
    """Perturb each element of a 2-element list.

    For position-style vec2 (e.g. a center) each component is clamped to the
    canonical [0.0, 1.0] range. For a gradient ``direction`` the components form
    a signed axis, so they are clamped to the symmetric [-1.0, 1.0] range — this
    preserves the sign and never collapses a negative axis to 0.0.
    """
    gauss = (rng or random).gauss
    lo = -1.0 if is_direction else 0.0
    return [
        max(lo, min(1.0, v + gauss(0, scale)))
        for v in value
    ]


def _is_direction_accessor(accessor_keys: list) -> bool:
    """True when the accessor points at a sign-bearing gradient direction."""
    return bool(accessor_keys) and accessor_keys[-1] in DIRECTION_PARAM_KEYS


# ---------------------------------------------------------------------------
# Nested dict/list accessor utilities
# ---------------------------------------------------------------------------

def _get_nested(d: Any, keys: list) -> Any:
    """Navigate a nested dict/list by a sequence of keys/indices."""
    current = d
    for k in keys:
        if isinstance(current, list):
            current = current[k]
        else:
            current = current[k]
    return current


def _set_nested(d: dict, keys: list, value: Any) -> dict:
    """Return a copy of *d* with the value at *keys* set to *value*.

    Only copies containers along the access path (structural sharing),
    which is much cheaper than a full deep copy for large DSL dicts.
    """
    if not keys:
        return d

    if isinstance(d, list):
        result = list(d)
    else:
        result = dict(d)

    current = result
    for k in keys[:-1]:
        child = current[k]
        if isinstance(child, list):
            child = list(child)
        elif isinstance(child, dict):
            child = dict(child)
        current[k] = child
        current = child

    last_key = keys[-1]
    current[last_key] = value
    return result


# ---------------------------------------------------------------------------
# Parameter collection
# ---------------------------------------------------------------------------

def _collect_optimizable_params(dsl: dict) -> list[tuple[str, list, Any]]:
    """Walk layers and collect all optimizable (path, accessor_keys, value) tuples.

    Returns:
        List of (param_path_str, accessor_keys, current_value) where
        accessor_keys is a list of keys/indices navigating into the DSL dict.
    """
    collected: list[tuple[str, list, Any]] = []
    layers = dsl.get("layers", [])

    for layer_idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue

        layer_id = layer.get("id", f"layer_{layer_idx}")
        base_keys = ["layers", layer_idx]

        # --- params dict ---
        params = layer.get("params")
        if isinstance(params, dict):
            for param_key in OPTIMIZABLE_LAYER_PARAMS - {"opacity"}:
                if param_key in params:
                    val = params[param_key]
                    path_str = f"layer[{layer_id}].params.{param_key}"
                    accessor = base_keys + ["params", param_key]
                    collected.append((path_str, accessor, val))

        # --- opacity ---
        if "opacity" in layer:
            val = layer["opacity"]
            if isinstance(val, (int, float)):
                path_str = f"layer[{layer_id}].opacity"
                accessor = base_keys + ["opacity"]
                collected.append((path_str, accessor, float(val)))

        # --- fill dict ---
        fill = layer.get("fill")
        if isinstance(fill, dict):
            fill_type = fill.get("type", "")
            for fill_param in OPTIMIZABLE_FILL_PARAMS:
                if fill_param in fill:
                    val = fill[fill_param]
                    path_str = f"layer[{layer_id}].fill.{fill_param}"
                    accessor = base_keys + ["fill", fill_param]
                    collected.append((path_str, accessor, val))

            # color stops: perturb stop positions
            if fill_type in ("linearGradient", "radialGradient"):
                stops = fill.get("stops", [])
                if isinstance(stops, list):
                    for stop_idx, stop in enumerate(stops):
                        if isinstance(stop, dict) and "position" in stop:
                            val = stop["position"]
                            if isinstance(val, (int, float)):
                                path_str = (
                                    f"layer[{layer_id}].fill.stops[{stop_idx}].position"
                                )
                                accessor = base_keys + [
                                    "fill", "stops", stop_idx, "position"
                                ]
                                collected.append((path_str, accessor, float(val)))

        # --- effects ---
        effects = layer.get("effects")
        if isinstance(effects, list):
            for effect_idx, effect in enumerate(effects):
                if not isinstance(effect, dict):
                    continue
                effect_type = effect.get("type", "")
                if effect_type == "glow":
                    for effect_param in OPTIMIZABLE_EFFECT_PARAMS:
                        if effect_param in effect:
                            val = effect[effect_param]
                            if isinstance(val, (int, float)):
                                path_str = (
                                    f"layer[{layer_id}].effects[{effect_idx}].{effect_param}"
                                )
                                accessor = base_keys + [
                                    "effects", effect_idx, effect_param
                                ]
                                collected.append((path_str, accessor, float(val)))

    return collected


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_dsl(
    dsl: dict,
    ref_path: "str | Path",
    render_fn: "Callable[[str], Path | None] | None" = None,
    *,
    render_dsl_fn: "Callable[[dict, str], Path | None] | None" = None,
) -> float:
    """Score a DSL dict against a reference image using the supplied render_fn.

    Returns a float in [0.0, 1.0].  Returns 0.0 when compilation or
    rendering fails.
    """
    compile_result = compile_dsl(dsl)
    if not compile_result.success:
        return 0.0

    if render_dsl_fn is not None:
        render_path = render_dsl_fn(dsl, compile_result.glsl)
    elif render_fn is not None:
        render_path = render_fn(compile_result.glsl)
    else:
        return 0.0
    if render_path is None:
        return 0.0

    metrics = compute_objective_metrics(ref_path, render_path)
    return compute_final_score(metrics)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_candidate(
    dsl: dict,
    ref_path: "str | Path",
    render_fn: "Callable[[str], Path | None] | None" = None,
    *,
    render_dsl_fn: "Callable[[dict, str], Path | None] | None" = None,
    max_iterations: int = 20,
    strategy: str = "random",
    protected_aspects: "list[str] | None" = None,
    seed: "int | None" = None,
    accept_epsilon: float = DEFAULT_ACCEPT_EPSILON,
) -> OptimizeResult:
    """Optimize DSL parameters to improve visual similarity to *ref_path*.

    Only numeric parameter values (center, radius, size, opacity, color stop
    positions, glow radius/intensity, gradient direction) are modified.
    Layer types, count, and order are never altered.

    Args:
        dsl:              Input DSL dict (not mutated).
        ref_path:         Path to the reference image.
        render_fn:        Callable accepting a GLSL string and returning a
                          rendered image Path (or None on failure).
        max_iterations:   Maximum number of perturbation evaluations.
        strategy:         "random" or "coordinate_descent".
        protected_aspects: Aspects that must not be changed (informational only;
                           enforced by not modifying types/structure).
        seed:             Optional random seed for reproducibility.
        accept_epsilon:   Minimum score gain for a perturbation to be accepted.
                          Gains below this are treated as noise and rejected, so
                          a full sweep of sub-epsilon trials triggers early-stop
                          (coordinate_descent) instead of running to the cap.

    Returns:
        OptimizeResult with the best DSL found and diagnostic information.
    """
    # Use a local RNG instance so the optimizer never mutates the process-global
    # random module — concurrent variant workers must not see each other's seed.
    rng = random.Random(seed)

    ref_path = Path(ref_path)
    best_dsl = copy.deepcopy(dsl)
    collected = _collect_optimizable_params(best_dsl)

    initial_score = score_dsl(
        best_dsl, ref_path, render_fn, render_dsl_fn=render_dsl_fn
    )
    best_score = initial_score
    loss_curve: list[float] = [initial_score]
    log: list[OptimizeStep] = []
    # Default: assume we ran to the cap; the coordinate-descent loop overrides
    # this when a full no-improvement sweep triggers convergence early-stop.
    stop_reason = "max_iterations"

    if strategy == "random":
        if not collected:
            stop_reason = "no_params"
        for i in range(max_iterations):
            if not collected:
                break

            # Pick a random optimizable param
            param_path, accessor_keys, current_value = rng.choice(collected)

            # Perturb the value
            if isinstance(current_value, list) and len(current_value) == 2:
                new_value = _perturb_vec2(
                    current_value,
                    is_direction=_is_direction_accessor(accessor_keys),
                    rng=rng,
                )
            elif isinstance(current_value, (int, float)):
                new_value = _perturb_scalar(float(current_value), rng=rng)
            else:
                continue  # Skip non-numeric params

            # Evaluate the perturbed DSL
            new_dsl = _set_nested(best_dsl, accessor_keys, new_value)
            new_score = score_dsl(
                new_dsl, ref_path, render_fn, render_dsl_fn=render_dsl_fn
            )

            # Acceptance epsilon: only a gain of at least ``accept_epsilon``
            # counts as a real improvement (noise-level gains are rejected).
            accepted = new_score >= best_score + accept_epsilon
            log.append(OptimizeStep(
                iteration=i,
                param_path=param_path,
                old_value=current_value,
                new_value=new_value,
                score_before=best_score,
                score_after=new_score,
                accepted=accepted,
            ))

            if accepted:
                best_dsl = new_dsl
                best_score = new_score
                loss_curve.append(new_score)
                # Re-collect params from the updated DSL
                collected = _collect_optimizable_params(best_dsl)

    elif strategy == "coordinate_descent":
        scale = 0.05
        step_count = 0
        param_idx = 0
        # Convergence tracker: how many consecutive params have been swept with
        # NO accepted improvement. Once this reaches len(collected) a full sweep
        # has produced nothing — the optimizer has converged and stops early.
        params_since_improvement = 0

        # Re-collect with fresh reference
        collected = _collect_optimizable_params(best_dsl)
        if not collected:
            stop_reason = "no_params"

        while step_count < max_iterations:
            if not collected:
                break

            param_idx = param_idx % len(collected)
            param_path, accessor_keys, current_value = collected[param_idx]

            best_direction_value = None
            best_direction_score = best_score

            # Try both +scale and -scale perturbations
            if isinstance(current_value, list) and len(current_value) == 2:
                # A gradient direction is a signed axis: clamp to [-1, 1] so a
                # negative component (e.g. [-1.0, 0.0]) keeps its sign instead of
                # degenerating to 0.0. Other vec2 params (center) stay in [0, 1].
                lo = -1.0 if _is_direction_accessor(accessor_keys) else 0.0
                plus = [max(lo, min(1.0, v + scale)) for v in current_value]
                minus = [max(lo, min(1.0, v - scale)) for v in current_value]
                candidates = [plus, minus]
            elif isinstance(current_value, (int, float)):
                val = float(current_value)
                perturb_plus = val + scale
                perturb_minus = val - scale
                if 0.0 <= val <= 1.0:
                    perturb_plus = max(0.0, min(1.0, perturb_plus))
                    perturb_minus = max(0.0, min(1.0, perturb_minus))
                else:
                    upper = max(val * 2, 0.01)
                    perturb_plus = max(0.0, min(upper, perturb_plus))
                    perturb_minus = max(0.0, min(upper, perturb_minus))
                candidates = [perturb_plus, perturb_minus]
            else:
                param_idx += 1
                step_count += 1
                continue

            for candidate_value in candidates:
                if step_count >= max_iterations:
                    break
                trial_dsl = _set_nested(best_dsl, accessor_keys, candidate_value)
                trial_score = score_dsl(
                    trial_dsl, ref_path, render_fn, render_dsl_fn=render_dsl_fn
                )
                step_count += 1

                # Acceptance epsilon: a trial must beat the running best for
                # this param by at least ``accept_epsilon`` to count. Noise-level
                # gains are rejected so they never block convergence early-stop.
                accepted = trial_score >= best_direction_score + accept_epsilon
                log.append(OptimizeStep(
                    iteration=step_count - 1,
                    param_path=param_path,
                    old_value=current_value,
                    new_value=candidate_value,
                    # Baseline the accept decision was made against: the running
                    # best for this param step. After the +scale trial is
                    # accepted, the -scale trial competes with that score, so the
                    # log stays consistent (accepted == score_after > score_before).
                    score_before=best_direction_score,
                    score_after=trial_score,
                    accepted=accepted,
                ))

                if accepted:
                    best_direction_value = candidate_value
                    best_direction_score = trial_score

            if best_direction_value is not None:
                best_dsl = _set_nested(best_dsl, accessor_keys, best_direction_value)
                best_score = best_direction_score
                loss_curve.append(best_score)
                collected = _collect_optimizable_params(best_dsl)
                params_since_improvement = 0
            else:
                params_since_improvement += 1
            param_idx += 1

            # Convergence / early-stop: once a full sweep over every parameter
            # has produced no accepted improvement, further sweeps would just
            # repeat the same wasted compile/render/score passes. Stop early and
            # record the reason instead of running to ``max_iterations``.
            if (
                collected
                and step_count > 0
                and params_since_improvement >= len(collected)
            ):
                stop_reason = "converged_no_improvement"
                break

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'random' or 'coordinate_descent'.")

    return OptimizeResult(
        best_dsl=best_dsl,
        initial_score=initial_score,
        best_score=best_score,
        improved=best_score > initial_score,
        iterations_run=len(log),  # actual evaluations performed, not the cap
        loss_curve=loss_curve,
        optimizer_log=log,
        protected_aspects_violations=[],
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------

def build_optimization_artifacts(result: OptimizeResult) -> dict:
    """Return a JSON-serializable summary dict for an OptimizeResult."""
    steps_accepted = sum(1 for s in result.optimizer_log if s.accepted)
    steps_rejected = len(result.optimizer_log) - steps_accepted

    return {
        "initial_score": result.initial_score,
        "best_score": result.best_score,
        "improved": result.improved,
        "iterations_run": result.iterations_run,
        "loss_curve": result.loss_curve,
        "steps_accepted": steps_accepted,
        "steps_rejected": steps_rejected,
        "protected_aspects_violations": result.protected_aspects_violations,
        "stop_reason": result.stop_reason,
    }

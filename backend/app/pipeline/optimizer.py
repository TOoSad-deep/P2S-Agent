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

from app.dsl.compiler import compile_dsl
from app.metrics.compute import compute_objective_metrics
from app.metrics.quality_router import compute_final_score

# ---------------------------------------------------------------------------
# Allowed optimizable parameter sets
# ---------------------------------------------------------------------------

OPTIMIZABLE_LAYER_PARAMS = {
    "center", "radius", "size", "opacity", "edge_softness"
}

OPTIMIZABLE_FILL_PARAMS = {
    "gradient_center", "gradient_direction",  # radialGradient
    "direction",                               # linearGradient direction
}

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


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def _perturb_scalar(value: float, scale: float = 0.05) -> float:
    """Add Gaussian noise to a scalar value.

    Clamps to [0.0, 1.0] for values already in that range;
    otherwise clamps to [0.0, max(value * 2, 0.01)].
    """
    new_val = value + random.gauss(0, scale)
    if 0.0 <= value <= 1.0:
        return max(0.0, min(1.0, new_val))
    else:
        upper = max(value * 2, 0.01)
        return max(0.0, min(upper, new_val))


def _perturb_vec2(value: list[float], scale: float = 0.05) -> list[float]:
    """Perturb each element of a 2-element list, each clamped to [0.0, 1.0]."""
    return [
        max(0.0, min(1.0, v + random.gauss(0, scale)))
        for v in value
    ]


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

    Returns:
        OptimizeResult with the best DSL found and diagnostic information.
    """
    if seed is not None:
        random.seed(seed)

    ref_path = Path(ref_path)
    best_dsl = copy.deepcopy(dsl)
    collected = _collect_optimizable_params(best_dsl)

    initial_score = score_dsl(
        best_dsl, ref_path, render_fn, render_dsl_fn=render_dsl_fn
    )
    best_score = initial_score
    loss_curve: list[float] = [initial_score]
    log: list[OptimizeStep] = []

    if strategy == "random":
        for i in range(max_iterations):
            if not collected:
                break

            # Pick a random optimizable param
            param_path, accessor_keys, current_value = random.choice(collected)

            # Perturb the value
            if isinstance(current_value, list) and len(current_value) == 2:
                new_value = _perturb_vec2(current_value)
            elif isinstance(current_value, (int, float)):
                new_value = _perturb_scalar(float(current_value))
            else:
                continue  # Skip non-numeric params

            # Evaluate the perturbed DSL
            new_dsl = _set_nested(best_dsl, accessor_keys, new_value)
            new_score = score_dsl(
                new_dsl, ref_path, render_fn, render_dsl_fn=render_dsl_fn
            )

            accepted = new_score > best_score
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

        # Re-collect with fresh reference
        collected = _collect_optimizable_params(best_dsl)

        while step_count < max_iterations:
            if not collected:
                break

            param_idx = param_idx % len(collected)
            param_path, accessor_keys, current_value = collected[param_idx]

            best_direction_value = None
            best_direction_score = best_score

            # Try both +scale and -scale perturbations
            if isinstance(current_value, list) and len(current_value) == 2:
                plus = [max(0.0, min(1.0, v + scale)) for v in current_value]
                minus = [max(0.0, min(1.0, v - scale)) for v in current_value]
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

                accepted = trial_score > best_direction_score
                log.append(OptimizeStep(
                    iteration=step_count - 1,
                    param_path=param_path,
                    old_value=current_value,
                    new_value=candidate_value,
                    score_before=best_score,
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
            param_idx += 1

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
    }

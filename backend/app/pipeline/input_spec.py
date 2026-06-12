"""Input specification builder for PNG-to-Shader.

Defines the contract between the caller and the pipeline:
what image, what target environment, what quality budget.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from app.strategy_config_loader import get_strategy_config

DEFAULT_INPUT_SPEC: dict[str, Any] = {
    "input_image": "",
    "target": {
        "backend": "glsl",
        "shader_env": "webgl2",
        "resolution": [512, 512],
        "allow_texture": False,
        "allow_sdf_texture": False,
        "max_shader_chars": 12000,
        "max_layers": 24,
        "max_render_time_ms": 8,
    },
    "quality": {
        "mode": "balanced",
        "max_iterations": 5,
        "optimization_budget": 300,
        "refinement_mode": "auto",
        "max_refinement_iterations": 3,
        "refinement_threshold": 0.80,
        "refinement_high_score_stop": 0.92,
        "refinement_min_improvement": 0.01,
        "refinement_patience": 2,
        "force_failure_type": None,
        "max_added_layers": 4,
        "protected_aspects": ["layer_count", "primitive_types", "background"],
    },
    "candidates": {
        "llm_enabled": False,
        "llm_implementation": "auto",
        "cv_enabled": True,
        "glsl_render_enabled": False,
    },
}

def _apply_strategy_defaults() -> None:
    cfg = get_strategy_config()
    quality = DEFAULT_INPUT_SPEC["quality"]
    for name, meta in cfg.params.items():
        if name in quality:
            quality[name] = meta.default


_apply_strategy_defaults()

_VALID_QUALITY_MODES = {"fast", "balanced", "quality", "aggressive"}
_VALID_REFINEMENT_MODES = {"off", "auto", "on"}
_VALID_LLM_IMPLEMENTATIONS = {"auto", "png_dsl", "shadertoy_glsl"}
_VALID_FAILURE_TYPES = {"color", "structure", "parameter", "layer_order", "budget", "unsupported"}
_VALID_PROTECTED_ASPECTS = {
    "layer_count", "primitive_types", "background",
    "visual_causality", "technique_plan", "tunable_parameters",
}


def build_input_spec(image_path: "str | Path", **overrides: Any) -> dict[str, Any]:
    """Return a deep copy of DEFAULT_INPUT_SPEC with input_image set and overrides applied.

    Only top-level keys ``target``, ``quality``, and ``candidates`` are merged from overrides.
    All other keyword arguments are ignored.

    Args:
        image_path: Path to the source PNG image.
        **overrides: Optional top-level key overrides. Recognised keys:
            ``target`` (dict), ``quality`` (dict), ``candidates`` (dict).

    Returns:
        A new dict conforming to the input-spec schema.
    """
    spec = copy.deepcopy(DEFAULT_INPUT_SPEC)
    spec["input_image"] = str(image_path)

    if "target" in overrides and isinstance(overrides["target"], dict):
        spec["target"].update(overrides["target"])

    if "quality" in overrides and isinstance(overrides["quality"], dict):
        spec["quality"].update(overrides["quality"])

    if "candidates" in overrides and isinstance(overrides["candidates"], dict):
        spec["candidates"].update(overrides["candidates"])

    return spec


def validate_input_spec(spec: dict[str, Any]) -> list[str]:
    """Validate an input spec dict and return a list of error strings.

    An empty list means the spec is valid.

    Checks:
    - ``input_image`` is a non-empty string.
    - ``target.backend`` equals "glsl".
    - ``target.resolution`` is a list of exactly two positive integers.
    - ``quality.mode`` is one of "fast", "balanced", "quality", "aggressive".
    - ``quality.refinement_mode`` is one of "off", "auto", "on".
    - ``quality.max_refinement_iterations`` is an int in [0, 20].
    - ``quality.refinement_threshold`` is a number in [0.5, 1.0].
    - ``quality.refinement_high_score_stop`` is a number in [0.7, 1.0]
      and is greater than or equal to ``quality.refinement_threshold``.
    - ``quality.refinement_min_improvement`` is a number in [0.001, 0.05].
    - ``quality.refinement_patience`` is an int in [1, 5].
    - ``quality.force_failure_type`` is ``None`` or one of the valid
      failure types.
    - ``quality.protected_aspects`` is a list of valid aspect names.
    - ``candidates.llm_enabled`` and ``candidates.cv_enabled`` are booleans.
    - ``candidates.glsl_render_enabled`` is a boolean.
    - ``candidates.llm_implementation`` is one of "auto", "png_dsl",
      "shadertoy_glsl".

    Args:
        spec: Dict previously produced (or shaped like) :func:`build_input_spec`.

    Returns:
        List of human-readable error strings (empty ↔ valid).
    """
    errors: list[str] = []

    # --- input_image ---
    image = spec.get("input_image", "")
    if not isinstance(image, str) or not image.strip():
        errors.append("input_image must be a non-empty string")

    # --- target ---
    target = spec.get("target", {})
    if not isinstance(target, dict):
        errors.append("target must be a dict")
    else:
        backend = target.get("backend")
        if backend != "glsl":
            errors.append(f"target.backend must be 'glsl', got {backend!r}")

        resolution = target.get("resolution")
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or not all(isinstance(v, int) and v > 0 for v in resolution)
        ):
            errors.append(
                "target.resolution must be a list of 2 positive integers, "
                f"got {resolution!r}"
            )

    # --- quality ---
    quality = spec.get("quality", {})
    if not isinstance(quality, dict):
        errors.append("quality must be a dict")
    else:
        mode = quality.get("mode")
        if mode not in _VALID_QUALITY_MODES:
            errors.append(
                f"quality.mode must be one of {sorted(_VALID_QUALITY_MODES)}, "
                f"got {mode!r}"
            )

        refinement_mode = quality.get("refinement_mode", "auto")
        if refinement_mode not in _VALID_REFINEMENT_MODES:
            errors.append(
                "quality.refinement_mode must be one of "
                f"{sorted(_VALID_REFINEMENT_MODES)}, got {refinement_mode!r}"
            )

        # --- numeric refinement parameters ---
        def _check_number(name: str, low: float, high: float, *, integer: bool = False) -> "float | None":
            value = quality.get(name, DEFAULT_INPUT_SPEC["quality"][name])
            if integer:
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"quality.{name} must be an int, got {value!r}")
                    return None
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(f"quality.{name} must be a number, got {value!r}")
                    return None
            if value < low or value > high:
                errors.append(
                    f"quality.{name} must be between {low} and {high}, got {value!r}"
                )
                return None
            return float(value)

        cfg = get_strategy_config()
        threshold_value = None
        high_stop_value = None
        for name, meta in cfg.params.items():
            result = _check_number(name, meta.min, meta.max, integer=meta.integer)
            if name == "refinement_threshold":
                threshold_value = result
            elif name == "refinement_high_score_stop":
                high_stop_value = result

        if threshold_value is not None and high_stop_value is not None:
            if high_stop_value < threshold_value:
                errors.append(
                    "quality.refinement_high_score_stop must be ≥ "
                    f"quality.refinement_threshold ({high_stop_value} < {threshold_value})"
                )

        # --- force_failure_type ---
        force_failure = quality.get("force_failure_type", None)
        if force_failure is not None and force_failure not in _VALID_FAILURE_TYPES:
            errors.append(
                "quality.force_failure_type must be None or one of "
                f"{sorted(_VALID_FAILURE_TYPES)}, got {force_failure!r}"
            )

        # --- protected_aspects ---
        protected = quality.get("protected_aspects", [])
        if not isinstance(protected, list):
            errors.append(
                f"quality.protected_aspects must be a list, got {protected!r}"
            )
        else:
            for item in protected:
                if item not in _VALID_PROTECTED_ASPECTS:
                    errors.append(
                        "quality.protected_aspects items must be in "
                        f"{sorted(_VALID_PROTECTED_ASPECTS)}, got {item!r}"
                    )
                    break

    # --- candidates ---
    candidates = spec.get("candidates", {})
    if not isinstance(candidates, dict):
        errors.append("candidates must be a dict")
    else:
        llm_enabled = candidates.get("llm_enabled")
        if not isinstance(llm_enabled, bool):
            errors.append(f"candidates.llm_enabled must be a bool, got {llm_enabled!r}")

        cv_enabled = candidates.get("cv_enabled")
        if not isinstance(cv_enabled, bool):
            errors.append(f"candidates.cv_enabled must be a bool, got {cv_enabled!r}")

        glsl_render_enabled = candidates.get("glsl_render_enabled")
        if not isinstance(glsl_render_enabled, bool):
            errors.append(
                f"candidates.glsl_render_enabled must be a bool, got {glsl_render_enabled!r}"
            )

        llm_implementation = candidates.get("llm_implementation")
        if llm_implementation not in _VALID_LLM_IMPLEMENTATIONS:
            errors.append(
                "candidates.llm_implementation must be one of "
                f"{sorted(_VALID_LLM_IMPLEMENTATIONS)}, got {llm_implementation!r}"
            )

    return errors

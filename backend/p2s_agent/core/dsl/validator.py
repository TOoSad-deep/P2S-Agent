"""DSL schema validator for PNG-to-Shader.

Validates that a DSL dict conforms to the Phase 3 schema.
Returns structured validation results, never raises on bad input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from p2s_agent.core.dsl.schema import (
    EFFECT_TYPES,
    FILL_TYPES,
    PRIMITIVE_TYPES,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_dsl(dsl: dict) -> ValidationResult:
    """Validate a DSL dict against the Phase 3 schema.

    Returns a ValidationResult with valid=True only when no errors are found.
    Warnings do not affect validity. Never raises on bad input.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(dsl, dict):
        return ValidationResult(valid=False, errors=["DSL must be a dict"])

    # 1. Top-level keys
    if "schema_version" not in dsl:
        errors.append("Missing required top-level field: 'schema_version'")
    elif not isinstance(dsl["schema_version"], int):
        errors.append("'schema_version' must be an int")

    if "canvas" not in dsl:
        errors.append("Missing required top-level field: 'canvas'")
    else:
        canvas = dsl["canvas"]
        if not isinstance(canvas, dict):
            errors.append("'canvas' must be a dict")
        else:
            for req in ("width", "height"):
                if req not in canvas:
                    errors.append(f"'canvas' missing required field: '{req}'")

    if "layers" not in dsl:
        errors.append("Missing required top-level field: 'layers'")
    else:
        layers = dsl["layers"]
        if not isinstance(layers, list):
            errors.append("'layers' must be a list")
        elif len(layers) == 0:
            errors.append("'layers' must be non-empty")
        else:
            seen_ids: set[str] = set()
            for idx, layer in enumerate(layers):
                layer_prefix = f"Layer[{idx}]"
                if not isinstance(layer, dict):
                    errors.append(f"{layer_prefix}: must be a dict")
                    continue

                # id
                if "id" not in layer:
                    errors.append(f"{layer_prefix}: missing required field 'id'")
                elif not isinstance(layer["id"], str) or not layer["id"].strip():
                    errors.append(f"{layer_prefix}: 'id' must be a non-empty string")
                else:
                    layer_id = layer["id"]
                    if layer_id in seen_ids:
                        warnings.append(f"Duplicate layer id: '{layer_id}'")
                    seen_ids.add(layer_id)

                # type
                if "type" not in layer:
                    errors.append(f"{layer_prefix}: missing required field 'type'")
                elif layer["type"] not in PRIMITIVE_TYPES:
                    errors.append(
                        f"{layer_prefix}: 'type' must be one of {PRIMITIVE_TYPES}, "
                        f"got '{layer['type']}'"
                    )

                # fill
                if "fill" not in layer:
                    errors.append(f"{layer_prefix}: missing required field 'fill'")
                else:
                    fill = layer["fill"]
                    if not isinstance(fill, dict):
                        errors.append(f"{layer_prefix}: 'fill' must be a dict")
                    else:
                        if "type" not in fill:
                            errors.append(f"{layer_prefix}: 'fill' missing 'type'")
                        elif fill["type"] not in FILL_TYPES:
                            errors.append(
                                f"{layer_prefix}: fill 'type' must be one of "
                                f"{FILL_TYPES}, got '{fill['type']}'"
                            )
                        else:
                            fill_type = fill["type"]
                            if fill_type == "linearGradient":
                                _validate_linear_gradient(fill, layer_prefix, errors)
                            elif fill_type == "radialGradient":
                                _validate_radial_gradient(fill, layer_prefix, errors)

                # effects
                if "effects" in layer:
                    effects = layer["effects"]
                    if not isinstance(effects, list):
                        errors.append(f"{layer_prefix}: 'effects' must be a list")
                    else:
                        for eidx, effect in enumerate(effects):
                            if not isinstance(effect, dict):
                                errors.append(
                                    f"{layer_prefix} effect[{eidx}]: must be a dict"
                                )
                                continue
                            if "type" not in effect:
                                errors.append(
                                    f"{layer_prefix} effect[{eidx}]: missing 'type'"
                                )
                            elif effect["type"] not in EFFECT_TYPES:
                                errors.append(
                                    f"{layer_prefix} effect[{eidx}]: 'type' must be "
                                    f"one of {EFFECT_TYPES}, got '{effect['type']}'"
                                )

                # opacity
                if "opacity" in layer:
                    opacity = layer["opacity"]
                    if not isinstance(opacity, (int, float)):
                        errors.append(
                            f"{layer_prefix}: 'opacity' must be a float in [0, 1]"
                        )
                    elif not (0.0 <= float(opacity) <= 1.0):
                        errors.append(
                            f"{layer_prefix}: 'opacity' must be in [0, 1], "
                            f"got {opacity}"
                        )

                # params (value-level): center/radius/size/ab/sides/etc.
                if "params" in layer:
                    params = layer["params"]
                    if not isinstance(params, dict):
                        errors.append(f"{layer_prefix}: 'params' must be a dict")
                    else:
                        _validate_params(
                            params, layer.get("type"), layer_prefix, errors
                        )
                else:
                    layer_id_str = layer.get("id", f"index {idx}")
                    warnings.append(
                        f"Layer '{layer_id_str}': missing recommended field 'params'"
                    )

                # transform (value-level): scale/rotate/translate shape
                if layer.get("transform") is not None:
                    _validate_transform(layer["transform"], layer_prefix, errors)

    valid = len(errors) == 0
    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


def _validate_linear_gradient(fill: dict, layer_prefix: str, errors: list[str]) -> None:
    """Validate linearGradient fill fields."""
    if "stops" not in fill:
        errors.append(f"{layer_prefix}: linearGradient fill missing 'stops'")
    else:
        stops = fill["stops"]
        if not isinstance(stops, list) or len(stops) < 2:
            errors.append(
                f"{layer_prefix}: linearGradient 'stops' must be a list of 2+"
            )
        else:
            for sidx, stop in enumerate(stops):
                if not isinstance(stop, dict):
                    errors.append(
                        f"{layer_prefix}: linearGradient stop[{sidx}] must be a dict"
                    )
                elif "color" not in stop or "position" not in stop:
                    errors.append(
                        f"{layer_prefix}: linearGradient stop[{sidx}] must have "
                        "'color' and 'position'"
                    )
                else:
                    _validate_stop_position(
                        stop["position"],
                        f"{layer_prefix}: linearGradient stop[{sidx}]",
                        errors,
                    )

    if "direction" not in fill:
        errors.append(f"{layer_prefix}: linearGradient fill missing 'direction'")
    else:
        direction = fill["direction"]
        if not isinstance(direction, (list, tuple)) or len(direction) != 2:
            errors.append(
                f"{layer_prefix}: linearGradient 'direction' must be [dx, dy]"
            )


def _validate_radial_gradient(fill: dict, layer_prefix: str, errors: list[str]) -> None:
    """Validate radialGradient fill fields."""
    if "stops" not in fill:
        errors.append(f"{layer_prefix}: radialGradient fill missing 'stops'")
    else:
        stops = fill["stops"]
        if not isinstance(stops, list) or len(stops) < 2:
            errors.append(
                f"{layer_prefix}: radialGradient 'stops' must be a list of 2+"
            )
        else:
            for sidx, stop in enumerate(stops):
                if not isinstance(stop, dict):
                    errors.append(
                        f"{layer_prefix}: radialGradient stop[{sidx}] must be a dict"
                    )
                elif "color" not in stop or "position" not in stop:
                    errors.append(
                        f"{layer_prefix}: radialGradient stop[{sidx}] must have "
                        "'color' and 'position'"
                    )
                else:
                    _validate_stop_position(
                        stop["position"],
                        f"{layer_prefix}: radialGradient stop[{sidx}]",
                        errors,
                    )

    if "center" not in fill:
        errors.append(f"{layer_prefix}: radialGradient fill missing 'center'")
    else:
        center = fill["center"]
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            errors.append(
                f"{layer_prefix}: radialGradient 'center' must be [cx, cy]"
            )


# ---------------------------------------------------------------------------
# Value-level helpers (Bug 1)
#
# These reject schema-valid-but-numerically-bad DSLs before they reach the
# compiler, where bad values would otherwise raise (crashing a worker) or
# compile success=True to GLSL that diverges from the raster renderer.
# ---------------------------------------------------------------------------

def _is_finite_number(v) -> bool:
    """True iff v is a real, finite number (rejects bool, inf, nan, strings)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def _validate_finite(v, label: str, errors: list[str]) -> None:
    if not _is_finite_number(v):
        errors.append(f"{label} must be a finite number, got {v!r}")


def _validate_number_pair(v, label: str, errors: list[str]) -> None:
    """Validate a 2-element list/tuple of finite numbers (center/size/ab/etc.)."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        errors.append(f"{label} must be a list of 2 numbers, got {v!r}")
        return
    for i, component in enumerate(v):
        if not _is_finite_number(component):
            errors.append(
                f"{label}[{i}] must be a finite number, got {component!r}"
            )


def _validate_stop_position(pos, label: str, errors: list[str]) -> None:
    """Gradient stop position must be a finite number in [0, 1]."""
    if not _is_finite_number(pos):
        errors.append(f"{label}: 'position' must be a number in [0, 1], got {pos!r}")
    elif not (0.0 <= float(pos) <= 1.0):
        errors.append(f"{label}: 'position' must be in [0, 1], got {pos}")


def _validate_params(params: dict, ptype, layer_prefix: str, errors: list[str]) -> None:
    """Value-level checks for a layer's params dict, keyed by primitive type."""
    if "center" in params:
        _validate_number_pair(params["center"], f"{layer_prefix}: params 'center'", errors)

    if ptype == "circle":
        if "radius" in params:
            _validate_finite(params["radius"], f"{layer_prefix}: params 'radius'", errors)
    elif ptype == "ellipse":
        if "ab" in params:
            _validate_number_pair(params["ab"], f"{layer_prefix}: params 'ab'", errors)
    elif ptype in ("box", "roundedBox"):
        if "size" in params:
            _validate_number_pair(params["size"], f"{layer_prefix}: params 'size'", errors)
        if ptype == "roundedBox" and "radius" in params:
            _validate_finite(params["radius"], f"{layer_prefix}: params 'radius'", errors)
    elif ptype == "ring":
        if "radius" in params:
            _validate_finite(params["radius"], f"{layer_prefix}: params 'radius'", errors)
        if "thickness" in params:
            _validate_finite(params["thickness"], f"{layer_prefix}: params 'thickness'", errors)
    elif ptype == "polygon":
        if "radius" in params:
            _validate_finite(params["radius"], f"{layer_prefix}: params 'radius'", errors)
        if "sides" in params:
            sides = params["sides"]
            if isinstance(sides, bool) or not isinstance(sides, int):
                errors.append(
                    f"{layer_prefix}: params 'sides' must be an int >= 3, got {sides!r}"
                )
            elif sides < 3:
                errors.append(
                    f"{layer_prefix}: params 'sides' must be >= 3, got {sides}"
                )


def _validate_transform(transform, layer_prefix: str, errors: list[str]) -> None:
    """Value-level checks for a layer's transform (translate/rotate/scale)."""
    if not isinstance(transform, dict):
        errors.append(f"{layer_prefix}: 'transform' must be a dict or null")
        return

    ttype = transform.get("type")
    if ttype == "translate":
        for axis in ("x", "y"):
            if axis in transform:
                _validate_finite(
                    transform[axis], f"{layer_prefix}: transform translate '{axis}'", errors
                )
    elif ttype == "rotate":
        if "angle" in transform:
            _validate_finite(
                transform["angle"], f"{layer_prefix}: transform rotate 'angle'", errors
            )
    elif ttype == "scale":
        for axis in ("x", "y"):
            if axis in transform:
                _validate_finite(
                    transform[axis], f"{layer_prefix}: transform scale '{axis}'", errors
                )

"""DSL schema validator for PNG-to-Shader.

Validates that a DSL dict conforms to the Phase 3 schema.
Returns structured validation results, never raises on bad input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.dsl.schema import (
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

                # params warning
                if "params" not in layer:
                    layer_id_str = layer.get("id", f"index {idx}")
                    warnings.append(
                        f"Layer '{layer_id_str}': missing recommended field 'params'"
                    )

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

    if "center" not in fill:
        errors.append(f"{layer_prefix}: radialGradient fill missing 'center'")
    else:
        center = fill["center"]
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            errors.append(
                f"{layer_prefix}: radialGradient 'center' must be [cx, cy]"
            )

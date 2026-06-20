"""Controlled revision system for PNG-to-Shader DSL.

Applies structured patch ops to a DSL dict. Never outputs raw GLSL.
Each revision targets one failure_type with one class of ops.
Automatically rolls back if the patch makes the score worse or
violates protected_aspects.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

from p2s_agent.core.dsl.validator import validate_dsl
from p2s_agent.core.metrics.quality_router import FAILURE_TYPE_VALUES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_OPS = [
    "update_layer_params",
    "update_layer_material",
    "update_layer_transform",
    "add_layer",
    "remove_layer",
    "reorder_layer",
    "bind_effect_source",
    "update_optimize_range",
]

REVISION_TYPES = ["structure", "parameter"]

# Ops that require a layer_id that references an existing layer
_OPS_REQUIRING_LAYER_ID = {
    "update_layer_params",
    "update_layer_material",
    "update_layer_transform",
    "remove_layer",
    "reorder_layer",
    "bind_effect_source",
    "update_optimize_range",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PatchOp:
    """A single atomic operation applied to the DSL.

    op:        One of ALLOWED_OPS.
    layer_id:  ID of the target layer (required by most ops).
    params:    Op-specific parameters (see module docstring for details).
    """

    op: str
    layer_id: str | None = None
    params: dict = field(default_factory=dict)


@dataclass
class RevisionPatch:
    """A structured revision patch describing one change class.

    revision_type:       "structure" | "parameter"
    failure_type:        From quality_router.FAILURE_TYPE_VALUES
    ops:                 Ordered list of PatchOp to apply in sequence.
    protected_aspects:   Aspects that must not be violated.
    expected_improvement: Human-readable list of expected gains.
    """

    revision_type: str
    failure_type: str
    ops: list[PatchOp]
    protected_aspects: list[str] = field(default_factory=list)
    expected_improvement: list[str] = field(default_factory=list)


@dataclass
class RevisionResult:
    """The outcome of applying a revision patch with rollback semantics."""

    success: bool
    final_dsl: dict
    final_score: float
    initial_score: float
    improved: bool
    rolled_back: bool
    violations: list[str]   # protected_aspect violations
    errors: list[str]       # apply / validation errors
    patch_log: list[dict]   # per-op log entries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_layer(
    layers: list[dict], layer_id: str
) -> tuple[int, dict] | tuple[None, None]:
    """Return (index, layer_dict) for the layer with matching id, or (None, None)."""
    for idx, layer in enumerate(layers):
        if isinstance(layer, dict) and layer.get("id") == layer_id:
            return idx, layer
    return None, None


def _apply_op(dsl: dict, op: PatchOp) -> tuple[dict, list[str]]:
    """Apply a single PatchOp to a deep copy of dsl.

    Returns (new_dsl, errors).  Errors are non-empty on failure.
    Never mutates the input dsl.
    """
    new_dsl = copy.deepcopy(dsl)
    layers: list[dict] = new_dsl.get("layers", [])
    errors: list[str] = []

    # ------------------------------------------------------------------
    if op.op == "update_layer_params":
        idx, layer = _find_layer(layers, op.layer_id)
        if layer is None:
            errors.append(f"update_layer_params: layer '{op.layer_id}' not found")
            return new_dsl, errors
        if "params" not in layer or not isinstance(layer["params"], dict):
            layer["params"] = {}
        if "updates" in op.params:
            layer["params"].update(op.params["updates"])
        elif "key" in op.params and "value" in op.params:
            layer["params"][op.params["key"]] = op.params["value"]

    # ------------------------------------------------------------------
    elif op.op == "update_layer_material":
        idx, layer = _find_layer(layers, op.layer_id)
        if layer is None:
            errors.append(f"update_layer_material: layer '{op.layer_id}' not found")
            return new_dsl, errors
        if "fill" not in op.params:
            errors.append("update_layer_material: 'fill' key missing from params")
            return new_dsl, errors
        layer["fill"] = copy.deepcopy(op.params["fill"])

    # ------------------------------------------------------------------
    elif op.op == "update_layer_transform":
        idx, layer = _find_layer(layers, op.layer_id)
        if layer is None:
            errors.append(f"update_layer_transform: layer '{op.layer_id}' not found")
            return new_dsl, errors
        layer["transform"] = copy.deepcopy(op.params.get("transform"))

    # ------------------------------------------------------------------
    elif op.op == "add_layer":
        new_layer = op.params.get("layer")
        if new_layer is None:
            errors.append("add_layer: 'layer' key missing from params")
            return new_dsl, errors
        if not isinstance(new_layer, dict):
            errors.append("add_layer: 'layer' must be a dict")
            return new_dsl, errors
        if "id" not in new_layer:
            errors.append("add_layer: new layer must have 'id'")
            return new_dsl, errors
        if "type" not in new_layer:
            errors.append("add_layer: new layer must have 'type'")
            return new_dsl, errors
        position = op.params.get("position", len(layers))
        position = max(0, min(position, len(layers)))
        layers.insert(position, copy.deepcopy(new_layer))

    # ------------------------------------------------------------------
    elif op.op == "remove_layer":
        idx, layer = _find_layer(layers, op.layer_id)
        if layer is None:
            errors.append(f"remove_layer: layer '{op.layer_id}' not found")
            return new_dsl, errors
        layers.pop(idx)

    # ------------------------------------------------------------------
    elif op.op == "reorder_layer":
        idx, layer = _find_layer(layers, op.layer_id)
        if layer is None:
            errors.append(f"reorder_layer: layer '{op.layer_id}' not found")
            return new_dsl, errors
        after = op.params.get("after")
        # Moving a layer after itself is a no-op (check before pop to avoid
        # losing the layer from the search list).
        if after is not None and after == op.layer_id:
            return new_dsl, errors  # no-op, no error
        moved = layers.pop(idx)
        if after is None:
            layers.insert(0, moved)
        else:
            after_idx, after_layer = _find_layer(layers, after)
            if after_layer is None:
                errors.append(f"reorder_layer: 'after' layer '{after}' not found")
                # restore
                layers.insert(idx, moved)
                return new_dsl, errors
            layers.insert(after_idx + 1, moved)

    # ------------------------------------------------------------------
    elif op.op == "bind_effect_source":
        idx, layer = _find_layer(layers, op.layer_id)
        if layer is None:
            errors.append(f"bind_effect_source: layer '{op.layer_id}' not found")
            return new_dsl, errors
        effect_type = op.params.get("effect_type")
        source_layer = op.params.get("source_layer")
        if effect_type is None:
            errors.append("bind_effect_source: 'effect_type' missing from params")
            return new_dsl, errors
        effects = layer.get("effects", [])
        found_effect = False
        for effect in effects:
            if isinstance(effect, dict) and effect.get("type") == effect_type:
                effect["source_layer"] = source_layer
                found_effect = True
                break
        if not found_effect:
            errors.append(
                f"bind_effect_source: no effect of type '{effect_type}' on layer '{op.layer_id}'"
            )
            return new_dsl, errors

    # ------------------------------------------------------------------
    elif op.op == "update_optimize_range":
        idx, layer = _find_layer(layers, op.layer_id)
        if layer is None:
            errors.append(f"update_optimize_range: layer '{op.layer_id}' not found")
            return new_dsl, errors
        if "_optimize_ranges" not in layer:
            layer["_optimize_ranges"] = {}
        key = op.params.get("key")
        if key is None:
            errors.append("update_optimize_range: 'key' missing from params")
            return new_dsl, errors
        layer["_optimize_ranges"][key] = {
            "min": op.params.get("min"),
            "max": op.params.get("max"),
        }

    # ------------------------------------------------------------------
    else:
        errors.append(f"Unknown op: '{op.op}'")

    return new_dsl, errors


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------


def validate_patch(patch: RevisionPatch, dsl: dict) -> tuple[bool, list[str]]:
    """Validate a RevisionPatch against the current DSL.

    Checks revision_type, failure_type, op names, and layer references.
    Returns (True, []) if valid, (False, errors) otherwise.
    """
    errors: list[str] = []

    # 1. revision_type
    if patch.revision_type not in REVISION_TYPES:
        errors.append(
            f"Invalid revision_type '{patch.revision_type}'; "
            f"must be one of {REVISION_TYPES}"
        )

    # 2. failure_type
    if patch.failure_type not in FAILURE_TYPE_VALUES:
        errors.append(
            f"Invalid failure_type '{patch.failure_type}'; "
            f"must be one of {FAILURE_TYPE_VALUES}"
        )

    layers = dsl.get("layers", []) if isinstance(dsl, dict) else []
    existing_ids = {
        layer["id"]
        for layer in layers
        if isinstance(layer, dict) and "id" in layer
    }

    for op_idx, op in enumerate(patch.ops):
        prefix = f"Op[{op_idx}] ({op.op!r})"

        # 3. op name
        if op.op not in ALLOWED_OPS:
            errors.append(f"{prefix}: op not in ALLOWED_OPS")
            continue  # skip further checks for this op

        # 4. layer_id existence for ops that require it
        if op.op in _OPS_REQUIRING_LAYER_ID:
            if not op.layer_id:
                errors.append(f"{prefix}: layer_id is required but not set")
            elif op.layer_id not in existing_ids:
                errors.append(
                    f"{prefix}: layer_id '{op.layer_id}' does not exist in dsl"
                )

        # 5. add_layer: must have "layer" with "id" and "type"
        if op.op == "add_layer":
            new_layer = op.params.get("layer")
            if new_layer is None:
                errors.append(f"{prefix}: params missing 'layer' key")
            else:
                if not isinstance(new_layer, dict):
                    errors.append(f"{prefix}: params['layer'] must be a dict")
                else:
                    if "id" not in new_layer:
                        errors.append(f"{prefix}: new layer missing 'id'")
                    if "type" not in new_layer:
                        errors.append(f"{prefix}: new layer missing 'type'")

        # 6. reorder_layer: if "after" specified, it must exist
        if op.op == "reorder_layer":
            after = op.params.get("after")
            if after is not None and after not in existing_ids:
                errors.append(
                    f"{prefix}: 'after' layer '{after}' does not exist in dsl"
                )

    if errors:
        return False, errors
    return True, []


def apply_patch(
    dsl: dict, patch: RevisionPatch
) -> tuple[dict, bool, list[str]]:
    """Apply all ops in patch sequentially.

    Stops on the first failing op and returns the original dsl copy.
    Returns (new_dsl, success, errors).
    """
    original = copy.deepcopy(dsl)
    current = copy.deepcopy(dsl)

    all_errors: list[str] = []

    for op in patch.ops:
        new_dsl, errors = _apply_op(current, op)
        if errors:
            all_errors.extend(errors)
            return original, False, all_errors
        current = new_dsl

    return current, True, []


def check_protected_aspects(
    old_dsl: dict,
    new_dsl: dict,
    protected_aspects: list[str],
) -> list[str]:
    """Return a list of protected_aspect strings that were violated.

    Known aspects:
        "layer_count"     — number of layers changed
        "layer_order"     — id sequence changed
        "primitive_types" — any layer type changed
        "background"      — canvas background changed
        "opacity"         — any layer opacity changed by more than 0.05
    """
    violations: list[str] = []

    old_layers: list[dict] = old_dsl.get("layers", []) if isinstance(old_dsl, dict) else []
    new_layers: list[dict] = new_dsl.get("layers", []) if isinstance(new_dsl, dict) else []

    for aspect in protected_aspects:
        if aspect == "layer_count":
            if len(old_layers) != len(new_layers):
                violations.append("layer_count")

        elif aspect == "layer_order":
            old_ids = [l.get("id") for l in old_layers if isinstance(l, dict)]
            new_ids = [l.get("id") for l in new_layers if isinstance(l, dict)]
            if old_ids != new_ids:
                violations.append("layer_order")

        elif aspect == "primitive_types":
            old_types = {
                l.get("id"): l.get("type")
                for l in old_layers
                if isinstance(l, dict)
            }
            new_types = {
                l.get("id"): l.get("type")
                for l in new_layers
                if isinstance(l, dict)
            }
            # Check if any common layer_id has a different type
            for lid, old_type in old_types.items():
                new_type = new_types.get(lid)
                if new_type is not None and new_type != old_type:
                    violations.append("primitive_types")
                    break

        elif aspect == "background":
            old_bg = (
                old_dsl.get("canvas", {}).get("background")
                if isinstance(old_dsl, dict)
                else None
            )
            new_bg = (
                new_dsl.get("canvas", {}).get("background")
                if isinstance(new_dsl, dict)
                else None
            )
            if old_bg != new_bg:
                violations.append("background")

        elif aspect == "opacity":
            old_opacities = {
                l.get("id"): float(l.get("opacity", 1.0))
                for l in old_layers
                if isinstance(l, dict) and "id" in l
            }
            new_opacities = {
                l.get("id"): float(l.get("opacity", 1.0))
                for l in new_layers
                if isinstance(l, dict) and "id" in l
            }
            for lid, old_op in old_opacities.items():
                new_op = new_opacities.get(lid)
                if new_op is not None and abs(new_op - old_op) > 0.05:
                    violations.append("opacity")
                    break

        # Unknown protected_aspect strings are NOT silently ignored —
        # we flag them as a violation to fail safe.
        else:
            violations.append(aspect)

    return violations


def apply_revision_with_rollback(
    dsl: dict,
    patch: RevisionPatch,
    score_fn: Callable[[dict], float],
) -> RevisionResult:
    """Apply a RevisionPatch with automatic rollback on regression or violation.

    Steps:
    1. Validate patch structure.
    2. Score original DSL.
    3. Apply ops.
    4. Check protected_aspects.
    5. Validate new DSL schema.
    6. Score new DSL; rollback if score dropped.
    """
    # 1. Score original first so every return path has the real initial_score
    initial_score = score_fn(dsl)

    # 2. Validate patch
    valid, val_errors = validate_patch(patch, dsl)
    if not valid:
        return RevisionResult(
            success=False,
            final_dsl=copy.deepcopy(dsl),
            final_score=initial_score,
            initial_score=initial_score,
            improved=False,
            rolled_back=False,
            violations=[],
            errors=val_errors,
            patch_log=[],
        )

    # 3. Apply patch
    new_dsl, apply_success, apply_errors = apply_patch(dsl, patch)
    if not apply_success:
        return RevisionResult(
            success=False,
            final_dsl=copy.deepcopy(dsl),
            final_score=initial_score,
            initial_score=initial_score,
            improved=False,
            rolled_back=False,
            violations=[],
            errors=apply_errors,
            patch_log=[],
        )

    # 4. Check protected_aspects
    violations = check_protected_aspects(dsl, new_dsl, patch.protected_aspects)
    if violations:
        return RevisionResult(
            success=False,
            final_dsl=copy.deepcopy(dsl),
            final_score=initial_score,
            initial_score=initial_score,
            improved=False,
            rolled_back=False,
            violations=violations,
            errors=[f"Protected aspect violated: {v}" for v in violations],
            patch_log=[],
        )

    # 5. Validate new DSL schema
    val_result = validate_dsl(new_dsl)
    if not val_result.valid:
        return RevisionResult(
            success=False,
            final_dsl=copy.deepcopy(dsl),
            final_score=initial_score,
            initial_score=initial_score,
            improved=False,
            rolled_back=True,
            violations=[],
            errors=[f"Schema validation failed after patch: {e}" for e in val_result.errors],
            patch_log=[],
        )

    # 6. Score new DSL
    new_score = score_fn(new_dsl)

    if new_score < initial_score:
        # Rollback
        return RevisionResult(
            success=False,
            final_dsl=copy.deepcopy(dsl),
            final_score=initial_score,
            initial_score=initial_score,
            improved=False,
            rolled_back=True,
            violations=[],
            errors=[
                f"Score dropped from {initial_score:.4f} to {new_score:.4f}; rolled back"
            ],
            patch_log=[{"ops_applied": len(patch.ops), "rolled_back": True}],
        )

    # Success
    return RevisionResult(
        success=True,
        final_dsl=new_dsl,
        final_score=new_score,
        initial_score=initial_score,
        improved=new_score > initial_score,
        rolled_back=False,
        violations=[],
        errors=[],
        patch_log=[{"ops_applied": len(patch.ops), "rolled_back": False}],
    )


def build_revision_log_entry(
    patch: RevisionPatch, result: RevisionResult
) -> dict:
    """Return a JSON-serialisable summary dict for this revision attempt."""
    return {
        "revision_type": patch.revision_type,
        "failure_type": patch.failure_type,
        "ops_count": len(patch.ops),
        "success": result.success,
        "improved": result.improved,
        "rolled_back": result.rolled_back,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "violations": list(result.violations),
        "errors": list(result.errors),
    }

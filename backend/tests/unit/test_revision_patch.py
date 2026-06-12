"""Unit tests for the Phase 7 controlled revision system.

No LLM calls, no browser, no renderer — pure data structure tests.
"""

from __future__ import annotations

import copy

import pytest

from app.pipeline.revision import (
    ALLOWED_OPS,
    REVISION_TYPES,
    PatchOp,
    RevisionPatch,
    RevisionResult,
    apply_patch,
    apply_revision_with_rollback,
    build_revision_log_entry,
    check_protected_aspects,
    validate_patch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_DSL = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#000000"},
    "layers": [
        {
            "id": "layer_a",
            "type": "circle",
            "fill": {"type": "solid", "color": "#ff0000"},
            "params": {"center": [0.5, 0.5], "radius": 0.3},
            "opacity": 1.0,
            "transform": None,
            "effects": [],
        },
        {
            "id": "layer_b",
            "type": "box",
            "fill": {"type": "solid", "color": "#00ff00"},
            "params": {"center": [0.5, 0.5], "size": [0.4, 0.2]},
            "opacity": 0.8,
            "transform": None,
            "effects": [],
        },
    ],
}


def mock_score_fn(score: float):
    """Return a score_fn that always returns the given score."""
    return lambda dsl: score


def alternating_score_fn(scores: list[float]):
    """Return a score_fn that returns successive values from scores."""
    it = iter(scores)
    return lambda dsl: next(it)


def _fresh() -> dict:
    """Deep copy of SIMPLE_DSL to avoid cross-test mutation."""
    return copy.deepcopy(SIMPLE_DSL)


# ---------------------------------------------------------------------------
# validate_patch tests
# ---------------------------------------------------------------------------


def test_validate_patch_valid_update_params():
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            PatchOp(
                op="update_layer_params",
                layer_id="layer_a",
                params={"updates": {"radius": 0.4}},
            )
        ],
    )
    valid, errors = validate_patch(patch, _fresh())
    assert valid is True
    assert errors == []


def test_validate_patch_invalid_op():
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            PatchOp(op="delete_layer", layer_id="layer_a", params={})
        ],
    )
    valid, errors = validate_patch(patch, _fresh())
    assert valid is False
    assert any("ALLOWED_OPS" in e or "not in" in e.lower() for e in errors)


def test_validate_patch_bad_revision_type():
    patch = RevisionPatch(
        revision_type="hack",
        failure_type="parameter",
        ops=[],
    )
    valid, errors = validate_patch(patch, _fresh())
    assert valid is False
    assert any("revision_type" in e for e in errors)


def test_validate_patch_bad_failure_type():
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="typo",
        ops=[],
    )
    valid, errors = validate_patch(patch, _fresh())
    assert valid is False
    assert any("failure_type" in e for e in errors)


def test_validate_patch_missing_layer_id():
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            PatchOp(
                op="update_layer_params",
                layer_id="nonexistent",
                params={"updates": {"radius": 0.1}},
            )
        ],
    )
    valid, errors = validate_patch(patch, _fresh())
    assert valid is False
    assert any("nonexistent" in e for e in errors)


def test_validate_patch_add_layer_no_layer_key():
    patch = RevisionPatch(
        revision_type="structure",
        failure_type="structure",
        ops=[
            PatchOp(op="add_layer", layer_id=None, params={})  # missing "layer" key
        ],
    )
    valid, errors = validate_patch(patch, _fresh())
    assert valid is False
    assert any("layer" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# apply_patch tests
# ---------------------------------------------------------------------------


def test_apply_update_layer_params():
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            PatchOp(
                op="update_layer_params",
                layer_id="layer_a",
                params={"updates": {"radius": 0.45}},
            )
        ],
    )
    new_dsl, success, errors = apply_patch(_fresh(), patch)
    assert success is True
    assert errors == []
    layer_a = next(l for l in new_dsl["layers"] if l["id"] == "layer_a")
    assert layer_a["params"]["radius"] == 0.45


def test_apply_update_layer_material():
    new_fill = {"type": "solid", "color": "#0000ff"}
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="color",
        ops=[
            PatchOp(
                op="update_layer_material",
                layer_id="layer_b",
                params={"fill": new_fill},
            )
        ],
    )
    new_dsl, success, errors = apply_patch(_fresh(), patch)
    assert success is True
    layer_b = next(l for l in new_dsl["layers"] if l["id"] == "layer_b")
    assert layer_b["fill"]["color"] == "#0000ff"


def test_apply_update_layer_transform():
    transform = {"type": "translate", "x": 0.1, "y": -0.05}
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            PatchOp(
                op="update_layer_transform",
                layer_id="layer_a",
                params={"transform": transform},
            )
        ],
    )
    new_dsl, success, errors = apply_patch(_fresh(), patch)
    assert success is True
    layer_a = next(l for l in new_dsl["layers"] if l["id"] == "layer_a")
    assert layer_a["transform"] == transform


def test_apply_add_layer():
    new_layer = {
        "id": "layer_c",
        "type": "ring",
        "fill": {"type": "solid", "color": "#ffffff"},
        "params": {"center": [0.5, 0.5], "radius": 0.2, "thickness": 0.02},
        "opacity": 1.0,
        "transform": None,
        "effects": [],
    }
    patch = RevisionPatch(
        revision_type="structure",
        failure_type="structure",
        ops=[
            PatchOp(op="add_layer", layer_id=None, params={"layer": new_layer, "position": 0})
        ],
    )
    new_dsl, success, errors = apply_patch(_fresh(), patch)
    assert success is True
    assert len(new_dsl["layers"]) == 3
    assert new_dsl["layers"][0]["id"] == "layer_c"


def test_apply_remove_layer():
    patch = RevisionPatch(
        revision_type="structure",
        failure_type="structure",
        ops=[PatchOp(op="remove_layer", layer_id="layer_b", params={})],
    )
    new_dsl, success, errors = apply_patch(_fresh(), patch)
    assert success is True
    assert len(new_dsl["layers"]) == 1
    assert new_dsl["layers"][0]["id"] == "layer_a"


def test_apply_reorder_layer():
    # Move layer_b to front (after=None)
    patch = RevisionPatch(
        revision_type="structure",
        failure_type="layer_order",
        ops=[
            PatchOp(
                op="reorder_layer",
                layer_id="layer_b",
                params={"after": None},
            )
        ],
    )
    new_dsl, success, errors = apply_patch(_fresh(), patch)
    assert success is True
    ids = [l["id"] for l in new_dsl["layers"]]
    assert ids == ["layer_b", "layer_a"]


def test_apply_does_not_mutate_original():
    original = _fresh()
    original_copy = copy.deepcopy(original)
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            PatchOp(
                op="update_layer_params",
                layer_id="layer_a",
                params={"updates": {"radius": 0.99}},
            )
        ],
    )
    apply_patch(original, patch)
    assert original == original_copy, "apply_patch must not mutate the input DSL"


def test_apply_bind_effect_source():
    dsl = _fresh()
    # Add a glow effect to layer_a first
    dsl["layers"][0]["effects"] = [{"type": "glow", "intensity": 2.0, "color": "#ffffff"}]

    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            PatchOp(
                op="bind_effect_source",
                layer_id="layer_a",
                params={"effect_type": "glow", "source_layer": "layer_b"},
            )
        ],
    )
    new_dsl, success, errors = apply_patch(dsl, patch)
    assert success is True, errors
    layer_a = next(l for l in new_dsl["layers"] if l["id"] == "layer_a")
    glow = next(e for e in layer_a["effects"] if e["type"] == "glow")
    assert glow["source_layer"] == "layer_b"


def test_apply_bad_op_stops_early():
    """A patch with an unknown op after a valid op must stop and return the original DSL."""
    patch = RevisionPatch(
        revision_type="parameter",
        failure_type="parameter",
        ops=[
            # This op is valid and would succeed
            PatchOp(
                op="update_layer_params",
                layer_id="layer_a",
                params={"updates": {"radius": 0.99}},
            ),
            # This op is unknown and should trigger failure
            PatchOp(op="explode_everything", layer_id="layer_a", params={}),
        ],
    )
    original = _fresh()
    new_dsl, success, errors = apply_patch(original, patch)
    assert success is False
    assert len(errors) > 0
    # The returned DSL must be the original, unchanged
    assert new_dsl == original


# ---------------------------------------------------------------------------
# check_protected_aspects tests
# ---------------------------------------------------------------------------


def test_no_violations_on_identical_dsl():
    dsl = _fresh()
    violations = check_protected_aspects(
        dsl, copy.deepcopy(dsl),
        ["layer_count", "layer_order", "primitive_types", "background", "opacity"],
    )
    assert violations == []


def test_layer_count_violation():
    old = _fresh()
    new = _fresh()
    new["layers"].append({
        "id": "layer_x",
        "type": "circle",
        "fill": {"type": "solid", "color": "#ffffff"},
        "params": {},
        "opacity": 1.0,
        "transform": None,
        "effects": [],
    })
    violations = check_protected_aspects(old, new, ["layer_count"])
    assert "layer_count" in violations


def test_layer_order_violation():
    old = _fresh()
    new = _fresh()
    # Swap the two layers
    new["layers"] = list(reversed(new["layers"]))
    violations = check_protected_aspects(old, new, ["layer_order"])
    assert "layer_order" in violations


def test_primitive_type_violation():
    old = _fresh()
    new = _fresh()
    # Change layer_a from "circle" to "box"
    new["layers"][0]["type"] = "box"
    violations = check_protected_aspects(old, new, ["primitive_types"])
    assert "primitive_types" in violations


# ---------------------------------------------------------------------------
# apply_revision_with_rollback tests
# ---------------------------------------------------------------------------

_VALID_UPDATE_PATCH = RevisionPatch(
    revision_type="parameter",
    failure_type="parameter",
    ops=[
        PatchOp(
            op="update_layer_params",
            layer_id="layer_a",
            params={"updates": {"radius": 0.4}},
        )
    ],
)


def test_rollback_when_score_drops():
    original = _fresh()
    # Initial call returns 0.9, second call (after patch) returns 0.5
    score_fn = alternating_score_fn([0.9, 0.5])
    result = apply_revision_with_rollback(original, _VALID_UPDATE_PATCH, score_fn)
    assert result.rolled_back is True
    assert result.final_dsl == original
    assert result.final_score == 0.9


def test_keep_when_score_improves():
    original = _fresh()
    score_fn = alternating_score_fn([0.3, 0.7])
    result = apply_revision_with_rollback(original, _VALID_UPDATE_PATCH, score_fn)
    assert result.rolled_back is False
    assert result.success is True
    # The final DSL should differ from original (radius updated)
    layer_a = next(l for l in result.final_dsl["layers"] if l["id"] == "layer_a")
    assert layer_a["params"]["radius"] == 0.4
    assert result.final_score == 0.7


def test_rollback_on_invalid_patch():
    original = _fresh()
    bad_patch = RevisionPatch(
        revision_type="bad_type",      # invalid
        failure_type="parameter",
        ops=[],
    )
    result = apply_revision_with_rollback(original, bad_patch, mock_score_fn(0.5))
    assert result.success is False
    assert result.final_dsl == original
    assert len(result.errors) > 0


def test_rollback_on_protected_aspect_violation():
    original = _fresh()
    # Reorder with "layer_order" protected → violation
    patch = RevisionPatch(
        revision_type="structure",
        failure_type="layer_order",
        ops=[
            PatchOp(
                op="reorder_layer",
                layer_id="layer_b",
                params={"after": None},
            )
        ],
        protected_aspects=["layer_order"],
    )
    result = apply_revision_with_rollback(original, patch, mock_score_fn(0.5))
    assert result.success is False
    assert "layer_order" in result.violations
    assert result.final_dsl == original


def test_revision_result_has_all_fields():
    original = _fresh()
    result = apply_revision_with_rollback(
        original, _VALID_UPDATE_PATCH, mock_score_fn(0.5)
    )
    assert isinstance(result, RevisionResult)
    assert hasattr(result, "success")
    assert hasattr(result, "final_dsl")
    assert hasattr(result, "final_score")
    assert hasattr(result, "initial_score")
    assert hasattr(result, "improved")
    assert hasattr(result, "rolled_back")
    assert hasattr(result, "violations")
    assert hasattr(result, "errors")
    assert hasattr(result, "patch_log")


# ---------------------------------------------------------------------------
# build_revision_log_entry test
# ---------------------------------------------------------------------------


def test_build_revision_log_has_required_keys():
    patch = _VALID_UPDATE_PATCH
    result = apply_revision_with_rollback(_fresh(), patch, mock_score_fn(0.5))
    log = build_revision_log_entry(patch, result)

    required_keys = {
        "revision_type",
        "failure_type",
        "ops_count",
        "success",
        "improved",
        "rolled_back",
        "initial_score",
        "final_score",
        "violations",
        "errors",
    }
    assert required_keys.issubset(set(log.keys()))
    assert log["revision_type"] == "parameter"
    assert log["failure_type"] == "parameter"
    assert log["ops_count"] == 1
    assert isinstance(log["violations"], list)
    assert isinstance(log["errors"], list)

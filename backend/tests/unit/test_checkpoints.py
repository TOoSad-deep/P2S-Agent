"""Tests for the human-in-loop checkpoint resolver (V1 + V2.1)."""

from __future__ import annotations

import json

import pytest

from app.pipeline.checkpoints import (
    CheckpointError,
    PipelineCheckpoint,
    build_timeline,
    checkpoint_metadata,
    list_checkpoints,
    resolve_checkpoint,
    resolve_checkpoint_artifact,
    save_timeline,
)

SELECTED_GLSL = "#define R 0.72\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"
CV_GLSL = "#define R 0.40\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"
ITER1_GLSL = "#define R 0.65\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"
ITER2_GLSL = "#define R 0.67\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"


def _result() -> dict:
    """A pipeline result/store-entry with candidates, iterations, and a final."""
    return {
        "selected_candidate_id": "llm_0",
        "selected_glsl": SELECTED_GLSL,
        "quality_router": {"final_score": 0.72},
        "scoreboard": {
            "selected_id": "llm_0",
            "candidates": [
                {
                    "id": "llm_0",
                    "source": "llm",
                    "selected": True,
                    "previewable": True,
                    "compile_glsl": SELECTED_GLSL,
                    "final_score": 0.61,
                    "objective_metrics": {"mse": 0.2},
                    "quality_router": {"final_score": 0.61},
                },
                {
                    "id": "cv_0",
                    "source": "cv",
                    "selected": False,
                    "previewable": True,
                    "compile_glsl": CV_GLSL,
                    "final_score": 0.40,
                    "objective_metrics": {},
                    "quality_router": {},
                },
                {
                    "id": "rule_0",
                    "source": "rule",
                    "selected": False,
                    "previewable": False,
                    "compile_glsl": "",
                    "final_score": None,
                    "objective_metrics": {},
                    "quality_router": {},
                },
            ],
        },
        "refinement_history": [
            {
                "iteration": 1,
                "compile_glsl": ITER1_GLSL,
                "score_after": 0.65,
                "improved": True,
                "accepted": True,
                "changes_summary": "1 changed line",
            },
            {
                "iteration": 2,
                "compile_glsl": ITER2_GLSL,
                "score_after": 0.67,
                "improved": True,
                "accepted": True,
            },
            {
                "iteration": 3,
                "compile_glsl": None,
                "score_after": None,
                "error": "LLM returned no usable GLSL",
            },
        ],
    }


# ---------------------------------------------------------------------------
# list_checkpoints
# ---------------------------------------------------------------------------

def test_list_includes_previewable_candidates_only():
    ids = {cp["id"] for cp in list_checkpoints(_result())}
    assert "candidate:llm_0" in ids
    assert "candidate:cv_0" in ids
    # rule_0 has no compiled GLSL -> not a branchable checkpoint
    assert "candidate:rule_0" not in ids


def test_list_marks_selected_candidate_as_baseline():
    by_id = {cp["id"]: cp for cp in list_checkpoints(_result())}
    selected = by_id["candidate:llm_0"]
    assert selected["accepted"] is True
    assert selected["label"] == "Selected baseline"
    assert by_id["candidate:cv_0"]["accepted"] is False


def test_list_includes_refinement_iterations_with_glsl():
    by_id = {cp["id"]: cp for cp in list_checkpoints(_result())}
    assert by_id["refinement:iter:1"]["kind"] == "refinement_iter"
    assert by_id["refinement:iter:1"]["iteration"] == 1
    assert by_id["refinement:iter:2"]["score"] == pytest.approx(0.67)
    # iteration 3 produced no GLSL -> excluded
    assert "refinement:iter:3" not in by_id


def test_list_includes_final_selected():
    by_id = {cp["id"]: cp for cp in list_checkpoints(_result())}
    final = by_id["final:selected"]
    assert final["kind"] == "final"
    assert final["score"] == pytest.approx(0.72)
    assert final["has_glsl"] is True


def test_list_omits_raw_glsl_payload():
    for cp in list_checkpoints(_result()):
        assert "glsl" not in cp
        assert cp["has_glsl"] is True
        assert set(cp) == {"id", "kind", "label", "score", "iteration", "accepted", "has_glsl"}


def test_list_handles_empty_result():
    assert list_checkpoints({}) == []


# ---------------------------------------------------------------------------
# resolve_checkpoint
# ---------------------------------------------------------------------------

def test_resolve_candidate_by_id():
    cp = resolve_checkpoint(_result(), "candidate:cv_0")
    assert isinstance(cp, PipelineCheckpoint)
    assert cp.kind == "candidate"
    assert cp.glsl == CV_GLSL
    assert cp.candidate_id == "cv_0"


def test_resolve_candidate_selected_alias():
    cp = resolve_checkpoint(_result(), "candidate:selected")
    assert cp.glsl == SELECTED_GLSL
    assert cp.kind == "candidate"
    assert cp.score == pytest.approx(0.61)


def test_resolve_refinement_iteration():
    cp = resolve_checkpoint(_result(), "refinement:iter:2")
    assert cp.kind == "refinement_iter"
    assert cp.iteration == 2
    assert cp.glsl == ITER2_GLSL


def test_resolve_final_selected():
    cp = resolve_checkpoint(_result(), "final:selected")
    assert cp.kind == "final"
    assert cp.glsl == SELECTED_GLSL
    assert cp.score == pytest.approx(0.72)


def test_resolve_unknown_candidate_raises():
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "candidate:does_not_exist")


def test_resolve_unknown_iteration_raises():
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "refinement:iter:99")


def test_resolve_malformed_id_raises():
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "totally-bogus")


def test_resolve_checkpoint_without_glsl_raises():
    # rule_0 compiled to empty GLSL -> cannot seed a branch
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "candidate:rule_0")


def test_resolve_final_without_selected_glsl_raises():
    result = _result()
    result["selected_glsl"] = None
    with pytest.raises(CheckpointError):
        resolve_checkpoint(result, "final:selected")


# ---------------------------------------------------------------------------
# checkpoint_metadata
# ---------------------------------------------------------------------------

def test_checkpoint_metadata_summary_shape():
    cp = resolve_checkpoint(_result(), "refinement:iter:1")
    meta = checkpoint_metadata(cp)
    assert meta["id"] == "refinement:iter:1"
    assert meta["has_glsl"] is True
    assert "glsl" not in meta
    assert set(meta) == {"id", "kind", "label", "score", "iteration", "accepted", "has_glsl"}


# ---------------------------------------------------------------------------
# V2.1 fixtures: richer result with score_before, rejected iterations, run_id
# ---------------------------------------------------------------------------

ITER3_GLSL = "#define R 0.55\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"


def _result_v2() -> dict:
    """Extended result with score_before/delta fields and a rejected iteration."""
    r = _result()
    r["run_id"] = "run-abc"
    # Add score_before to existing iterations and inject a rejected iter
    r["refinement_history"][0]["score_before"] = 0.61  # iter 1
    r["refinement_history"][1]["score_before"] = 0.65  # iter 2
    # Rejected iteration with GLSL (score declined, not accepted)
    r["refinement_history"].insert(
        2,
        {
            "iteration": 3,
            "compile_glsl": ITER3_GLSL,
            "score_before": 0.67,
            "score_after": 0.55,
            "improved": False,
            "accepted": False,
            "changes_summary": "overfit patch",
            "human_goal_override": None,
        },
    )
    # Shift the old iteration-3 (no glsl) to iteration 4
    r["refinement_history"][3]["iteration"] = 4
    return r


# ---------------------------------------------------------------------------
# build_timeline
# ---------------------------------------------------------------------------


def test_build_timeline_includes_selected_candidate_entry():
    tl = build_timeline(_result_v2(), run_id="run-abc")
    ids = [e["id"] for e in tl]
    assert "candidate:selected" in ids
    sel = next(e for e in tl if e["id"] == "candidate:selected")
    assert sel["kind"] == "candidate"
    assert sel["label"] == "Selected baseline"
    assert sel["accepted"] is True
    assert sel["has_glsl"] is True
    assert sel["artifact_ids"]["shader"] == "checkpoint:candidate:selected:shader"
    assert sel["artifact_ids"]["render"] == "checkpoint:candidate:selected:render"


def test_build_timeline_includes_non_selected_candidate():
    tl = build_timeline(_result_v2())
    ids = [e["id"] for e in tl]
    assert "candidate:cv_0" in ids
    cv = next(e for e in tl if e["id"] == "candidate:cv_0")
    assert cv["accepted"] is False
    assert cv["artifact_ids"]["render"] == "checkpoint:candidate:cv_0:render"
    assert cv["artifact_ids"]["llm_io"] == "checkpoint:candidate:cv_0:llm_io"
    # Non-selected candidates must NOT have a shader artifact_id
    assert "shader" not in cv["artifact_ids"]


def test_build_timeline_ordering():
    """Candidates come before refinement iters, final is last."""
    tl = build_timeline(_result_v2())
    kinds = [e["kind"] for e in tl]
    # Find last candidate index and first refinement index
    last_candidate_idx = max(i for i, k in enumerate(kinds) if k == "candidate")
    first_iter_idx = min(i for i, k in enumerate(kinds) if k == "refinement_iter")
    final_idx = next(i for i, k in enumerate(kinds) if k == "final")
    assert last_candidate_idx < first_iter_idx
    assert first_iter_idx < final_idx


def test_build_timeline_rejected_iteration_present():
    """A rejected iteration (accepted=False) must still appear in the timeline."""
    tl = build_timeline(_result_v2())
    iter3 = next((e for e in tl if e["id"] == "refinement:iter:3"), None)
    assert iter3 is not None, "rejected iteration 3 must appear in timeline"
    assert iter3["kind"] == "refinement_iter"
    assert iter3["accepted"] is False


def test_build_timeline_delta_computed_when_both_scores_present():
    tl = build_timeline(_result_v2())
    iter1 = next(e for e in tl if e["id"] == "refinement:iter:1")
    assert iter1["score_before"] == pytest.approx(0.61)
    assert iter1["score"] == pytest.approx(0.65)
    assert iter1["delta"] == pytest.approx(0.65 - 0.61)


def test_build_timeline_delta_none_when_score_missing():
    r = _result_v2()
    # Remove score_before from iter 2
    del r["refinement_history"][1]["score_before"]
    tl = build_timeline(r)
    iter2 = next(e for e in tl if e["id"] == "refinement:iter:2")
    assert iter2["delta"] is None


def test_build_timeline_final_entry():
    tl = build_timeline(_result_v2(), run_id="run-abc")
    final = next(e for e in tl if e["id"] == "final:selected")
    assert final["kind"] == "final"
    assert final["label"] == "Current best"
    assert final["accepted"] is True
    assert final["score"] == pytest.approx(0.72)
    assert final["artifact_ids"]["shader"] == "checkpoint:final:selected:shader"
    assert final["run_id"] == "run-abc"


def test_build_timeline_run_id_propagation():
    tl = build_timeline(_result_v2(), run_id="explicit-id")
    assert all(e["run_id"] == "explicit-id" for e in tl)


def test_build_timeline_run_id_from_result_when_not_given():
    tl = build_timeline(_result_v2())
    assert all(e["run_id"] == "run-abc" for e in tl)


def test_build_timeline_empty_result():
    tl = build_timeline({})
    assert tl == []


def test_build_timeline_no_glsl_iteration_excluded():
    """Iterations without compile_glsl are excluded (same rule as list_checkpoints)."""
    tl = build_timeline(_result_v2())
    ids = [e["id"] for e in tl]
    # The old no-GLSL iteration (now iter 4) must not appear
    assert "refinement:iter:4" not in ids


def test_build_timeline_entry_fields_complete():
    """Every timeline entry has all required fields, none missing."""
    required = {
        "id", "run_id", "kind", "label", "iteration", "score",
        "score_before", "delta", "accepted", "human_goal_override",
        "changes_summary", "has_glsl", "artifact_ids",
    }
    tl = build_timeline(_result_v2())
    for entry in tl:
        missing = required - set(entry)
        assert not missing, f"Entry {entry['id']} missing fields: {missing}"


# ---------------------------------------------------------------------------
# save_timeline
# ---------------------------------------------------------------------------


def test_save_timeline_writes_timeline_json(tmp_path):
    saved = save_timeline(tmp_path, _result_v2(), run_id="run-abc")
    assert saved == tmp_path / "timeline.json"
    assert saved.exists()
    content = json.loads(saved.read_text())
    assert content["run_id"] == "run-abc"
    assert isinstance(content["timeline"], list)
    assert len(content["timeline"]) > 0


def test_save_timeline_accepts_string_path(tmp_path):
    saved = save_timeline(str(tmp_path), _result_v2())
    assert saved.exists()


def test_save_timeline_accepts_rundir_like(tmp_path):
    """run_dir with a .path attribute (like RunDir dataclass) is handled."""
    from types import SimpleNamespace
    fake_run_dir = SimpleNamespace(path=tmp_path)
    saved = save_timeline(fake_run_dir, _result_v2())
    assert saved.exists()


# ---------------------------------------------------------------------------
# resolve_checkpoint_artifact
# ---------------------------------------------------------------------------


def _make_run_dir(tmp_path):
    """Create a minimal fake run_dir with expected files."""
    (tmp_path / "candidates").mkdir()
    (tmp_path / "selected_shader.glsl").write_text("void main(){}")
    (tmp_path / "candidates" / "llm_0_render.png").write_bytes(b"\x89PNG")
    (tmp_path / "candidates" / "cv_0_render.png").write_bytes(b"\x89PNG")
    (tmp_path / "candidates" / "cv_0.json").write_text("{}")
    return tmp_path


def test_resolve_artifact_final_shader(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    p = resolve_checkpoint_artifact(_result(), "final:selected", "shader", run_dir=run_dir)
    assert p.name == "selected_shader.glsl"
    assert str(p).startswith(str(run_dir.resolve()))


def test_resolve_artifact_candidate_selected_shader(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    p = resolve_checkpoint_artifact(_result(), "candidate:selected", "shader", run_dir=run_dir)
    assert p.name == "selected_shader.glsl"


def test_resolve_artifact_candidate_selected_render(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    p = resolve_checkpoint_artifact(_result(), "candidate:selected", "render", run_dir=run_dir)
    assert p.name == "llm_0_render.png"
    assert "candidates" in str(p)


def test_resolve_artifact_non_selected_candidate_render(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    p = resolve_checkpoint_artifact(_result(), "candidate:cv_0", "render", run_dir=run_dir)
    assert p.name == "cv_0_render.png"
    assert str(p).startswith(str(run_dir.resolve()))


def test_resolve_artifact_non_selected_candidate_llm_io(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    p = resolve_checkpoint_artifact(_result(), "candidate:cv_0", "llm_io", run_dir=run_dir)
    assert p.name == "cv_0.json"


def test_resolve_artifact_candidate_shader_raises(tmp_path):
    """Non-selected candidate shader kind has no file — must raise."""
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(CheckpointError):
        resolve_checkpoint_artifact(_result(), "candidate:cv_0", "shader", run_dir=run_dir)


def test_resolve_artifact_refinement_any_kind_raises(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(CheckpointError):
        resolve_checkpoint_artifact(_result(), "refinement:iter:1", "shader", run_dir=run_dir)


def test_resolve_artifact_unknown_kind_raises(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(CheckpointError):
        resolve_checkpoint_artifact(_result(), "final:selected", "video", run_dir=run_dir)


def test_resolve_artifact_no_run_dir_raises():
    """run_dir=None and absent from result must raise CheckpointError."""
    with pytest.raises(CheckpointError):
        resolve_checkpoint_artifact(_result(), "final:selected", "shader", run_dir=None)


def test_resolve_artifact_path_traversal_rejected(tmp_path):
    """Path traversal in candidate id must be rejected."""
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(CheckpointError):
        resolve_checkpoint_artifact(
            _result(), "candidate:../../etc/passwd", "render", run_dir=run_dir
        )


def test_resolve_artifact_unknown_candidate_id_raises(tmp_path):
    """A well-formed but unknown candidate id must raise."""
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(CheckpointError):
        resolve_checkpoint_artifact(_result(), "candidate:ghost_0", "render", run_dir=run_dir)


def test_resolve_artifact_path_stays_inside_run_dir(tmp_path):
    """Resolved path must be contained within run_dir (containment check)."""
    run_dir = _make_run_dir(tmp_path)
    p = resolve_checkpoint_artifact(_result(), "candidate:cv_0", "render", run_dir=run_dir)
    # Must not raise and must be inside run_dir
    p.relative_to(run_dir.resolve())  # raises ValueError if not inside


# ---------------------------------------------------------------------------
# Important 1: build_timeline selected candidate must be first among candidates
# ---------------------------------------------------------------------------


def test_build_timeline_selected_candidate_is_first_entry():
    result = _result_v2()
    cands = result["scoreboard"]["candidates"]
    cands.append(cands.pop(0))  # move the selected candidate (llm_0) to the end of the list
    tl = build_timeline(result)
    assert tl[0]["id"] == "candidate:selected"


# ---------------------------------------------------------------------------
# Important 2: resolve_checkpoint_artifact accepts RunDir-like object in result
# ---------------------------------------------------------------------------


def test_resolve_artifact_rundir_from_result(tmp_path):
    from types import SimpleNamespace
    run_dir = _make_run_dir(tmp_path)
    result = _result()
    result["run_dir"] = SimpleNamespace(path=str(run_dir))
    p = resolve_checkpoint_artifact(result, "final:selected", "shader")
    assert p.name == "selected_shader.glsl"

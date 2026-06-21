"""Unit tests for store.rehydrate_run — rebuild a completed run's result from
its persisted run_dir when it has fallen out of the in-memory _run_store
(LRU eviction at 100, or a uvicorn --reload wiping the process).

TDD: written before implementation; each first fails for the right reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2s_agent import store
from p2s_agent.orchestration.run_index import RunLineageRecord, append_run_created


def _seed_index(tmp_index: Path, run_id: str, run_dir: Path, *, status: str = "completed") -> None:
    rec = RunLineageRecord(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        source_checkpoint_id=None,
        source_checkpoint_label=None,
        mode=None,
        feedback=None,
        title=None,
        status=status,
        run_dir=str(run_dir),
        created_at=1.0,
    )
    append_run_created(rec, path=str(tmp_index))


@pytest.fixture
def idx(tmp_path, monkeypatch):
    p = tmp_path / "run_index.jsonl"
    monkeypatch.setattr(store, "_RUN_INDEX_PATH", str(p))
    return p


def _make_run_dir(tmp_path: Path, name: str = "2026-06-21_png-shader_single_run_x") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    return d


def test_rehydrate_run_loads_result_json_exactly(tmp_path, idx):
    rd = _make_run_dir(tmp_path)
    stored = {
        "run_id": "run_x",
        "run_dir": str(rd),
        "selected_glsl": "void main(){}",
        "scoreboard": {"selected_id": "seed_0"},
        "quality_router": {"final_score": 0.77},
        "status": "completed",
        "current_phase": "done",
    }
    (rd / "result.json").write_text(json.dumps(stored), encoding="utf-8")
    _seed_index(idx, "run_x", rd)

    out = store.rehydrate_run("run_x")

    assert out is not None
    assert out["selected_glsl"] == "void main(){}"
    assert out["status"] == "completed"
    assert out["quality_router"]["final_score"] == 0.77


def test_rehydrate_run_reconstructs_from_artifacts_when_no_result_json(tmp_path, idx):
    rd = _make_run_dir(tmp_path)
    (rd / "selected_shader.glsl").write_text("void main(){ gl_FragColor=vec4(1.0); }", encoding="utf-8")
    (rd / "scoreboard.json").write_text(json.dumps({"selected_id": "llm_0"}), encoding="utf-8")
    (rd / "objective_metrics.json").write_text(json.dumps({"ssim": 0.9}), encoding="utf-8")
    (rd / "quality_router.json").write_text(json.dumps({"final_score": 0.81}), encoding="utf-8")
    _seed_index(idx, "run_y", rd)

    out = store.rehydrate_run("run_y")

    assert out is not None
    assert out["selected_glsl"].startswith("void main")
    assert out["status"] == "completed"
    assert out["scoreboard"]["selected_id"] == "llm_0"
    assert out["objective_metrics"]["ssim"] == 0.9
    assert out["quality_router"]["final_score"] == 0.81
    assert out["run_id"] == "run_y"


def test_rehydrate_run_unknown_run_id_returns_none(idx):
    assert store.rehydrate_run("nope") is None


def test_rehydrate_run_missing_dir_returns_none(tmp_path, idx):
    gone = tmp_path / "2026-06-21_png-shader_single_run_gone"
    _seed_index(idx, "run_gone", gone)  # index points at a dir that doesn't exist
    assert store.rehydrate_run("run_gone") is None


def test_rehydrate_run_no_glsl_and_no_result_json_returns_none(tmp_path, idx):
    rd = _make_run_dir(tmp_path)
    (rd / "manifest.json").write_text("{}", encoding="utf-8")  # present but no shader
    _seed_index(idx, "run_empty", rd)
    assert store.rehydrate_run("run_empty") is None

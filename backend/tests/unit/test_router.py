"""API contract tests for the PNG-to-Shader router."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from app.routers.png_shader import _run_store, router


# ---------------------------------------------------------------------------
# Autouse fixture: isolate every test from the real run_index.jsonl and
# variant_groups directory.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_run_index(tmp_path, monkeypatch):
    """Redirect all run-index writes to a per-test temp file so the real
    backend/test_results/run_index.jsonl is never touched during tests.
    Also redirect variant_groups, draw_sessions and preferences writes to
    isolated temp directories."""
    monkeypatch.setattr(
        "app.routers.png_shader._RUN_INDEX_PATH",
        str(tmp_path / "run_index.jsonl"),
    )
    monkeypatch.setattr(
        "app.routers.png_shader._VARIANT_GROUPS_ROOT",
        str(tmp_path / "vg"),
    )
    monkeypatch.setattr(
        "app.routers.png_shader._DRAW_SESSIONS_ROOT",
        str(tmp_path / "ds"),
    )
    monkeypatch.setattr(
        "app.routers.png_shader._PREFERENCES_ROOT",
        str(tmp_path / "prefs"),
    )
    monkeypatch.setattr(
        "app.routers.png_shader._FUSIONS_ROOT",
        str(tmp_path / "fusions_root"),
    )

FastAPI = fastapi.FastAPI
TestClient = testclient.TestClient


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _png_bytes(tmp_path) -> bytes:
    path = tmp_path / "input.png"
    Image.new("RGBA", (32, 32), (180, 80, 40, 255)).save(path)
    return path.read_bytes()


def _wait_for_completion(client: TestClient, run_id: str, timeout_seconds: float = 10.0) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/png-shader/status/{run_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] != "running":
            return data
        time.sleep(0.05)
    raise AssertionError(f"PNG shader run {run_id} did not complete within {timeout_seconds}s")


def test_run_returns_run_id_then_status_returns_full_pipeline_result(tmp_path):
    _run_store.clear()
    client = _client()

    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"]
    assert data["status"] == "running"

    data = _wait_for_completion(client, data["run_id"])
    assert data["status"] == "completed"
    assert data["preprocess"]["width"] == 32
    assert data["selected_glsl"].startswith("#version 300 es")
    assert data["quality_router"]["final_score"] >= 0.0


def test_run_accepts_input_spec_overrides(tmp_path):
    _run_store.clear()
    client = _client()
    overrides = {"target": {"resolution": [256, 128], "max_shader_chars": 6000}}

    response = client.post(
        "/png-shader/run",
        data={"input_spec_json": json.dumps(overrides)},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )

    assert response.status_code == 200
    data = _wait_for_completion(client, response.json()["run_id"])
    assert data["input_spec"]["target"]["resolution"] == [256, 128]
    assert data["input_spec"]["target"]["max_shader_chars"] == 6000
    assert data["input_spec"]["input_image"].endswith("input.png")


def test_status_returns_cached_full_result(tmp_path):
    _run_store.clear()
    client = _client()

    run_response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = run_response.json()["run_id"]

    data = _wait_for_completion(client, run_id)
    assert data["run_id"] == run_id
    assert data["selected_glsl"]


def test_run_initialises_strategy_and_stop_flag_in_store(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = response.json()["run_id"]
    _wait_for_completion(client, run_id)
    data = client.get(f"/png-shader/status/{run_id}").json()
    assert "strategy" in data
    assert "stop_requested" in data
    assert data["stop_requested"] is False
    assert data["strategy_revision"] == 1
    assert data["strategy"]["refinement_threshold"] == 0.80


def test_patch_strategy_updates_running_run(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = response.json()["run_id"]
    patch = client.patch(
        f"/png-shader/runs/{run_id}/strategy",
        json={"quality": {"refinement_threshold": 0.65}},
    )
    if patch.status_code == 409:
        pytest.skip("Pipeline finished before patch could land — timing-sensitive on fast machines")
    assert patch.status_code == 200, patch.text
    body = patch.json()
    assert body["strategy"]["refinement_threshold"] == 0.65
    assert body["strategy_revision"] >= 2


def test_patch_strategy_rejects_invalid_value(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = response.json()["run_id"]
    patch = client.patch(
        f"/png-shader/runs/{run_id}/strategy",
        json={"quality": {"refinement_threshold": 2.0}},
    )
    if patch.status_code == 409:
        pytest.skip("Pipeline finished before patch could land")
    assert patch.status_code == 422


def test_patch_strategy_404_for_unknown_run():
    _run_store.clear()
    client = _client()
    patch = client.patch(
        "/png-shader/runs/missing_run/strategy",
        json={"quality": {"refinement_threshold": 0.7}},
    )
    assert patch.status_code == 404


def test_patch_strategy_409_for_completed_run(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = response.json()["run_id"]
    _wait_for_completion(client, run_id)
    patch = client.patch(
        f"/png-shader/runs/{run_id}/strategy",
        json={"quality": {"refinement_threshold": 0.65}},
    )
    assert patch.status_code == 409


def test_stop_sets_flag_for_running_run(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = response.json()["run_id"]
    stop = client.post(f"/png-shader/runs/{run_id}/stop")
    if stop.status_code == 409:
        pytest.skip("Pipeline finished before stop request could land")
    assert stop.status_code == 200
    assert stop.json() == {"stopping": True}
    # idempotent
    stop2 = client.post(f"/png-shader/runs/{run_id}/stop")
    assert stop2.status_code in (200, 409)
    _wait_for_completion(client, run_id)
    data = client.get(f"/png-shader/status/{run_id}").json()
    assert data["stop_requested"] is True


def test_stop_404_for_unknown_run():
    _run_store.clear()
    client = _client()
    stop = client.post("/png-shader/runs/missing/stop")
    assert stop.status_code == 404


def test_stop_409_for_completed_run(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = response.json()["run_id"]
    _wait_for_completion(client, run_id)
    stop = client.post(f"/png-shader/runs/{run_id}/stop")
    assert stop.status_code == 409


def test_run_accepts_seed_glsl_and_defaults_refinement_on(tmp_path, monkeypatch):
    _run_store.clear()
    captured: dict = {}

    def fake_pipeline(image_path, input_spec=None, run_id=None, *, seed_glsl=None, **kwargs):
        captured["seed_glsl"] = seed_glsl
        captured["refinement_mode"] = (
            (input_spec or {}).get("quality", {}).get("refinement_mode")
        )
        return {
            "run_id": run_id,
            "selected_glsl": seed_glsl or "",
            "scoreboard": {},
            "quality_router": {},
            "refinement_summary": {},
        }

    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", fake_pipeline
    )
    client = _client()
    seed = "void mainImage(out vec4 c, in vec2 p){ c = vec4(0.3); }"

    response = client.post(
        "/png-shader/run",
        data={"seed_glsl": seed},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert response.status_code == 200
    data = _wait_for_completion(client, response.json()["run_id"])
    assert data["status"] == "completed"
    assert "mainImage" in captured["seed_glsl"]
    assert captured["refinement_mode"] == "on"


def test_run_rejects_blank_seed_glsl(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        data={"seed_glsl": "   "},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert response.status_code == 422


def test_run_rejects_non_object_input_spec_with_seed_glsl(tmp_path):
    _run_store.clear()
    client = _client()
    seed = "void mainImage(out vec4 c, in vec2 p){ c = vec4(0.3); }"
    response = client.post(
        "/png-shader/run",
        data={"seed_glsl": seed, "input_spec_json": "[]"},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert response.status_code == 422
    assert "input_spec_json must decode to an object" in response.text


def test_publish_partial_merges_into_running_store():
    from app.routers.png_shader import _publish_partial_to_store

    _run_store.clear()
    _run_store["run_x"] = {
        "run_id": "run_x",
        "status": "running",
        "strategy": {"refinement_threshold": 0.8},
        "stop_requested": False,
        "strategy_revision": 1,
    }

    _publish_partial_to_store(
        "run_x",
        {"scoreboard": {"selected_id": "seed_0"}, "refinement_history": [{"iteration": 1}]},
    )

    stored = _run_store["run_x"]
    assert stored["status"] == "running"
    assert stored["scoreboard"]["selected_id"] == "seed_0"
    assert stored["refinement_history"] == [{"iteration": 1}]
    assert stored["strategy"] == {"refinement_threshold": 0.8}
    assert stored["stop_requested"] is False
    assert stored["strategy_revision"] == 1


def test_publish_partial_noop_when_terminal():
    from app.routers.png_shader import _publish_partial_to_store

    _run_store.clear()
    _run_store["run_done"] = {"run_id": "run_done", "status": "completed"}
    _publish_partial_to_store("run_done", {"scoreboard": {"x": 1}})
    assert "scoreboard" not in _run_store["run_done"]

    # A partial arriving after a crash must not mutate a failed run either.
    _run_store["run_failed"] = {"run_id": "run_failed", "status": "failed"}
    _publish_partial_to_store("run_failed", {"scoreboard": {"x": 1}})
    assert "scoreboard" not in _run_store["run_failed"]


def test_publish_partial_noop_when_missing():
    from app.routers.png_shader import _publish_partial_to_store

    _run_store.clear()
    _publish_partial_to_store("ghost", {"scoreboard": {}})
    assert "ghost" not in _run_store


# ---------------------------------------------------------------------------
# Human-in-loop: checkpoints + branch-refine (V1.1)
# ---------------------------------------------------------------------------

_BRANCH_GLSL = (
    "#version 300 es\nprecision highp float;\n"
    "void mainImage(out vec4 c, in vec2 p){ c = vec4(0.7); }"
)


def _seed_parent(tmp_path, run_id="run_parent", status="completed", with_reference=True):
    parent_dir = tmp_path / run_id
    parent_dir.mkdir(parents=True, exist_ok=True)
    if with_reference:
        Image.new("RGBA", (32, 32), (180, 80, 40, 255)).save(parent_dir / "reference_input.png")
    _run_store[run_id] = {
        "run_id": run_id,
        "status": status,
        "run_dir": str(parent_dir),
        "selected_glsl": _BRANCH_GLSL,
        "quality_router": {"final_score": 0.7},
        "scoreboard": {
            "selected_id": "llm_0",
            "candidates": [{
                "id": "llm_0",
                "source": "llm",
                "selected": True,
                "previewable": True,
                "compile_glsl": _BRANCH_GLSL,
                "final_score": 0.7,
            }],
        },
        "refinement_history": [],
        "strategy": {"refinement_threshold": 0.8},
        "stop_requested": False,
        "strategy_revision": 1,
    }
    return parent_dir


def test_checkpoints_lists_branchable_points(tmp_path):
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.get("/png-shader/runs/run_parent/checkpoints")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run_parent"
    ids = {cp["id"] for cp in body["checkpoints"]}
    assert "candidate:llm_0" in ids
    assert "final:selected" in ids


def test_checkpoints_404_for_unknown_run():
    _run_store.clear()
    client = _client()
    assert client.get("/png-shader/runs/ghost/checkpoints").status_code == 404


def _fake_branch_pipeline(captured):
    def fake_pipeline(image_path, input_spec=None, run_id=None, *, seed_glsl=None,
                      human_feedback_notes=None, directed_acceptance=None,
                      force_first_refinement_iteration=False, lineage=None,
                      extra_artifacts=None, **kwargs):
        captured["seed_glsl"] = seed_glsl
        captured["human_feedback_notes"] = human_feedback_notes
        captured["force_first"] = force_first_refinement_iteration
        captured["lineage"] = lineage
        captured["directed_acceptance"] = directed_acceptance
        captured["extra_artifacts"] = extra_artifacts
        captured["image_path"] = str(image_path)
        return {
            "run_id": run_id,
            "selected_glsl": seed_glsl or "",
            "scoreboard": {},
            "quality_router": {},
            "refinement_summary": {},
            "lineage": lineage,
        }
    return fake_pipeline


def test_branch_refine_creates_child_run(tmp_path, monkeypatch):
    _run_store.clear()
    parent_dir = _seed_parent(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "make the reflection stronger",
            "mode": "refine",
            "locks": {"preserve_layout": True},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    child_id = body["run_id"]
    assert child_id != "run_parent"
    assert body["status"] == "running"
    assert body["parent_run_id"] == "run_parent"
    assert body["lineage"]["source_checkpoint_id"] == "final:selected"

    _wait_for_completion(client, child_id)
    assert captured["seed_glsl"] == _BRANCH_GLSL
    assert any("[HUMAN GOAL] make the reflection stronger" in n
               for n in captured["human_feedback_notes"])
    assert any("[LOCK] Preserve layout" in n for n in captured["human_feedback_notes"])
    assert captured["force_first"] is True
    assert captured["lineage"]["parent_run_id"] == "run_parent"
    # branch worker must NOT delete the parent reference image
    assert (parent_dir / "reference_input.png").exists()


def test_branch_refine_404_for_unknown_parent():
    _run_store.clear()
    client = _client()
    resp = client.post(
        "/png-shader/runs/ghost/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "x", "mode": "refine"},
    )
    assert resp.status_code == 404


def test_branch_refine_422_for_unknown_checkpoint(tmp_path):
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "refinement:iter:99", "feedback": "x", "mode": "refine"},
    )
    assert resp.status_code == 422


def test_branch_refine_422_for_empty_feedback_in_refine(tmp_path):
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "   ", "mode": "refine"},
    )
    assert resp.status_code == 422


def test_branch_refine_409_when_parent_has_no_run_dir():
    _run_store.clear()
    _run_store["run_norun"] = {
        "run_id": "run_norun", "status": "completed", "selected_glsl": _BRANCH_GLSL,
        "scoreboard": {"selected_id": "llm_0", "candidates": [
            {"id": "llm_0", "selected": True, "previewable": True, "compile_glsl": _BRANCH_GLSL, "final_score": 0.7}]},
        "refinement_history": [], "quality_router": {"final_score": 0.7},
    }
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_norun/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "x", "mode": "refine"},
    )
    assert resp.status_code == 409


def test_branch_refine_409_when_reference_missing(tmp_path):
    _run_store.clear()
    _seed_parent(tmp_path, with_reference=False)
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "x", "mode": "refine"},
    )
    assert resp.status_code == 409


def test_branch_refine_continue_mode_allows_empty_feedback(tmp_path, monkeypatch):
    _run_store.clear()
    _seed_parent(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "", "mode": "continue"},
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])
    assert any("[MODE] Continue" in n for n in captured["human_feedback_notes"])


def test_branch_refine_stop_parent_sets_flag(tmp_path, monkeypatch):
    _run_store.clear()
    _seed_parent(tmp_path, status="running")
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "brighter",
              "mode": "refine", "stop_parent": True},
    )
    assert resp.status_code == 200, resp.text
    assert _run_store["run_parent"]["stop_requested"] is True


def test_branch_refine_refine_mode_enables_directed_acceptance(tmp_path, monkeypatch):
    _run_store.clear()
    _seed_parent(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "brighter water", "mode": "refine"},
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])
    da = captured["directed_acceptance"]
    assert da["enabled"] is True
    assert da["feedback"] == "brighter water"
    assert da["score_drop_tolerance"] > 0.0
    # directed_acceptance must be JSON-serialisable (no callables)
    import json as _json
    _json.dumps(da)


def test_branch_refine_polish_mode_zero_tolerance(tmp_path, monkeypatch):
    _run_store.clear()
    _seed_parent(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "cleaner edges", "mode": "polish"},
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])
    assert captured["directed_acceptance"]["enabled"] is True
    assert captured["directed_acceptance"]["score_drop_tolerance"] == 0.0


def test_branch_refine_continue_mode_disables_directed_acceptance(tmp_path, monkeypatch):
    _run_store.clear()
    _seed_parent(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"checkpoint_id": "final:selected", "feedback": "", "mode": "continue"},
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])
    da = captured["directed_acceptance"]
    assert da is None or da.get("enabled") is False


# ---------------------------------------------------------------------------
# M3-3: run index lifecycle tests
# ---------------------------------------------------------------------------

def test_run_index_root_run_completed(tmp_path, monkeypatch):
    """POST /run with a fake pipeline that emits a run_dir partial; after completion
    the folded run-index record should show status=completed with run_dir and
    root/parent lineage for a root (non-branch) run."""
    _run_store.clear()

    # Use a deterministic index path so we can read it back.
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    fake_run_dir = str(tmp_path / "rd")

    def fake_pipeline(image_path, input_spec=None, run_id=None, *, publish_partial=None, **kwargs):
        if publish_partial:
            publish_partial({"run_dir": fake_run_dir})
        return {
            "run_id": run_id,
            "run_dir": fake_run_dir,
            "selected_glsl": "",
            "scoreboard": {},
            "quality_router": {"final_score": 0.85},
            "refinement_summary": {},
        }

    monkeypatch.setattr("app.routers.png_shader.run_png_shader_pipeline", fake_pipeline)
    client = _client()

    resp = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    _wait_for_completion(client, run_id)

    from app.pipeline.run_index import load_run_index
    records = load_run_index(path=idx)

    assert run_id in records, f"run_id {run_id!r} not in index; keys={list(records)}"
    rec = records[run_id]
    assert rec.status == "completed"
    assert rec.run_dir == fake_run_dir
    assert rec.root_run_id == run_id
    assert rec.parent_run_id is None


def test_run_index_branch_run_completed(tmp_path, monkeypatch):
    """branch-refine child's folded record should carry correct lineage fields."""
    _run_store.clear()

    idx = str(tmp_path / "ri_branch.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    _seed_parent(tmp_path)
    captured: dict = {}

    def fake_pipeline_branch(image_path, input_spec=None, run_id=None, *,
                             seed_glsl=None, publish_partial=None, lineage=None,
                             human_feedback_notes=None, directed_acceptance=None,
                             force_first_refinement_iteration=False,
                             extra_artifacts=None, **kwargs):
        captured["lineage"] = lineage
        return {
            "run_id": run_id,
            "selected_glsl": seed_glsl or "",
            "scoreboard": {},
            "quality_router": {"final_score": 0.72},
            "refinement_summary": {},
            "lineage": lineage,
        }

    monkeypatch.setattr("app.routers.png_shader.run_png_shader_pipeline", fake_pipeline_branch)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "more contrast",
            "mode": "refine",
        },
    )
    assert resp.status_code == 200, resp.text
    child_id = resp.json()["run_id"]
    _wait_for_completion(client, child_id)

    from app.pipeline.run_index import load_run_index
    records = load_run_index(path=idx)

    assert child_id in records, f"child {child_id!r} not in index; keys={list(records)}"
    rec = records[child_id]
    assert rec.parent_run_id == "run_parent"
    assert rec.root_run_id == "run_parent"
    assert rec.source_checkpoint_id == "final:selected"
    assert rec.mode == "refine"
    assert rec.status == "completed"


def test_run_index_failed_run(tmp_path, monkeypatch):
    """A pipeline that raises should result in status=failed in the index."""
    _run_store.clear()

    idx = str(tmp_path / "ri_fail.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    def failing_pipeline(image_path, input_spec=None, run_id=None, **kwargs):
        raise RuntimeError("simulated pipeline crash")

    monkeypatch.setattr("app.routers.png_shader.run_png_shader_pipeline", failing_pipeline)
    client = _client()

    resp = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    _wait_for_completion(client, run_id)

    from app.pipeline.run_index import load_run_index
    records = load_run_index(path=idx)

    assert run_id in records, f"run_id {run_id!r} not in index; keys={list(records)}"
    rec = records[run_id]
    assert rec.status == "failed"
    assert rec.run_dir is None


# ---------------------------------------------------------------------------
# M3-4: timeline / branches / metadata / artifacts endpoints
# ---------------------------------------------------------------------------

def _seed_index(idx_path, **fields):
    """Helper: append a RunLineageRecord to the isolated test index."""
    from app.pipeline.run_index import RunLineageRecord, append_run_created
    rec = RunLineageRecord(
        run_id=fields["run_id"],
        root_run_id=fields.get("root_run_id", fields["run_id"]),
        parent_run_id=fields.get("parent_run_id"),
        source_checkpoint_id=fields.get("source_checkpoint_id"),
        source_checkpoint_label=fields.get("source_checkpoint_label"),
        mode=fields.get("mode"),
        feedback=fields.get("feedback"),
        title=fields.get("title"),
        status=fields.get("status", "completed"),
        run_dir=fields.get("run_dir"),
        created_at=fields.get("created_at", 1.0),
        final_score=fields.get("final_score"),
    )
    append_run_created(rec, path=idx_path)


# --- /timeline ---

def test_timeline_running_from_store(tmp_path):
    """GET /timeline for a store entry returns timeline with candidate + final entries."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.get("/png-shader/runs/run_parent/timeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run_parent"
    ids = {e["id"] for e in body["timeline"]}
    assert "candidate:selected" in ids
    assert "final:selected" in ids


def test_timeline_404_unknown_run(tmp_path, monkeypatch):
    """GET /timeline for an unknown run_id returns 404."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    client = _client()
    resp = client.get("/png-shader/runs/run_missing/timeline")
    assert resp.status_code == 404


def test_timeline_from_timeline_json_on_disk(tmp_path, monkeypatch):
    """GET /timeline for an evicted run reads timeline.json from disk."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    run_dir = tmp_path / "run_ev"
    run_dir.mkdir()
    timeline_data = {
        "run_id": "run_ev",
        "timeline": [{"id": "final:selected", "kind": "final", "label": "Current best"}],
    }
    (run_dir / "timeline.json").write_text(json.dumps(timeline_data))
    _seed_index(idx, run_id="run_ev", run_dir=str(run_dir))

    client = _client()
    resp = client.get("/png-shader/runs/run_ev/timeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run_ev"
    assert any(e["id"] == "final:selected" for e in body["timeline"])


def test_timeline_store_missing_run_dir_none(tmp_path, monkeypatch):
    """GET /timeline for an index entry with run_dir=None returns empty timeline (no path access)."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    _seed_index(idx, run_id="run_pending", run_dir=None, status="pending")

    client = _client()
    resp = client.get("/png-shader/runs/run_pending/timeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["timeline"] == []
    assert body["status"] == "pending"


# --- /branches ---

def test_branches_child_tree(tmp_path, monkeypatch):
    """GET /branches for a child run returns tree rooted at root_a."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    _seed_index(idx, run_id="root_a", run_dir=str(tmp_path / "root_a"), created_at=1.0)
    _seed_index(idx, run_id="child_b", root_run_id="root_a", parent_run_id="root_a",
                source_checkpoint_id="final:selected", created_at=2.0)

    client = _client()
    resp = client.get("/png-shader/runs/child_b/branches")
    assert resp.status_code == 200
    body = resp.json()
    assert body["root_run_id"] == "root_a"
    assert body["active_run_id"] == "child_b"
    tree = body["tree"]
    assert tree["run_id"] == "root_a"
    child_ids = [c["run_id"] for c in tree["children"]]
    assert "child_b" in child_ids


def test_branches_404_unknown(tmp_path, monkeypatch):
    """GET /branches for unknown run_id returns 404."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    client = _client()
    resp = client.get("/png-shader/runs/ghost/branches")
    assert resp.status_code == 404


def test_branches_404_when_root_missing(tmp_path, monkeypatch):
    """GET /branches for a child whose root_run_id points to a non-existent root
    must return 404 (not 500) — build_branch_tree raises RunIndexError in this case."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    _seed_index(idx, run_id="orphan", root_run_id="ghost_root", parent_run_id="ghost_root")
    client = _client()
    assert client.get("/png-shader/runs/orphan/branches").status_code == 404


def test_branches_store_only_single_node(tmp_path, monkeypatch):
    """GET /branches for a run that exists only in _run_store returns a synthesised
    single-node tree with no children."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    _run_store["run_solo"] = {"run_id": "run_solo", "status": "completed", "run_dir": None}
    client = _client()
    body = client.get("/png-shader/runs/run_solo/branches").json()
    assert body["root_run_id"] == "run_solo"
    assert body["active_run_id"] == "run_solo"
    assert body["tree"]["run_id"] == "run_solo"
    assert body["tree"]["children"] == []


# --- /metadata ---

def test_metadata_update(tmp_path, monkeypatch):
    """PATCH /metadata with allowed fields persists them to the index."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    _seed_index(idx, run_id="run_m")

    client = _client()
    resp = client.patch(
        "/png-shader/runs/run_m/metadata",
        json={"title": "my branch", "favorite": True, "tags": ["a"]},
    )
    assert resp.status_code == 200

    from app.pipeline.run_index import load_run_index
    rec = load_run_index(path=idx)["run_m"]
    assert rec.title == "my branch"
    assert rec.favorite is True
    assert "a" in rec.tags


def test_metadata_422_disallowed_key(tmp_path, monkeypatch):
    """PATCH /metadata with a disallowed key returns 422."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    _seed_index(idx, run_id="run_m2")

    client = _client()
    resp = client.patch("/png-shader/runs/run_m2/metadata", json={"status": "x"})
    assert resp.status_code == 422


def test_metadata_404_unknown(tmp_path, monkeypatch):
    """PATCH /metadata for unknown run_id returns 404."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    client = _client()
    resp = client.patch("/png-shader/runs/ghost_run/metadata", json={"title": "x"})
    assert resp.status_code == 404


# --- /artifacts ---

def test_artifacts_selected_shader(tmp_path, monkeypatch):
    """GET /artifacts/selected_shader returns the file content."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    parent_dir = _seed_parent(tmp_path)
    shader_text = _BRANCH_GLSL
    (parent_dir / "selected_shader.glsl").write_text(shader_text)

    client = _client()
    resp = client.get("/png-shader/runs/run_parent/artifacts/selected_shader")
    assert resp.status_code == 200
    assert shader_text in resp.text


def test_artifacts_409_when_run_dir_none(tmp_path, monkeypatch):
    """GET /artifacts for a run with no run_dir returns 409."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    # A store entry with no run_dir
    _run_store["run_nodir"] = {
        "run_id": "run_nodir",
        "status": "running",
        "run_dir": None,
        "selected_glsl": _BRANCH_GLSL,
        "scoreboard": {},
        "strategy": {},
        "stop_requested": False,
        "strategy_revision": 1,
    }
    client = _client()
    resp = client.get("/png-shader/runs/run_nodir/artifacts/selected_shader")
    assert resp.status_code == 409


def test_artifacts_404_file_missing(tmp_path, monkeypatch):
    """GET /artifacts for a valid artifact_id but missing file returns 404."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    parent_dir = _seed_parent(tmp_path)
    # Do NOT write selected_shader.glsl — it should 404.
    client = _client()
    resp = client.get("/png-shader/runs/run_parent/artifacts/selected_shader")
    assert resp.status_code == 404


def test_artifacts_422_unknown_artifact_id(tmp_path, monkeypatch):
    """GET /artifacts with a completely unknown artifact_id returns 422."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    parent_dir = _seed_parent(tmp_path)
    client = _client()
    resp = client.get("/png-shader/runs/run_parent/artifacts/bogus_artifact")
    assert resp.status_code == 422


def test_artifacts_422_bad_candidate_id_in_scoreboard(tmp_path, monkeypatch):
    """GET /artifacts with a candidate id not in scoreboard is rejected by
    resolve_checkpoint_artifact → 422.

    We use a checkpoint: prefix with an id that is syntactically valid but
    not present in the scoreboard (which has only 'llm_0').
    """
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    parent_dir = _seed_parent(tmp_path)
    # candidate:ghost_99 is not in the scoreboard — resolver should reject it.
    client = _client()
    resp = client.get("/png-shader/runs/run_parent/artifacts/checkpoint:candidate:ghost_99:render")
    assert resp.status_code == 422


# Traversal note: Starlette's {artifact_id:path} converter decodes %2e%2e%2f to "../"
# but does NOT collapse it — the decoded string reaches this handler. Traversal is blocked
# at the APPLICATION layer: artifact_id must begin with "selected_shader"/"selected_render"/
# "checkpoint:", and resolve_checkpoint_artifact enforces a candidate-id regex + path
# containment + suffix allowlist (unit-tested in test_checkpoints.py).
# Here we verify the 422 path for a malformed checkpoint: prefix that doesn't split into
# two non-empty parts.
def test_artifacts_422_malformed_checkpoint_prefix(tmp_path, monkeypatch):
    """GET /artifacts/checkpoint:<id_with_no_trailing_kind> returns 422."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    parent_dir = _seed_parent(tmp_path)
    client = _client()
    # Note: "checkpoint:final:selected" is NOT a valid full artifact id — it parses to
    # checkpoint_id="final", kind="selected", but the resolver rejects it (the valid final-shader
    # artifact id is "checkpoint:final:selected:shader"). We test a structurally malformed case:
    # "checkpoint:onlyone" has no colon after "checkpoint:", so rsplit(":", 1) on "onlyone"
    # gives a single-element list → len == 1, not 2 → 422.
    resp = client.get("/png-shader/runs/run_parent/artifacts/checkpoint:onlyone")
    # rsplit(":", 1) on "onlyone" gives ["onlyone"] — len == 1, not 2 → 422.
    assert resp.status_code == 422


# --- save_timeline wiring ---

def test_save_timeline_written_on_success(tmp_path, monkeypatch):
    """Worker writes timeline.json to run_dir on successful completion."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    fake_run_dir = tmp_path / "rd_timeline"
    fake_run_dir.mkdir()

    def fake_pipeline(image_path, input_spec=None, run_id=None, *, publish_partial=None, **kwargs):
        if publish_partial:
            publish_partial({"run_dir": str(fake_run_dir)})
        return {
            "run_id": run_id,
            "run_dir": str(fake_run_dir),
            "selected_glsl": _BRANCH_GLSL,
            "scoreboard": {
                "selected_id": "llm_0",
                "candidates": [{"id": "llm_0", "selected": True, "previewable": True,
                                "compile_glsl": _BRANCH_GLSL, "final_score": 0.8}],
            },
            "quality_router": {"final_score": 0.8},
            "refinement_summary": {},
            "refinement_history": [],
        }

    monkeypatch.setattr("app.routers.png_shader.run_png_shader_pipeline", fake_pipeline)
    client = _client()
    resp = client.post(
        "/png-shader/run",
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    run_id = resp.json()["run_id"]
    _wait_for_completion(client, run_id)

    timeline_file = fake_run_dir / "timeline.json"
    assert timeline_file.exists(), "save_timeline must write timeline.json to run_dir"
    data = json.loads(timeline_file.read_text())
    assert "timeline" in data


def test_run_index_completed_run_dir_from_result(tmp_path, monkeypatch):
    """run_dir that arrives only in the pipeline RESULT (not via publish_partial)
    must still be persisted in the index via the final_run_dir fallback path."""
    _run_store.clear()
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    rd = str(tmp_path / "rd_from_result")

    def fake_pipeline(image_path, input_spec=None, run_id=None, *, seed_glsl=None, **kwargs):
        # NOTE: deliberately does NOT call publish_partial — run_dir only in the result
        return {"run_id": run_id, "selected_glsl": "x", "scoreboard": {},
                "quality_router": {"final_score": 0.5}, "refinement_summary": {}, "run_dir": rd}

    monkeypatch.setattr("app.routers.png_shader.run_png_shader_pipeline", fake_pipeline)
    client = _client()
    resp = client.post("/png-shader/run", files={"image": ("input.png", _png_bytes(tmp_path), "image/png")})
    run_id = resp.json()["run_id"]
    _wait_for_completion(client, run_id)
    from app.pipeline.run_index import load_run_index
    rec = load_run_index(path=idx)[run_id]
    assert rec.status == "completed"
    assert rec.run_dir == rd
    assert rec.final_score == 0.5


# ---------------------------------------------------------------------------
# V3-3: explore-variants endpoint + variant worker semaphore
# ---------------------------------------------------------------------------

def _wait_for_variant_completion(
    client: TestClient, run_id: str, timeout_seconds: float = 15.0
) -> dict:
    """Like _wait_for_completion but also waits through 'queued' status."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/png-shader/status/{run_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] not in ("running", "queued"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"Variant run {run_id} did not complete within {timeout_seconds}s")


def _fake_variant_pipeline(captured_list):
    """Return a fast fake pipeline that records per-call captures in a list."""
    def fake_pipeline(image_path, input_spec=None, run_id=None, *, seed_glsl=None,
                      human_feedback_notes=None, directed_acceptance=None,
                      force_first_refinement_iteration=False, lineage=None,
                      extra_artifacts=None, **kwargs):
        captured_list.append({
            "run_id": run_id,
            "seed_glsl": seed_glsl,
            "human_feedback_notes": human_feedback_notes,
            "force_first": force_first_refinement_iteration,
            "lineage": lineage,
            "directed_acceptance": directed_acceptance,
            "extra_artifacts": extra_artifacts,
            "image_path": str(image_path),
        })
        return {
            "run_id": run_id,
            "selected_glsl": seed_glsl or "",
            "scoreboard": {},
            "quality_router": {},
            "refinement_summary": {},
            "lineage": lineage,
        }
    return fake_pipeline


def test_explore_variants_creates_4_children(tmp_path, monkeypatch):
    """POST /explore-variants with default count returns 4 children, all complete,
    with correct variant_group_id / variant_label / lineage.variant_index in store.
    The group record is persisted with all 4 child_run_ids."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH",
                        str(tmp_path / "ri.jsonl"))

    _seed_parent(tmp_path)
    captures: list[dict] = []
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline(captures),
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={
            "feedback": "make the lighting warmer and softer",
            "variant_count": 4,
            "diversity": "medium",
            "mode": "explore",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "running"
    assert body["parent_run_id"] == "run_parent"
    child_run_ids = body["child_run_ids"]
    assert len(child_run_ids) == 4
    group_id = body["group_id"]
    assert group_id.startswith("group_")

    # Wait for all children to complete.
    for cid in child_run_ids:
        result = _wait_for_variant_completion(client, cid)
        assert result["status"] == "completed", f"{cid} ended with status {result['status']}"

    # Verify store fields on each child.
    for idx, cid in enumerate(child_run_ids):
        stored = _run_store[cid]
        assert stored["variant_group_id"] == group_id
        assert stored["variant_label"] is not None
        assert stored["lineage"]["variant_index"] == idx
        assert stored["lineage"]["variant_group_id"] == group_id

    # Verify the group record was persisted.
    from app.pipeline.variant_groups import load_group
    rec = load_group(group_id, root=vg_root)
    assert rec is not None
    assert set(rec.child_run_ids) == set(child_run_ids)
    assert rec.parent_run_id == "run_parent"
    assert rec.feedback == "make the lighting warmer and softer"


def test_explore_variants_run_index_has_variant_fields(tmp_path, monkeypatch):
    """Each child's run-index record carries variant_group_id, variant_index, variant_label."""
    _run_store.clear()
    idx = str(tmp_path / "ri_v.jsonl")
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    _seed_parent(tmp_path)
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline([]),
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "warmer tones", "variant_count": 2},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    child_run_ids = body["child_run_ids"]
    group_id = body["group_id"]

    for cid in child_run_ids:
        _wait_for_variant_completion(client, cid)

    from app.pipeline.run_index import load_run_index
    records = load_run_index(path=idx)
    for idx_child, cid in enumerate(child_run_ids):
        assert cid in records, f"{cid} missing from run index"
        rec = records[cid]
        assert rec.variant_group_id == group_id
        assert rec.variant_index == idx_child
        assert rec.variant_label is not None
        assert rec.parent_run_id == "run_parent"
        assert rec.mode == "explore"


def test_explore_variants_count_out_of_range(tmp_path, monkeypatch):
    """variant_count outside [2, 6] returns 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    resp_low = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "more color", "variant_count": 1},
    )
    assert resp_low.status_code == 422

    resp_high = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "more color", "variant_count": 7},
    )
    assert resp_high.status_code == 422


def test_explore_variants_404_unknown_parent():
    """Unknown parent run_id returns 404."""
    _run_store.clear()
    client = _client()
    resp = client.post(
        "/png-shader/runs/ghost/explore-variants",
        json={"feedback": "brighter"},
    )
    assert resp.status_code == 404


def test_explore_variants_422_bad_checkpoint(tmp_path):
    """Bad checkpoint_id returns 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "brighter", "checkpoint_id": "refinement:iter:99"},
    )
    assert resp.status_code == 422


def test_explore_variants_422_empty_feedback(tmp_path):
    """Empty feedback returns 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "   "},
    )
    assert resp.status_code == 422


def test_explore_variants_409_parent_no_run_dir():
    """Parent without run_dir returns 409."""
    _run_store.clear()
    _run_store["run_norun"] = {
        "run_id": "run_norun", "status": "completed", "selected_glsl": _BRANCH_GLSL,
        "scoreboard": {"selected_id": "llm_0", "candidates": [
            {"id": "llm_0", "selected": True, "previewable": True,
             "compile_glsl": _BRANCH_GLSL, "final_score": 0.7}]},
        "refinement_history": [], "quality_router": {"final_score": 0.7},
    }
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_norun/explore-variants",
        json={"feedback": "brighter"},
    )
    assert resp.status_code == 409


def test_explore_variants_409_no_checkpoints():
    """Parent with no scoreboard / no checkpoints returns 409."""
    _run_store.clear()
    _run_store["run_empty"] = {
        "run_id": "run_empty", "status": "running",
        "strategy": {}, "stop_requested": False, "strategy_revision": 1,
    }
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_empty/explore-variants",
        json={"feedback": "brighter"},
    )
    assert resp.status_code == 409


def test_explore_variants_concurrency_all_complete(tmp_path, monkeypatch):
    """All 4 children complete even though semaphore allows only 2 concurrent workers."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH",
                        str(tmp_path / "ri_conc.jsonl"))

    _seed_parent(tmp_path)
    captures: list[dict] = []
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline(captures),
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "richer colors", "variant_count": 4},
    )
    assert resp.status_code == 200, resp.text
    child_run_ids = resp.json()["child_run_ids"]

    # All 4 must reach a terminal state (the semaphore serialises but all get through).
    for cid in child_run_ids:
        result = _wait_for_variant_completion(client, cid, timeout_seconds=20.0)
        assert result["status"] == "completed", f"{cid} did not complete: {result['status']}"

    # All 4 pipelines were invoked.
    assert len(captures) == 4


# ---------------------------------------------------------------------------
# V3.1: stoppable queued variants + post-acquire stop recheck + concurrency cap
# ---------------------------------------------------------------------------

def test_stop_queued_variant_returns_200(tmp_path):
    """POST /stop on a variant child with status='queued' must return 200 (not 409)
    and set stop_requested=True."""
    import threading as _threading
    from app.routers.png_shader import _run_store as _rs

    _rs.clear()
    run_id = "run_qstop"
    _rs[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "current_phase": "queued",
        "strategy": {},
        "stop_requested": False,
        "strategy_revision": 1,
        "variant_group_id": "group_x",
        "variant_index": 0,
        "variant_label": "A",
    }

    client = _client()
    resp = client.post(f"/png-shader/runs/{run_id}/stop")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"stopping": True}
    assert _rs[run_id]["stop_requested"] is True


def test_stop_before_acquire_cancels_without_acquiring(tmp_path, monkeypatch):
    """A variant worker that finds stop_requested=True before acquiring the semaphore
    must cancel immediately without calling acquire(), leaving status='cancelled' and
    group identity intact."""
    import threading as _threading
    from app.routers.png_shader import _run_store as _rs, _run_png_shader_background, _variant_preserved

    _rs.clear()
    run_id = "run_preacq"
    _rs[run_id] = {
        "run_id": run_id,
        "status": "queued",
        "current_phase": "queued",
        "strategy": {"refinement_threshold": 0.8},
        "stop_requested": True,  # pre-set before worker checks
        "strategy_revision": 1,
        "variant_group_id": "group_preacq",
        "variant_index": 1,
        "variant_label": "B",
        "lineage": {"variant_group_id": "group_preacq", "variant_index": 1},
    }

    acquire_called = []

    class _RecordingSemaphore:
        def acquire(self):
            acquire_called.append(True)
            return True
        def release(self):
            pass

    recording_sem = _RecordingSemaphore()

    # Seed a fake image path (worker reads stop_requested before touching the pipeline)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        fake_image = pathlib.Path(td) / "input.png"
        from PIL import Image as _Img
        _Img.new("RGBA", (8, 8)).save(fake_image)

        _run_png_shader_background(
            run_id=run_id,
            image_path=fake_image,
            upload_dir=None,
            pipeline_input_spec=None,
            seed_glsl=None,
            model_config=None,
            trace_input={},
            trace_metadata={},
            pipeline_extra=None,
            variant_semaphore=recording_sem,
        )

    # Semaphore must NOT have been acquired (cancelled before acquire).
    assert acquire_called == [], "acquire() must not be called when stop_requested is already set"

    stored = _rs[run_id]
    assert stored["status"] == "cancelled"
    # Group identity fields must be preserved.
    assert stored["variant_group_id"] == "group_preacq"
    assert stored["variant_index"] == 1
    assert stored["variant_label"] == "B"


def test_variant_concurrency_peak_le_2(tmp_path, monkeypatch):
    """Peak concurrent variant pipeline executions must never exceed _MAX_VARIANT_CONCURRENCY=2."""
    import threading as _threading

    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH",
                        str(tmp_path / "ri_peak.jsonl"))

    _seed_parent(tmp_path)

    peak_lock = _threading.Lock()
    active = [0]
    peak = [0]
    done_event = _threading.Event()
    # Barrier: each worker sets its ready event; test releases them all once N are blocked.
    WORKERS = 4
    ready_count = [0]
    ready_lock = _threading.Lock()
    gate = _threading.Event()

    def measuring_pipeline(image_path, input_spec=None, run_id=None, **kwargs):
        with peak_lock:
            active[0] += 1
            if active[0] > peak[0]:
                peak[0] = active[0]
        # Signal readiness and wait for gate (bounded: 2 s max so test can't hang).
        with ready_lock:
            ready_count[0] += 1
        gate.wait(timeout=2.0)
        with peak_lock:
            active[0] -= 1
        return {
            "run_id": run_id,
            "selected_glsl": "",
            "scoreboard": {},
            "quality_router": {},
            "refinement_summary": {},
        }

    monkeypatch.setattr("app.routers.png_shader.run_png_shader_pipeline", measuring_pipeline)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "vivid contrast", "variant_count": WORKERS},
    )
    assert resp.status_code == 200, resp.text
    child_run_ids = resp.json()["child_run_ids"]

    # Wait until at least 2 workers are inside the pipeline (i.e. past the semaphore).
    import time as _time
    deadline = _time.time() + 5.0
    while _time.time() < deadline:
        with ready_lock:
            cnt = ready_count[0]
        if cnt >= 2:
            break
        _time.sleep(0.02)

    # Release all waiting workers.
    gate.set()

    for cid in child_run_ids:
        _wait_for_variant_completion(client, cid, timeout_seconds=15.0)

    assert peak[0] <= 2, f"Peak concurrency {peak[0]} exceeded _MAX_VARIANT_CONCURRENCY=2"


# ---------------------------------------------------------------------------
# V3-4: variant-group status / stop / winner / ratings endpoints
# ---------------------------------------------------------------------------

def _seed_group(vg_root, group_id="group_x", children=None):
    """Seed a VariantGroupRecord + child entries in _run_store directly."""
    from app.pipeline.variant_groups import VariantGroupRecord, save_group as _save_group
    children = children or []
    rec = VariantGroupRecord(
        group_id=group_id,
        root_run_id="run_parent",
        parent_run_id="run_parent",
        source_checkpoint_id="final:selected",
        feedback="make it brighter",
        mode="explore",
        variant_count=len(children),
        diversity="medium",
        status="running",
        child_run_ids=[c["run_id"] for c in children],
        created_at=1.0,
    )
    _save_group(rec, root=vg_root)
    for c in children:
        _run_store[c["run_id"]] = {"run_id": c["run_id"], **c}
    return rec


def test_get_variant_group_aggregates_and_sorts(tmp_path, monkeypatch):
    """GET /variant-groups/{id} returns correct aggregate status and sorted variants."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    children = [
        {
            "run_id": "v_comp",
            "status": "completed",
            "variant_index": 0,
            "variant_label": "conservative",
            "quality_router": {"final_score": 0.7},
            "selected_glsl": "void main(){}",
            "refinement_history": [{"changes_summary": "improved brightness"}],
        },
        {
            "run_id": "v_run",
            "status": "running",
            "variant_index": 1,
            "variant_label": "semantic",
            "quality_router": {"final_score": 0.5},
            "selected_glsl": None,
            "refinement_history": [],
        },
        {
            "run_id": "v_fail",
            "status": "failed",
            "variant_index": 2,
            "variant_label": "alt_technique",
            "quality_router": {},
            "selected_glsl": None,
            "refinement_history": [],
            "error": "pipeline crashed",
        },
    ]
    _seed_group(vg_root, group_id="group_sort", children=children)
    client = _client()
    resp = client.get("/png-shader/variant-groups/group_sort")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Aggregate status: one running → "running"
    assert body["status"] == "running"
    assert body["group_id"] == "group_sort"
    assert body["parent_run_id"] == "run_parent"

    variants = body["variants"]
    assert len(variants) == 3

    # Ordering: completed(0) < running(1) < failed(4)
    assert variants[0]["run_id"] == "v_comp"
    assert variants[1]["run_id"] == "v_run"
    assert variants[2]["run_id"] == "v_fail"

    # Check required fields present on each variant.
    for v in variants:
        assert "run_id" in v
        assert "label" in v
        assert "status" in v
        assert "thumbnail_url" in v
        assert v["thumbnail_url"] == f"/png-shader/runs/{v['run_id']}/artifacts/selected_render"

    # completed variant details
    comp = variants[0]
    assert comp["final_score"] == 0.7
    assert comp["selected_glsl"] == "void main(){}"
    assert comp["changes_summary"] == "improved brightness"

    # running variant: current_score set, final_score None
    run_v = variants[1]
    assert run_v["final_score"] is None
    assert run_v["current_score"] == 0.5

    # failed variant: error exposed
    fail_v = variants[2]
    assert fail_v["error"] == "pipeline crashed"


def test_get_variant_group_404_unknown(tmp_path, monkeypatch):
    """GET /variant-groups/{id} returns 404 for unknown group_id."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    client = _client()
    resp = client.get("/png-shader/variant-groups/ghost_group")
    assert resp.status_code == 404


def test_stop_variant_group_sets_stop_requested(tmp_path, monkeypatch):
    """POST /variant-groups/{id}/stop sets stop_requested on queued/running children
    but not on completed ones."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    children = [
        {"run_id": "s_queued", "status": "queued", "stop_requested": False,
         "variant_index": 0, "variant_label": "A"},
        {"run_id": "s_running", "status": "running", "stop_requested": False,
         "variant_index": 1, "variant_label": "B"},
        {"run_id": "s_done", "status": "completed", "stop_requested": False,
         "variant_index": 2, "variant_label": "C"},
    ]
    _seed_group(vg_root, group_id="group_stop", children=children)
    client = _client()

    resp = client.post("/png-shader/variant-groups/group_stop/stop")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stopping"] is True
    assert body["group_id"] == "group_stop"

    # Queued and running must have stop_requested set.
    assert _run_store["s_queued"]["stop_requested"] is True
    assert _run_store["s_running"]["stop_requested"] is True
    # Completed must remain untouched.
    assert _run_store["s_done"]["stop_requested"] is False


def test_stop_variant_group_404_unknown(tmp_path, monkeypatch):
    """POST /variant-groups/{id}/stop returns 404 for unknown group."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    client = _client()
    resp = client.post("/png-shader/variant-groups/ghost_grp/stop")
    assert resp.status_code == 404


def test_winner_marks_group_and_favorite(tmp_path, monkeypatch):
    """POST /variant-groups/{id}/winner updates winner_run_id in the group record
    and sets favorite=True in the store entry."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    children = [
        {"run_id": "w_a", "status": "completed", "variant_index": 0, "variant_label": "A",
         "quality_router": {"final_score": 0.8}},
        {"run_id": "w_b", "status": "completed", "variant_index": 1, "variant_label": "B",
         "quality_router": {"final_score": 0.6}},
    ]
    _seed_group(vg_root, group_id="group_win", children=children)
    client = _client()

    resp = client.post(
        "/png-shader/variant-groups/group_win/winner",
        json={"winner_run_id": "w_a", "reason": "best score"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["group_id"] == "group_win"
    assert body["winner_run_id"] == "w_a"

    # Group record must be updated on disk.
    from app.pipeline.variant_groups import load_group as _load_group
    rec = _load_group("group_win", root=vg_root)
    assert rec is not None
    assert rec.winner_run_id == "w_a"

    # Store entry must have favorite=True.
    assert _run_store["w_a"]["favorite"] is True

    # Other child must not be marked favorite.
    assert _run_store["w_b"].get("favorite", False) is False


def test_winner_422_non_member_run_id(tmp_path, monkeypatch):
    """POST /winner with a run_id not in child_run_ids returns 422."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    children = [{"run_id": "w_only", "status": "completed", "variant_index": 0, "variant_label": "A"}]
    _seed_group(vg_root, group_id="group_422w", children=children)
    client = _client()

    resp = client.post(
        "/png-shader/variant-groups/group_422w/winner",
        json={"winner_run_id": "foreign_run"},
    )
    assert resp.status_code == 422


def test_winner_404_unknown_group(tmp_path, monkeypatch):
    """POST /winner for unknown group returns 404."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    client = _client()
    resp = client.post("/png-shader/variant-groups/ghost/winner", json={"winner_run_id": "x"})
    assert resp.status_code == 404


def test_ratings_appends_event(tmp_path, monkeypatch):
    """POST /ratings appends a rating event that is readable via load_group_events."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    children = [
        {"run_id": "r_a", "status": "completed", "variant_index": 0, "variant_label": "A"},
    ]
    _seed_group(vg_root, group_id="group_rate", children=children)
    client = _client()

    resp = client.post(
        "/png-shader/variant-groups/group_rate/ratings",
        json={"run_id": "r_a", "rating": 1, "reason": "looks great", "tags": ["color"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["group_id"] == "group_rate"
    assert body["run_id"] == "r_a"
    assert body["rating"] == 1

    # Verify event was appended.
    from app.pipeline.variant_groups import load_group_events
    events = load_group_events("group_rate", root=vg_root)
    rating_events = [e for e in events if e.get("event") == "rating"]
    assert len(rating_events) == 1
    ev = rating_events[0]
    assert ev["run_id"] == "r_a"
    assert ev["rating"] == 1
    assert ev["reason"] == "looks great"
    assert "color" in ev["tags"]


def test_ratings_422_non_member_run_id(tmp_path, monkeypatch):
    """POST /ratings with a run_id not in child_run_ids returns 422."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    children = [{"run_id": "r_only", "status": "completed", "variant_index": 0, "variant_label": "A"}]
    _seed_group(vg_root, group_id="group_422r", children=children)
    client = _client()

    resp = client.post(
        "/png-shader/variant-groups/group_422r/ratings",
        json={"run_id": "foreign_run", "rating": 1},
    )
    assert resp.status_code == 422


def test_ratings_422_invalid_rating(tmp_path, monkeypatch):
    """POST /ratings with rating outside {-1, 0, 1} returns 422."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    children = [{"run_id": "r_val", "status": "completed", "variant_index": 0, "variant_label": "A"}]
    _seed_group(vg_root, group_id="group_422rv", children=children)
    client = _client()

    # rating=5 is invalid
    resp = client.post(
        "/png-shader/variant-groups/group_422rv/ratings",
        json={"run_id": "r_val", "rating": 5},
    )
    assert resp.status_code == 422

    # rating=2 is also invalid
    resp2 = client.post(
        "/png-shader/variant-groups/group_422rv/ratings",
        json={"run_id": "r_val", "rating": 2},
    )
    assert resp2.status_code == 422


def test_get_variant_group_winner_sorted_first(tmp_path, monkeypatch):
    """When winner_run_id is set, that variant appears first in sorted output."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)

    # Build group with 3 completed children; set winner to the one with lowest score.
    from app.pipeline.variant_groups import VariantGroupRecord, save_group as _save_group
    children = [
        {"run_id": "ws_a", "status": "completed", "variant_index": 0, "variant_label": "A",
         "quality_router": {"final_score": 0.9}},
        {"run_id": "ws_b", "status": "completed", "variant_index": 1, "variant_label": "B",
         "quality_router": {"final_score": 0.5}},
        {"run_id": "ws_c", "status": "completed", "variant_index": 2, "variant_label": "C",
         "quality_router": {"final_score": 0.7}},
    ]
    rec = VariantGroupRecord(
        group_id="group_ws",
        root_run_id="run_parent",
        parent_run_id="run_parent",
        source_checkpoint_id="final:selected",
        feedback="test",
        mode="explore",
        variant_count=3,
        diversity="medium",
        status="completed",
        child_run_ids=["ws_a", "ws_b", "ws_c"],
        winner_run_id="ws_b",  # winner has lowest score but must sort first
        created_at=1.0,
    )
    _save_group(rec, root=vg_root)
    for c in children:
        _run_store[c["run_id"]] = {"run_id": c["run_id"], **c}

    client = _client()
    resp = client.get("/png-shader/variant-groups/group_ws")
    assert resp.status_code == 200
    variants = resp.json()["variants"]
    assert variants[0]["run_id"] == "ws_b", "Winner must appear first regardless of score"


def test_winner_persists_run_index_favorite_and_appends_event(tmp_path, monkeypatch):
    """POST /winner seeds the winner into the run index, then verifies
    favorite=True is persisted there and a 'winner' event is appended."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    # Seed children and group.
    children = [
        {"run_id": "wi_a", "status": "completed", "variant_index": 0, "variant_label": "A",
         "quality_router": {"final_score": 0.85}},
        {"run_id": "wi_b", "status": "completed", "variant_index": 1, "variant_label": "B",
         "quality_router": {"final_score": 0.60}},
    ]
    _seed_group(vg_root, group_id="group_windex", children=children)

    # Seed the winner run into the run index before calling the endpoint.
    from app.pipeline.run_index import RunLineageRecord, append_run_created, load_run_index
    idx_path = tmp_path / "ri.jsonl"
    winner_rec = RunLineageRecord(
        run_id="wi_a",
        root_run_id="run_parent",
        parent_run_id="run_parent",
        source_checkpoint_id="final:selected",
        source_checkpoint_label=None,
        mode="explore",
        feedback="make it brighter",
        title=None,
        status="completed",
        run_dir=None,
        created_at=1.0,
        final_score=0.85,
        variant_group_id="group_windex",
        variant_index=0,
        variant_label="A",
    )
    append_run_created(winner_rec, path=idx_path)

    client = _client()
    resp = client.post(
        "/png-shader/variant-groups/group_windex/winner",
        json={"winner_run_id": "wi_a", "reason": "highest score"},
    )
    assert resp.status_code == 200, resp.text

    # Verify favorite=True written into the run index.
    index = load_run_index(path=idx_path)
    assert "wi_a" in index, "winner run_id must be in index"
    assert index["wi_a"].favorite is True, "favorite must be True after /winner"

    # Verify 'winner' event was appended to the group event log.
    from app.pipeline.variant_groups import load_group_events
    evs = load_group_events("group_windex", root=vg_root)
    assert any(
        e.get("event") == "winner" and e.get("run_id") == "wi_a"
        for e in evs
    ), f"Expected a 'winner' event for wi_a in events: {evs}"


def test_winner_save_group_failure_returns_500(tmp_path, monkeypatch):
    """If save_group raises, POST /winner must return 500 (not silently succeed)."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    children = [
        {"run_id": "wf_a", "status": "completed", "variant_index": 0, "variant_label": "A"},
    ]
    _seed_group(vg_root, group_id="group_wfail", children=children)

    # Patch save_group to simulate an I/O failure.
    monkeypatch.setattr(
        "app.routers.png_shader.save_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    client = _client()
    resp = client.post(
        "/png-shader/variant-groups/group_wfail/winner",
        json={"winner_run_id": "wf_a"},
    )
    assert resp.status_code == 500, resp.text
    assert "persist" in resp.json().get("detail", "").lower()


def test_get_variant_group_evicted_child_fallback(tmp_path, monkeypatch):
    """GET /variant-groups/{id}: a child evicted from _run_store but present in
    the run index must appear in variants using index-derived status and favorite."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    # Seed a group with one child, but do NOT put that child in _run_store.
    from app.pipeline.variant_groups import VariantGroupRecord, save_group as _save_group
    rec = VariantGroupRecord(
        group_id="group_evict",
        root_run_id="run_parent",
        parent_run_id="run_parent",
        source_checkpoint_id="final:selected",
        feedback="test eviction",
        mode="explore",
        variant_count=1,
        diversity="medium",
        status="completed",
        child_run_ids=["ev_run"],
        created_at=1.0,
    )
    _save_group(rec, root=vg_root)
    # _run_store intentionally left empty — ev_run is "evicted".

    # Seed ev_run into the run index (as if it was previously completed).
    from app.pipeline.run_index import RunLineageRecord, append_run_created
    idx_path = tmp_path / "ri.jsonl"
    ev_rec = RunLineageRecord(
        run_id="ev_run",
        root_run_id="run_parent",
        parent_run_id="run_parent",
        source_checkpoint_id="final:selected",
        source_checkpoint_label=None,
        mode="explore",
        feedback="test eviction",
        title=None,
        status="completed",
        run_dir=None,
        created_at=1.0,
        final_score=0.72,
        favorite=True,
        variant_group_id="group_evict",
        variant_index=0,
        variant_label="evicted_variant",
    )
    append_run_created(ev_rec, path=idx_path)

    client = _client()
    resp = client.get("/png-shader/variant-groups/group_evict")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    variants = body["variants"]
    assert len(variants) == 1

    v = variants[0]
    assert v["run_id"] == "ev_run"
    assert v["status"] == "completed", f"expected 'completed', got {v['status']!r}"
    assert v["final_score"] == 0.72, f"expected 0.72, got {v['final_score']!r}"
    assert v["favorite"] is True, "run-index favorite must be reflected in evicted child"


# ---------------------------------------------------------------------------
# V3.5-B4: draw-session endpoints
#   POST   /runs/{run_id}/draw-session
#   GET    /draw-sessions/{draw_id}
#   POST   /draw-sessions/{draw_id}/draw-more
#   POST   /draw-sessions/{draw_id}/redraw
#   POST   /draw-sessions/{draw_id}/cards/{run_id}/event
# ---------------------------------------------------------------------------


def _ds_root(tmp_path) -> str:
    """The draw-sessions root the autouse fixture monkeypatched onto the router."""
    return str(tmp_path / "ds")


def test_draw_session_create_8_two_groups(tmp_path, monkeypatch):
    """POST /draw-session card_count=8 -> 2 groups, 8 card_run_ids, persisted record;
    all cards complete after workers drain."""
    _run_store.clear()
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline([]),
    )
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "draw me some bold neon variants", "card_count": 8},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "running"
    assert body["parent_run_id"] == "run_parent"
    assert body["source_checkpoint_id"] == "final:selected"
    draw_id = body["draw_id"]
    assert draw_id.startswith("draw_")
    assert len(body["group_ids"]) == 2
    assert len(body["card_run_ids"]) == 8
    assert len(set(body["card_run_ids"])) == 8

    # Record round-trips with parent / feedback / group_ids / card_run_ids.
    from app.pipeline.draw_sessions import load_session
    rec = load_session(draw_id, root=_ds_root(tmp_path))
    assert rec is not None
    assert rec.parent_run_id == "run_parent"
    assert rec.feedback == "draw me some bold neon variants"
    assert rec.requested_count == 8
    assert list(rec.group_ids) == list(body["group_ids"])
    assert list(rec.card_run_ids) == list(body["card_run_ids"])

    for cid in body["card_run_ids"]:
        result = _wait_for_variant_completion(client, cid, timeout_seconds=20.0)
        assert result["status"] == "completed", f"{cid} ended {result['status']}"


def test_draw_session_create_12_batches_6_6(tmp_path, monkeypatch):
    """card_count=12 -> batches [6,6] => 2 groups of 6, 12 cards."""
    _run_store.clear()
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline([]),
    )
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "twelve please", "card_count": 12},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["group_ids"]) == 2
    assert len(body["card_run_ids"]) == 12

    from app.pipeline.variant_groups import load_group
    sizes = sorted(
        len(load_group(gid, root=str(tmp_path / "vg")).child_run_ids)
        for gid in body["group_ids"]
    )
    assert sizes == [6, 6]


def test_draw_session_create_4_one_group(tmp_path, monkeypatch):
    """card_count=4 -> a single group of 4 cards."""
    _run_store.clear()
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline([]),
    )
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "four cards", "card_count": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["group_ids"]) == 1
    assert len(body["card_run_ids"]) == 4


def test_draw_session_create_card_count_out_of_range(tmp_path):
    """card_count of 1 and 13 both 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    low = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "x", "card_count": 1},
    )
    assert low.status_code == 422

    high = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "x", "card_count": 13},
    )
    assert high.status_code == 422


def test_draw_session_create_404_unknown_parent():
    """Unknown parent run -> 404."""
    _run_store.clear()
    client = _client()
    resp = client.post(
        "/png-shader/runs/ghost/draw-session",
        json={"feedback": "x"},
    )
    assert resp.status_code == 404


def test_draw_session_create_422_empty_feedback(tmp_path):
    """Blank feedback -> 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "   "},
    )
    assert resp.status_code == 422


def test_draw_session_create_422_bad_checkpoint(tmp_path):
    """Unresolvable checkpoint_id -> 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "x", "checkpoint_id": "refinement:iter:99"},
    )
    assert resp.status_code == 422


def _seed_draw_session(tmp_path, *, draw_id="draw_test1", card_run_ids, feedback="brighten",
                       requested_count=None, group_ids=None):
    """Persist a DrawSessionRecord directly (no worker spawn)."""
    from app.pipeline.draw_sessions import DrawSessionRecord, save_session
    rec = DrawSessionRecord(
        draw_id=draw_id,
        root_run_id="run_parent",
        parent_run_id="run_parent",
        source_checkpoint_id="final:selected",
        feedback=feedback,
        status="running",
        requested_count=requested_count if requested_count is not None else len(card_run_ids),
        diversity="medium",
        mode="batch_draw",
        group_ids=group_ids if group_ids is not None else ["group_seed"],
        card_run_ids=list(card_run_ids),
        created_at=1.0,
        metadata={"locks": {}, "quality": {}, "mode": "batch_draw"},
    )
    save_session(rec, root=_ds_root(tmp_path))
    return rec


def test_get_draw_session_mixed_statuses(tmp_path):
    """GET /draw-sessions/{id} aggregates counts + status from mixed child statuses."""
    _run_store.clear()
    _run_store["c_done"] = {
        "run_id": "c_done", "status": "completed", "variant_index": 0,
        "variant_label": "card", "variant_group_id": "group_seed",
        "quality_router": {"final_score": 0.81}, "refinement_history": [],
    }
    _run_store["c_fail"] = {
        "run_id": "c_fail", "status": "failed", "variant_index": 1,
        "variant_label": "card", "variant_group_id": "group_seed",
        "quality_router": {}, "refinement_history": [], "error": "boom",
    }
    _seed_draw_session(tmp_path, draw_id="draw_mix", card_run_ids=["c_done", "c_fail"])

    client = _client()
    resp = client.get("/png-shader/draw-sessions/draw_mix")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["draw_id"] == "draw_mix"
    assert body["parent_run_id"] == "run_parent"
    assert body["source_checkpoint_id"] == "final:selected"
    assert body["completed_count"] == 1
    assert body["failed_count"] == 1
    assert body["running_count"] == 0
    # one completed + one failed -> partial_failed
    assert body["status"] == "partial_failed"

    cards = body["cards"]
    assert len(cards) == 2
    by_run = {c["run_id"]: c for c in cards}
    done = by_run["c_done"]
    assert done["card_id"] == "c_done"
    assert done["status"] == "completed"
    assert done["final_score"] == 0.81
    assert done["can_use_for_fusion"] is True
    assert done["thumbnail_url"] == "/png-shader/runs/c_done/artifacts/selected_render"
    assert done["feedback"] == "brighten"
    fail = by_run["c_fail"]
    assert fail["status"] == "failed"
    assert fail["error"] == "boom"
    assert fail["can_use_for_fusion"] is False


def test_get_draw_session_running_counts(tmp_path):
    """running + queued cards count toward running_count; aggregate -> running."""
    _run_store.clear()
    _run_store["c_run"] = {
        "run_id": "c_run", "status": "running", "variant_index": 0,
        "variant_label": "card", "variant_group_id": "group_seed",
        "quality_router": {"final_score": 0.4}, "refinement_history": [],
    }
    _run_store["c_q"] = {
        "run_id": "c_q", "status": "queued", "variant_index": 1,
        "variant_label": "card", "variant_group_id": "group_seed",
        "quality_router": {}, "refinement_history": [],
    }
    _seed_draw_session(tmp_path, draw_id="draw_run", card_run_ids=["c_run", "c_q"])
    client = _client()
    resp = client.get("/png-shader/draw-sessions/draw_run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["running_count"] == 2
    assert body["completed_count"] == 0
    assert body["status"] == "running"
    by_run = {c["run_id"]: c for c in body["cards"]}
    assert by_run["c_run"]["current_score"] == 0.4
    assert by_run["c_run"]["final_score"] is None


def test_get_draw_session_404_unknown():
    """Unknown draw_id -> 404."""
    _run_store.clear()
    client = _client()
    resp = client.get("/png-shader/draw-sessions/ghost")
    assert resp.status_code == 404


def test_draw_session_draw_more_extends_record(tmp_path, monkeypatch):
    """draw-more keeps the original group, grows card_run_ids, adds distinct group."""
    _run_store.clear()
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline([]),
    )
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "first draw", "card_count": 4},
    )
    assert resp.status_code == 200, resp.text
    created = resp.json()
    draw_id = created["draw_id"]
    orig_groups = list(created["group_ids"])
    orig_cards = list(created["card_run_ids"])

    more = client.post(
        f"/png-shader/draw-sessions/{draw_id}/draw-more",
        json={"card_count": 4},
    )
    assert more.status_code == 200, more.text
    mbody = more.json()
    assert mbody["draw_id"] == draw_id
    assert len(mbody["group_ids"]) == 1
    new_gid = mbody["group_ids"][0]
    assert new_gid not in orig_groups
    assert len(mbody["card_run_ids"]) == 4

    from app.pipeline.draw_sessions import load_session
    rec = load_session(draw_id, root=_ds_root(tmp_path))
    assert rec is not None
    # original group still present
    for gid in orig_groups:
        assert gid in rec.group_ids
    assert new_gid in rec.group_ids
    # card_run_ids grew
    assert len(rec.card_run_ids) == len(orig_cards) + 4
    for c in orig_cards:
        assert c in rec.card_run_ids


def test_draw_session_draw_more_404_unknown():
    """draw-more on unknown draw_id -> 404."""
    _run_store.clear()
    client = _client()
    resp = client.post(
        "/png-shader/draw-sessions/ghost/draw-more",
        json={"card_count": 4},
    )
    assert resp.status_code == 404


def test_draw_session_draw_more_409_parent_gone(tmp_path):
    """draw-more when parent run is no longer in the store -> 409."""
    _run_store.clear()  # parent absent
    _seed_draw_session(tmp_path, draw_id="draw_orphan", card_run_ids=["c1"])
    client = _client()
    resp = client.post(
        "/png-shader/draw-sessions/draw_orphan/draw-more",
        json={"card_count": 4},
    )
    assert resp.status_code == 409


def test_draw_session_redraw_links_replacement(tmp_path, monkeypatch):
    """redraw spawns 1 replacement card, links replacement_of_run_id, eliminates
    original (kept in record), and logs a draw_card_redrawn event."""
    _run_store.clear()
    idx = str(tmp_path / "run_index.jsonl")  # matches autouse fixture path
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline([]),
    )
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "draw set", "card_count": 4},
    )
    assert resp.status_code == 200, resp.text
    created = resp.json()
    draw_id = created["draw_id"]
    original = created["card_run_ids"][0]

    redraw = client.post(
        f"/png-shader/draw-sessions/{draw_id}/redraw",
        json={"run_id": original, "reason": "too dark"},
    )
    assert redraw.status_code == 200, redraw.text
    rbody = redraw.json()
    assert rbody["replaced_run_id"] == original
    new_run_id = rbody["replacement_run_id"]
    assert new_run_id and new_run_id != original
    assert rbody["group_id"]

    # replacement's run-index record points back to the original.
    from app.pipeline.run_index import load_run_index
    records = load_run_index(path=idx)
    assert new_run_id in records
    assert records[new_run_id].replacement_of_run_id == original

    # original still tracked in the record's card_run_ids.
    from app.pipeline.draw_sessions import load_session, load_session_events
    rec = load_session(draw_id, root=_ds_root(tmp_path))
    assert original in rec.card_run_ids
    assert new_run_id in rec.card_run_ids

    # a draw_card_redrawn event exists.
    events = load_session_events(draw_id, root=_ds_root(tmp_path))
    redrawn = [e for e in events if e.get("event") == "draw_card_redrawn"]
    assert any(
        e.get("run_id") == original and e.get("replacement_run_id") == new_run_id
        for e in redrawn
    )


def test_draw_session_redraw_422_non_member(tmp_path):
    """redraw with run_id not in the record -> 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    _seed_draw_session(tmp_path, draw_id="draw_rd", card_run_ids=["c_known"])
    client = _client()
    resp = client.post(
        "/png-shader/draw-sessions/draw_rd/redraw",
        json={"run_id": "not_a_member"},
    )
    assert resp.status_code == 422


def test_draw_session_card_event_favorite(tmp_path, monkeypatch):
    """favorite=true writes a session event, mirrors to run-index favorite, and a
    subsequent GET shows that card favorite:true."""
    _run_store.clear()
    idx = str(tmp_path / "run_index.jsonl")
    # seed a run-index 'created' record so update_run_metadata can patch it.
    from app.pipeline.run_index import RunLineageRecord, append_run_created
    append_run_created(
        RunLineageRecord(
            run_id="c_fav", root_run_id="run_parent", parent_run_id="run_parent",
            source_checkpoint_id="final:selected", source_checkpoint_label="final",
            mode="batch_draw", feedback="brighten", title=None,
            status="completed", run_dir=None, created_at=1.0,
            variant_group_id="group_seed", variant_index=0, variant_label="card",
        ),
        path=idx,
    )
    _run_store["c_fav"] = {
        "run_id": "c_fav", "status": "completed", "variant_index": 0,
        "variant_label": "card", "variant_group_id": "group_seed",
        "quality_router": {"final_score": 0.6}, "refinement_history": [],
    }
    _seed_draw_session(tmp_path, draw_id="draw_fav", card_run_ids=["c_fav"])
    client = _client()

    ev = client.post(
        "/png-shader/draw-sessions/draw_fav/cards/c_fav/event",
        json={"event_type": "favorite", "value": True},
    )
    assert ev.status_code == 200, ev.text
    assert ev.json()["ok"] is True

    # session event written
    from app.pipeline.draw_sessions import load_session_events
    events = load_session_events(draw_id="draw_fav", root=_ds_root(tmp_path))
    assert any(e.get("event") == "favorite" and e.get("run_id") == "c_fav" for e in events)

    # run-index favorite mirrored
    from app.pipeline.run_index import load_run_index
    assert load_run_index(path=idx)["c_fav"].favorite is True

    # GET reflects favorite
    body = client.get("/png-shader/draw-sessions/draw_fav").json()
    fav_card = next(c for c in body["cards"] if c["run_id"] == "c_fav")
    assert fav_card["favorite"] is True


def test_draw_session_card_event_eliminate_and_tag(tmp_path):
    """eliminate=true and tag events are reflected by a subsequent GET overlay."""
    _run_store.clear()
    _run_store["c_x"] = {
        "run_id": "c_x", "status": "completed", "variant_index": 0,
        "variant_label": "card", "variant_group_id": "group_seed",
        "quality_router": {"final_score": 0.5}, "refinement_history": [],
    }
    _seed_draw_session(tmp_path, draw_id="draw_et", card_run_ids=["c_x"])
    client = _client()

    elim = client.post(
        "/png-shader/draw-sessions/draw_et/cards/c_x/event",
        json={"event_type": "eliminate", "value": True},
    )
    assert elim.status_code == 200, elim.text
    tag = client.post(
        "/png-shader/draw-sessions/draw_et/cards/c_x/event",
        json={"event_type": "tag", "tags": ["keep-color"]},
    )
    assert tag.status_code == 200, tag.text

    body = client.get("/png-shader/draw-sessions/draw_et").json()
    card = next(c for c in body["cards"] if c["run_id"] == "c_x")
    assert card["eliminated"] is True
    assert "keep-color" in card["tags"]


def test_draw_session_card_event_422_invalid_type(tmp_path):
    """Unknown event_type -> 422."""
    _run_store.clear()
    _run_store["c_y"] = {"run_id": "c_y", "status": "completed", "variant_group_id": "group_seed"}
    _seed_draw_session(tmp_path, draw_id="draw_inv", card_run_ids=["c_y"])
    client = _client()
    resp = client.post(
        "/png-shader/draw-sessions/draw_inv/cards/c_y/event",
        json={"event_type": "bogus"},
    )
    assert resp.status_code == 422


def test_draw_session_card_event_422_non_member(tmp_path):
    """Event for a run_id not in the record -> 422."""
    _run_store.clear()
    _seed_draw_session(tmp_path, draw_id="draw_nm", card_run_ids=["c_member"])
    client = _client()
    resp = client.post(
        "/png-shader/draw-sessions/draw_nm/cards/c_outsider/event",
        json={"event_type": "favorite", "value": True},
    )
    assert resp.status_code == 422


def test_draw_session_card_event_un_eliminate(tmp_path):
    """eliminate=true followed by eliminate=false -> later-overrides-earlier fold:
    card eliminated must be false after the second event."""
    _run_store.clear()
    _run_store["c_ue"] = {
        "run_id": "c_ue", "status": "completed", "variant_index": 0,
        "variant_label": "card", "variant_group_id": "group_seed",
        "quality_router": {"final_score": 0.5}, "refinement_history": [],
    }
    _seed_draw_session(tmp_path, draw_id="draw_ue", card_run_ids=["c_ue"])
    client = _client()

    # First: eliminate the card.
    ev1 = client.post(
        "/png-shader/draw-sessions/draw_ue/cards/c_ue/event",
        json={"event_type": "eliminate", "value": True},
    )
    assert ev1.status_code == 200, ev1.text
    assert ev1.json()["ok"] is True

    body1 = client.get("/png-shader/draw-sessions/draw_ue").json()
    card1 = next(c for c in body1["cards"] if c["run_id"] == "c_ue")
    assert card1["eliminated"] is True, "card must be eliminated after first event"

    # Second: clear the elimination.
    ev2 = client.post(
        "/png-shader/draw-sessions/draw_ue/cards/c_ue/event",
        json={"event_type": "eliminate", "value": False},
    )
    assert ev2.status_code == 200, ev2.text
    assert ev2.json()["ok"] is True

    body2 = client.get("/png-shader/draw-sessions/draw_ue").json()
    card2 = next(c for c in body2["cards"] if c["run_id"] == "c_ue")
    assert card2["eliminated"] is False, "card must not be eliminated after value=false event"


# ---------------------------------------------------------------------------
# V4.1 Structured Constraints — branch_refine + explore_variants + draw-session
# ---------------------------------------------------------------------------

_VALID_CONSTRAINTS = {
    "locks": {"preserve_layout": True},
    "targets": {"reflection": "increase"},
    "edit_strength": 0.35,
}


def test_branch_refine_with_constraints_adds_notes_and_artifacts(tmp_path, monkeypatch):
    """branch_refine WITH constraints → human_feedback_notes include [GLOBAL LOCK] and
    [TARGET] notes; extra_artifacts contains constraints.json; directed_acceptance has
    a 'constraints' key."""
    _run_store.clear()
    _seed_parent(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "make the reflection stronger",
            "mode": "refine",
            "constraints": _VALID_CONSTRAINTS,
        },
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])

    notes = captured["human_feedback_notes"]
    assert any("[GLOBAL LOCK]" in n for n in notes), f"no [GLOBAL LOCK] note in {notes}"
    assert any("[TARGET]" in n for n in notes), f"no [TARGET] note in {notes}"
    assert any("[EDIT STRENGTH]" in n for n in notes), f"no [EDIT STRENGTH] note in {notes}"

    artifacts = captured["extra_artifacts"]
    assert "constraints.json" in artifacts, "constraints.json missing from extra_artifacts"
    cj = artifacts["constraints.json"]
    assert cj["locks"] == {"preserve_layout": True}
    assert cj["targets"] == {"reflection": "increase"}
    assert abs(cj["edit_strength"] - 0.35) < 1e-9

    da = captured["directed_acceptance"]
    assert "constraints" in da, "directed_acceptance missing 'constraints' key"
    assert da["constraints"]["locks"] == {"preserve_layout": True}


def test_branch_refine_without_constraints_unchanged(tmp_path, monkeypatch):
    """branch_refine WITHOUT constraints → no [GLOBAL LOCK]/[TARGET]/[EDIT STRENGTH] notes;
    no constraints.json artifact; no 'constraints' key in directed_acceptance
    (proves backward compat)."""
    _run_store.clear()
    _seed_parent(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "make the reflection stronger",
            "mode": "refine",
        },
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])

    notes = captured["human_feedback_notes"]
    assert not any("[GLOBAL LOCK]" in n for n in notes), "unexpected [GLOBAL LOCK] in notes"
    assert not any("[TARGET]" in n for n in notes), "unexpected [TARGET] in notes"
    assert not any("[EDIT STRENGTH]" in n for n in notes), "unexpected [EDIT STRENGTH] in notes"

    artifacts = captured["extra_artifacts"]
    assert "constraints.json" not in artifacts, "unexpected constraints.json in extra_artifacts"

    da = captured["directed_acceptance"]
    assert "constraints" not in da, "unexpected 'constraints' key in directed_acceptance"


def test_branch_refine_invalid_constraints_returns_422(tmp_path):
    """branch_refine with invalid constraints (edit_strength out of range) → 422
    with constraint_errors."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "make it brighter",
            "mode": "refine",
            "constraints": {"edit_strength": 1.5},
        },
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert "constraint_errors" in body["detail"], f"expected constraint_errors in {body}"


def test_explore_variants_with_constraints_each_child_has_notes_and_artifact(tmp_path, monkeypatch):
    """explore_variants WITH constraints → each child's human_feedback_notes include
    [GLOBAL LOCK] and constraints.json appears in each child's extra_artifacts."""
    _run_store.clear()
    _seed_parent(tmp_path)
    captures: list[dict] = []
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline(captures),
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={
            "feedback": "add more warmth",
            "variant_count": 2,
            "constraints": _VALID_CONSTRAINTS,
        },
    )
    assert resp.status_code == 200, resp.text
    child_run_ids = resp.json()["child_run_ids"]
    for cid in child_run_ids:
        _wait_for_variant_completion(client, cid)

    assert len(captures) == 2
    for cap in captures:
        notes = cap["human_feedback_notes"]
        assert any("[GLOBAL LOCK]" in n for n in notes), f"missing [GLOBAL LOCK] in {notes}"
        artifacts = cap["extra_artifacts"]
        assert "constraints.json" in artifacts, "constraints.json missing from extra_artifacts"
        assert artifacts["constraints.json"]["locks"] == {"preserve_layout": True}
        da = cap["directed_acceptance"]
        assert "constraints" in da, "directed_acceptance missing 'constraints'"


def test_explore_variants_invalid_constraints_returns_422(tmp_path):
    """explore_variants with invalid constraints → 422 with constraint_errors."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={
            "feedback": "more depth",
            "variant_count": 2,
            "constraints": {"edit_strength": 2.0},
        },
    )
    assert resp.status_code == 422, resp.text
    assert "constraint_errors" in resp.json()["detail"]


def test_draw_session_create_with_constraints_children_carry_notes(tmp_path, monkeypatch):
    """draw-session create WITH constraints → children carry [GLOBAL LOCK] notes
    and constraints.json in extra_artifacts."""
    _run_store.clear()
    _seed_parent(tmp_path)
    captures: list[dict] = []
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline(captures),
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={
            "feedback": "warmer palette",
            "card_count": 2,
            "constraints": _VALID_CONSTRAINTS,
        },
    )
    assert resp.status_code == 200, resp.text
    card_run_ids = resp.json()["card_run_ids"]
    for cid in card_run_ids:
        _wait_for_variant_completion(client, cid)

    assert len(captures) == 2
    for cap in captures:
        notes = cap["human_feedback_notes"]
        assert any("[GLOBAL LOCK]" in n for n in notes), f"missing [GLOBAL LOCK] in {notes}"
        artifacts = cap["extra_artifacts"]
        assert "constraints.json" in artifacts, "constraints.json missing"
        assert artifacts["constraints.json"]["locks"] == {"preserve_layout": True}
        da = cap["directed_acceptance"]
        assert "constraints" in da, "directed_acceptance missing 'constraints'"


# ---------------------------------------------------------------------------
# V4.2 region-mask endpoint tests
# ---------------------------------------------------------------------------

_VALID_REGION_BODY = {
    "region_id": "region_water",
    "geometry_type": "rect",
    "geometry": {"x": 0.05, "y": 0.58, "w": 0.90, "h": 0.34},
    "label": "water",
    "mode": "modify",
    "instruction": "make reflection clearer",
    "strength": 0.45,
}


def test_region_mask_valid_rect_200(tmp_path):
    """Valid rect → 200, correct response shape, geometry JSON persisted."""
    _run_store.clear()
    parent_dir = _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/region-mask",
        json=_VALID_REGION_BODY,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["region_id"] == "region_water"
    assert body["mask_artifact_id"] == "mask:region_water"
    assert body["mask_url"] == "/png-shader/runs/run_parent/artifacts/mask:region_water"
    assert body["geometry"] == {"x": 0.05, "y": 0.58, "w": 0.90, "h": 0.34}

    # geometry JSON must be persisted under run_dir/region_masks/
    mask_file = parent_dir / "region_masks" / "region_water.json"
    assert mask_file.exists(), f"mask file not written: {mask_file}"
    import json as _json
    data = _json.loads(mask_file.read_text())
    assert data["id"] == "region_water"
    assert data["geometry"] == {"x": 0.05, "y": 0.58, "w": 0.90, "h": 0.34}


def test_region_mask_out_of_bounds_rect_422(tmp_path):
    """Out-of-bounds rect (x+w > 1) → 422 with region_errors mentioning region_id."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    bad_body = dict(_VALID_REGION_BODY)
    bad_body["geometry"] = {"x": 0.8, "y": 0.0, "w": 0.5, "h": 0.5}  # x+w = 1.3 > 1

    resp = client.post(
        "/png-shader/runs/run_parent/region-mask",
        json=bad_body,
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "region_errors" in detail
    errors = detail["region_errors"]
    assert any("region_water" in e for e in errors), f"region_id missing from errors: {errors}"


def test_region_mask_unknown_run_404(tmp_path):
    """Unknown run_id → 404."""
    _run_store.clear()
    client = _client()

    resp = client.post(
        "/png-shader/runs/ghost_run/region-mask",
        json=_VALID_REGION_BODY,
    )
    assert resp.status_code == 404, resp.text


def test_region_mask_with_render_computes_metrics(tmp_path):
    """With reference + render PNG present → metrics["regions"] contains region_id,
    and region_metrics/<id>.json is persisted."""
    _run_store.clear()
    parent_dir = _seed_parent(tmp_path)

    # Write the render PNG that resolve_checkpoint_artifact expects:
    # selected_id = "llm_0" → <run_dir>/candidates/llm_0_render.png
    cands_dir = parent_dir / "candidates"
    cands_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (32, 32), (80, 180, 40, 255)).save(cands_dir / "llm_0_render.png")

    client = _client()
    resp = client.post(
        "/png-shader/runs/run_parent/region-mask",
        json=_VALID_REGION_BODY,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["metrics"] is not None, "metrics should not be null when render present"
    assert "region_water" in body["metrics"]["regions"], (
        f"region_water missing from metrics regions: {body['metrics']['regions']}"
    )

    # region_metrics JSON must be persisted
    metrics_file = parent_dir / "region_metrics" / "region_water.json"
    assert metrics_file.exists(), f"metrics file not written: {metrics_file}"


def test_region_mask_without_render_metrics_null(tmp_path):
    """No render PNG → metrics is null and request still 200."""
    _run_store.clear()
    _seed_parent(tmp_path)
    # Do NOT write any render PNG — candidates/ dir does not exist.
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/region-mask",
        json=_VALID_REGION_BODY,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["metrics"] is None, f"expected metrics=null when render absent, got {body['metrics']}"


# ---------------------------------------------------------------------------
# V4.2 security + robustness fixes
# ---------------------------------------------------------------------------

def test_region_mask_traversal_region_id_422(tmp_path):
    """region_id values containing path-traversal characters → 422, no file written outside run_dir."""
    _run_store.clear()
    parent_dir = _seed_parent(tmp_path)
    client = _client()

    bad_ids = ["../../etc/shadow", "foo/bar", "../x", ".."]
    for bad_id in bad_ids:
        body = dict(_VALID_REGION_BODY)
        body["region_id"] = bad_id
        resp = client.post("/png-shader/runs/run_parent/region-mask", json=body)
        assert resp.status_code == 422, (
            f"Expected 422 for region_id={bad_id!r}, got {resp.status_code}: {resp.text}"
        )

    # No stray .json files must have been written outside run_dir's parent.
    parent_of_run_dir = parent_dir.parent
    stray = list(parent_of_run_dir.glob("*.json"))
    assert not stray, f"stray .json files written outside run_dir: {stray}"


def test_region_mask_blank_region_id_422(tmp_path):
    """Missing or whitespace-only region_id → 422."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    # Empty body (no region_id key)
    resp = client.post("/png-shader/runs/run_parent/region-mask", json={})
    assert resp.status_code == 422, f"Expected 422 for missing region_id, got {resp.status_code}"

    # Whitespace-only region_id
    resp2 = client.post(
        "/png-shader/runs/run_parent/region-mask",
        json={"region_id": "  "},
    )
    assert resp2.status_code == 422, (
        f"Expected 422 for whitespace region_id, got {resp2.status_code}"
    )


def test_region_mask_bad_strength_422(tmp_path):
    """Non-numeric strength value → 422, not 500."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    body = dict(_VALID_REGION_BODY)
    body["strength"] = "abc"
    resp = client.post("/png-shader/runs/run_parent/region-mask", json=body)
    assert resp.status_code == 422, (
        f"Expected 422 for non-numeric strength, got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# V4.3 Preference endpoints tests
# ---------------------------------------------------------------------------


def test_preference_profile_get_default(tmp_path):
    """GET /preferences/profile when nothing is saved returns the default profile."""
    client = _client()
    resp = client.get("/png-shader/preferences/profile")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema_version"] == 1
    assert body["enabled"] is True
    assert body["positive_preferences"] == []
    assert body["negative_preferences"] == []
    assert body["summary_source_event_count"] == 0


def test_preference_profile_patch_editable_fields(tmp_path, monkeypatch):
    """PATCH editable fields persists; subsequent GET reflects the change."""
    prefs_root = str(tmp_path / "prefs")
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)
    client = _client()

    patch_body = {"enabled": False, "positive_preferences": ["x"]}
    resp = client.patch("/png-shader/preferences/profile", json=patch_body)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["enabled"] is False
    assert body["positive_preferences"] == ["x"]

    # Subsequent GET should reflect the patch.
    get_resp = client.get("/png-shader/preferences/profile")
    assert get_resp.status_code == 200, get_resp.text
    get_body = get_resp.json()
    assert get_body["enabled"] is False
    assert get_body["positive_preferences"] == ["x"]


def test_preference_profile_patch_disallowed_key_422(tmp_path):
    """PATCH with a disallowed key (schema_version) returns 422."""
    client = _client()
    resp = client.patch("/png-shader/preferences/profile", json={"schema_version": 2})
    assert resp.status_code == 422, resp.text


def test_preference_events_post_and_readable(tmp_path, monkeypatch):
    """POST /preferences/events stores an event; load_preference_events shows it."""
    prefs_root = str(tmp_path / "prefs")
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)
    client = _client()

    resp = client.post(
        "/png-shader/preferences/events",
        json={"event_type": "manual_note", "feedback": "looks great", "reason": "vivid colors"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    event_id = body["event_id"]
    assert event_id.startswith("pref_")

    # Verify via load_preference_events.
    from app.pipeline.preferences import load_preference_events as _load_pref_events
    events = _load_pref_events(root=prefs_root)
    assert len(events) == 1
    assert events[0].event_id == event_id
    assert events[0].event_type == "manual_note"
    assert events[0].feedback == "looks great"


def test_preference_rebuild_reflects_winner_event(tmp_path, monkeypatch):
    """POST rebuild after a winner_selected event updates positive_preferences."""
    prefs_root = str(tmp_path / "prefs")
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)
    client = _client()

    # Post a winner_selected event with reason and tags.
    post_resp = client.post(
        "/png-shader/preferences/events",
        json={
            "event_type": "winner_selected",
            "reason": "sharp edges",
            "tags": ["clean", "minimal"],
            "group_id": "g1",
        },
    )
    assert post_resp.status_code == 200, post_resp.text

    # Trigger rebuild.
    rebuild_resp = client.post("/png-shader/preferences/rebuild")
    assert rebuild_resp.status_code == 200, rebuild_resp.text
    profile = rebuild_resp.json()

    # positive_preferences should include reason and tags.
    assert "sharp edges" in profile["positive_preferences"]
    assert "clean" in profile["positive_preferences"]
    assert "minimal" in profile["positive_preferences"]
    assert profile["summary_source_event_count"] == 1


def test_preference_clear_resets_events_and_profile(tmp_path, monkeypatch):
    """POST /preferences/clear empties events and resets profile to defaults."""
    prefs_root = str(tmp_path / "prefs")
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)
    client = _client()

    # Add some data first.
    client.post(
        "/png-shader/preferences/events",
        json={"event_type": "manual_note", "reason": "temporary"},
    )
    client.patch("/png-shader/preferences/profile", json={"enabled": False})

    # Clear.
    clear_resp = client.post("/png-shader/preferences/clear")
    assert clear_resp.status_code == 200, clear_resp.text
    assert clear_resp.json()["ok"] is True

    # Events should be gone.
    from app.pipeline.preferences import load_preference_events as _load_pref_events
    events = _load_pref_events(root=prefs_root)
    assert events == []

    # Profile should be default.
    get_resp = client.get("/png-shader/preferences/profile")
    profile = get_resp.json()
    assert profile["enabled"] is True
    assert profile["positive_preferences"] == []


def test_variant_winner_mirrors_preference_event(tmp_path, monkeypatch):
    """set_variant_winner also appends a winner_selected preference event."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    prefs_root = str(tmp_path / "prefs")
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)

    children = [
        {"run_id": "pw_a", "status": "completed", "variant_index": 0, "variant_label": "A"},
        {"run_id": "pw_b", "status": "completed", "variant_index": 1, "variant_label": "B"},
    ]
    _seed_group(vg_root, group_id="group_pref_win", children=children)
    client = _client()

    resp = client.post(
        "/png-shader/variant-groups/group_pref_win/winner",
        json={"winner_run_id": "pw_a", "reason": "best contrast"},
    )
    assert resp.status_code == 200, resp.text

    # A preference event must have been appended.
    from app.pipeline.preferences import load_preference_events as _load_pref_events
    events = _load_pref_events(root=prefs_root)
    winner_evts = [e for e in events if e.event_type == "winner_selected"]
    assert len(winner_evts) == 1
    assert winner_evts[0].winner_run_id == "pw_a"
    assert winner_evts[0].reason == "best contrast"
    assert winner_evts[0].context.get("source") == "variant_winner"


def test_rate_variant_mirrors_preference_event(tmp_path, monkeypatch):
    """rate_variant with rating -1 appends a variant_rated preference event."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    prefs_root = str(tmp_path / "prefs")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)

    children = [
        {"run_id": "pr_a", "status": "completed", "variant_index": 0, "variant_label": "A"},
    ]
    _seed_group(vg_root, group_id="group_pref_rate", children=children)
    client = _client()

    resp = client.post(
        "/png-shader/variant-groups/group_pref_rate/ratings",
        json={"run_id": "pr_a", "rating": -1, "reason": "too dark", "tags": ["dark"]},
    )
    assert resp.status_code == 200, resp.text

    # A preference event must have been appended.
    from app.pipeline.preferences import load_preference_events as _load_pref_events
    events = _load_pref_events(root=prefs_root)
    rated_evts = [e for e in events if e.event_type == "variant_rated"]
    assert len(rated_evts) == 1
    assert rated_evts[0].run_id == "pr_a"
    assert rated_evts[0].rating == -1
    assert rated_evts[0].reason == "too dark"
    assert "dark" in rated_evts[0].tags
    assert rated_evts[0].context.get("source") == "variant_rating"


# ---------------------------------------------------------------------------
# V4.4 Preference injection — branch_refine + explore_variants
# ---------------------------------------------------------------------------

def _seed_nonempty_profile(positive_prefs: list[str] | None = None) -> None:
    """Write an enabled profile with at least one positive preference into the
    router's current _PREFERENCES_ROOT (already monkeypatched by autouse fixture)."""
    from app.routers import png_shader as ps
    from app.pipeline import preferences as prefs

    root = ps._PREFERENCES_ROOT
    prof = prefs.default_profile()
    prof["positive_preferences"] = positive_prefs or ["clearer reflections without darkening"]
    prefs.save_profile(prof, root=root)


def test_branch_refine_injects_preference_notes_and_snapshot(tmp_path, monkeypatch):
    """branch_refine with an enabled non-empty profile → human_feedback_notes contains
    a [PREFERENCE+] note; extra_artifacts has preference_profile_snapshot.json."""
    _run_store.clear()
    _seed_parent(tmp_path)
    _seed_nonempty_profile()
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "make the lighting warmer",
            "mode": "refine",
        },
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])

    notes = captured["human_feedback_notes"]
    assert any(n.startswith("[PREFERENCE") for n in notes), (
        f"expected a [PREFERENCE...] note but got: {notes}"
    )
    assert any("[PREFERENCE+] clearer reflections without darkening" in n for n in notes), (
        f"expected specific positive pref note in: {notes}"
    )

    artifacts = captured["extra_artifacts"]
    assert "preference_profile_snapshot.json" in artifacts, (
        f"preference_profile_snapshot.json missing from extra_artifacts: {list(artifacts)}"
    )
    snap = artifacts["preference_profile_snapshot.json"]
    assert "clearer reflections without darkening" in snap.get("positive_preferences", [])


def test_branch_refine_use_preferences_false_no_injection(tmp_path, monkeypatch):
    """branch_refine with use_preferences=false in constraints → no [PREFERENCE] note,
    no preference_profile_snapshot.json, even when profile is non-empty."""
    _run_store.clear()
    _seed_parent(tmp_path)
    _seed_nonempty_profile()
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "make the lighting warmer",
            "mode": "refine",
            "constraints": {"use_preferences": False},
        },
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])

    notes = captured["human_feedback_notes"]
    assert not any(n.startswith("[PREFERENCE") for n in notes), (
        f"unexpected [PREFERENCE...] note with use_preferences=false: {notes}"
    )

    artifacts = captured["extra_artifacts"]
    assert "preference_profile_snapshot.json" not in artifacts, (
        "unexpected preference_profile_snapshot.json when use_preferences=false"
    )

    da = captured["directed_acceptance"]
    assert "preference_score_drop_tolerance_hint" not in da, (
        "unexpected preference_score_drop_tolerance_hint in directed_acceptance"
    )


def test_branch_refine_empty_profile_no_preference_injection(tmp_path, monkeypatch):
    """branch_refine with the default empty profile (no seeded notes) → no [PREFERENCE]
    note injected (backward-compat: empty profile yields no notes)."""
    _run_store.clear()
    _seed_parent(tmp_path)
    # Do NOT seed a non-empty profile — the router will load the default (empty) profile.
    captured: dict = {}
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", _fake_branch_pipeline(captured)
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "checkpoint_id": "final:selected",
            "feedback": "make the lighting warmer",
            "mode": "refine",
        },
    )
    assert resp.status_code == 200, resp.text
    _wait_for_completion(client, resp.json()["run_id"])

    notes = captured["human_feedback_notes"]
    assert not any(n.startswith("[PREFERENCE") for n in notes), (
        f"unexpected [PREFERENCE...] note with empty profile: {notes}"
    )

    artifacts = captured["extra_artifacts"]
    assert "preference_profile_snapshot.json" not in artifacts, (
        "unexpected preference_profile_snapshot.json with empty profile"
    )


def test_explore_variants_injects_preference_notes(tmp_path, monkeypatch):
    """explore_variants with an enabled non-empty profile → each child's
    human_feedback_notes includes a [PREFERENCE+] note and extra_artifacts has
    preference_profile_snapshot.json."""
    _run_store.clear()
    _seed_parent(tmp_path)
    _seed_nonempty_profile(["clearer reflections without darkening"])
    captures: list[dict] = []
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline(captures),
    )
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={
            "feedback": "add more warmth to the lighting",
            "variant_count": 2,
        },
    )
    assert resp.status_code == 200, resp.text
    child_run_ids = resp.json()["child_run_ids"]
    for cid in child_run_ids:
        _wait_for_variant_completion(client, cid)

    assert len(captures) == 2, f"expected 2 captured pipelines, got {len(captures)}"
    for cap in captures:
        notes = cap["human_feedback_notes"]
        assert any(n.startswith("[PREFERENCE") for n in notes), (
            f"expected [PREFERENCE...] note in child notes: {notes}"
        )
        artifacts = cap["extra_artifacts"]
        assert "preference_profile_snapshot.json" in artifacts, (
            f"preference_profile_snapshot.json missing from child extra_artifacts: {list(artifacts)}"
        )
        snap = artifacts["preference_profile_snapshot.json"]
        assert "clearer reflections without darkening" in snap.get("positive_preferences", [])


# ---------------------------------------------------------------------------
# V4.4-B2: preference-assisted ranking surfaced in GET /variant-groups/{id}
# ---------------------------------------------------------------------------

def test_get_variant_group_preference_ranking_enabled(tmp_path, monkeypatch):
    """GET /variant-groups/{id}: matching preferred label → recommended=true;
    winner_run_id is unchanged (None); preference_enabled=true at group level."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    prefs_root = str(tmp_path / "prefs")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)

    import app.routers.png_shader as ps
    from app.pipeline.preferences import save_profile, default_profile

    # Seed an enabled profile that prefers "conservative".
    prof = default_profile()
    prof["enabled"] = True
    prof["preferred_variant_labels"] = ["conservative"]
    save_profile(prof, root=prefs_root)

    children = [
        {
            "run_id": "pr_a",
            "status": "completed",
            "variant_index": 0,
            "variant_label": "conservative",
            "quality_router": {"final_score": 0.7},
        },
        {
            "run_id": "pr_b",
            "status": "completed",
            "variant_index": 1,
            "variant_label": "semantic",
            "quality_router": {"final_score": 0.8},
        },
    ]
    _seed_group(vg_root, group_id="group_pref", children=children)

    client = _client()
    resp = client.get("/png-shader/variant-groups/group_pref")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Group-level field.
    assert body["preference_enabled"] is True

    # winner_run_id must still be None (no winner set via /winner).
    assert body["winner_run_id"] is None

    variants_by_id = {v["run_id"]: v for v in body["variants"]}

    # The "conservative" variant matches the preferred label.
    assert variants_by_id["pr_a"]["recommended"] is True
    assert variants_by_id["pr_a"]["preference_score"] >= 1.0

    # The "semantic" variant has no label match.
    assert variants_by_id["pr_b"]["recommended"] is False
    assert variants_by_id["pr_b"]["preference_score"] == 0.0

    # Winner sort order is unchanged (highest-score "semantic" is first when
    # no winner is set, because score sort still governs).
    # Most importantly, winner_run_id was never auto-set.
    assert body["winner_run_id"] is None


def test_get_variant_group_preference_ranking_disabled(tmp_path, monkeypatch):
    """GET /variant-groups/{id}: with disabled/default profile → recommended=false,
    preference_enabled=false."""
    _run_store.clear()
    vg_root = str(tmp_path / "vg")
    prefs_root = str(tmp_path / "prefs")
    monkeypatch.setattr("app.routers.png_shader._VARIANT_GROUPS_ROOT", vg_root)
    monkeypatch.setattr("app.routers.png_shader._PREFERENCES_ROOT", prefs_root)

    from app.pipeline.preferences import save_profile, default_profile

    # Seed a disabled profile.
    prof = default_profile()
    prof["enabled"] = False
    prof["preferred_variant_labels"] = ["conservative"]
    save_profile(prof, root=prefs_root)

    children = [
        {
            "run_id": "pd_a",
            "status": "completed",
            "variant_index": 0,
            "variant_label": "conservative",
            "quality_router": {"final_score": 0.9},
        },
    ]
    _seed_group(vg_root, group_id="group_pref_dis", children=children)

    client = _client()
    resp = client.get("/png-shader/variant-groups/group_pref_dis")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["preference_enabled"] is False

    v = body["variants"][0]
    assert v["recommended"] is False
    assert v["preference_score"] == 0.0


# ===========================================================================
# V4.5 fusion endpoints
# ===========================================================================


def _seed_fusion_run(tmp_path, run_id, *, with_render=True, with_glsl=True):
    """Seed a run in _run_store with a run_dir, reference image, optional selected
    GLSL, and an optional candidates/llm_0_render.png render PNG (mirrors the
    region-mask render setup). Returns the run_dir Path."""
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (32, 32), (180, 80, 40, 255)).save(run_dir / "reference_input.png")
    if with_render:
        cands_dir = run_dir / "candidates"
        cands_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (32, 32), (40, 120, 200, 255)).save(cands_dir / "llm_0_render.png")
    _run_store[run_id] = {
        "run_id": run_id,
        "status": "completed",
        "run_dir": str(run_dir),
        "selected_glsl": _BRANCH_GLSL if with_glsl else None,
        "quality_router": {"final_score": 0.7},
        "scoreboard": {
            "selected_id": "llm_0",
            "candidates": [{
                "id": "llm_0",
                "source": "llm",
                "selected": True,
                "previewable": True,
                "compile_glsl": _BRANCH_GLSL,
                "final_score": 0.7,
            }],
        },
        "refinement_history": [],
        "strategy": {"refinement_threshold": 0.8},
        "stop_requested": False,
        "strategy_revision": 1,
    }
    return run_dir


def _fusion_body(base_run_id="run_base", source_run_id="run_src",
                 *, region_geometry=None):
    return {
        "base_run_id": base_run_id,
        "feedback": "blend the water reflection from the source into the base",
        "source_run_ids": [source_run_id],
        "regions": [{
            "id": "region_water",
            "label": "water",
            "source_run_id": source_run_id,
            "instruction": "borrow the crisp water reflection",
            "geometry_type": "rect",
            "geometry": region_geometry or {"x": 0.05, "y": 0.55, "w": 0.9, "h": 0.4},
            "strength": 0.6,
            "blend_mode": "soft",
            "feather": 0.08,
        }],
    }


def test_create_fusion_draft_200(tmp_path, monkeypatch):
    """POST /fusions → 200 {fusion_id, status:"draft"}; plan persisted with
    base/source/regions."""
    from app.pipeline.fusion_plans import load_plan
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src")
    fusions_root = str(tmp_path / "fusions_root")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT", fusions_root)

    client = _client()
    resp = client.post("/png-shader/fusions", json=_fusion_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "draft"
    fusion_id = body["fusion_id"]
    assert fusion_id.startswith("fusion_")

    record = load_plan(fusion_id, root=fusions_root)
    assert record is not None
    assert record.base_run_id == "run_base"
    assert "run_src" in record.source_run_ids
    assert len(record.regions) == 1
    assert record.regions[0].id == "region_water"


def test_create_fusion_source_missing_render_422(tmp_path, monkeypatch):
    """A source run with no render → 422 naming that run_id."""
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src", with_render=False)
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))

    client = _client()
    resp = client.post("/png-shader/fusions", json=_fusion_body())
    assert resp.status_code == 422, resp.text
    assert "run_src" in resp.text


def test_create_fusion_base_missing_glsl_422(tmp_path, monkeypatch):
    """A base run lacking selected_glsl → 422."""
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base", with_glsl=False)
    _seed_fusion_run(tmp_path, "run_src")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))

    client = _client()
    resp = client.post("/png-shader/fusions", json=_fusion_body())
    assert resp.status_code == 422, resp.text


def test_create_fusion_unknown_base_404(tmp_path, monkeypatch):
    """Unknown base run → 404."""
    _run_store.clear()
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))
    client = _client()
    resp = client.post("/png-shader/fusions",
                       json=_fusion_body(base_run_id="ghost_base"))
    assert resp.status_code == 404, resp.text


def test_create_fusion_missing_base_run_id_422(tmp_path, monkeypatch):
    """Missing base_run_id → 422."""
    _run_store.clear()
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))
    client = _client()
    resp = client.post("/png-shader/fusions", json={"feedback": "x", "regions": []})
    assert resp.status_code == 422, resp.text


def test_create_fusion_out_of_bounds_region_422(tmp_path, monkeypatch):
    """Out-of-bounds region rect → 422 with fusion_errors."""
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))

    client = _client()
    body = _fusion_body(region_geometry={"x": 0.8, "y": 0.0, "w": 0.5, "h": 0.5})
    resp = client.post("/png-shader/fusions", json=body)
    assert resp.status_code == 422, resp.text
    assert "fusion_errors" in resp.json()["detail"]


def test_get_fusion_status_shape(tmp_path, monkeypatch):
    """GET /fusions/{id} → FusionStatus shape; composite_target_url null before composite."""
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))

    client = _client()
    fusion_id = client.post("/png-shader/fusions", json=_fusion_body()).json()["fusion_id"]

    resp = client.get(f"/png-shader/fusions/{fusion_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["fusion_id"] == fusion_id
    assert body["status"] == "draft"
    assert body["base_run_id"] == "run_base"
    assert "run_src" in body["source_run_ids"]
    assert body["output_run_id"] is None
    assert body["composite_target_url"] is None
    assert isinstance(body["regions"], list) and len(body["regions"]) == 1


def test_get_fusion_unknown_404(tmp_path, monkeypatch):
    _run_store.clear()
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))
    client = _client()
    assert client.get("/png-shader/fusions/fusion_ghost").status_code == 404


def test_composite_target_creates_png_and_url(tmp_path, monkeypatch):
    """POST composite-target → 200 status "target_ready"; composite_target.png
    written; GET now has composite_target_url; artifacts serves the PNG bytes."""
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src")
    fusions_root = str(tmp_path / "fusions_root")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT", fusions_root)

    client = _client()
    fusion_id = client.post("/png-shader/fusions", json=_fusion_body()).json()["fusion_id"]

    resp = client.post(f"/png-shader/fusions/{fusion_id}/composite-target")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "target_ready"
    assert body["composite_target_url"] == f"/png-shader/fusions/{fusion_id}/artifacts/composite_target"

    composite_png = Path(fusions_root) / "fusions" / fusion_id / "composite_target.png"
    assert composite_png.exists(), f"composite not written: {composite_png}"

    # GET now reports the composite_target_url.
    status = client.get(f"/png-shader/fusions/{fusion_id}").json()
    assert status["composite_target_url"] == f"/png-shader/fusions/{fusion_id}/artifacts/composite_target"

    # Artifacts endpoint serves the PNG bytes.
    art = client.get(f"/png-shader/fusions/{fusion_id}/artifacts/composite_target")
    assert art.status_code == 200, art.text
    assert art.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_fusion_run_creates_child(tmp_path, monkeypatch):
    """POST /fusions/{id}/run → 200 {output_run_id, status:"running"}; the output
    run's run-index record has fusion_id/base_run_id/source_run_ids; base & source
    runs unchanged; after the fake worker completes the output run reaches completed."""
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src")
    fusions_root = str(tmp_path / "fusions_root")
    idx = str(tmp_path / "ri.jsonl")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT", fusions_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", idx)
    captures: list[dict] = []
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline(captures),
    )

    client = _client()
    fusion_id = client.post("/png-shader/fusions", json=_fusion_body()).json()["fusion_id"]
    # composite target first (mirrors the real flow producing a target_ready plan).
    client.post(f"/png-shader/fusions/{fusion_id}/composite-target")

    resp = client.post(f"/png-shader/fusions/{fusion_id}/run", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "running"
    output_run_id = body["output_run_id"]
    assert output_run_id.startswith("run_")

    # Output run reaches completed (fake worker).
    final = _wait_for_completion(client, output_run_id)
    assert final["status"] == "completed"

    # Run-index record carries fusion lineage fields.
    from app.pipeline.run_index import load_run_index
    records = load_run_index(path=idx)
    assert output_run_id in records
    rec = records[output_run_id]
    assert rec.fusion_id == fusion_id
    assert rec.base_run_id == "run_base"
    assert "run_src" in rec.source_run_ids
    assert rec.mode == "fusion"

    # The fake pipeline received the fusion notes + directed_acceptance.
    cap = next(c for c in captures if c["run_id"] == output_run_id)
    assert cap["directed_acceptance"]["mode"] == "fusion"
    assert cap["directed_acceptance"]["fusion_id"] == fusion_id
    assert any("FUSION GOAL" in n for n in cap["human_feedback_notes"])

    # Base & source runs are never overwritten.
    assert _run_store["run_base"]["selected_glsl"] == _BRANCH_GLSL
    assert _run_store["run_src"]["run_dir"].endswith("run_src")
    assert _run_store["run_base"]["status"] == "completed"


def test_fusion_run_unknown_404(tmp_path, monkeypatch):
    _run_store.clear()
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))
    client = _client()
    assert client.post("/png-shader/fusions/fusion_ghost/run", json={}).status_code == 404


def test_fusion_artifacts_traversal_422(tmp_path, monkeypatch):
    """region_mask:../etc artifact id → 422 (bad region_id)."""
    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT",
                        str(tmp_path / "fusions_root"))

    client = _client()
    fusion_id = client.post("/png-shader/fusions", json=_fusion_body()).json()["fusion_id"]
    resp = client.get(f"/png-shader/fusions/{fusion_id}/artifacts/region_mask:../etc")
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Bug 3: non-integer count fields → 422 (not uncaught ValueError → 500)
# ---------------------------------------------------------------------------

def test_explore_variants_non_numeric_count_returns_422(tmp_path, monkeypatch):
    """A non-numeric variant_count must yield 422 (validated), never a 500 from
    an uncaught ValueError inside int(...)."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/explore-variants",
        json={"feedback": "more color", "variant_count": "abc"},
    )
    assert resp.status_code == 422, resp.text


def test_draw_session_non_numeric_card_count_returns_422(tmp_path, monkeypatch):
    """A non-numeric card_count must yield 422, not 500."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/draw-session",
        json={"feedback": "warmer", "card_count": "not-a-number"},
    )
    assert resp.status_code == 422, resp.text


def test_branch_refine_non_numeric_max_iter_returns_422(tmp_path, monkeypatch):
    """A non-numeric quality.max_refinement_iterations must yield 422, not 500."""
    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={
            "feedback": "make it warmer and brighter overall",
            "mode": "refine",
            "quality": {"max_refinement_iterations": "lots"},
        },
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Bug 4: /status returns a deep copy taken under the lock (no shared mutable ref)
# ---------------------------------------------------------------------------

def test_status_returns_deep_copy_not_live_store_ref(tmp_path):
    """GET /status must not hand back the live store dict; mutating the returned
    payload (or the store) afterward must not affect the other."""
    _run_store.clear()
    _run_store["run_copy"] = {
        "run_id": "run_copy",
        "status": "running",
        "nested": {"phase": "p0", "items": [1, 2, 3]},
    }
    client = _client()

    resp = client.get("/png-shader/status/run_copy")
    assert resp.status_code == 200

    # Mutate the live store entry's nested structure AFTER the read.
    _run_store["run_copy"]["nested"]["phase"] = "MUTATED"
    _run_store["run_copy"]["nested"]["items"].append(999)

    # The JSON the client already received reflects the pre-mutation snapshot.
    body = resp.json()
    assert body["nested"]["phase"] == "p0"
    assert body["nested"]["items"] == [1, 2, 3]


def test_status_concurrent_mutation_does_not_500(tmp_path):
    """Hammer /status while a background thread mutates the live store entry;
    the deep-copy-under-lock snapshot must never raise 'dictionary changed size
    during iteration'."""
    import threading as _threading

    _run_store.clear()
    _run_store["run_race"] = {"run_id": "run_race", "status": "running", "d": {}}
    client = _client()

    stop = _threading.Event()

    def _mutator():
        i = 0
        while not stop.is_set():
            # Mutate under the same lock the reader uses, churning dict size.
            from app.routers.png_shader import _run_store_lock
            with _run_store_lock:
                d = _run_store["run_race"]["d"]
                d[str(i)] = i
                if len(d) > 50:
                    d.clear()
            i += 1

    t = _threading.Thread(target=_mutator, daemon=True)
    t.start()
    try:
        for _ in range(100):
            resp = client.get("/png-shader/status/run_race")
            assert resp.status_code == 200, resp.text
    finally:
        stop.set()
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Bug 5: a completed fusion run drives its FusionPlanRecord to a terminal status
# ---------------------------------------------------------------------------

def test_fusion_run_completion_marks_plan_completed(tmp_path, monkeypatch):
    """After the fusion output run completes, the fusion plan record status must
    leave 'running' and become 'completed' (so the frontend poll terminates)."""
    from app.pipeline.fusion_plans import load_plan

    _run_store.clear()
    _seed_fusion_run(tmp_path, "run_base")
    _seed_fusion_run(tmp_path, "run_src")
    fusions_root = str(tmp_path / "fusions_root")
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT", fusions_root)
    monkeypatch.setattr("app.routers.png_shader._RUN_INDEX_PATH", str(tmp_path / "ri.jsonl"))
    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline",
        _fake_variant_pipeline([]),
    )

    client = _client()
    fusion_id = client.post("/png-shader/fusions", json=_fusion_body()).json()["fusion_id"]
    client.post(f"/png-shader/fusions/{fusion_id}/composite-target")

    resp = client.post(f"/png-shader/fusions/{fusion_id}/run", json={})
    assert resp.status_code == 200, resp.text
    output_run_id = resp.json()["output_run_id"]

    _wait_for_completion(client, output_run_id)

    # Give the worker's best-effort fusion finalize a moment to land.
    deadline = time.time() + 3.0
    record = None
    while time.time() < deadline:
        record = load_plan(fusion_id, root=fusions_root)
        if record is not None and record.status in ("completed", "failed"):
            break
        time.sleep(0.02)

    assert record is not None
    assert record.status == "completed", f"fusion status stuck at {record.status!r}"


# ---------------------------------------------------------------------------
# Security hardening: path-traversal on fusion_id (Item 2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_id", ["..", "....", "%2e%2e", "a%2Fb", "a.b", "a:b"])
def test_get_fusion_rejects_unsafe_fusion_id(tmp_path, monkeypatch, bad_id):
    """A fusion_id with path-traversal / non-allowlisted chars is rejected 4xx
    and never touches the filesystem outside the fusions root."""
    import app.routers.png_shader as mod

    fusions_root = tmp_path / "fusions_root"
    monkeypatch.setattr("app.routers.png_shader._FUSIONS_ROOT", str(fusions_root))

    # Spy on load_plan to prove no filesystem lookup happens for a rejected id.
    calls: list = []
    orig = mod.load_plan
    monkeypatch.setattr(
        mod, "load_plan", lambda *a, **k: (calls.append(a), orig(*a, **k))[1]
    )

    client = _client()
    resp = client.get(f"/png-shader/fusions/{bad_id}")
    # Either our validator (422) or the router not matching a "/" id (404) — both
    # are safe rejections; what matters is it is a 4xx and load_plan never ran for
    # ids that DID reach the handler.
    assert resp.status_code in (404, 422), resp.text
    if resp.status_code == 422:
        assert calls == [], "load_plan must not run for a rejected fusion_id"


def test_run_fusion_rejects_unsafe_fusion_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.routers.png_shader._FUSIONS_ROOT", str(tmp_path / "fusions_root")
    )
    client = _client()
    resp = client.post("/png-shader/fusions/..%2e/run", json={})
    assert resp.status_code in (404, 422), resp.text


def test_fusion_artifact_rejects_unsafe_fusion_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.routers.png_shader._FUSIONS_ROOT", str(tmp_path / "fusions_root")
    )
    client = _client()
    resp = client.get("/png-shader/fusions/..../artifacts/composite_target")
    assert resp.status_code in (404, 422), resp.text


def test_valid_fusion_id_still_accepted(tmp_path, monkeypatch):
    """A normal fusion_id passes validate_safe_id (regression for Item 2)."""
    monkeypatch.setattr(
        "app.routers.png_shader._FUSIONS_ROOT", str(tmp_path / "fusions_root")
    )
    client = _client()
    # Unknown-but-safe id → 404 (not a 422 validation error).
    resp = client.get("/png-shader/fusions/fusion_abcd1234")
    assert resp.status_code == 404, resp.text


def test_validate_safe_id_helper():
    """Unit-level check of the allowlist helper (Item 2)."""
    from fastapi import HTTPException

    from app.routers.png_shader import validate_safe_id

    assert validate_safe_id("fusion_abcd1234") == "fusion_abcd1234"
    assert validate_safe_id("Run-01_x") == "Run-01_x"
    for bad in ["", "..", "../x", "a/b", "a.b", "a b", "a:b", None, 5]:
        with pytest.raises(HTTPException) as ei:
            validate_safe_id(bad)
        assert ei.value.status_code == 422


# ---------------------------------------------------------------------------
# Security hardening: free-text / code length caps (Item 3)
# ---------------------------------------------------------------------------

def test_run_rejects_oversized_seed_glsl(tmp_path, monkeypatch):
    """An over-cap seed_glsl is rejected 422 before any pipeline work."""
    from app.routers.png_shader import _MAX_SEED_GLSL_CHARS

    _run_store.clear()
    client = _client()
    oversized = "a" * (_MAX_SEED_GLSL_CHARS + 1)

    resp = client.post(
        "/png-shader/run",
        data={"seed_glsl": oversized},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert resp.status_code == 422, resp.text


def test_run_rejects_oversized_input_spec_json(tmp_path):
    from app.routers.png_shader import _MAX_INPUT_SPEC_CHARS

    _run_store.clear()
    client = _client()
    oversized = "x" * (_MAX_INPUT_SPEC_CHARS + 1)

    resp = client.post(
        "/png-shader/run",
        data={"input_spec_json": oversized},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert resp.status_code == 422, resp.text


def test_branch_refine_rejects_oversized_feedback(tmp_path):
    from app.routers.png_shader import _MAX_FEEDBACK_CHARS

    _run_store.clear()
    _seed_parent(tmp_path)
    client = _client()
    oversized = "f" * (_MAX_FEEDBACK_CHARS + 1)

    resp = client.post(
        "/png-shader/runs/run_parent/branch-refine",
        json={"feedback": oversized, "mode": "refine"},
    )
    assert resp.status_code == 422, resp.text

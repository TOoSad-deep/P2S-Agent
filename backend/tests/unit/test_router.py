"""API contract tests for the PNG-to-Shader router."""

from __future__ import annotations

import json
import time

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from app.routers.png_shader import _run_store, router


# ---------------------------------------------------------------------------
# Autouse fixture: isolate every test from the real run_index.jsonl
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_run_index(tmp_path, monkeypatch):
    """Redirect all run-index writes to a per-test temp file so the real
    backend/test_results/run_index.jsonl is never touched during tests."""
    monkeypatch.setattr(
        "app.routers.png_shader._RUN_INDEX_PATH",
        str(tmp_path / "run_index.jsonl"),
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
    assert rec.status in ("completed", "failed")  # terminal


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

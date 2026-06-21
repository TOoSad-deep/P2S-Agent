"""Phase 0 unit tests: PNG-to-Shader artifacts helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from p2s_agent.core.pipeline import artifacts


# === create_run_dir ===


def test_default_results_root_is_backend_test_results():
    assert artifacts.DEFAULT_RESULTS_ROOT.name == "test_results"
    assert artifacts.DEFAULT_RESULTS_ROOT.parent.name == "backend"


# === P2S_RESULTS_ROOT env override ===


def test_results_root_from_env_falls_back_to_backend_test_results():
    root = artifacts._results_root_from_env({})
    assert root.name == "test_results"
    assert root.parent.name == "backend"


def test_results_root_from_env_honors_p2s_results_root():
    root = artifacts._results_root_from_env({"P2S_RESULTS_ROOT": "/tmp/p2s-data"})
    assert root == Path("/tmp/p2s-data")


def test_results_root_from_env_ignores_blank_value():
    # A blank/whitespace-only override must NOT collapse the root to "" — fall
    # back to the packaged default instead.
    root = artifacts._results_root_from_env({"P2S_RESULTS_ROOT": "   "})
    assert root.name == "test_results"
    assert root.parent.name == "backend"


def test_create_run_dir_uses_date_and_slugified_label(tmp_path):
    fixed = datetime(2026, 6, 7, 12, 30, 0, tzinfo=timezone.utc)
    handle = artifacts.create_run_dir(
        run_id="abc123",
        label="smoke 10 / batch",
        root=tmp_path,
        clock=lambda: fixed,
    )

    assert handle.path.is_dir()
    assert handle.path.parent == tmp_path
    # Slashes and spaces collapse into single hyphens.
    assert handle.path.name == "2026-06-07_png-shader_smoke-10-batch_abc123"
    assert handle.run_id == "abc123"
    assert handle.label == "smoke-10-batch"
    assert handle.created_at == "2026-06-07T12:30:00+00:00"


def test_create_run_dir_is_idempotent_with_same_inputs(tmp_path):
    fixed = datetime(2026, 6, 7, tzinfo=timezone.utc)
    h1 = artifacts.create_run_dir("r1", "baseline", root=tmp_path, clock=lambda: fixed)
    h2 = artifacts.create_run_dir("r1", "baseline", root=tmp_path, clock=lambda: fixed)
    assert h1.path == h2.path
    assert h1.path.is_dir()


def test_create_run_dir_empty_label_falls_back(tmp_path):
    handle = artifacts.create_run_dir("r", "   ", root=tmp_path)
    assert "_unlabeled_" in handle.path.name


# === save_json ===


def test_save_json_writes_sorted_pretty_utf8(tmp_path):
    target = tmp_path / "out" / "data.json"
    payload = {"z": 1, "a": {"nested": "héllo"}}

    written = artifacts.save_json(target, payload)

    assert written == target
    text = target.read_text(encoding="utf-8")
    # Keys must be sorted (so diffs across runs are stable).
    assert text.index('"a"') < text.index('"z"')
    # UTF-8 stays intact (ensure_ascii=False).
    assert "héllo" in text
    # Ends with newline so POSIX tools don't complain.
    assert text.endswith("\n")
    # Round-trip.
    assert json.loads(text) == payload


def test_save_json_does_not_leave_tempfile(tmp_path):
    target = tmp_path / "data.json"
    artifacts.save_json(target, {"x": 1})

    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_save_json_overwrites_existing(tmp_path):
    target = tmp_path / "data.json"
    artifacts.save_json(target, {"v": 1})
    artifacts.save_json(target, {"v": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


# === copy_artifact ===


def test_copy_artifact_to_directory_keeps_basename(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    dest_dir = tmp_path / "run"
    dest_dir.mkdir()

    out = artifacts.copy_artifact(src, dest_dir)

    assert out == dest_dir / "src.png"
    assert out.read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_copy_artifact_to_explicit_file_creates_parent(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"data")
    dest = tmp_path / "run" / "renamed.png"

    out = artifacts.copy_artifact(src, dest)

    assert out == dest
    assert dest.read_bytes() == b"data"


def test_copy_artifact_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifacts.copy_artifact(tmp_path / "nope.png", tmp_path / "out.png")


# === write_manifest ===


def test_write_manifest_contains_expected_fields(tmp_path):
    fixed = datetime(2026, 6, 7, tzinfo=timezone.utc)
    handle = artifacts.create_run_dir("r1", "phase0", root=tmp_path, clock=lambda: fixed)

    spec = {"input_image": "icon.png", "target": {"backend": "glsl"}}
    cfg = {"max_iterations": 5}

    path = artifacts.write_manifest(
        handle,
        input_spec=spec,
        git_sha="deadbeef",
        config=cfg,
        extras={"notes": "phase 0 smoke"},
    )

    assert path == handle.path / "manifest.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["schema_version"] == artifacts.MANIFEST_SCHEMA_VERSION
    assert body["run_id"] == "r1"
    assert body["label"] == "phase0"
    assert body["created_at"] == "2026-06-07T00:00:00+00:00"
    assert body["git_sha"] == "deadbeef"
    assert body["input_spec"] == spec
    assert body["config"] == cfg
    assert body["notes"] == "phase 0 smoke"


def test_write_manifest_allows_missing_input_and_config(tmp_path):
    handle = artifacts.create_run_dir("r2", "baseline", root=tmp_path)
    path = artifacts.write_manifest(handle, input_spec=None, git_sha=None, config=None)

    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["input_spec"] is None
    assert body["config"] is None
    # git_sha may be a real SHA from the repo or None if git is missing.
    assert "git_sha" in body


def test_write_manifest_rejects_extras_key_collision(tmp_path):
    handle = artifacts.create_run_dir("r3", "x", root=tmp_path)
    with pytest.raises(ValueError):
        artifacts.write_manifest(
            handle,
            input_spec=None,
            extras={"run_id": "should-not-override"},
        )


def test_write_manifest_accepts_raw_path(tmp_path):
    """Caller may already own a directory; we don't require RunDir."""
    plain = tmp_path / "preexisting"
    plain.mkdir()
    path = artifacts.write_manifest(plain, input_spec={"x": 1})
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["run_id"] == "preexisting"
    assert body["input_spec"] == {"x": 1}

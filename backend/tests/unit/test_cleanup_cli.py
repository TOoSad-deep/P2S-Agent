"""Unit tests for the `python -m p2s_agent.tools.cleanup_runs` CLI.

TDD: written before the implementation. Each test first fails for the right
reason (ImportError), then passes after implementation.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from p2s_agent.tools import cleanup_runs
from p2s_agent.orchestration.run_index import (
    RunLineageRecord,
    append_run_created,
    load_run_index,
)

_NOW = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)


def _at(days_ago: float) -> float:
    return (_NOW - timedelta(days=days_ago)).timestamp()


def _seed_run(root: Path, run_id: str, *, created_at: float, status: str = "completed") -> None:
    d = root / f"2026-06-01_png-shader_single_run_{run_id}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    rec = RunLineageRecord(
        run_id=run_id, root_run_id=run_id, parent_run_id=None,
        source_checkpoint_id=None, source_checkpoint_label=None,
        mode=None, feedback=None, title=None, status=status,
        run_dir=str(d), created_at=created_at,
    )
    append_run_created(rec, path=root / "run_index.jsonl")


def _dir(root: Path, run_id: str) -> Path:
    return root / f"2026-06-01_png-shader_single_run_{run_id}"


def test_cli_dry_run_lists_without_deleting(tmp_path, capsys):
    for i in range(3):
        _seed_run(tmp_path, f"r{i}", created_at=_at(10 - i))

    rc = cleanup_runs.main(
        ["--root", str(tmp_path), "--max-runs", "1"], env={}, now=_NOW
    )

    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "dry-run" in out
    # nothing deleted in dry-run
    assert all(_dir(tmp_path, f"r{i}").exists() for i in range(3))


def test_cli_apply_deletes_oldest_and_prunes_index(tmp_path):
    for i in range(3):
        _seed_run(tmp_path, f"r{i}", created_at=_at(10 - i))

    rc = cleanup_runs.main(
        ["--root", str(tmp_path), "--max-runs", "1", "--apply"], env={}, now=_NOW
    )

    assert rc == 0
    assert not _dir(tmp_path, "r0").exists()
    assert not _dir(tmp_path, "r1").exists()
    assert _dir(tmp_path, "r2").exists()
    assert set(load_run_index(path=tmp_path / "run_index.jsonl").keys()) == {"r2"}


def test_cli_runs_even_when_env_disables_retention(tmp_path):
    # An explicit CLI invocation must act regardless of P2S_RETENTION_ENABLED.
    for i in range(2):
        _seed_run(tmp_path, f"r{i}", created_at=_at(10 - i))

    rc = cleanup_runs.main(
        ["--root", str(tmp_path), "--max-runs", "1", "--apply"],
        env={"P2S_RETENTION_ENABLED": "false"},
        now=_NOW,
    )

    assert rc == 0
    assert not _dir(tmp_path, "r0").exists()
    assert _dir(tmp_path, "r1").exists()


def test_cli_include_orphans_deletes_old_recordless_dir(tmp_path):
    _seed_run(tmp_path, "kept", created_at=_at(1))
    orphan = tmp_path / "2026-01-01_png-shader_single_run_orphan"
    orphan.mkdir()
    (orphan / "f").write_text("x")
    os.utime(orphan, (_at(100), _at(100)))

    rc = cleanup_runs.main(
        [
            "--root", str(tmp_path),
            "--max-runs", "10",
            "--include-orphans",
            "--max-age-days", "30",
            "--apply",
        ],
        env={},
        now=_NOW,
    )

    assert rc == 0
    assert not orphan.exists()
    assert _dir(tmp_path, "kept").exists()

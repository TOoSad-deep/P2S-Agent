"""Run-level artifact helpers for PNG-to-Shader.

Phase 0 deliverable: provide a small, well-tested file-system layer so
every later phase emits comparable artifacts under a single run
directory.

Contract (per phased plan §3):
    - ``create_run_dir(run_id, label)``
    - ``save_json(path, data)``
    - ``copy_artifact(src, dest)``
    - ``write_manifest(run_dir, input_spec, git_sha, config)``

The layout is::

    backend/test_results/<YYYY-MM-DD>_png-shader_<label>_<run_id>/
        manifest.json
        input_spec.json (later phases)
        baseline_result.json (later phases)
        ...

The module deliberately depends only on the standard library so it can
be imported without pulling in fastapi, langgraph, or playwright.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


# Default root for all PNG-to-Shader run directories.
# Tests can override by passing ``root=`` explicitly.
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "test_results"

# Manifest schema version. Bump on breaking changes so downstream
# tooling can detect old layouts.
MANIFEST_SCHEMA_VERSION = 1

_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(label: str) -> str:
    """Make a label safe for use as a directory component."""
    label = label.strip()
    if not label:
        return "unlabeled"
    return _SAFE_LABEL_RE.sub("-", label).strip("-") or "unlabeled"


@dataclass(frozen=True)
class RunDir:
    """Handle to a created run directory.

    Use ``path`` for filesystem operations and ``run_id``/``label`` when
    logging or building cross-run reports.
    """

    path: Path
    run_id: str
    label: str
    created_at: str  # ISO-8601 UTC


def create_run_dir(
    run_id: str,
    label: str,
    *,
    root: Path | str | None = None,
    clock: "Callable[[], datetime] | None" = None,
) -> RunDir:
    """Create (and return) a directory for a single PNG-to-Shader run.

    The directory name follows ``<YYYY-MM-DD>_png-shader_<label>_<run_id>``
    so that ``ls`` produces a chronological listing without external
    tooling.

    Args:
        run_id: Unique identifier for this run. Caller decides format
            (uuid, monotonic counter, etc.); only characters allowed in
            ``_SAFE_LABEL_RE`` are kept.
        label: Human-readable short label (e.g. ``"baseline"`` or
            ``"smoke-10"``).
        root: Optional override for the results root. Defaults to
            ``DEFAULT_RESULTS_ROOT``. Useful for tests.
        clock: Optional callable returning a ``datetime`` for the run's
            timestamp. Defaults to ``datetime.now(timezone.utc)``.
    """

    root_path = Path(root) if root is not None else DEFAULT_RESULTS_ROOT
    now = clock() if clock is not None else datetime.now(timezone.utc)

    date_part = now.strftime("%Y-%m-%d")
    safe_label = _slugify(label)
    safe_run_id = _slugify(run_id)
    dir_name = f"{date_part}_png-shader_{safe_label}_{safe_run_id}"

    run_path = root_path / dir_name
    run_path.mkdir(parents=True, exist_ok=True)

    return RunDir(
        path=run_path,
        run_id=safe_run_id,
        label=safe_label,
        created_at=now.replace(microsecond=0).isoformat(),
    )


def save_json(path: Path | str, data: Any, *, indent: int = 2) -> Path:
    """Serialize ``data`` as JSON at ``path``.

    Writes through a sibling temp file then ``os.replace`` so concurrent
    readers never see a half-written file. Parent directories are
    created on demand.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, target)
    finally:
        # Best-effort cleanup if replace failed.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    return target


def copy_artifact(src: Path | str, dest: Path | str) -> Path:
    """Copy ``src`` to ``dest`` preserving metadata.

    ``dest`` may be a directory or a target file path. Parent
    directories are created when needed. Returns the final destination
    path.
    """

    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"copy_artifact: source does not exist: {src_path}")

    dest_path = Path(dest)
    if dest_path.exists() and dest_path.is_dir():
        dest_path = dest_path / src_path.name
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_path, dest_path)
    return dest_path


def _current_git_sha() -> str | None:
    """Return ``git rev-parse HEAD`` or ``None`` if unavailable.

    Used as a default for :func:`write_manifest` so callers do not need
    to plumb the SHA through every layer.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None


def write_manifest(
    run_dir: RunDir | Path | str,
    input_spec: Mapping[str, Any] | None,
    git_sha: str | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    extras: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` into the run directory.

    The manifest is the single source of truth that ties artifacts to
    a particular code revision, input, and config. Later phases append
    candidate/metric files alongside it; the manifest itself stays
    small and human-readable.

    Args:
        run_dir: Result of :func:`create_run_dir` or a path-like.
        input_spec: PNG-to-Shader input spec (Phase 2 schema). May be
            ``None`` in Phase 0 when only the baseline is being saved.
        git_sha: Optional SHA override. Defaults to ``git rev-parse HEAD``.
        config: Optional config snapshot (model settings, thresholds).
        extras: Optional free-form mapping merged into the manifest.
            Useful for one-off debug fields without bumping the schema.
    """

    if isinstance(run_dir, RunDir):
        run_path = run_dir.path
        run_id = run_dir.run_id
        label = run_dir.label
        created_at = run_dir.created_at
    else:
        run_path = Path(run_dir)
        run_id = run_path.name
        label = run_path.name
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "label": label,
        "created_at": created_at,
        "git_sha": git_sha if git_sha is not None else _current_git_sha(),
        "input_spec": dict(input_spec) if input_spec is not None else None,
        "config": dict(config) if config is not None else None,
    }
    if extras:
        for key, value in extras.items():
            if key in manifest:
                raise ValueError(f"extras key collides with manifest field: {key!r}")
            manifest[key] = value

    return save_json(run_path / "manifest.json", manifest)

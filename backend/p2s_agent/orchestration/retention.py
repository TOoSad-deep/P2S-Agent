"""Run-directory retention & cleanup for the PNG-to-Shader data root.

Run directories under the data root (``backend/test_results`` by default, or
``$P2S_RESULTS_ROOT``) are self-contained and never pruned by the pipeline, so
they accumulate without bound. This module provides:

* :class:`RetentionPolicy` + :func:`policy_from_env` — configuration parsed from
  ``P2S_RETENTION_*`` environment variables.
* :func:`plan_cleanup` — a **pure** function that decides which run directories
  may be deleted, honoring a set of safety invariants (see below). It does NOT
  touch the filesystem except to ``stat``/size directories when a byte-cap is in
  play.
* :func:`apply_cleanup` — executes a plan: ``rmtree`` each target and prune the
  matching ``run_index`` records so the branch tree never references a missing
  directory.

Safety invariants (a run dir is deletable only if ALL hold):
  1. it is a ``*_png-shader_*`` directory under the root;
  2. it has a terminal ``run_index`` record (completed/failed/cancelled);
     directories without a record are reported, never deleted (unless the CLI
     opts in via ``include_orphans``);
  3. it is a standalone run — not a member of, nor referenced by, any
     variant-group / draw-session / fusion;
  4. no surviving run references it as an ancestor (parent/root) — i.e. we delete
     from the leaves up, protecting ancestors of kept runs;
  5. it falls outside the retention window (§ policy).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from p2s_agent.orchestration.run_index import (
    RunLineageRecord,
    _TERMINAL_STATUSES,
    prune_run_index,
)

logger = logging.getLogger(__name__)

# Substring identifying a pipeline run directory (e.g.
# ``2026-06-21_png-shader_single_run_abc123``).
_RUN_DIR_MARKER = "png-shader"

_SIZE_UNITS = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*$")

_TRUEY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def parse_size(text: str | None) -> int | None:
    """Parse a human size (``2GB``/``500MB``/``1024``) into bytes (base 1024).

    Blank / ``None`` → ``None`` (limit disabled). Unrecognised → ``ValueError``.
    """
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    m = _SIZE_RE.match(stripped)
    if not m:
        raise ValueError(f"unparseable size: {text!r}")
    number, unit = m.group(1), m.group(2).upper()
    if unit not in _SIZE_UNITS:
        raise ValueError(f"unknown size unit in {text!r}: {unit!r}")
    return int(float(number) * _SIZE_UNITS[unit])


def _parse_bool(text: str, *, default: bool) -> bool:
    norm = text.strip().lower()
    if norm in _TRUEY:
        return True
    if norm in _FALSEY:
        return False
    return default


def _parse_positive_int(text: str | None) -> int | None:
    """Return a positive int, or ``None`` when unset / blank / ``<= 0``."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    value = int(stripped)
    return value if value > 0 else None


@dataclass(frozen=True)
class RetentionPolicy:
    """Resolved retention configuration. ``None`` on a limit means *disabled*."""

    enabled: bool = True
    max_runs: int | None = 1000
    max_age_days: int | None = None
    max_bytes: int | None = None

    @property
    def any_limit_active(self) -> bool:
        return any(
            limit is not None
            for limit in (self.max_runs, self.max_age_days, self.max_bytes)
        )


def policy_from_env(env: Mapping[str, str]) -> RetentionPolicy:
    """Build a :class:`RetentionPolicy` from ``P2S_RETENTION_*`` env vars.

    Defaults: enabled, keep newest 1000 runs, age/byte caps disabled.
    """
    enabled = _parse_bool(env.get("P2S_RETENTION_ENABLED", "true"), default=True)
    raw_runs = env.get("P2S_RETENTION_MAX_RUNS")
    max_runs = _parse_positive_int(raw_runs) if raw_runs is not None else 1000
    max_age_days = _parse_positive_int(env.get("P2S_RETENTION_MAX_AGE_DAYS"))
    max_bytes = parse_size(env.get("P2S_RETENTION_MAX_BYTES"))
    if max_bytes is not None and max_bytes <= 0:
        max_bytes = None
    return RetentionPolicy(
        enabled=enabled,
        max_runs=max_runs,
        max_age_days=max_age_days,
        max_bytes=max_bytes,
    )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeletionTarget:
    run_id: str
    run_dir: Path
    created_at: float
    size_bytes: int | None
    reason: str


@dataclass
class CleanupPlan:
    root: Path
    delete: list[DeletionTarget] = field(default_factory=list)
    freed_bytes: int = 0
    kept_count: int = 0
    skipped_no_record: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)


def _dir_size(path: Path) -> int:
    """Sum of regular-file ``st_size`` under *path* (deterministic, not blocks)."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and _RUN_DIR_MARKER in path.name


def _ancestor_refs(rec: RunLineageRecord) -> set[str]:
    """Run-ids *rec* depends on as ancestors (branch lineage + fusion inputs)."""
    refs: set[str] = set()
    if rec.parent_run_id:
        refs.add(rec.parent_run_id)
    if rec.root_run_id and rec.root_run_id != rec.run_id:
        refs.add(rec.root_run_id)
    if rec.base_run_id:
        refs.add(rec.base_run_id)
    refs.update(rec.source_run_ids or [])
    return refs


def plan_cleanup(
    root: Path | str,
    policy: RetentionPolicy,
    *,
    now: datetime,
    index_records: Mapping[str, RunLineageRecord],
    compute_sizes: bool = False,
) -> CleanupPlan:
    """Decide which run directories under *root* may be deleted (pure).

    Honors the five safety invariants documented at module level. Does not touch
    the filesystem beyond reading directory listings (and ``stat`` sizes when a
    byte-cap is active or *compute_sizes* is set). Returns a :class:`CleanupPlan`.
    """
    root = Path(root)
    plan = CleanupPlan(root=root)

    if not root.exists() or not policy.enabled or not policy.any_limit_active:
        # Still report how many run dirs exist so callers can log "kept N".
        if root.exists():
            plan.kept_count = sum(1 for d in root.iterdir() if _is_run_dir(d))
        return plan

    # 1. Discover run directories and pair them with lineage records by dirname.
    dirs = sorted((d for d in root.iterdir() if _is_run_dir(d)), key=lambda d: d.name)
    rec_by_dirname: dict[str, RunLineageRecord] = {
        Path(r.run_dir).name: r for r in index_records.values() if r.run_dir
    }
    paired: list[tuple[Path, RunLineageRecord]] = []
    for d in dirs:
        rec = rec_by_dirname.get(d.name)
        if rec is None:
            plan.skipped_no_record.append(d.name)
        else:
            paired.append((d, rec))

    # Runs referenced as a fusion base/source by ANY record → protected wholesale.
    referenced_as_input: set[str] = set()
    for r in index_records.values():
        if r.base_run_id:
            referenced_as_input.add(r.base_run_id)
        referenced_as_input.update(r.source_run_ids or [])

    def _eligible(rec: RunLineageRecord) -> bool:
        if rec.status not in _TERMINAL_STATUSES:
            return False  # invariant 2
        if (  # invariant 3a: not a session member itself
            rec.variant_group_id is not None
            or rec.draw_session_id is not None
            or rec.fusion_id is not None
            or rec.base_run_id is not None
        ):
            return False
        if rec.run_id in referenced_as_input:  # invariant 3b: fusion input
            return False
        return True

    eligible = [(d, rec) for (d, rec) in paired if _eligible(rec)]
    eligible_by_id = {rec.run_id: (d, rec) for (d, rec) in eligible}

    # 2. Retention window → candidate run-ids with a reason (count, then age).
    reason: dict[str, str] = {}

    if policy.max_runs is not None:
        # Rank known runs newest-first; everything past the keep-window is a
        # candidate (only eligible ones can actually be deleted).
        ranked = sorted(paired, key=lambda dr: dr[1].created_at, reverse=True)
        for d, rec in ranked[policy.max_runs:]:
            if rec.run_id in eligible_by_id:
                reason.setdefault(
                    rec.run_id, f"beyond max_runs (keep newest {policy.max_runs})"
                )

    if policy.max_age_days is not None:
        cutoff = (now - timedelta(days=policy.max_age_days)).timestamp()
        for d, rec in eligible:
            if rec.created_at < cutoff:
                reason.setdefault(rec.run_id, f"older than {policy.max_age_days}d")

    delete_ids = set(reason)

    # 3. Lineage protection (invariant 4): never delete a run still referenced as
    #    an ancestor by a SURVIVING run. Iterate to a fixpoint (delete set only
    #    shrinks). Survivors = every record not currently in delete_ids.
    protected: set[str] = set()
    changed = True
    while changed:
        changed = False
        needed: set[str] = set()
        for r in index_records.values():
            if r.run_id in delete_ids:
                continue
            needed |= _ancestor_refs(r)
        rescued = delete_ids & needed
        if rescued:
            delete_ids -= rescued
            protected |= rescued
            changed = True

    # 4. Byte-cap backstop: if still over, delete oldest survivors (eligible,
    #    unprotected, no surviving descendant) until under the cap.
    sizes: dict[str, int] = {}
    want_sizes = compute_sizes or policy.max_bytes is not None
    if want_sizes:
        for d, rec in paired:
            sizes[rec.run_id] = _dir_size(d)

    if policy.max_bytes is not None:
        surviving_total = sum(
            sizes[rec.run_id] for d, rec in paired if rec.run_id not in delete_ids
        )
        if surviving_total > policy.max_bytes:
            survivors_oldest = sorted(
                (
                    (d, rec)
                    for d, rec in eligible
                    if rec.run_id not in delete_ids and rec.run_id not in protected
                ),
                key=lambda dr: dr[1].created_at,
            )
            for d, rec in survivors_oldest:
                if surviving_total <= policy.max_bytes:
                    break
                # Re-check lineage: a survivor (outside delete_ids ∪ {this}) must
                # not reference this run as an ancestor.
                still_referenced = any(
                    rec.run_id in _ancestor_refs(other)
                    for other in index_records.values()
                    if other.run_id not in delete_ids and other.run_id != rec.run_id
                )
                if still_referenced:
                    continue
                delete_ids.add(rec.run_id)
                reason.setdefault(rec.run_id, "over size cap")
                surviving_total -= sizes[rec.run_id]

    # 5. Materialise the plan.
    targets: list[DeletionTarget] = []
    for run_id in delete_ids:
        d, rec = eligible_by_id[run_id]
        targets.append(
            DeletionTarget(
                run_id=run_id,
                run_dir=d,
                created_at=rec.created_at,
                size_bytes=sizes.get(run_id),
                reason=reason[run_id],
            )
        )
    targets.sort(key=lambda t: t.created_at)  # oldest first for display

    plan.delete = targets
    plan.freed_bytes = sum(t.size_bytes or 0 for t in targets)
    plan.kept_count = len(dirs) - len(targets)
    plan.protected = sorted(protected)
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class CleanupResult:
    deleted: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def apply_cleanup(plan: CleanupPlan, *, index_path: Path | str | None) -> CleanupResult:
    """Execute *plan*: ``rmtree`` each target, then prune the matching records.

    A single ``rmtree`` failure is recorded in ``errors`` and does not abort the
    rest. Only run-ids whose directory was actually removed are pruned from the
    index, so a failed delete keeps its lineage record intact.
    """
    result = CleanupResult()
    for target in plan.delete:
        try:
            shutil.rmtree(target.run_dir)
        except OSError as exc:
            logger.warning(
                "retention: failed to delete %s (%s); leaving index record",
                target.run_dir,
                exc,
            )
            result.errors.append((target.run_id, str(exc)))
            continue
        result.deleted.append(target.run_id)
        result.freed_bytes += target.size_bytes or 0

    if result.deleted:
        prune_run_index(set(result.deleted), path=index_path)
    return result


def session_referenced_run_ids(root: Path | str) -> set[str]:
    """Run-ids referenced by the variant-group / draw-session / fusion JSON.

    Record-less directories can still be live members of a session subsystem (a
    variant card, a draw winner, a fusion source). The branch tree never sees
    them, but the subsystem JSON does — so ``--include-orphans`` must protect any
    dir whose run-id appears here. Best-effort: unreadable/garbage JSON is
    skipped.
    """
    root = Path(root)
    refs: set[str] = set()

    def _walk(obj: object) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, str) and obj.startswith("run_"):
            refs.add(obj)

    for sub in ("variant_groups", "draw_sessions", "fusions"):
        d = root / sub
        if not d.is_dir():
            continue
        for jf in d.glob("*.json"):
            try:
                _walk(json.loads(jf.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return refs


def orphan_targets(
    root: Path | str,
    skipped_no_record: list[str],
    *,
    now: datetime,
    max_age_days: int | None,
    protected_run_ids: set[str] | None = None,
) -> list[DeletionTarget]:
    """Deletion targets for record-less run dirs (CLI ``--include-orphans``).

    Gated on an explicit age bound: without *max_age_days* this returns ``[]``,
    so a record-less directory is never deleted purely on a count cap. Only dirs
    whose mtime predates ``now - max_age_days`` qualify. Dirs whose name ends
    with a *protected_run_ids* entry (live session members) are skipped. These
    have no lineage record, so there is nothing to prune from the index.
    """
    if not max_age_days:
        return []
    root = Path(root)
    protected = protected_run_ids or set()
    cutoff = (now - timedelta(days=max_age_days)).timestamp()
    targets: list[DeletionTarget] = []
    for name in skipped_no_record:
        if any(name.endswith(rid) for rid in protected):
            continue
        d = root / name
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            targets.append(
                DeletionTarget(
                    run_id=name,
                    run_dir=d,
                    created_at=mtime,
                    size_bytes=_dir_size(d),
                    reason=f"orphan dir older than {max_age_days}d",
                )
            )
    targets.sort(key=lambda t: t.created_at)
    return targets


def cleanup_at_startup(
    *,
    env: Mapping[str, str] | None = None,
    root: Path | str | None = None,
    now: datetime | None = None,
) -> CleanupResult | None:
    """Run one retention pass at process startup. Never raises.

    Returns the :class:`CleanupResult` when the policy is enabled and acted,
    ``None`` when disabled / no active limit / on any error (so a failure here
    can never block server startup).
    """
    try:
        import os
        from datetime import timezone

        from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT

        resolved_env = env if env is not None else os.environ
        policy = policy_from_env(resolved_env)
        if not policy.enabled or not policy.any_limit_active:
            return None

        base = Path(root) if root is not None else DEFAULT_RESULTS_ROOT
        index_path = base / "run_index.jsonl"
        # Local import avoids a module-load cycle (run_index imports artifacts).
        from p2s_agent.orchestration.run_index import load_run_index

        records = load_run_index(path=index_path)
        plan = plan_cleanup(
            base, policy, now=now or datetime.now(timezone.utc),
            index_records=records,
        )
        result = apply_cleanup(plan, index_path=index_path)
        if result.deleted or result.errors:
            # freed_bytes is only meaningful when a byte-cap forced a size walk;
            # the default keep-newest-N policy skips sizing for speed.
            freed = (
                f", freed {result.freed_bytes / 1e6:.1f} MB"
                if result.freed_bytes
                else ""
            )
            logger.info(
                "retention: deleted %d run(s)%s, %d error(s); %d kept",
                len(result.deleted),
                freed,
                len(result.errors),
                plan.kept_count,
            )
        return result
    except Exception:  # noqa: BLE001 — startup cleanup must never break boot
        logger.exception("retention: startup cleanup failed; continuing")
        return None

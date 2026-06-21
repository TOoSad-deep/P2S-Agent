"""CLI: prune old PNG-to-Shader run directories under the data root.

Default is a **dry-run** — it prints what would be deleted and frees nothing.
Pass ``--apply`` to actually delete.

    python -m p2s_agent.tools.cleanup_runs                 # dry-run, env policy
    python -m p2s_agent.tools.cleanup_runs --max-runs 500  # preview keep-newest-500
    python -m p2s_agent.tools.cleanup_runs --apply         # execute env policy
    python -m p2s_agent.tools.cleanup_runs --root /data/p2s --max-age-days 30 --apply
    python -m p2s_agent.tools.cleanup_runs --include-orphans --max-age-days 30 --apply

Retention policy comes from the ``P2S_RETENTION_*`` env vars (see
:mod:`p2s_agent.orchestration.retention`); CLI flags override per-invocation.
An explicit CLI run always acts even if ``P2S_RETENTION_ENABLED=false``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT
from p2s_agent.orchestration.retention import (
    DeletionTarget,
    apply_cleanup,
    orphan_targets,
    plan_cleanup,
    policy_from_env,
    session_referenced_run_ids,
)
from p2s_agent.orchestration.run_index import load_run_index


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cleanup_runs",
        description="Prune old PNG-to-Shader run directories (dry-run by default).",
    )
    p.add_argument("--root", default=None, help="data root (default: $P2S_RESULTS_ROOT or backend/test_results)")
    p.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    p.add_argument("--max-runs", type=int, default=None, help="keep newest N runs (0 disables)")
    p.add_argument("--max-age-days", type=int, default=None, help="delete runs older than D days (0 disables)")
    p.add_argument("--max-bytes", default=None, help="cap total size, e.g. 2GB / 500MB (0 disables)")
    p.add_argument(
        "--include-orphans",
        action="store_true",
        help="also delete record-less dirs older than --max-age-days",
    )
    return p


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def _fmt_date(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return "????-??-??"


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> int:
    env = env if env is not None else os.environ
    args = _build_parser().parse_args(argv)
    now = now or datetime.now(timezone.utc)

    root = Path(args.root).expanduser() if args.root else DEFAULT_RESULTS_ROOT

    # Env policy, then per-invocation CLI overrides. An explicit CLI run always
    # acts, so force enabled regardless of P2S_RETENTION_ENABLED.
    policy = policy_from_env(env)
    overrides: dict[str, object] = {"enabled": True}
    if args.max_runs is not None:
        overrides["max_runs"] = args.max_runs if args.max_runs > 0 else None
    if args.max_age_days is not None:
        overrides["max_age_days"] = args.max_age_days if args.max_age_days > 0 else None
    if args.max_bytes is not None:
        from p2s_agent.orchestration.retention import parse_size

        parsed = parse_size(args.max_bytes)
        overrides["max_bytes"] = parsed if (parsed or 0) > 0 else None
    policy = replace(policy, **overrides)

    index_path = root / "run_index.jsonl"
    records = load_run_index(path=index_path)
    plan = plan_cleanup(root, policy, now=now, index_records=records, compute_sizes=True)

    orphans: list[DeletionTarget] = []
    if args.include_orphans:
        orphans = orphan_targets(
            root,
            plan.skipped_no_record,
            now=now,
            max_age_days=policy.max_age_days,
            protected_run_ids=session_referenced_run_ids(root),
        )

    _print_report(root, policy, plan, orphans, apply=args.apply)

    if not args.apply:
        return 0

    result = apply_cleanup(plan, index_path=index_path)
    orphan_freed = 0
    orphan_deleted = 0
    for t in orphans:
        try:
            shutil.rmtree(t.run_dir)
        except OSError as exc:  # pragma: no cover - defensive
            print(f"  ! failed to delete orphan {t.run_dir.name}: {exc}", file=sys.stderr)
            continue
        orphan_freed += t.size_bytes or 0
        orphan_deleted += 1

    print(
        f"\nDeleted {len(result.deleted)} run(s)"
        + (f" + {orphan_deleted} orphan(s)" if orphans else "")
        + f", freed {_human(result.freed_bytes + orphan_freed)}."
    )
    if result.errors:
        print(f"  {len(result.errors)} deletion error(s); see warnings above.", file=sys.stderr)
    return 0


def _print_report(
    root: Path,
    policy,
    plan,
    orphans: list[DeletionTarget],
    *,
    apply: bool,
) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] retention cleanup on {root}")
    print(
        f"  policy: max_runs={policy.max_runs} max_age_days={policy.max_age_days} "
        f"max_bytes={policy.max_bytes}"
    )
    rows = list(plan.delete) + list(orphans)
    if not rows:
        print(f"  nothing to delete; {plan.kept_count} run(s) kept.")
        if plan.skipped_no_record and not orphans:
            print(f"  ({len(plan.skipped_no_record)} record-less dir(s) skipped; "
                  f"use --include-orphans --max-age-days N to prune them)")
        return
    print(f"  {len(rows)} dir(s) to delete (oldest first):")
    for t in rows:
        size = _human(t.size_bytes) if t.size_bytes is not None else "?"
        print(f"    {_fmt_date(t.created_at)}  {size:>9}  {t.run_dir.name}  [{t.reason}]")
    total = sum((t.size_bytes or 0) for t in rows)
    print(f"  total to free: {_human(total)};  {plan.kept_count} run(s) kept;  "
          f"{len(plan.protected)} protected by lineage.")
    if not apply:
        print("  (dry-run — pass --apply to delete)")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

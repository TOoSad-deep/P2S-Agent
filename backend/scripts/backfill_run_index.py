"""One-time: backfill the real run_index.jsonl into the SQLite runs table.

    cd backend && python3 scripts/backfill_run_index.py

Idempotent (upsert by run_id); safe to re-run. Reads the default JSONL and
writes the default DB (backend/data/p2s.db).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2s_agent.orchestration.run_index import (  # noqa: E402
    backfill_runs_to_db,
    reconcile_runs_with_db,
)


def main() -> None:
    n = backfill_runs_to_db()  # default JSONL → default DB
    mism = reconcile_runs_with_db()
    print(f"[backfill] upserted {n} runs; reconcile mismatches: {len(mism)}")
    if mism:
        print("  mismatched run_ids:", mism[:20])


if __name__ == "__main__":
    main()

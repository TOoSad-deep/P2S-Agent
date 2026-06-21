# SQLite 数据层 · 计划 03d：run_index 读切换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.

**Goal:** 让 `load_run_index` 从 SQLite `runs` 表读，而不再折叠 JSONL。分两步小走，**先回填达 parity，再翻读**。

**Architecture:** 影子双写已让新 run 实时进 DB；读切换前需把**历史 JSONL** 回填进 DB（否则翻读会丢历史）。Step 1（本计划）：`backfill_runs_to_db` + `reconcile_runs_with_db` + 脚本，纯加法、零读路径改动。Step 2（后续）：翻转 `load_run_index` 读 DB、在影子写入处强制 terminal-stickiness、丢弃 fold 缓存、把折叠语义测试重定向到 `_fold_index_file`、JSONL 写入与 compact/prune 作为次要 artifact 保留。

**Tech Stack:** Python 3.9 · SQLAlchemy 2.0 · pytest

---

## Step 1（本计划）: 回填 + 对账

### Task 1: backfill_runs_to_db + reconcile_runs_with_db

**Files:** Modify `p2s_agent/orchestration/run_index.py`; Test `backend/tests/unit/test_run_index_backfill.py`

- [ ] **Step 1: 失败测试**
```python
# backend/tests/unit/test_run_index_backfill.py
from p2s_agent.orchestration.run_index import (
    RunLineageRecord, append_run_created, backfill_runs_to_db, reconcile_runs_with_db)


def _rec(run_id, status="pending"):
    return RunLineageRecord(run_id=run_id, root_run_id=run_id, parent_run_id=None,
        source_checkpoint_id=None, source_checkpoint_label=None, mode=None, feedback=None,
        title=None, status=status, run_dir=None, created_at=1.0)


def test_backfill_populates_db_from_jsonl(tmp_path, monkeypatch):
    import p2s_agent.orchestration.run_index as ri
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)  # simulate pre-shadow history
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("h1"), path=idx)
    append_run_created(_rec("h2"), path=idx)
    from p2s_agent.core.db.engine import get_engine, init_db
    from p2s_agent.core.db.repositories import runs as runs_repo
    init_db(get_engine(tmp_path))
    assert runs_repo.get_run(get_engine(tmp_path), "h1") is None  # not mirrored (shadow off)
    assert backfill_runs_to_db(path=idx) == 2
    assert runs_repo.get_run(get_engine(tmp_path), "h1")["run_id"] == "h1"
    assert reconcile_runs_with_db(path=idx) == []  # parity


def test_backfill_idempotent(tmp_path, monkeypatch):
    import p2s_agent.orchestration.run_index as ri
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("h1"), path=idx)
    backfill_runs_to_db(path=idx)
    assert backfill_runs_to_db(path=idx) == 1   # re-run: still 1, no dup
    assert reconcile_runs_with_db(path=idx) == []


def test_backfill_absent_jsonl_noop(tmp_path):
    assert backfill_runs_to_db(path=tmp_path / "nope.jsonl") == 0
```

- [ ] **Step 2: 运行确认失败** → FAIL (ImportError: backfill_runs_to_db)

- [ ] **Step 3: 实现** — 在 `run_index.py` 末尾加：
```python
# ---------------------------------------------------------------------------
# Read-cutover step 1: backfill the runs table from the JSONL + reconcile.
# Additive — does NOT change any read path. Idempotent (upsert by run_id).
# ---------------------------------------------------------------------------


def backfill_runs_to_db(*, path: Path | str | None = None, engine=None) -> int:
    """Fold the JSONL and upsert every record into the runs table. Idempotent.
    Returns the number of records upserted; 0 if the JSONL is absent."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        return 0
    records = _fold_index_file(resolved)
    eng = engine if engine is not None else _shadow_engine(path)
    from p2s_agent.core.db.repositories import runs as runs_repo
    for rec in records.values():
        runs_repo.upsert_run(eng, asdict(rec))
    return len(records)


def reconcile_runs_with_db(*, path: Path | str | None = None, engine=None) -> list[str]:
    """Return run_ids that differ between the JSONL fold and the DB (empty = parity)."""
    resolved = _resolve_path(path)
    folded = _fold_index_file(resolved) if resolved.exists() else {}
    eng = engine if engine is not None else _shadow_engine(path)
    from p2s_agent.core.db.repositories import runs as runs_repo
    db_rows = runs_repo.get_all_runs(eng)
    mismatches: list[str] = []
    for rid, rec in folded.items():
        db = db_rows.get(rid)
        if db is None or _dict_to_record(db) != rec:
            mismatches.append(rid)
    mismatches.extend(rid for rid in db_rows if rid not in folded)
    return mismatches
```

- [ ] **Step 4: 运行确认通过** → PASS (3)
- [ ] **Step 5: 提交** `git commit -m "feat(db): add run_index backfill + reconcile (read-cutover step 1)"`

### Task 2: 回填脚本 + 门禁

- [ ] **Step 1:** 写 `backend/scripts/backfill_run_index.py`：
```python
"""One-time: backfill the real run_index.jsonl into the SQLite runs table.

    cd backend && python3 scripts/backfill_run_index.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2s_agent.orchestration.run_index import backfill_runs_to_db, reconcile_runs_with_db  # noqa: E402


def main() -> None:
    n = backfill_runs_to_db()  # default JSONL → default DB
    mism = reconcile_runs_with_db()
    print(f"[backfill] upserted {n} runs; reconcile mismatches: {len(mism)}")
    if mism:
        print("  mismatched run_ids:", mism[:20])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** 后端全量 `cd backend && python3 -m pytest tests/unit/ -q` → 0 失败（仅加法）
- [ ] **Step 3:** `cd frontend && npm run build` → 成功
- [ ] **Step 4: 提交** `git commit -m "feat(db): add backfill_run_index script"`

---

## Step 2（后续独立计划，本计划不做）: 翻转 load_run_index 读 DB
- `load_run_index` → `runs_repo.get_all_runs` → `_dict_to_record`；DB 不可用时回退折叠 JSONL。
- 影子 `_shadow_update` 强制 terminal-stickiness（DB 读需匹配 fold 契约）。
- fold 语义测试重定向到 `_fold_index_file`；移除/改写 fold-cache 测试。
- compact/prune/JSONL 写保留为次要 artifact（或后续退役）。
- 高风险、大测试面 —— 单独慎重计划，并发更稳时做。

## Self-Review
- Step 1 纯加法、零读路径改动，由 3 个新测试覆盖回填/幂等/缺文件，对账验证 parity。
- 范围外：翻读、stickiness、测试重写（Step 2）。

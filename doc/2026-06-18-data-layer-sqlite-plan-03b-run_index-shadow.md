# SQLite 数据层 · 计划 03b：run_index 影子双写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. 加法式接线，**JSONL 行为/读路径/既有测试全部不变**；新增独立测试验证影子库被写入。

**Goal:** 让 `p2s_agent/orchestration/run_index.py` 在写 JSONL 的同时 best-effort 镜像写入 SQLite `runs` 表，零语义风险地把 DB 接进实时写路径。读仍走 JSONL（DB 暂非权威）。

**Architecture:** 在 `append_run_created`/`append_run_updated` 末尾各加一行影子写。影子库放在 JSONL 文件的**父目录** `p2s.db`（避免同一路径既当 JSONL 又当 sqlite）。影子首次使用某引擎时惰性 `init_db`（幂等）。全程 `try/except`，DEBUG 日志，**永不抛错**。

**Tech Stack:** Python 3.9 · SQLAlchemy 2.0（`p2s_agent.core.db`）· pytest

**已知影子期偏差（留待将来读切换）:** ① update-before-create 在 DB 端被丢弃（无行可更新）；② terminal-status stickiness 不在 DB 端强制；③ compact/prune 不同步到 DB。读切换前 DB 非权威，故均可接受 —— 写进代码注释。

---

## File Structure
- 改：`backend/p2s_agent/orchestration/run_index.py`（加影子 helper + 两处 append 末尾各 1 行；既有逻辑不动）
- 新：`backend/tests/unit/test_run_index_shadow.py`

---

## Task 1: 影子双写实现 + 新测试

**Files:**
- Modify: `backend/p2s_agent/orchestration/run_index.py`
- Test: `backend/tests/unit/test_run_index_shadow.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/test_run_index_shadow.py`:
```python
"""Shadow dual-write: run_index also mirrors into SQLite (additive).

JSONL stays authoritative; these tests assert the shadow runs table is
populated and that a shadow failure never breaks the JSONL path.
"""
from p2s_agent.orchestration.run_index import (
    RunLineageRecord,
    append_run_created,
    append_run_updated,
    load_run_index,
)


def _rec(run_id="r1", status="pending"):
    return RunLineageRecord(
        run_id=run_id, root_run_id=run_id, parent_run_id=None,
        source_checkpoint_id=None, source_checkpoint_label=None, mode=None,
        feedback=None, title=None, status=status, run_dir=None, created_at=1.0,
        tags=["a"], source_run_ids=["s1"],
    )


def test_shadow_created_populates_runs_table(tmp_path):
    from p2s_agent.core.db.engine import get_engine
    from p2s_agent.core.db.repositories import runs as runs_repo
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r1"), path=idx)
    # JSONL still authoritative
    assert idx.exists()
    assert load_run_index(path=idx)["r1"].run_id == "r1"
    # shadow DB (beside the JSONL) populated
    row = runs_repo.get_run(get_engine(tmp_path), "r1")
    assert row is not None and row["run_id"] == "r1"
    assert row["tags"] == ["a"] and row["source_run_ids"] == ["s1"]


def test_shadow_update_mirrors_fields(tmp_path):
    from p2s_agent.core.db.engine import get_engine
    from p2s_agent.core.db.repositories import runs as runs_repo
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r2", status="pending"), path=idx)
    append_run_updated("r2", {"status": "completed", "final_score": 0.9}, path=idx)
    row = runs_repo.get_run(get_engine(tmp_path), "r2")
    assert row["status"] == "completed" and row["final_score"] == 0.9


def test_shadow_failure_never_breaks_jsonl(tmp_path, monkeypatch):
    # Force the shadow engine to raise; JSONL write+read must still work.
    import p2s_agent.orchestration.run_index as ri
    monkeypatch.setattr(ri, "_shadow_engine",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r3"), path=idx)
    assert idx.exists()
    assert load_run_index(path=idx)["r3"].run_id == "r3"  # JSONL unaffected


def test_shadow_disabled_flag_skips_db(tmp_path, monkeypatch):
    import p2s_agent.orchestration.run_index as ri
    from p2s_agent.core.db.engine import get_engine, init_db
    from p2s_agent.core.db.repositories import runs as runs_repo
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)
    init_db(get_engine(tmp_path))  # table exists, but shadow disabled
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r4"), path=idx)
    assert runs_repo.get_run(get_engine(tmp_path), "r4") is None  # not mirrored
    assert idx.exists()  # JSONL still written
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/test_run_index_shadow.py -q`
Expected: FAIL（`AttributeError: _shadow_engine` / 影子未实现）

- [ ] **Step 3: 实现影子双写**

在 `run_index.py` 的常量区（`_TERMINAL_STATUSES` 之后）插入：
```python
# ---------------------------------------------------------------------------
# Shadow SQLite mirror (additive; JSONL remains authoritative).
# Best-effort: a DB failure NEVER breaks the JSONL write path or reads. Reads
# still fold JSONL; this only populates the `runs` table so the eventual
# read-cutover has live data. Known shadow-mode divergences (resolved at
# read-cutover): update-before-create is dropped (no row to update),
# terminal-status stickiness is not enforced DB-side, and compact/prune are
# not mirrored.
# ---------------------------------------------------------------------------
_SHADOW_DB_ENABLED = True
_SHADOW_INIT_LOCK = threading.Lock()
_SHADOW_INITED: set[str] = set()


def _shadow_engine(path: Path | str | None):
    """Engine for the shadow DB sitting BESIDE the JSONL (never the same file).

    path=None → canonical app DB (backend/data/p2s.db);
    path=<jsonl file> → <its parent dir>/p2s.db (per-test isolation).
    Lazily ensures the schema exists (idempotent).
    """
    from p2s_agent.core.db.engine import get_engine, init_db
    eng = get_engine(None) if path is None else get_engine(Path(path).parent)
    key = str(eng.url)
    if key not in _SHADOW_INITED:
        with _SHADOW_INIT_LOCK:
            if key not in _SHADOW_INITED:
                init_db(eng)  # create_all: only adds missing tables
                _SHADOW_INITED.add(key)
    return eng


def _shadow_upsert_created(record: "RunLineageRecord", path: Path | str | None) -> None:
    if not _SHADOW_DB_ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import runs as runs_repo
        runs_repo.upsert_run(_shadow_engine(path), asdict(record))
    except Exception:
        logger.debug("run_index shadow upsert (created) failed", exc_info=True)


def _shadow_update(run_id: str, fields: dict[str, Any], path: Path | str | None) -> None:
    if not _SHADOW_DB_ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import runs as runs_repo
        runs_repo.update_run(_shadow_engine(path), run_id, fields)
    except Exception:
        logger.debug("run_index shadow update failed", exc_info=True)
```

在 `append_run_created` 末尾（`_append_line(data, resolved)` 之后）加：
```python
    _shadow_upsert_created(record, path)
```

在 `append_run_updated` 末尾（`_append_line(data, resolved)` 之后）加：
```python
    _shadow_update(run_id, fields, path)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/test_run_index_shadow.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add backend/p2s_agent/orchestration/run_index.py backend/tests/unit/test_run_index_shadow.py
git commit -m "feat(db): shadow-mirror run_index writes into SQLite (additive)"
```

---

## Task 2: 全门禁回归（核心：既有 run_index 测试零变化仍全绿）

- [ ] **Step 1: 既有 run_index 测试不受影响**

Run: `cd backend && python3 -m pytest tests/unit/test_run_index.py tests/unit/test_run_index_compaction_wiring.py tests/unit/test_retention.py tests/unit/test_cleanup_cli.py -q`
Expected: 全绿（影子是 best-effort 加法，JSONL 行为不变）。若有因影子副作用（tmp 下多出 p2s.db / 日志噪声）导致的失败 → 复核影子的 best-effort 包裹与 DEBUG 日志级别，修至绿。

- [ ] **Step 2: 后端全量 + 边界测试**

Run: `cd backend && python3 -m pytest tests/unit/ -q`
Expected: 0 失败（既有 + 4 个影子新测试；边界测试绿 —— run_index 只 import 了 `p2s_agent.core.db`，未碰 app.*）。

- [ ] **Step 3: 前端 build**

Run: `cd frontend && npm run build`
Expected: 成功。

---

## Self-Review
- **覆盖:** 影子写 created/updated、JSONL 不变、失败不破坏、禁用开关 —— 4 个新测试覆盖。
- **零回归保证:** 不改 load/compact/prune/build_branch_tree/update_run_metadata 的既有逻辑；只在两个 append 末尾各加 1 行 best-effort 调用。
- **范围外:** 读切换（load 改读 DB + 重写 format-coupled 测试）、compact/prune 同步 DB、其余 4 个 orchestration 模块 —— 均后续计划。

# SQLite 数据层 · 计划 01：数据库地基 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `backend/data/p2s.db` 立起一个可查询的 SQLite 库，含 spec §2 的全部 7 张表、索引与约束，配 engine（WAL/PRAGMA）与 Alembic 基线，全部由单测覆盖。

**Architecture:** 新增 `app/db/` 包：`schema.py`（SQLAlchemy 2.0 Core，`MetaData`/`Table` 为 schema 唯一真相源）+ `engine.py`（Engine 工厂、PRAGMA、`get_engine(override)` 目录覆盖）。Alembic 以 `schema.metadata` 为 `target_metadata` 生成基线。本计划**不**改动任何现有持久化模块、路由或前端 —— 只新增地基。

**Tech Stack:** Python 3.9（系统解释器，无 venv）· SQLAlchemy 2.0 · Alembic · stdlib `sqlite3` · pytest

参考 spec：[doc/2026-06-18-data-layer-sqlite-design.md](2026-06-18-data-layer-sqlite-design.md)（§1 边界、§2 DDL、§3 engine/并发）。

---

## File Structure（本计划新建/修改）

| 文件 | 职责 |
|---|---|
| `backend/requirements.txt`（改） | 加 `SQLAlchemy>=2.0,<2.1`、`alembic>=1.13` |
| `backend/app/db/__init__.py`（新） | 包标记 |
| `backend/app/db/schema.py`（新） | `MetaData` + 7 张 `Table` + 索引 + 约束（schema 真相源） |
| `backend/app/db/engine.py`（新） | Engine 工厂、PRAGMA、`get_engine`、`init_db`、`DEFAULT_DB_PATH` |
| `backend/data/.gitignore`（新） | 忽略 `p2s.db*`（库文件不入库） |
| `backend/alembic.ini`（新） | Alembic 配置 |
| `backend/alembic/env.py`（新） | `target_metadata = schema.metadata` |
| `backend/alembic/versions/<rev>_baseline.py`（新） | 基线迁移（autogenerate 后 review） |
| `backend/tests/unit/db/__init__.py`（新） | 测试包标记 |
| `backend/tests/unit/db/test_schema.py`（新） | 建表/列/索引/约束行为 |
| `backend/tests/unit/db/test_engine.py`（新） | PRAGMA、`get_engine` 覆盖、缓存 |
| `backend/tests/unit/db/test_alembic_baseline.py`（新） | `upgrade head` 后表集合 == metadata |

约定：测试自 `backend/` 运行（`python3 -m pytest tests/unit/ -v`），`app` 与 `tests` 可直接 import（沿用现有约定）。

---

## Task 1: 依赖与 db 包骨架

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/app/db/__init__.py`
- Create: `backend/data/.gitignore`
- Test: `backend/tests/unit/db/__init__.py`, `backend/tests/unit/db/test_imports.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_imports.py`:
```python
def test_sqlalchemy_and_db_package_import():
    import sqlalchemy
    assert sqlalchemy.__version__.startswith("2.")
    import app.db  # noqa: F401
```
同时创建空文件 `backend/tests/unit/db/__init__.py`。

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_imports.py -v`
Expected: FAIL（`ModuleNotFoundError: app.db` 或 sqlalchemy 缺失）

- [ ] **Step 3: 加依赖并安装**

在 `backend/requirements.txt` 末尾追加：
```
SQLAlchemy>=2.0,<2.1
alembic>=1.13
```
安装：`cd backend && python3 -m pip install "SQLAlchemy>=2.0,<2.1" "alembic>=1.13"`

- [ ] **Step 4: 建包与 .gitignore**

创建 `backend/app/db/__init__.py`（空文件，含一行模块 docstring）：
```python
"""SQLite-backed data layer for P2S-Agent (schema + engine)."""
```
创建 `backend/data/.gitignore`：
```
# 本地数据库文件不入库（含 WAL/shm 边车文件）
p2s.db
p2s.db-wal
p2s.db-shm
```

- [ ] **Step 5: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_imports.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add backend/requirements.txt backend/app/db/__init__.py backend/data/.gitignore backend/tests/unit/db/__init__.py backend/tests/unit/db/test_imports.py
git commit -m "feat(db): add SQLAlchemy/Alembic deps and app.db package skeleton"
```

---

## Task 2: schema.py — 7 张表 + 索引

**Files:**
- Create: `backend/app/db/schema.py`
- Test: `backend/tests/unit/db/test_schema.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_schema.py`:
```python
from sqlalchemy import create_engine, inspect


def _fresh_engine():
    # 纯内存库，仅验证 DDL 结构（不涉及 PRAGMA/FK 行为）
    eng = create_engine("sqlite://")
    from app.db.schema import metadata
    metadata.create_all(eng)
    return eng


def test_all_seven_tables_created():
    insp = inspect(_fresh_engine())
    expected = {
        "runs", "variant_groups", "draw_sessions",
        "fusion_plans", "fusion_regions", "preference_profile", "events",
    }
    assert expected.issubset(set(insp.get_table_names()))


def test_runs_has_key_columns():
    insp = inspect(_fresh_engine())
    cols = {c["name"] for c in insp.get_columns("runs")}
    for required in (
        "run_id", "root_run_id", "parent_run_id", "status", "final_score",
        "favorite", "tags", "variant_group_id", "draw_session_id",
        "fusion_id", "source_run_ids", "run_dir", "created_at",
    ):
        assert required in cols


def test_runs_indexes_present():
    insp = inspect(_fresh_engine())
    names = {ix["name"] for ix in insp.get_indexes("runs")}
    for required in (
        "idx_runs_root", "idx_runs_status", "idx_runs_score",
        "idx_runs_fav", "idx_runs_vgroup", "idx_runs_draw", "idx_runs_fusion",
    ):
        assert required in names


def test_events_table_shape():
    insp = inspect(_fresh_engine())
    cols = {c["name"] for c in insp.get_columns("events")}
    assert {"event_id", "entity_type", "entity_id", "event_type", "payload", "ts"}.issubset(cols)
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_schema.py -v`
Expected: FAIL（`ModuleNotFoundError: app.db.schema`）

- [ ] **Step 3: 写 schema.py**

`backend/app/db/schema.py`:
```python
"""SQLAlchemy Core schema — single source of truth for the P2S-Agent DB.

Portable on purpose: only generic types (Text/Integer/Float/Boolean/JSON),
a partial index, a CHECK, and an ON DELETE CASCADE FK — all of which map
cleanly onto PostgreSQL/openGauss if we migrate later.
"""
from __future__ import annotations

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, Column, Float, ForeignKey, Index,
    Integer, MetaData, Table, Text, text,
)

metadata = MetaData()

# 1) runs — 血缘 + 元数据主表（取代 run_index.jsonl 折叠）
runs = Table(
    "runs", metadata,
    Column("run_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text),
    Column("source_checkpoint_id", Text),
    Column("source_checkpoint_label", Text),
    Column("mode", Text),
    Column("feedback", Text),
    Column("title", Text),
    Column("status", Text, nullable=False, server_default="unknown"),
    Column("run_dir", Text),
    Column("created_at", Float, nullable=False),
    Column("completed_at", Float),
    Column("final_score", Float),
    Column("favorite", Boolean, nullable=False, server_default="0"),
    Column("tags", JSON, nullable=False, server_default=text("'[]'")),
    Column("variant_group_id", Text),
    Column("variant_index", Integer),
    Column("variant_label", Text),
    Column("draw_session_id", Text),
    Column("draw_card_index", Integer),
    Column("replacement_of_run_id", Text),
    Column("fusion_id", Text),
    Column("base_run_id", Text),
    Column("source_run_ids", JSON, nullable=False, server_default=text("'[]'")),
)
Index("idx_runs_root", runs.c.root_run_id)
Index("idx_runs_parent", runs.c.parent_run_id)
Index("idx_runs_status", runs.c.status)
Index("idx_runs_score", runs.c.final_score)
Index("idx_runs_created", runs.c.created_at)
Index("idx_runs_fav", runs.c.favorite, sqlite_where=runs.c.favorite == True)  # noqa: E712 部分索引
Index("idx_runs_vgroup", runs.c.variant_group_id)
Index("idx_runs_draw", runs.c.draw_session_id)
Index("idx_runs_fusion", runs.c.fusion_id)

# 2) variant_groups
variant_groups = Table(
    "variant_groups", metadata,
    Column("group_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text, nullable=False),
    Column("source_checkpoint_id", Text, nullable=False),
    Column("feedback", Text, nullable=False, server_default=""),
    Column("mode", Text, nullable=False, server_default=""),
    Column("variant_count", Integer, nullable=False, server_default="0"),
    Column("diversity", Text, nullable=False, server_default="medium"),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("child_run_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("winner_run_id", Text),
    Column("created_at", Float, nullable=False, server_default="0"),
    Column("completed_at", Float),
    Column("draw_session_id", Text),
)
Index("idx_vg_root", variant_groups.c.root_run_id)
Index("idx_vg_draw", variant_groups.c.draw_session_id)

# 3) draw_sessions
draw_sessions = Table(
    "draw_sessions", metadata,
    Column("draw_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text, nullable=False),
    Column("source_checkpoint_id", Text, nullable=False),
    Column("feedback", Text, nullable=False, server_default=""),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("requested_count", Integer, nullable=False, server_default="0"),
    Column("diversity", Text, nullable=False, server_default="medium"),
    Column("mode", Text, nullable=False, server_default="batch_draw"),
    Column("group_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("card_run_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("winner_run_id", Text),
    Column("created_at", Float, nullable=False, server_default="0"),
    Column("updated_at", Float),
    Column("completed_at", Float),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
)
Index("idx_draw_root", draw_sessions.c.root_run_id)

# 4) fusion_plans
fusion_plans = Table(
    "fusion_plans", metadata,
    Column("fusion_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text, nullable=False),
    Column("base_run_id", Text, nullable=False, server_default=""),
    Column("source_run_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("draw_session_id", Text),
    Column("feedback", Text, nullable=False, server_default=""),
    Column("status", Text, nullable=False, server_default="draft"),
    Column("composite_target_artifact_id", Text),
    Column("output_run_id", Text),
    Column("created_at", Float, nullable=False, server_default="0"),
    Column("updated_at", Float),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
)
Index("idx_fusion_root", fusion_plans.c.root_run_id)

# 5) fusion_regions（子表；删 plan 级联清理）
fusion_regions = Table(
    "fusion_regions", metadata,
    Column("fusion_id", Text,
           ForeignKey("fusion_plans.fusion_id", ondelete="CASCADE"),
           nullable=False),
    Column("region_id", Text, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("label", Text, nullable=False, server_default=""),
    Column("source_run_id", Text, nullable=False, server_default=""),
    Column("instruction", Text, nullable=False, server_default=""),
    Column("geometry_type", Text, nullable=False, server_default="rect"),
    Column("geometry", JSON, nullable=False, server_default=text("'{}'")),
    Column("strength", Float, nullable=False, server_default="0.5"),
    Column("blend_mode", Text, nullable=False, server_default="soft"),
    Column("feather", Float, nullable=False, server_default="0.08"),
    CheckConstraint("fusion_id IS NOT NULL", name="ck_region_has_plan"),
)
# 复合主键
from sqlalchemy import PrimaryKeyConstraint  # noqa: E402
fusion_regions.append_constraint(
    PrimaryKeyConstraint("fusion_id", "region_id", name="pk_fusion_regions")
)

# 6) preference_profile（单行单例）
preference_profile = Table(
    "preference_profile", metadata,
    Column("id", Integer, primary_key=True),
    Column("schema_version", Integer, nullable=False, server_default="1"),
    Column("updated_at", Float, nullable=False, server_default="0"),
    Column("enabled", Boolean, nullable=False, server_default="1"),
    Column("default_locks", JSON, nullable=False, server_default=text("'{}'")),
    Column("positive_preferences", JSON, nullable=False, server_default=text("'[]'")),
    Column("negative_preferences", JSON, nullable=False, server_default=text("'[]'")),
    Column("preferred_variant_labels", JSON, nullable=False, server_default=text("'[]'")),
    Column("score_drop_tolerance_hint", Float, nullable=False, server_default="0.02"),
    Column("summary_source_event_count", Integer, nullable=False, server_default="0"),
    CheckConstraint("id = 1", name="ck_pref_singleton"),
)

# 7) events（通用事件流；取代所有 *_events.jsonl + preference events.jsonl）
events = Table(
    "events", metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("entity_type", Text, nullable=False),
    Column("entity_id", Text),
    Column("event_type", Text, nullable=False),
    Column("payload", JSON, nullable=False, server_default=text("'{}'")),
    Column("ts", Float, nullable=False),
)
Index("idx_events_entity", events.c.entity_type, events.c.entity_id, events.c.event_id)
Index("idx_events_ts", events.c.entity_type, events.c.ts)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_schema.py -v`
Expected: PASS（4 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/schema.py backend/tests/unit/db/test_schema.py
git commit -m "feat(db): define 7-table SQLAlchemy Core schema with indexes"
```

---

## Task 3: engine.py — PRAGMA / get_engine / init_db

**Files:**
- Create: `backend/app/db/engine.py`
- Test: `backend/tests/unit/db/test_engine.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_engine.py`:
```python
from sqlalchemy import text


def test_pragmas_wal_and_fk_on(tmp_path):
    from app.db.engine import get_engine
    eng = get_engine(tmp_path)
    with eng.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_get_engine_dir_override_creates_db_file(tmp_path):
    from app.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    assert (tmp_path / "p2s.db").exists()


def test_get_engine_is_cached_per_dir(tmp_path):
    from app.db.engine import get_engine
    assert get_engine(tmp_path) is get_engine(tmp_path)


def test_init_db_creates_all_tables(tmp_path):
    from sqlalchemy import inspect
    from app.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    tables = set(inspect(eng).get_table_names())
    assert {"runs", "events", "fusion_regions"}.issubset(tables)
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_engine.py -v`
Expected: FAIL（`ModuleNotFoundError: app.db.engine`）

- [ ] **Step 3: 写 engine.py**

`backend/app/db/engine.py`:
```python
"""SQLite engine factory: WAL + PRAGMAs + directory-keyed get_engine override.

`get_engine(override)`:
  - None  → 默认库 backend/data/p2s.db
  - 目录  → <目录>/p2s.db（让旧 `root=tmp_path` 风格的测试继续隔离）
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from app.db.schema import metadata

# backend/ 根：engine.py 在 backend/app/db/ → parents[2] == backend/
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "p2s.db"

_engines: dict[str, Engine] = {}


def _resolve_db_url(override: "Path | str | None") -> str:
    if override is None:
        path = DEFAULT_DB_PATH
    else:
        p = Path(override)
        # 目录 → 目录下 p2s.db；显式 .db 文件 → 原样
        path = (p / "p2s.db") if (p.is_dir() or p.suffix == "") else p
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _make_engine(url: str) -> Engine:
    engine = create_engine(url, connect_args={"check_same_thread": False}, future=True)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    return engine


def get_engine(override: "Path | str | None" = None) -> Engine:
    url = _resolve_db_url(override)
    if url not in _engines:
        _engines[url] = _make_engine(url)
    return _engines[url]


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables (idempotent). For the real DB prefer Alembic; this is
    the fast path for tests and first-run bootstrap."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    return engine
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_engine.py -v`
Expected: PASS（4 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/engine.py backend/tests/unit/db/test_engine.py
git commit -m "feat(db): add engine factory with WAL PRAGMAs and get_engine override"
```

---

## Task 4: 约束行为测试（FK 级联 / 单例 CHECK / 默认值）

**Files:**
- Test: `backend/tests/unit/db/test_constraints.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_constraints.py`:
```python
import pytest
from sqlalchemy import insert, select, delete
from sqlalchemy.exc import IntegrityError


def _eng(tmp_path):
    from app.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    return eng


def test_fusion_region_cascade_delete(tmp_path):
    from app.db.schema import fusion_plans, fusion_regions
    eng = _eng(tmp_path)
    with eng.begin() as conn:
        conn.execute(insert(fusion_plans).values(
            fusion_id="f1", root_run_id="r", parent_run_id="p", created_at=1.0))
        conn.execute(insert(fusion_regions).values(
            fusion_id="f1", region_id="reg1", ordinal=0))
    with eng.begin() as conn:
        conn.execute(delete(fusion_plans).where(fusion_plans.c.fusion_id == "f1"))
    with eng.connect() as conn:
        remaining = conn.execute(select(fusion_regions)).fetchall()
    assert remaining == []  # 级联清理


def test_preference_profile_singleton(tmp_path):
    from app.db.schema import preference_profile
    eng = _eng(tmp_path)
    with eng.begin() as conn:
        conn.execute(insert(preference_profile).values(id=1))
    with pytest.raises(IntegrityError):  # id=2 触发 CHECK(id=1)
        with eng.begin() as conn:
            conn.execute(insert(preference_profile).values(id=2))


def test_runs_defaults_applied(tmp_path):
    from app.db.schema import runs
    eng = _eng(tmp_path)
    with eng.begin() as conn:
        conn.execute(insert(runs).values(
            run_id="x", root_run_id="x", created_at=1.0))
    with eng.connect() as conn:
        row = conn.execute(select(runs).where(runs.c.run_id == "x")).mappings().one()
    assert row["status"] == "unknown"
    assert row["favorite"] in (0, False)
    assert row["tags"] == []          # JSON server_default '[]'
    assert row["source_run_ids"] == []
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_constraints.py -v`
Expected: FAIL（表/约束行为未就绪，或 schema 需微调）

- [ ] **Step 3: 让测试通过**

预期 schema 已支持。若 `test_runs_defaults_applied` 因 JSON server_default 读回为字符串 `'[]'` 而非 `[]`，将该列 server_default 改为 Python 端默认：把 `tags`/`source_run_ids` 等 JSON 列的 `server_default=text("'[]'")` 替换为 `default=list`（`metadata` 重新建表后生效），并保留 `nullable=False`。对应 `'{}'` 列用 `default=dict`。重跑测试至通过。

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_constraints.py -v`
Expected: PASS（3 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/schema.py backend/tests/unit/db/test_constraints.py
git commit -m "test(db): verify FK cascade, singleton CHECK, column defaults"
```

---

## Task 5: 生成真实库 backend/data/p2s.db

**Files:**
- Create: `backend/scripts/init_db.py`
- Test: 手动验证（生成的库文件被 .gitignore 忽略，不提交）

- [ ] **Step 1: 写 init 脚本**

`backend/scripts/init_db.py`:
```python
"""一次性引导：在 backend/data/p2s.db 建好全部表。

    python3 scripts/init_db.py

幂等；真实 schema 演进请走 Alembic。
"""
from app.db.engine import DEFAULT_DB_PATH, get_engine, init_db


def main() -> None:
    eng = get_engine()  # 默认 backend/data/p2s.db
    init_db(eng)
    print(f"[init_db] tables created at {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行脚本**

Run: `cd backend && python3 scripts/init_db.py`
Expected: 打印 `tables created at .../backend/data/p2s.db`

- [ ] **Step 3: 验证库结构**

Run: `cd backend && python3 -c "import sqlite3; c=sqlite3.connect('data/p2s.db'); print(sorted(r[0] for r in c.execute(\"select name from sqlite_master where type='table'\")))"`
Expected: 含 `['draw_sessions', 'events', 'fusion_plans', 'fusion_regions', 'preference_profile', 'runs', 'variant_groups']`（可能另含 alembic 元表，本步尚无）

- [ ] **Step 4: 确认库文件未被 git 跟踪**

Run: `cd backend && git status --porcelain data/`
Expected: 无 `p2s.db` 输出（已被 `backend/data/.gitignore` 忽略）

- [ ] **Step 5: 提交脚本**

```bash
git add backend/scripts/init_db.py
git commit -m "feat(db): add init_db bootstrap script"
```

---

## Task 6: Alembic 基线

**Files:**
- Create: `backend/alembic.ini`, `backend/alembic/env.py`, `backend/alembic/script.py.mako`, `backend/alembic/versions/<rev>_baseline.py`
- Test: `backend/tests/unit/db/test_alembic_baseline.py`

- [ ] **Step 1: 初始化 Alembic**

Run: `cd backend && python3 -m alembic init alembic`
Expected: 生成 `alembic.ini` 与 `alembic/`。

- [ ] **Step 2: 配置 url 与 target_metadata**

`backend/alembic.ini` 中设：
```
sqlalchemy.url = sqlite:///data/p2s.db
```
`backend/alembic/env.py` 顶部加入并替换 target_metadata：
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/
from app.db.schema import metadata as target_metadata  # noqa: E402
# 删除自动生成的 `target_metadata = None`，改用上面的 import
# 为让 SQLite 批处理迁移可用（未来 ALTER）：在 context.configure(...) 调用里加 render_as_batch=True
```
在 `run_migrations_online()` 的两处 `context.configure(...)` 中加 `render_as_batch=True`。

- [ ] **Step 3: 生成基线迁移**

先删除上一步手动建的真实库以避免“无变更”：`cd backend && rm -f data/p2s.db data/p2s.db-wal data/p2s.db-shm`
Run: `cd backend && python3 -m alembic revision --autogenerate -m "baseline schema"`
Expected: 在 `alembic/versions/` 生成一个迁移文件，`upgrade()` 内含 7 张 `op.create_table(...)` 与各索引。**人工 review**：确认表名/列/索引/FK CASCADE/CHECK 与 `schema.py` 一致；部分索引 `idx_runs_fav` 若 autogenerate 漏了 where 子句，手动补 `sqlite_where=sa.text("favorite = 1")`。

- [ ] **Step 4: 写失败测试**

`backend/tests/unit/db/test_alembic_baseline.py`:
```python
import subprocess
from pathlib import Path
from sqlalchemy import create_engine, inspect

BACKEND = Path(__file__).resolve().parents[3]  # backend/


def test_alembic_upgrade_head_builds_full_schema(tmp_path):
    db = tmp_path / "p2s.db"
    subprocess.run(
        ["python3", "-m", "alembic", "-x", f"db={db}", "upgrade", "head"],
        cwd=BACKEND, check=True,
        env={**__import__("os").environ, "ALEMBIC_DB_URL": f"sqlite:///{db}"},
    )
    tables = set(inspect(create_engine(f"sqlite:///{db}")).get_table_names())
    from app.db.schema import metadata
    expected = set(metadata.tables.keys())
    assert expected.issubset(tables)
```
为让测试能指定库，`env.py` 读取覆盖（在 `config.get_main_option("sqlalchemy.url")` 之后加）：
```python
import os
_override = os.environ.get("ALEMBIC_DB_URL")
if _override:
    config.set_main_option("sqlalchemy.url", _override)
```

- [ ] **Step 5: 运行测试**

Run: `cd backend && python3 -m pytest tests/unit/db/test_alembic_baseline.py -v`
Expected: PASS（`upgrade head` 后表集合覆盖 metadata 全部表）

- [ ] **Step 6: 重建真实库（走 Alembic）并提交**

Run: `cd backend && python3 -m alembic upgrade head`（在 `data/p2s.db` 重建，含 `alembic_version` 表）
```bash
git add backend/alembic.ini backend/alembic/ backend/tests/unit/db/test_alembic_baseline.py
git commit -m "feat(db): add Alembic baseline migration for full schema"
```

---

## Task 7: 全门禁回归

- [ ] **Step 1: 后端全量测试**

Run: `cd backend && python3 -m pytest tests/unit/ -q`
Expected: 全绿（新增 db 测试 + 既有测试不受影响；本计划未改任何现有模块）

- [ ] **Step 2: 前端构建门禁（确认未被波及）**

Run: `cd frontend && npm run build`
Expected: 构建成功（本计划不动前端）

- [ ] **Step 3: 收尾提交（如有未决变更）**

```bash
git status
# 若有遗留改动，按上文分类 add 并 commit
```

---

## Self-Review

**1. Spec coverage（对照 §2/§3）:**
- §2.3 全部 7 张表 → Task 2 ✅；索引 → Task 2 ✅；FK CASCADE / 单例 CHECK / 部分索引 / 默认值 → Task 4 ✅
- §3.2 engine + PRAGMA(WAL/synchronous/busy_timeout/foreign_keys) + `get_engine` 目录覆盖 → Task 3 ✅
- §4.1 依赖 SQLAlchemy/Alembic → Task 1 ✅；§4.2① Alembic 基线 → Task 6 ✅
- §5.4 门禁（pytest + npm build）→ Task 7 ✅
- **本计划范围外**（后续计划）：repositories、5 个模块内部改写、JSONL→SQLite 导入脚本与对账 —— 见 spec §8 步骤 2–5。

**2. Placeholder scan:** 无 TBD/TODO；每个代码步骤含完整代码；命令含预期输出。Task 6 的 autogenerate 是标准命令 + 人工 review 步骤，非占位。

**3. Type consistency:** `get_engine` / `init_db` / `_resolve_db_url` / `DEFAULT_DB_PATH` / `metadata` 在 Task 2/3 定义，Task 4/5/6 引用一致；表对象名（`runs`/`fusion_plans`/`fusion_regions`/`preference_profile`/`events`）全程一致。

**4. 已知风险点（执行时注意）:**
- JSON `server_default` 读回可能是字符串 → Task 4 Step 3 已给出改用 Python 端 `default=list/dict` 的修法。
- autogenerate 可能漏部分索引的 `where` → Task 6 Step 3 要求人工补全。
- Alembic `env.py` 的 sys.path 注入确保能 import `app.db.schema`（从 backend/ 运行）。

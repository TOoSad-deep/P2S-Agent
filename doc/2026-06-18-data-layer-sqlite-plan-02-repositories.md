# SQLite 数据层 · 计划 02：仓储层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 plan 01 的 schema/engine 之上建一层 **纯 dict 仓储层**（`app/db/repositories/`），为 7 张表提供 CRUD/upsert/事件流读写，全部单测覆盖。不改任何现有持久化模块（那是 plan 03）。

**Architecture:** 仓储层只处理 `dict ↔ 表行`，不依赖 `app/pipeline` 的 dataclass（避免 plan 03 接线时的循环 import）。一个泛型 `_base`（upsert/update/get/get_all）服务 runs / variant_groups / draw_sessions / fusion_plans / preference_profile；`events`（append-only）与 fusion 的 `regions` 子表（拆/并）为特例。所有仓储函数以 `engine` 为首参，便于 plan 03 用 `get_engine(path)` 接入。

**Tech Stack:** Python 3.9 · SQLAlchemy 2.0 Core（`sqlite_insert(...).on_conflict_do_update`）· pytest

参考：[plan 01](2026-06-18-data-layer-sqlite-plan-01-foundation.md) · [spec §3 仓储层](2026-06-18-data-layer-sqlite-design.md)

---

## File Structure（本计划新建）

| 文件 | 职责 |
|---|---|
| `backend/app/db/repositories/__init__.py` | 包标记 |
| `backend/app/db/repositories/_base.py` | 泛型 `upsert/update_by_pk/get_by_pk/get_all` |
| `backend/app/db/repositories/events.py` | 通用事件流 `append_event/load_events` |
| `backend/app/db/repositories/runs.py` | runs CRUD（薄封装 _base） |
| `backend/app/db/repositories/variant_groups.py` | variant_groups CRUD |
| `backend/app/db/repositories/draw_sessions.py` | draw_sessions CRUD |
| `backend/app/db/repositories/fusions.py` | fusion_plans + fusion_regions（拆/并） |
| `backend/app/db/repositories/preferences.py` | preference_profile 单例 get/save |
| `backend/tests/unit/db/conftest.py` | `repo_engine` fixture（每测试一个建好表的 tmp 库） |
| `backend/tests/unit/db/test_repo_*.py` | 各仓储单测 |

---

## Task 1: 泛型 _base + 测试 fixture

**Files:**
- Create: `backend/app/db/repositories/__init__.py`, `backend/app/db/repositories/_base.py`
- Create: `backend/tests/unit/db/conftest.py`
- Test: `backend/tests/unit/db/test_repo_base.py`

- [ ] **Step 1: 写 fixture 与失败测试**

`backend/tests/unit/db/conftest.py`:
```python
import pytest


@pytest.fixture
def repo_engine(tmp_path):
    from app.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    return eng
```

`backend/tests/unit/db/test_repo_base.py`:
```python
def test_upsert_insert_then_update(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    _base.upsert(repo_engine, runs, "run_id",
                 {"run_id": "a", "root_run_id": "a", "created_at": 1.0, "status": "pending"})
    _base.upsert(repo_engine, runs, "run_id",
                 {"run_id": "a", "root_run_id": "a", "created_at": 1.0, "status": "completed"})
    got = _base.get_by_pk(repo_engine, runs, "run_id", "a")
    assert got["status"] == "completed"  # 第二次 upsert 覆盖


def test_update_by_pk_partial(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    _base.upsert(repo_engine, runs, "run_id",
                 {"run_id": "b", "root_run_id": "b", "created_at": 1.0})
    n = _base.update_by_pk(repo_engine, runs, "run_id", "b", {"final_score": 0.9, "favorite": True})
    assert n == 1
    got = _base.get_by_pk(repo_engine, runs, "run_id", "b")
    assert got["final_score"] == 0.9 and got["favorite"] in (1, True)


def test_get_all_keyed_by_pk(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    for rid in ("x", "y"):
        _base.upsert(repo_engine, runs, "run_id",
                     {"run_id": rid, "root_run_id": rid, "created_at": 1.0})
    allrows = _base.get_all(repo_engine, runs, "run_id")
    assert set(allrows) == {"x", "y"} and allrows["x"]["run_id"] == "x"


def test_get_by_pk_missing_returns_none(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    assert _base.get_by_pk(repo_engine, runs, "run_id", "nope") is None
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_base.py -q`
Expected: FAIL（`ModuleNotFoundError: app.db.repositories`）

- [ ] **Step 3: 写 _base**

`backend/app/db/repositories/__init__.py`:
```python
"""Dict-based repository layer over the SQLite schema."""
```

`backend/app/db/repositories/_base.py`:
```python
"""Generic dict<->row CRUD helpers shared by single-PK tables."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert


def upsert(engine, table, pk: str, row: dict) -> None:
    """INSERT row; on PK conflict, UPDATE the non-PK columns present in *row*."""
    with engine.begin() as conn:
        stmt = sqlite_insert(table).values(**row)
        set_ = {k: stmt.excluded[k] for k in row if k != pk}
        if set_:
            stmt = stmt.on_conflict_do_update(index_elements=[pk], set_=set_)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=[pk])
        conn.execute(stmt)


def update_by_pk(engine, table, pk: str, key, fields: dict) -> int:
    """UPDATE *fields* WHERE pk==key. Returns affected row count (0 if no row / empty fields)."""
    if not fields:
        return 0
    with engine.begin() as conn:
        res = conn.execute(sa_update(table).where(table.c[pk] == key).values(**fields))
        return res.rowcount


def get_by_pk(engine, table, pk: str, key) -> "dict | None":
    with engine.connect() as conn:
        row = conn.execute(select(table).where(table.c[pk] == key)).mappings().first()
    return dict(row) if row else None


def get_all(engine, table, pk: str) -> dict:
    with engine.connect() as conn:
        rows = conn.execute(select(table)).mappings().all()
    return {r[pk]: dict(r) for r in rows}
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_base.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/repositories/__init__.py backend/app/db/repositories/_base.py backend/tests/unit/db/conftest.py backend/tests/unit/db/test_repo_base.py
git commit -m "feat(db): add generic _base repository CRUD + test fixture"
```

---

## Task 2: events 仓储（通用事件流）

**Files:**
- Create: `backend/app/db/repositories/events.py`
- Test: `backend/tests/unit/db/test_repo_events.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_repo_events.py`:
```python
def test_append_and_load_in_order(repo_engine):
    from app.db.repositories import events
    events.append_event(repo_engine, entity_type="variant_group", entity_id="g1",
                        event_type="created", payload={"n": 1}, ts=1.0)
    events.append_event(repo_engine, entity_type="variant_group", entity_id="g1",
                        event_type="winner", payload={"run": "r2"}, ts=2.0)
    got = events.load_events(repo_engine, entity_type="variant_group", entity_id="g1")
    assert [e["event_type"] for e in got] == ["created", "winner"]
    assert got[0]["payload"] == {"n": 1}


def test_load_scoped_by_entity(repo_engine):
    from app.db.repositories import events
    events.append_event(repo_engine, entity_type="fusion", entity_id="f1",
                        event_type="x", payload={}, ts=1.0)
    events.append_event(repo_engine, entity_type="fusion", entity_id="f2",
                        event_type="y", payload={}, ts=1.0)
    assert len(events.load_events(repo_engine, entity_type="fusion", entity_id="f1")) == 1


def test_preference_events_null_entity(repo_engine):
    from app.db.repositories import events
    events.append_event(repo_engine, entity_type="preference", entity_id=None,
                        event_type="winner_selected", payload={"run_id": "r"}, ts=3.0)
    got = events.load_events(repo_engine, entity_type="preference", entity_id=None)
    assert len(got) == 1 and got[0]["payload"]["run_id"] == "r"
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_events.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 events 仓储**

`backend/app/db/repositories/events.py`:
```python
"""Generic append-only event stream over the `events` table."""
from __future__ import annotations

from sqlalchemy import select

from app.db.schema import events as _events


def append_event(engine, *, entity_type: str, entity_id, event_type: str,
                 payload: dict, ts: float) -> None:
    with engine.begin() as conn:
        conn.execute(_events.insert().values(
            entity_type=entity_type, entity_id=entity_id,
            event_type=event_type, payload=payload, ts=ts))


def load_events(engine, *, entity_type: str, entity_id) -> list:
    """Events for one entity, ordered by insertion (event_id). entity_id=None
    matches NULL-entity streams (e.g. preference)."""
    cond = _events.c.entity_type == entity_type
    cond = cond & (_events.c.entity_id.is_(None) if entity_id is None
                   else _events.c.entity_id == entity_id)
    with engine.connect() as conn:
        rows = conn.execute(
            select(_events).where(cond).order_by(_events.c.event_id)
        ).mappings().all()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_events.py -v`
Expected: PASS（3 个）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/repositories/events.py backend/tests/unit/db/test_repo_events.py
git commit -m "feat(db): add generic events repository (append/load by entity)"
```

---

## Task 3: runs 仓储

**Files:**
- Create: `backend/app/db/repositories/runs.py`
- Test: `backend/tests/unit/db/test_repo_runs.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_repo_runs.py`:
```python
def test_upsert_get_run_roundtrip_with_json(repo_engine):
    from app.db.repositories import runs
    runs.upsert_run(repo_engine, {
        "run_id": "r1", "root_run_id": "r1", "created_at": 1.0,
        "tags": ["a", "b"], "source_run_ids": ["s1"], "favorite": True,
    })
    got = runs.get_run(repo_engine, "r1")
    assert got["tags"] == ["a", "b"]          # JSON 往返
    assert got["source_run_ids"] == ["s1"]
    assert got["favorite"] in (1, True)


def test_update_run_fields(repo_engine):
    from app.db.repositories import runs
    runs.upsert_run(repo_engine, {"run_id": "r2", "root_run_id": "r2", "created_at": 1.0})
    assert runs.update_run(repo_engine, "r2", {"status": "completed", "final_score": 0.8}) == 1
    assert runs.get_run(repo_engine, "r2")["status"] == "completed"


def test_get_all_runs_keyed(repo_engine):
    from app.db.repositories import runs
    for rid in ("a", "b"):
        runs.upsert_run(repo_engine, {"run_id": rid, "root_run_id": rid, "created_at": 1.0})
    assert set(runs.get_all_runs(repo_engine)) == {"a", "b"}
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_runs.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 runs 仓储**

`backend/app/db/repositories/runs.py`:
```python
"""runs table repository (thin wrappers over _base)."""
from __future__ import annotations

from app.db.repositories import _base
from app.db.schema import runs as _runs


def upsert_run(engine, row: dict) -> None:
    _base.upsert(engine, _runs, "run_id", row)


def update_run(engine, run_id: str, fields: dict) -> int:
    return _base.update_by_pk(engine, _runs, "run_id", run_id, fields)


def get_run(engine, run_id: str) -> "dict | None":
    return _base.get_by_pk(engine, _runs, "run_id", run_id)


def get_all_runs(engine) -> dict:
    return _base.get_all(engine, _runs, "run_id")
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_runs.py -v`
Expected: PASS（3 个）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/repositories/runs.py backend/tests/unit/db/test_repo_runs.py
git commit -m "feat(db): add runs repository"
```

---

## Task 4: variant_groups + draw_sessions 仓储

**Files:**
- Create: `backend/app/db/repositories/variant_groups.py`, `backend/app/db/repositories/draw_sessions.py`
- Test: `backend/tests/unit/db/test_repo_groups_draws.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_repo_groups_draws.py`:
```python
def test_variant_group_roundtrip(repo_engine):
    from app.db.repositories import variant_groups as vg
    vg.upsert_group(repo_engine, {
        "group_id": "g1", "root_run_id": "r", "parent_run_id": "p",
        "source_checkpoint_id": "c", "variant_count": 3,
        "child_run_ids": ["a", "b", "c"], "created_at": 1.0,
    })
    got = vg.get_group(repo_engine, "g1")
    assert got["child_run_ids"] == ["a", "b", "c"] and got["variant_count"] == 3


def test_draw_session_roundtrip(repo_engine):
    from app.db.repositories import draw_sessions as ds
    ds.upsert_session(repo_engine, {
        "draw_id": "d1", "root_run_id": "r", "parent_run_id": "p",
        "source_checkpoint_id": "c", "requested_count": 8,
        "group_ids": ["g1", "g2"], "metadata": {"k": "v"}, "created_at": 1.0,
    })
    got = ds.get_session(repo_engine, "d1")
    assert got["group_ids"] == ["g1", "g2"] and got["metadata"] == {"k": "v"}
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_groups_draws.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写两个仓储**

`backend/app/db/repositories/variant_groups.py`:
```python
"""variant_groups table repository."""
from __future__ import annotations

from app.db.repositories import _base
from app.db.schema import variant_groups as _vg


def upsert_group(engine, row: dict) -> None:
    _base.upsert(engine, _vg, "group_id", row)


def get_group(engine, group_id: str) -> "dict | None":
    return _base.get_by_pk(engine, _vg, "group_id", group_id)
```

`backend/app/db/repositories/draw_sessions.py`:
```python
"""draw_sessions table repository."""
from __future__ import annotations

from app.db.repositories import _base
from app.db.schema import draw_sessions as _ds


def upsert_session(engine, row: dict) -> None:
    _base.upsert(engine, _ds, "draw_id", row)


def get_session(engine, draw_id: str) -> "dict | None":
    return _base.get_by_pk(engine, _ds, "draw_id", draw_id)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_groups_draws.py -v`
Expected: PASS（2 个）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/repositories/variant_groups.py backend/app/db/repositories/draw_sessions.py backend/tests/unit/db/test_repo_groups_draws.py
git commit -m "feat(db): add variant_groups and draw_sessions repositories"
```

---

## Task 5: fusions 仓储（plan + regions 子表）

**Files:**
- Create: `backend/app/db/repositories/fusions.py`
- Test: `backend/tests/unit/db/test_repo_fusions.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_repo_fusions.py`:
```python
def _plan(regions):
    return {
        "fusion_id": "f1", "root_run_id": "r", "parent_run_id": "p",
        "base_run_id": "b", "source_run_ids": ["s1"], "feedback": "x",
        "status": "draft", "created_at": 1.0, "regions": regions,
    }


def test_upsert_get_fusion_with_regions_ordered(repo_engine):
    from app.db.repositories import fusions
    fusions.upsert_fusion(repo_engine, _plan([
        {"id": "reg2", "source_run_id": "s1", "geometry": {"x": 0, "y": 0, "w": 1, "h": 1}},
        {"id": "reg1", "source_run_id": "s1", "strength": 0.7},
    ]))
    got = fusions.get_fusion(repo_engine, "f1")
    assert [r["id"] for r in got["regions"]] == ["reg2", "reg1"]   # 保序
    assert got["regions"][0]["geometry"] == {"x": 0, "y": 0, "w": 1, "h": 1}
    assert got["regions"][1]["strength"] == 0.7
    assert got["source_run_ids"] == ["s1"]


def test_reupsert_replaces_regions(repo_engine):
    from app.db.repositories import fusions
    fusions.upsert_fusion(repo_engine, _plan([{"id": "a", "source_run_id": "s1"},
                                              {"id": "b", "source_run_id": "s1"}]))
    fusions.upsert_fusion(repo_engine, _plan([{"id": "c", "source_run_id": "s1"}]))
    got = fusions.get_fusion(repo_engine, "f1")
    assert [r["id"] for r in got["regions"]] == ["c"]   # 整体替换


def test_get_missing_fusion_none(repo_engine):
    from app.db.repositories import fusions
    assert fusions.get_fusion(repo_engine, "nope") is None
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_fusions.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 fusions 仓储**

`backend/app/db/repositories/fusions.py`:
```python
"""fusion_plans + fusion_regions repository.

The plan dict carries nested `regions` (list of region dicts using the
FusionRegion field `id`). On write we split regions into the child table
(adding ordinal); on read we re-attach them ordered by ordinal.
"""
from __future__ import annotations

from sqlalchemy import select

from app.db.repositories import _base
from app.db.schema import fusion_plans as _fp
from app.db.schema import fusion_regions as _fr

_REGION_DEFAULTS = {
    "label": "", "source_run_id": "", "instruction": "",
    "geometry_type": "rect", "geometry": {},
    "strength": 0.5, "blend_mode": "soft", "feather": 0.08,
}


def upsert_fusion(engine, plan: dict) -> None:
    regions = plan.get("regions") or []
    plan_row = {k: v for k, v in plan.items() if k != "regions"}
    fusion_id = plan_row["fusion_id"]
    with engine.begin() as conn:
        # plan row upsert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        stmt = sqlite_insert(_fp).values(**plan_row)
        set_ = {k: stmt.excluded[k] for k in plan_row if k != "fusion_id"}
        if set_:
            stmt = stmt.on_conflict_do_update(index_elements=["fusion_id"], set_=set_)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=["fusion_id"])
        conn.execute(stmt)
        # replace regions wholesale
        conn.execute(_fr.delete().where(_fr.c.fusion_id == fusion_id))
        for ordinal, reg in enumerate(regions):
            values = {"fusion_id": fusion_id, "region_id": reg["id"], "ordinal": ordinal}
            for key, default in _REGION_DEFAULTS.items():
                values[key] = reg.get(key, default)
            conn.execute(_fr.insert().values(**values))


def _region_row_to_dict(row) -> dict:
    return {
        "id": row["region_id"],
        "label": row["label"],
        "source_run_id": row["source_run_id"],
        "instruction": row["instruction"],
        "geometry_type": row["geometry_type"],
        "geometry": row["geometry"],
        "strength": row["strength"],
        "blend_mode": row["blend_mode"],
        "feather": row["feather"],
    }


def get_fusion(engine, fusion_id: str) -> "dict | None":
    plan = _base.get_by_pk(engine, _fp, "fusion_id", fusion_id)
    if plan is None:
        return None
    with engine.connect() as conn:
        rows = conn.execute(
            select(_fr).where(_fr.c.fusion_id == fusion_id).order_by(_fr.c.ordinal)
        ).mappings().all()
    plan["regions"] = [_region_row_to_dict(r) for r in rows]
    return plan
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_fusions.py -v`
Expected: PASS（3 个）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/repositories/fusions.py backend/tests/unit/db/test_repo_fusions.py
git commit -m "feat(db): add fusions repository with region child-table split"
```

---

## Task 6: preferences 仓储（profile 单例）

**Files:**
- Create: `backend/app/db/repositories/preferences.py`
- Test: `backend/tests/unit/db/test_repo_preferences.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/unit/db/test_repo_preferences.py`:
```python
def test_profile_absent_then_save_load(repo_engine):
    from app.db.repositories import preferences as pref
    assert pref.load_profile(repo_engine) is None
    pref.save_profile(repo_engine, {
        "schema_version": 1, "updated_at": 5.0, "enabled": True,
        "default_locks": {"palette": True}, "positive_preferences": ["bright"],
        "negative_preferences": [], "preferred_variant_labels": ["semantic"],
        "score_drop_tolerance_hint": 0.03, "summary_source_event_count": 4,
    })
    got = pref.load_profile(repo_engine)
    assert got["positive_preferences"] == ["bright"]
    assert got["default_locks"] == {"palette": True}
    assert got["enabled"] in (1, True)


def test_save_profile_is_singleton(repo_engine):
    from app.db.repositories import preferences as pref
    from app.db.schema import preference_profile
    from sqlalchemy import select, func
    pref.save_profile(repo_engine, {"updated_at": 1.0, "enabled": True})
    pref.save_profile(repo_engine, {"updated_at": 2.0, "enabled": False})
    with repo_engine.connect() as conn:
        n = conn.execute(select(func.count()).select_from(preference_profile)).scalar()
    assert n == 1                                   # 始终单行
    assert pref.load_profile(repo_engine)["updated_at"] == 2.0
```

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_preferences.py -q`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 写 preferences 仓储**

`backend/app/db/repositories/preferences.py`:
```python
"""preference_profile singleton repository (row id is always 1).

Preference *events* go through the generic events repository with
entity_type='preference'; this module only owns the profile row.
"""
from __future__ import annotations

from app.db.repositories import _base
from app.db.schema import preference_profile as _pp

# 表里除 id 外的列（save 时只取这些键，忽略外来键）
_PROFILE_COLS = (
    "schema_version", "updated_at", "enabled", "default_locks",
    "positive_preferences", "negative_preferences", "preferred_variant_labels",
    "score_drop_tolerance_hint", "summary_source_event_count",
)


def load_profile(engine) -> "dict | None":
    return _base.get_by_pk(engine, _pp, "id", 1)


def save_profile(engine, profile: dict) -> None:
    row = {k: profile[k] for k in _PROFILE_COLS if k in profile}
    row["id"] = 1
    _base.upsert(engine, _pp, "id", row)
```

- [ ] **Step 4: 运行，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/db/test_repo_preferences.py -v`
Expected: PASS（2 个）

- [ ] **Step 5: 提交**

```bash
git add backend/app/db/repositories/preferences.py backend/tests/unit/db/test_repo_preferences.py
git commit -m "feat(db): add preference_profile singleton repository"
```

---

## Task 7: 全门禁回归

- [ ] **Step 1: db 仓储全测**

Run: `cd backend && python3 -m pytest tests/unit/db/ -q`
Expected: 全绿（plan 01 的 13 个 + 本计划新增）

- [ ] **Step 2: 后端全量回归**

Run: `cd backend && python3 -m pytest tests/unit/ -q`
Expected: 仍只有 2 个预存 explore-variants 失败，无新增失败

- [ ] **Step 3: 前端 build 门禁（确认未波及）**

Run: `cd frontend && npm run build`
Expected: 成功

---

## Self-Review

**1. Spec coverage（§3 仓储层）:** 7 张表全部有仓储入口 —— runs/variant_groups/draw_sessions/fusion_plans(+regions)/preference_profile/events ✅。dict-only、engine 首参、保未来 plan 03 易接 ✅。

**2. Placeholder scan:** 无 TBD；每步含完整代码与预期。

**3. Type consistency:** `_base.upsert/update_by_pk/get_by_pk/get_all` 在 Task 1 定义，Task 3-6 一致引用；fusion region 字段名 `id`（dataclass）↔ 表列 `region_id` 的映射在 `upsert_fusion`/`_region_row_to_dict` 两处对称 ✅；preferences 列清单 `_PROFILE_COLS` 与 schema 列一致 ✅。

**4. 范围外（plan 03/04）:** 不改 run_index/variant_groups/draw_sessions/preferences/fusion_plans 这些**现有 pipeline 模块**；不做 dataclass↔dict 接线；不做数据导入。

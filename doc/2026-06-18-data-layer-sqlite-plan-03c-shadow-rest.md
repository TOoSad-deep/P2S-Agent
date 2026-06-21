# SQLite 数据层 · 计划 03c：其余 4 模块影子双写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. 与 03b 同构的加法式影子双写；文件行为/读路径/既有测试全部不变。

**Goal:** 把 `variant_groups` / `draw_sessions` / `preferences` / `fusion_plans` 四个 orchestration 模块的写（`save_*` + `append_*_event`）best-effort 镜像进 SQLite，复用 plan 02 的仓储。读仍走文件。

**Architecture:** 新增共享 `p2s_agent/core/db/shadow.py::shadow_engine(results_root)`（None→`backend/data/p2s.db`；目录→`<dir>/p2s.db`；惰性 init_db）。每个模块加 `_SHADOW_DB_ENABLED` + 两个 best-effort 影子 helper，在 `save_*`/`append_*_event` 末尾各调一次。全程 `try/except`+DEBUG，永不抛错。

**Tech Stack:** Python 3.9 · SQLAlchemy 2.0 · pytest

**已知影子期偏差（同 03b，留待读切换）:** 文件侧删除/重写不同步到 DB；events 的 `event_type`/`ts` 为 best-effort 提取（payload 完整）。

---

## File Structure
- 新：`backend/p2s_agent/core/db/shadow.py`
- 改：`p2s_agent/orchestration/{variant_groups,draw_sessions,preferences,fusion_plans}.py`（各加影子 helper + 末尾调用）
- 新测试：`backend/tests/unit/test_orchestration_shadow.py`

---

## Task 1: 共享 shadow_engine

**Files:** Create `backend/p2s_agent/core/db/shadow.py`; Test `backend/tests/unit/test_shadow_engine.py`

- [ ] **Step 1: 失败测试**
```python
# backend/tests/unit/test_shadow_engine.py
def test_shadow_engine_dir_inits_and_isolates(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    from sqlalchemy import inspect
    eng = shadow_engine(tmp_path)
    assert (tmp_path / "p2s.db").exists()
    assert "runs" in set(inspect(eng).get_table_names())

def test_shadow_engine_caches(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    assert shadow_engine(tmp_path) is shadow_engine(tmp_path)
```

- [ ] **Step 2: 运行确认失败**
Run: `cd backend && python3 -m pytest tests/unit/test_shadow_engine.py -q` → FAIL (ModuleNotFound)

- [ ] **Step 3: 实现**
```python
# backend/p2s_agent/core/db/shadow.py
"""Best-effort shadow-mirror helpers shared by orchestration persistence modules.

Let the file-based modules mirror writes into SQLite without changing their
authoritative file behavior. Callers wrap usage in try/except so a DB issue
never breaks the file write path.
"""
from __future__ import annotations

import threading
from pathlib import Path

_INIT_LOCK = threading.Lock()
_INITED: set[str] = set()


def shadow_engine(results_root):
    """Engine for the shadow DB. results_root=None → canonical backend/data/p2s.db;
    a directory → <dir>/p2s.db (per-test isolation). Lazily init_db (idempotent)."""
    from p2s_agent.core.db.engine import get_engine, init_db
    eng = get_engine(None) if results_root is None else get_engine(Path(results_root))
    key = str(eng.url)
    if key not in _INITED:
        with _INIT_LOCK:
            if key not in _INITED:
                init_db(eng)
                _INITED.add(key)
    return eng
```

- [ ] **Step 4: 运行确认通过** → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(db): add shared shadow_engine helper"`

---

## Task 2: variant_groups 影子

**Files:** Modify `p2s_agent/orchestration/variant_groups.py`; tests in `test_orchestration_shadow.py`

- [ ] **Step 1: 失败测试**（写进 `backend/tests/unit/test_orchestration_shadow.py`）
```python
def test_variant_group_shadow(tmp_path):
    from p2s_agent.orchestration.variant_groups import (
        VariantGroupRecord, save_group, append_group_event)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import variant_groups as vg, events as ev
    rec = VariantGroupRecord(group_id="g1", root_run_id="r", parent_run_id="p",
        source_checkpoint_id="c", feedback="f", mode="m", variant_count=2,
        diversity="medium", status="queued", child_run_ids=["a"], created_at=1.0)
    save_group(rec, root=tmp_path)
    append_group_event("g1", {"type": "winner", "ts": 2.0, "run_id": "a"}, root=tmp_path)
    eng = shadow_engine(tmp_path)
    assert vg.get_group(eng, "g1")["child_run_ids"] == ["a"]
    evs = ev.load_events(eng, entity_type="variant_group", entity_id="g1")
    assert len(evs) == 1 and evs[0]["payload"]["run_id"] == "a"
```

- [ ] **Step 2: 运行确认失败** → FAIL（影子未实现，get_group 返回 None）

- [ ] **Step 3: 实现** — 在 `variant_groups.py` 顶部常量区加：
```python
_SHADOW_DB_ENABLED = True


def _shadow_save_group(record, root) -> None:
    if not _SHADOW_DB_ENABLED:
        return
    try:
        from p2s_agent.core.db.shadow import shadow_engine
        from p2s_agent.core.db.repositories import variant_groups as _vg_repo
        _vg_repo.upsert_group(shadow_engine(root), dataclasses.asdict(record))
    except Exception:
        logger.debug("variant_groups shadow upsert failed", exc_info=True)


def _shadow_group_event(group_id, event, root) -> None:
    if not _SHADOW_DB_ENABLED:
        return
    try:
        from p2s_agent.core.db.shadow import shadow_engine
        from p2s_agent.core.db.repositories import events as _ev_repo
        _ev_repo.append_event(
            shadow_engine(root), entity_type="variant_group", entity_id=group_id,
            event_type=str(event.get("type", "")),
            payload=event, ts=_shadow_ts(event))
    except Exception:
        logger.debug("variant_groups shadow event failed", exc_info=True)


def _shadow_ts(event) -> float:
    v = event.get("ts", event.get("timestamp", 0.0))
    return float(v) if isinstance(v, (int, float)) else 0.0
```
变量需求：模块需有 `logger`（若无则加 `import logging; logger = logging.getLogger(__name__)`）。`dataclasses` 已导入。
在 `save_group` 的 `return target` 之前加 `_shadow_save_group(record, root)`；在 `append_group_event` 末尾加 `_shadow_group_event(group_id, event, root)`。

- [ ] **Step 4: 运行确认通过** → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(db): shadow-mirror variant_groups writes"`

---

## Task 3: draw_sessions 影子

**Files:** Modify `p2s_agent/orchestration/draw_sessions.py`

- [ ] **Step 1: 失败测试**（追加到 `test_orchestration_shadow.py`）
```python
def test_draw_session_shadow(tmp_path):
    from p2s_agent.orchestration.draw_sessions import (
        DrawSessionRecord, save_session, append_session_event)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import draw_sessions as ds, events as ev
    rec = DrawSessionRecord(draw_id="d1", root_run_id="r", parent_run_id="p",
        source_checkpoint_id="c", feedback="f", status="queued", requested_count=8,
        diversity="medium", group_ids=["g1"], created_at=1.0)
    save_session(rec, root=tmp_path)
    append_session_event("d1", {"type": "card", "ts": 2.0}, root=tmp_path)
    eng = shadow_engine(tmp_path)
    assert ds.get_session(eng, "d1")["group_ids"] == ["g1"]
    assert len(ev.load_events(eng, entity_type="draw_session", entity_id="d1")) == 1
```

- [ ] **Step 2: 失败** → FAIL
- [ ] **Step 3: 实现** — 同 Task 2 模式，`entity_type="draw_session"`，repo 用 `draw_sessions.upsert_session`，helper 名 `_shadow_save_session`/`_shadow_session_event`；钩 `save_session`/`append_session_event`。
- [ ] **Step 4: 通过** → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(db): shadow-mirror draw_sessions writes"`

---

## Task 4: preferences 影子

**Files:** Modify `p2s_agent/orchestration/preferences.py`

- [ ] **Step 1: 失败测试**
```python
def test_preferences_shadow(tmp_path):
    from p2s_agent.orchestration.preferences import (
        PreferenceEvent, save_profile, append_preference_event, default_profile)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import preferences as pf, events as ev
    prof = default_profile(); prof["positive_preferences"] = ["bright"]
    save_profile(prof, root=tmp_path)
    append_preference_event(PreferenceEvent(event_id="e1", event_type="winner_selected",
        timestamp=2.0, run_id="r"), root=tmp_path)
    eng = shadow_engine(tmp_path)
    assert pf.load_profile(eng)["positive_preferences"] == ["bright"]
    evs = ev.load_events(eng, entity_type="preference", entity_id=None)
    assert len(evs) == 1 and evs[0]["event_type"] == "winner_selected"
```

- [ ] **Step 2: 失败** → FAIL
- [ ] **Step 3: 实现** — profile：`_shadow_save_profile(profile, root)` → `preferences_repo.save_profile(shadow_engine(root), profile)`；event：`_shadow_pref_event(event, root)` → `events_repo.append_event(..., entity_type="preference", entity_id=None, event_type=event.event_type, payload=dataclasses.asdict(event), ts=event.timestamp)`。钩 `save_profile`/`append_preference_event`。（`clear_preferences` 暂不镜像 —— 删除属读切换范围。）
- [ ] **Step 4: 通过** → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(db): shadow-mirror preferences writes"`

---

## Task 5: fusion_plans 影子

**Files:** Modify `p2s_agent/orchestration/fusion_plans.py`

- [ ] **Step 1: 失败测试**
```python
def test_fusion_plan_shadow(tmp_path):
    from p2s_agent.orchestration.fusion_plans import (
        FusionPlanRecord, FusionRegion, save_plan, append_plan_event)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import fusions as fz, events as ev
    rec = FusionPlanRecord(fusion_id="f1", root_run_id="r", parent_run_id="p",
        base_run_id="b", source_run_ids=["s1"], draw_session_id=None, feedback="x",
        status="draft", regions=[FusionRegion(id="reg1", label="", source_run_id="s1",
            instruction="", geometry_type="rect", geometry={"x":0,"y":0,"w":1,"h":1})],
        created_at=1.0)
    save_plan(rec, root=tmp_path)
    append_plan_event("f1", {"type": "target_ready", "ts": 2.0}, root=tmp_path)
    eng = shadow_engine(tmp_path)
    got = fz.get_fusion(eng, "f1")
    assert [r["id"] for r in got["regions"]] == ["reg1"]
    assert len(ev.load_events(eng, entity_type="fusion", entity_id="f1")) == 1
```

- [ ] **Step 2: 失败** → FAIL
- [ ] **Step 3: 实现** — `_shadow_save_plan(record, root)` → `fusions_repo.upsert_fusion(shadow_engine(root), dataclasses.asdict(record))`（asdict 递归把 regions 转 dict，含 `id`）；`_shadow_plan_event` → events `entity_type="fusion"`。钩 `save_plan`/`append_plan_event`。
- [ ] **Step 4: 通过** → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(db): shadow-mirror fusion_plans writes"`

---

## Task 6: 全门禁回归

- [ ] **Step 1:** `cd backend && python3 -m pytest tests/unit/test_variant_groups.py tests/unit/test_draw_sessions.py tests/unit/test_preferences.py tests/unit/test_fusion_plans.py -q` → 既有测试零变化全绿
- [ ] **Step 2:** `cd backend && python3 -m pytest tests/unit/ -q` → 0 失败（+新影子测试；边界测试绿）
- [ ] **Step 3:** `cd frontend && npm run build` → 成功

---

## Self-Review
- **覆盖:** 4 模块 × (save + event) 全部影子化，共享 `shadow_engine`；每模块独立 `_SHADOW_DB_ENABLED`。
- **零回归:** 不改任何 `save_*`/`load_*`/`append_*` 既有逻辑，只在 `save_*`/`append_*_event` 末尾各加 1 行 best-effort 调用。
- **范围外:** 读切换、删除/prune 同步、数据导入。

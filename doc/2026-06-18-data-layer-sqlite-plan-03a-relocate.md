# SQLite 数据层 · 计划 03a：平移到 p2s_agent/core/db Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. 纯机械重构（移动 + 改 import），**验收门禁 = 既有测试 + 边界测试全绿**，无新行为。

**Goal:** 把数据层从 `backend/app/db/`（Web 层，违反 L1 边界）平移到 `backend/p2s_agent/core/db/`（与既有文件持久化 `core/pipeline/artifacts.py` 同层），为 plan 03b 的「orchestration→db」接线扫清分层障碍。

**Architecture:** `git mv` 保历史；`app.db` → `p2s_agent.core.db` 全量改 import；修 `engine.py` 的 `DEFAULT_DB_PATH` 深度（从 `app/db/` 的 parents[2] 变成 `p2s_agent/core/db/` 的 parents[3]，仍指向 `backend/data/p2s.db`）。blast radius 完全自包含（仅我自己的文件引用 `app.db`，orchestration/core 尚未依赖）。

**Tech Stack:** Python 3.9 · SQLAlchemy 2.0 · Alembic · pytest

**Why core/db not store:** 边界测试 `test_agent_web_boundary.py` 禁止 `p2s_agent` import `app.*`；且 store 的设计约定「orchestration 不得 import store」。core 是 orchestration 已允许依赖的下层（`run_index.py` 已 import `core.pipeline.artifacts`），故 db 放 core 最一致。

---

## 受影响文件（全部是本数据层自身的引用，已 grep 穷举）
- 移动：`app/db/{__init__,schema,engine}.py` + `app/db/repositories/*` → `p2s_agent/core/db/...`
- 改 import：上述被移动文件内部 + `alembic/env.py` + `scripts/init_db.py` + `tests/unit/db/*.py`
- 改一行：`engine.py` 的 `DEFAULT_DB_PATH` parents 深度

---

## Task 1: 移动文件 + 修 engine 路径 + 内部 import

- [ ] **Step 1: git mv 全部文件**

```bash
cd backend
mkdir -p p2s_agent/core/db
git mv app/db/__init__.py        p2s_agent/core/db/__init__.py
git mv app/db/schema.py          p2s_agent/core/db/schema.py
git mv app/db/engine.py          p2s_agent/core/db/engine.py
git mv app/db/repositories       p2s_agent/core/db/repositories
rmdir app/db 2>/dev/null || true
```

- [ ] **Step 2: 改被移动文件内部 import（`app.db` → `p2s_agent.core.db`）**

```bash
cd backend
grep -rl "app\.db" p2s_agent/core/db/ | xargs sed -i '' 's/app\.db/p2s_agent.core.db/g'
```

- [ ] **Step 3: 修 engine.py 的 DEFAULT_DB_PATH 深度**

`p2s_agent/core/db/engine.py`：把
```python
# backend/ 根：engine.py 在 backend/app/db/ → parents[2] == backend/
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "p2s.db"
```
改为
```python
# backend/ 根：engine.py 在 backend/p2s_agent/core/db/ → parents[3] == backend/
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "p2s.db"
```

- [ ] **Step 4: 验证 db 包 import + 默认路径仍指向 backend/data**

```bash
cd backend && python3 -c "from p2s_agent.core.db.engine import DEFAULT_DB_PATH; print(DEFAULT_DB_PATH)"
```
Expected: 打印 `.../backend/data/p2s.db`（不是 `.../backend/p2s_agent/data/p2s.db`）

- [ ] **Step 5: 提交**

```bash
git add -A backend/app backend/p2s_agent/core/db
git commit -m "refactor(db): relocate data layer app/db -> p2s_agent/core/db"
```

---

## Task 2: 更新 alembic / scripts / tests 的 import

- [ ] **Step 1: alembic env.py**

`backend/alembic/env.py`：`from app.db.schema import metadata as target_metadata` → `from p2s_agent.core.db.schema import metadata as target_metadata`（注释里的 `app.db` 一并改）。

- [ ] **Step 2: scripts/init_db.py**

`from app.db.engine import ...` → `from p2s_agent.core.db.engine import ...`（注释里的 `app.db` 一并改）。

- [ ] **Step 3: tests 批量改 import**

```bash
cd backend
grep -rl "app\.db" tests/unit/db/ | xargs sed -i '' 's/app\.db/p2s_agent.core.db/g'
```

- [ ] **Step 4: 验证 db 套件 + alembic + 边界测试**

```bash
cd backend
python3 -m pytest tests/unit/db/ -q
python3 -m pytest tests/unit/test_agent_web_boundary.py -q
```
Expected: db 套件 30 全绿；边界测试绿（db 代码进了 p2s_agent 但不 import app.*）。

- [ ] **Step 5: 重建真实库（路径未变，验证 alembic 仍工作）+ 提交**

```bash
cd backend && rm -f data/p2s.db data/p2s.db-wal data/p2s.db-shm && python3 -m alembic upgrade head
python3 -c "import sqlite3; c=sqlite3.connect('data/p2s.db'); print(len(list(c.execute(\"select name from sqlite_master where type='table'\"))), 'tables')"
git add -A backend/alembic backend/scripts backend/tests/unit/db
git commit -m "refactor(db): point alembic/scripts/tests at p2s_agent.core.db"
```

---

## Task 3: 全门禁回归

- [ ] **Step 1: 后端全量**

Run: `cd backend && python3 -m pytest tests/unit/ -q`
Expected: 仅 2 个预存 explore-variants 失败，无新增（db 套件全绿、边界测试绿）

- [ ] **Step 2: 前端 build**

Run: `cd frontend && npm run build`
Expected: 成功

---

## Self-Review
- **覆盖:** 移动 + 内部 import + engine 路径深度 + alembic + scripts + tests，已 grep 穷举 `app.db` 引用，无遗漏。
- **风险:** 唯一实质代码改动是 `DEFAULT_DB_PATH` 的 parents 深度（Task 1 Step 4 显式验证）；其余是路径/import 平移，由既有 30 个 db 测试 + 边界测试守住。
- **范围外:** 不接线 orchestration（plan 03b）、不导数据（plan 04）。

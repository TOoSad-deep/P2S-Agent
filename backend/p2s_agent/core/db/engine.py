"""SQLite engine factory: WAL + PRAGMAs + directory-keyed get_engine override.

`get_engine(override)`:
  - None  → 默认库 backend/data/p2s.db
  - 目录  → <目录>/p2s.db（让旧 `root=tmp_path` 风格的测试继续隔离）
"""
from __future__ import annotations

import threading
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from p2s_agent.core.db.schema import metadata

# backend/ 根：engine.py 在 backend/p2s_agent/core/db/ → parents[3] == backend/
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "p2s.db"

_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


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
        cur.execute("PRAGMA busy_timeout=5000;")   # FIRST: lets journal_mode wait out a lock
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    return engine


def get_engine(override: "Path | str | None" = None) -> Engine:
    url = _resolve_db_url(override)
    eng = _engines.get(url)
    if eng is None:
        # Double-checked locking: without it, concurrent first-touch (FastAPI
        # background workers) created duplicate engines/pools over the SAME
        # sqlite file — leaking pooled connections and racing PRAGMA/create_all
        # on a cold DB (which silently dropped the first writes).
        with _engines_lock:
            eng = _engines.get(url)
            if eng is None:
                eng = _make_engine(url)
                _engines[url] = eng
    return eng


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables (idempotent). For the real DB prefer Alembic; this is
    the fast path for tests and first-run bootstrap."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    return engine

"""一次性引导：在 backend/data/p2s.db 建好全部表。

    cd backend && python3 scripts/init_db.py

幂等；真实 schema 演进请走 Alembic。
"""
import os
import sys

# 让 `import app.db...` 在以脚本方式运行时也可用（backend/ 入 sys.path）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.engine import DEFAULT_DB_PATH, get_engine, init_db  # noqa: E402


def main() -> None:
    eng = get_engine()  # 默认 backend/data/p2s.db
    init_db(eng)
    print(f"[init_db] tables created at {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()

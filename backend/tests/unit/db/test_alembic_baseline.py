import os
import subprocess
from pathlib import Path

from sqlalchemy import create_engine, inspect

BACKEND = Path(__file__).resolve().parents[3]  # backend/


def test_alembic_upgrade_head_builds_full_schema(tmp_path):
    db = tmp_path / "p2s.db"
    env = {**os.environ, "ALEMBIC_DB_URL": f"sqlite:///{db}"}
    result = subprocess.run(
        ["python3", "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"

    tables = set(inspect(create_engine(f"sqlite:///{db}")).get_table_names())
    from app.db.schema import metadata
    expected = set(metadata.tables.keys())
    assert expected.issubset(tables), f"missing tables: {expected - tables}"

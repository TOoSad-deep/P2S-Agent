from sqlalchemy import text


def test_pragmas_wal_and_fk_on(tmp_path):
    from p2s_agent.core.db.engine import get_engine
    eng = get_engine(tmp_path)
    with eng.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_get_engine_dir_override_creates_db_file(tmp_path):
    from p2s_agent.core.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    assert (tmp_path / "p2s.db").exists()


def test_get_engine_is_cached_per_dir(tmp_path):
    from p2s_agent.core.db.engine import get_engine
    assert get_engine(tmp_path) is get_engine(tmp_path)


def test_init_db_creates_all_tables(tmp_path):
    from sqlalchemy import inspect
    from p2s_agent.core.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    tables = set(inspect(eng).get_table_names())
    assert {"runs", "events", "fusion_regions"}.issubset(tables)

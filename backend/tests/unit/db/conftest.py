import pytest


@pytest.fixture
def repo_engine(tmp_path):
    from app.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    return eng

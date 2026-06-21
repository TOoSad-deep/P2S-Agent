def test_sqlalchemy_and_db_package_import():
    import sqlalchemy
    assert sqlalchemy.__version__.startswith("2.")
    import p2s_agent.core.db  # noqa: F401

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

"""preference_profile singleton repository (row id is always 1).

Preference *events* go through the generic events repository with
entity_type='preference'; this module only owns the profile row.
"""
from __future__ import annotations

from p2s_agent.core.db.repositories import _base
from p2s_agent.core.db.schema import preference_profile as _pp

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

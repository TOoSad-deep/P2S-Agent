"""SQLAlchemy Core schema — single source of truth for the P2S-Agent DB.

Portable on purpose: only generic types (Text/Integer/Float/Boolean/JSON),
a partial index, a CHECK, and an ON DELETE CASCADE FK — all of which map
cleanly onto PostgreSQL/openGauss if we migrate later.
"""
from __future__ import annotations

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, Column, Float, ForeignKey, Index,
    Integer, MetaData, PrimaryKeyConstraint, Table, Text, text,
)

metadata = MetaData()

# 1) runs — 血缘 + 元数据主表（取代 run_index.jsonl 折叠）
runs = Table(
    "runs", metadata,
    Column("run_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text),
    Column("source_checkpoint_id", Text),
    Column("source_checkpoint_label", Text),
    Column("mode", Text),
    Column("feedback", Text),
    Column("title", Text),
    Column("status", Text, nullable=False, server_default="unknown"),
    Column("run_dir", Text),
    Column("created_at", Float, nullable=False),
    Column("completed_at", Float),
    Column("final_score", Float),
    Column("favorite", Boolean, nullable=False, server_default="0"),
    Column("tags", JSON, nullable=False, server_default=text("'[]'")),
    Column("variant_group_id", Text),
    Column("variant_index", Integer),
    Column("variant_label", Text),
    Column("draw_session_id", Text),
    Column("draw_card_index", Integer),
    Column("replacement_of_run_id", Text),
    Column("fusion_id", Text),
    Column("base_run_id", Text),
    Column("source_run_ids", JSON, nullable=False, server_default=text("'[]'")),
)
Index("idx_runs_root", runs.c.root_run_id)
Index("idx_runs_parent", runs.c.parent_run_id)
Index("idx_runs_status", runs.c.status)
Index("idx_runs_score", runs.c.final_score)
Index("idx_runs_created", runs.c.created_at)
Index("idx_runs_fav", runs.c.favorite, sqlite_where=runs.c.favorite == True)  # noqa: E712 部分索引
Index("idx_runs_vgroup", runs.c.variant_group_id)
Index("idx_runs_draw", runs.c.draw_session_id)
Index("idx_runs_fusion", runs.c.fusion_id)

# 2) variant_groups
variant_groups = Table(
    "variant_groups", metadata,
    Column("group_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text, nullable=False),
    Column("source_checkpoint_id", Text, nullable=False),
    Column("feedback", Text, nullable=False, server_default=""),
    Column("mode", Text, nullable=False, server_default=""),
    Column("variant_count", Integer, nullable=False, server_default="0"),
    Column("diversity", Text, nullable=False, server_default="medium"),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("child_run_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("winner_run_id", Text),
    Column("created_at", Float, nullable=False, server_default="0"),
    Column("completed_at", Float),
    Column("draw_session_id", Text),
)
Index("idx_vg_root", variant_groups.c.root_run_id)
Index("idx_vg_draw", variant_groups.c.draw_session_id)

# 3) draw_sessions
draw_sessions = Table(
    "draw_sessions", metadata,
    Column("draw_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text, nullable=False),
    Column("source_checkpoint_id", Text, nullable=False),
    Column("feedback", Text, nullable=False, server_default=""),
    Column("status", Text, nullable=False, server_default="queued"),
    Column("requested_count", Integer, nullable=False, server_default="0"),
    Column("diversity", Text, nullable=False, server_default="medium"),
    Column("mode", Text, nullable=False, server_default="batch_draw"),
    Column("group_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("card_run_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("winner_run_id", Text),
    Column("created_at", Float, nullable=False, server_default="0"),
    Column("updated_at", Float),
    Column("completed_at", Float),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
)
Index("idx_draw_root", draw_sessions.c.root_run_id)

# 4) fusion_plans
fusion_plans = Table(
    "fusion_plans", metadata,
    Column("fusion_id", Text, primary_key=True),
    Column("root_run_id", Text, nullable=False),
    Column("parent_run_id", Text, nullable=False),
    Column("base_run_id", Text, nullable=False, server_default=""),
    Column("source_run_ids", JSON, nullable=False, server_default=text("'[]'")),
    Column("draw_session_id", Text),
    Column("feedback", Text, nullable=False, server_default=""),
    Column("status", Text, nullable=False, server_default="draft"),
    Column("composite_target_artifact_id", Text),
    Column("output_run_id", Text),
    Column("created_at", Float, nullable=False, server_default="0"),
    Column("updated_at", Float),
    Column("metadata", JSON, nullable=False, server_default=text("'{}'")),
)
Index("idx_fusion_root", fusion_plans.c.root_run_id)

# 5) fusion_regions（子表；删 plan 级联清理；复合主键 (fusion_id, region_id)）
fusion_regions = Table(
    "fusion_regions", metadata,
    Column("fusion_id", Text,
           ForeignKey("fusion_plans.fusion_id", ondelete="CASCADE"),
           nullable=False),
    Column("region_id", Text, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("label", Text, nullable=False, server_default=""),
    Column("source_run_id", Text, nullable=False, server_default=""),
    Column("instruction", Text, nullable=False, server_default=""),
    Column("geometry_type", Text, nullable=False, server_default="rect"),
    Column("geometry", JSON, nullable=False, server_default=text("'{}'")),
    Column("strength", Float, nullable=False, server_default="0.5"),
    Column("blend_mode", Text, nullable=False, server_default="soft"),
    Column("feather", Float, nullable=False, server_default="0.08"),
    PrimaryKeyConstraint("fusion_id", "region_id", name="pk_fusion_regions"),
)

# 6) preference_profile（单行单例）
preference_profile = Table(
    "preference_profile", metadata,
    Column("id", Integer, primary_key=True),
    Column("schema_version", Integer, nullable=False, server_default="1"),
    Column("updated_at", Float, nullable=False, server_default="0"),
    Column("enabled", Boolean, nullable=False, server_default="1"),
    Column("default_locks", JSON, nullable=False, server_default=text("'{}'")),
    Column("positive_preferences", JSON, nullable=False, server_default=text("'[]'")),
    Column("negative_preferences", JSON, nullable=False, server_default=text("'[]'")),
    Column("preferred_variant_labels", JSON, nullable=False, server_default=text("'[]'")),
    Column("score_drop_tolerance_hint", Float, nullable=False, server_default="0.02"),
    Column("summary_source_event_count", Integer, nullable=False, server_default="0"),
    CheckConstraint("id = 1", name="ck_pref_singleton"),
)

# 7) events（通用事件流；取代所有 *_events.jsonl + preference events.jsonl）
events = Table(
    "events", metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("entity_type", Text, nullable=False),
    Column("entity_id", Text),
    Column("event_type", Text, nullable=False),
    Column("payload", JSON, nullable=False, server_default=text("'{}'")),
    Column("ts", Float, nullable=False),
)
Index("idx_events_entity", events.c.entity_type, events.c.entity_id, events.c.event_id)
Index("idx_events_ts", events.c.entity_type, events.c.ts)

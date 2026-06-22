"""Regression: event readers are file-first (the *_events.jsonl is the complete
append-only log) so a swallowed DB mirror can't yield a partial event stream;
the DB is read only when the file is absent (post-retirement)."""
import p2s_agent.core.db.shadow as shadow


def test_group_events_complete_when_db_partial(tmp_path, monkeypatch):
    from p2s_agent.orchestration.variant_groups import append_group_event, load_group_events
    append_group_event("g", {"type": "a", "ts": 1.0}, root=tmp_path)       # file + DB
    monkeypatch.setattr(shadow, "_ENABLED", False)
    append_group_event("g", {"type": "b", "ts": 2.0}, root=tmp_path)       # file only (swallowed)
    monkeypatch.setattr(shadow, "_ENABLED", True)
    append_group_event("g", {"type": "c", "ts": 3.0}, root=tmp_path)       # file + DB
    # file = [a,b,c] complete; DB = [a,c] partial. File-first returns complete.
    evs = load_group_events("g", root=tmp_path)
    assert [e["type"] for e in evs] == ["a", "b", "c"]


def test_preference_events_complete_when_db_partial(tmp_path, monkeypatch):
    from p2s_agent.orchestration.preferences import (
        PreferenceEvent, append_preference_event, load_preference_events)
    append_preference_event(PreferenceEvent("e1", "winner_selected", 1.0), root=tmp_path)
    monkeypatch.setattr(shadow, "_ENABLED", False)
    append_preference_event(PreferenceEvent("e2", "variant_rated", 2.0), root=tmp_path)
    monkeypatch.setattr(shadow, "_ENABLED", True)
    evs = load_preference_events(root=tmp_path)
    assert [e.event_id for e in evs] == ["e1", "e2"]   # file complete despite DB partial


def test_snapshot_file_wins_over_stale_db(tmp_path, monkeypatch):
    """A swallowed snapshot UPDATE leaves the DB stale; file-first reads must
    return the fresh file value, not the stale DB row."""
    from p2s_agent.orchestration.variant_groups import (
        VariantGroupRecord, save_group, load_group)

    def rec(status):
        return VariantGroupRecord(group_id="g", root_run_id="r", parent_run_id="p",
            source_checkpoint_id="c", feedback="", mode="", variant_count=2,
            diversity="medium", status=status, created_at=1.0)

    save_group(rec("queued"), root=tmp_path)            # file + DB = queued
    monkeypatch.setattr(shadow, "_ENABLED", False)
    save_group(rec("completed"), root=tmp_path)         # file=completed, DB stays queued
    monkeypatch.setattr(shadow, "_ENABLED", True)
    assert load_group("g", root=tmp_path).status == "completed"   # file-first → fresh


def test_events_fall_back_to_db_when_file_absent(tmp_path):
    from p2s_agent.orchestration.variant_groups import load_group_events
    from p2s_agent.core.db.repositories import events as ev
    # only the DB has events (no file) — e.g. after file retirement
    ev.append_event(shadow.shadow_engine(tmp_path), entity_type="variant_group",
                    entity_id="g2", event_type="x", payload={"k": "v"}, ts=1.0)
    evs = load_group_events("g2", root=tmp_path)
    assert len(evs) == 1 and evs[0]["k"] == "v"

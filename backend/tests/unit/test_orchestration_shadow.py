"""Shadow dual-write for the 4 file-based orchestration modules.

Each module's save_*/append_*_event additionally mirrors into SQLite
(best-effort). Files stay authoritative; these assert the shadow tables and
event stream are populated.
"""


def test_variant_group_shadow(tmp_path):
    from p2s_agent.orchestration.variant_groups import (
        VariantGroupRecord, save_group, append_group_event)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import variant_groups as vg, events as ev
    rec = VariantGroupRecord(group_id="g1", root_run_id="r", parent_run_id="p",
        source_checkpoint_id="c", feedback="f", mode="m", variant_count=2,
        diversity="medium", status="queued", child_run_ids=["a"], created_at=1.0)
    save_group(rec, root=tmp_path)
    append_group_event("g1", {"type": "winner", "ts": 2.0, "run_id": "a"}, root=tmp_path)
    eng = shadow_engine(tmp_path)
    assert vg.get_group(eng, "g1")["child_run_ids"] == ["a"]
    evs = ev.load_events(eng, entity_type="variant_group", entity_id="g1")
    assert len(evs) == 1 and evs[0]["payload"]["run_id"] == "a"


def test_draw_session_shadow(tmp_path):
    from p2s_agent.orchestration.draw_sessions import (
        DrawSessionRecord, save_session, append_session_event)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import draw_sessions as ds, events as ev
    rec = DrawSessionRecord(draw_id="d1", root_run_id="r", parent_run_id="p",
        source_checkpoint_id="c", feedback="f", status="queued", requested_count=8,
        diversity="medium", group_ids=["g1"], created_at=1.0)
    save_session(rec, root=tmp_path)
    append_session_event("d1", {"type": "card", "ts": 2.0}, root=tmp_path)
    eng = shadow_engine(tmp_path)
    assert ds.get_session(eng, "d1")["group_ids"] == ["g1"]
    assert len(ev.load_events(eng, entity_type="draw_session", entity_id="d1")) == 1


def test_preferences_shadow(tmp_path):
    from p2s_agent.orchestration.preferences import (
        PreferenceEvent, save_profile, append_preference_event, default_profile)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import preferences as pf, events as ev
    prof = default_profile()
    prof["positive_preferences"] = ["bright"]
    save_profile(prof, root=tmp_path)
    append_preference_event(PreferenceEvent(event_id="e1", event_type="winner_selected",
        timestamp=2.0, run_id="r"), root=tmp_path)
    eng = shadow_engine(tmp_path)
    assert pf.load_profile(eng)["positive_preferences"] == ["bright"]
    evs = ev.load_events(eng, entity_type="preference", entity_id=None)
    assert len(evs) == 1 and evs[0]["event_type"] == "winner_selected"


def test_fusion_plan_shadow(tmp_path):
    from p2s_agent.orchestration.fusion_plans import (
        FusionPlanRecord, FusionRegion, save_plan, append_plan_event)
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import fusions as fz, events as ev
    rec = FusionPlanRecord(fusion_id="f1", root_run_id="r", parent_run_id="p",
        base_run_id="b", source_run_ids=["s1"], draw_session_id=None, feedback="x",
        status="draft", regions=[FusionRegion(id="reg1", label="", source_run_id="s1",
            instruction="", geometry_type="rect", geometry={"x": 0, "y": 0, "w": 1, "h": 1})],
        created_at=1.0)
    save_plan(rec, root=tmp_path)
    append_plan_event("f1", {"type": "target_ready", "ts": 2.0}, root=tmp_path)
    eng = shadow_engine(tmp_path)
    got = fz.get_fusion(eng, "f1")
    assert [r["id"] for r in got["regions"]] == ["reg1"]
    assert len(ev.load_events(eng, entity_type="fusion", entity_id="f1")) == 1

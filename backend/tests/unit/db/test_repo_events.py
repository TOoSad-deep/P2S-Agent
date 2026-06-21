def test_append_and_load_in_order(repo_engine):
    from app.db.repositories import events
    events.append_event(repo_engine, entity_type="variant_group", entity_id="g1",
                        event_type="created", payload={"n": 1}, ts=1.0)
    events.append_event(repo_engine, entity_type="variant_group", entity_id="g1",
                        event_type="winner", payload={"run": "r2"}, ts=2.0)
    got = events.load_events(repo_engine, entity_type="variant_group", entity_id="g1")
    assert [e["event_type"] for e in got] == ["created", "winner"]
    assert got[0]["payload"] == {"n": 1}


def test_load_scoped_by_entity(repo_engine):
    from app.db.repositories import events
    events.append_event(repo_engine, entity_type="fusion", entity_id="f1",
                        event_type="x", payload={}, ts=1.0)
    events.append_event(repo_engine, entity_type="fusion", entity_id="f2",
                        event_type="y", payload={}, ts=1.0)
    assert len(events.load_events(repo_engine, entity_type="fusion", entity_id="f1")) == 1


def test_preference_events_null_entity(repo_engine):
    from app.db.repositories import events
    events.append_event(repo_engine, entity_type="preference", entity_id=None,
                        event_type="winner_selected", payload={"run_id": "r"}, ts=3.0)
    got = events.load_events(repo_engine, entity_type="preference", entity_id=None)
    assert len(got) == 1 and got[0]["payload"]["run_id"] == "r"

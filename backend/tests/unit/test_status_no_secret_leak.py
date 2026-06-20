"""/status must never serialize api keys / ModelConfig. Anchors the store split (later task)."""
import json
from app.config import ModelConfig
from app.routers import png_shader as ps


def test_status_snapshot_excludes_model_secrets():
    run_id = "leak-probe"
    ps._store_run(run_id, {"status": "completed", "result": {"ok": True}})
    # Register a per-run model carrying a sentinel secret via the model store:
    ps._store_run_model(run_id, ModelConfig(
        api_key="SENTINEL_SECRET_KEY",
        base_url="http://sentinel.invalid",
        model="m",
    ))
    snap = ps._snapshot_run(run_id)
    blob = json.dumps(snap)
    assert "SENTINEL_SECRET_KEY" not in blob, "/status snapshot leaked the api_key"
    for needle in ("api_key", "base_url", "ModelConfig"):
        assert needle not in blob, f"/status snapshot leaked {needle!r}"

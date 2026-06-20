"""Guards the L1 refactor: the public HTTP surface must not drift while we move code."""
from app.main import app

EXPECTED_ROUTES = set(tuple(x) for x in [
    ["GET", "/api/models"],
    ["GET", "/api/strategy-config"],
    ["GET", "/docs"],
    ["GET", "/docs/oauth2-redirect"],
    ["GET", "/health"],
    ["GET", "/openapi.json"],
    ["GET", "/png-shader/draw-sessions/{draw_id}"],
    ["GET", "/png-shader/fusions/{fusion_id}"],
    ["GET", "/png-shader/fusions/{fusion_id}/artifacts/{artifact_id:path}"],
    ["GET", "/png-shader/preferences/profile"],
    ["GET", "/png-shader/runs/{run_id}/artifacts/{artifact_id:path}"],
    ["GET", "/png-shader/runs/{run_id}/branches"],
    ["GET", "/png-shader/runs/{run_id}/checkpoints"],
    ["GET", "/png-shader/runs/{run_id}/timeline"],
    ["GET", "/png-shader/status/{run_id}"],
    ["GET", "/png-shader/variant-groups/{group_id}"],
    ["GET", "/redoc"],
    ["HEAD", "/docs"],
    ["HEAD", "/docs/oauth2-redirect"],
    ["HEAD", "/openapi.json"],
    ["HEAD", "/redoc"],
    ["PATCH", "/png-shader/preferences/profile"],
    ["PATCH", "/png-shader/runs/{run_id}/metadata"],
    ["PATCH", "/png-shader/runs/{run_id}/strategy"],
    ["POST", "/png-shader/draw-sessions/{draw_id}/cards/{run_id}/event"],
    ["POST", "/png-shader/draw-sessions/{draw_id}/draw-more"],
    ["POST", "/png-shader/draw-sessions/{draw_id}/redraw"],
    ["POST", "/png-shader/fusions"],
    ["POST", "/png-shader/fusions/{fusion_id}/composite-target"],
    ["POST", "/png-shader/fusions/{fusion_id}/run"],
    ["POST", "/png-shader/parameterize/{run_id}"],
    ["POST", "/png-shader/preferences/clear"],
    ["POST", "/png-shader/preferences/events"],
    ["POST", "/png-shader/preferences/rebuild"],
    ["POST", "/png-shader/refine/{run_id}"],
    ["POST", "/png-shader/run"],
    ["POST", "/png-shader/runs/{run_id}/branch-refine"],
    ["POST", "/png-shader/runs/{run_id}/draw-session"],
    ["POST", "/png-shader/runs/{run_id}/explore-variants"],
    ["POST", "/png-shader/runs/{run_id}/region-mask"],
    ["POST", "/png-shader/runs/{run_id}/stop"],
    ["POST", "/png-shader/variant-groups/{group_id}/ratings"],
    ["POST", "/png-shader/variant-groups/{group_id}/stop"],
    ["POST", "/png-shader/variant-groups/{group_id}/winner"],
])


def test_route_inventory_is_stable():
    got = {(m, r.path) for r in app.routes for m in (getattr(r, "methods", None) or [])}
    missing = EXPECTED_ROUTES - got
    added = got - EXPECTED_ROUTES
    assert not missing and not added, f"route surface drifted: missing={missing} added={added}"

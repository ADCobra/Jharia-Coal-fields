import base64
import json

import pytest
from fastapi.testclient import TestClient

from firewatch.api.app import create_app

AUTH = {"Authorization": "Basic " + base64.b64encode(b"demo:demo").decode()}


def test_demo_runs_end_to_end(demo_ctx):
    summaries = demo_ctx.summaries
    assert len(summaries) == 6
    assert all(s["status"] == "done" for s in summaries)
    with_agent = [s for s in summaries if s["agent"]]
    assert len(with_agent) == 4
    assert all(s["alignment"]["ok"] for s in summaries[1:])
    assert all((s["alignment"]["residual_px"] or 0) < 0.6 for s in summaries[1:])
    assert sum(s["hotspots"] for s in summaries) > 0
    repo = demo_ctx.services.repo
    run_id = with_agent[-1]["run_id"]
    events = demo_ctx.services.blobs.get_bytes(f"runs/{run_id}/trace.jsonl").decode().splitlines()
    nodes = {json.loads(e)["node"] for e in events}
    assert "perceive" in nodes and nodes & {"plan", "skip", "decide"}
    assert repo.list_hotspots(run_id)
    for name in ("lst.webp", "anomaly.webp", "hotspots.geojson", "cones.geojson", "maps.npz"):
        assert demo_ctx.services.blobs.exists(f"runs/{run_id}/{name}"), name


def test_industrial_site_is_never_treated_as_a_fire(demo_ctx):
    repo = demo_ctx.services.repo
    for summary in demo_ctx.summaries:
        for h in repo.list_hotspots(summary["run_id"]):
            if h["industrial_overlap"] >= 0.3 and summary["agent"]:
                assert h["decision"] == "exclude", (summary["scene_id"], h["hotspot_id"], h["decision"])


@pytest.fixture(scope="module")
def client(demo_ctx):
    app = create_app(demo_ctx.settings, demo_ctx.base_dir, ctx=demo_ctx)
    with TestClient(app) as c:
        yield c


def test_health_is_public_and_the_rest_needs_a_login(client):
    assert client.get("/healthz").json()["ok"]
    assert client.get("/api/summary").status_code == 401
    assert client.get("/api/summary", headers={"Authorization": "Basic " + base64.b64encode(b"demo:wrong").decode()}).status_code == 401


def test_dashboard_and_api(client, demo_ctx):
    assert client.get("/dashboard", headers=AUTH).status_code == 200
    away = client.get("/dashboard", follow_redirects=False)
    assert away.status_code == 303 and away.headers["location"] == "/login"
    assert client.get("/").status_code == 200
    assert client.get("/help").status_code == 200
    public = client.get("/api/public/status").json()
    assert public["photos"] >= 1 and "closed_roads" in public
    assert public["rules"]["close_buffer_m"] > 0 and public["hot_area_km2"] >= 0
    assert all({"name", "lon", "lat", "level"} <= set(f) for f in public["fire_list"])
    assert client.get("/api/public/layers/lst").headers["content-type"] == "image/webp"
    assert client.get("/api/public/layers/nope").status_code == 404
    assert isinstance(client.get("/api/public/examples").json(), list)
    assert client.get("/api/places").status_code == 401
    assert isinstance(client.get("/api/places", headers=AUTH).json(), list)
    summary = client.get("/api/summary", headers=AUTH).json()
    assert summary["runs"] == 6 and summary["source"] == "synthetic"
    runs = client.get("/api/runs", headers=AUTH).json()
    latest = runs[-1]["run_id"]
    assert client.get(f"/api/runs/{latest}", headers=AUTH).json()["status"] == "done"
    hotspots = client.get(f"/api/runs/{latest}/hotspots", headers=AUTH).json()
    assert all("risk" in h for h in hotspots)
    layer = client.get(f"/api/runs/{latest}/layers/lst", headers=AUTH)
    assert layer.status_code == 200 and layer.headers["content-type"] == "image/webp"
    assert client.get(f"/api/runs/{latest}/geojson/hotspots", headers=AUTH).json()["type"] == "FeatureCollection"
    assert client.get(f"/api/runs/{latest}/geojson/secret", headers=AUTH).status_code == 404
    trace = client.get(f"/api/runs/{latest}/trace", headers=AUTH).json()
    assert trace and trace[0]["node"] == "perceive"
    roads = client.get("/api/roads", headers=AUTH).json()
    assert len(roads["features"]) == len(demo_ctx.roads)
    assert "report" in client.get("/api/routes", headers=AUTH).json()
    assert client.get("/static/style.css").status_code == 200


def test_approval_endpoint_rejects_unknown_ids_and_double_decisions(client, demo_ctx):
    assert client.post("/api/approvals/A-nope", json={"approve": True}, headers=AUTH).status_code == 409
    pending = client.get("/api/approvals", headers=AUTH).json()
    if not pending:
        pytest.skip("this short demo produced nothing to approve")
    first = pending[0]
    decided = client.post(f"/api/approvals/{first['approval_id']}", json={"approve": False, "note": "not yet"}, headers=AUTH)
    assert decided.status_code == 200 and decided.json()["status"] == "rejected"
    again = client.post(f"/api/approvals/{first['approval_id']}", json={"approve": True}, headers=AUTH)
    assert again.status_code == 409


def test_layers_carry_their_provenance_as_xmp(client, demo_ctx):
    latest = demo_ctx.summaries[-1]
    for name in ("lst", "anomaly", "frame"):
        xmp = client.get(f"/api/runs/{latest['run_id']}/layers/{name}/provenance", headers=AUTH).json()["xmp"]
        assert xmp["runId"] == latest["run_id"] and xmp["sceneId"] == latest["scene_id"]
        assert xmp["acquired"].startswith(latest["acquired"][:10])
    lst = client.get(f"/api/runs/{latest['run_id']}/layers/lst/provenance", headers=AUTH).json()["xmp"]
    west, south, east, north = latest["layers"]["bounds"]
    assert float(lst["west"]) == pytest.approx(west, abs=1e-5) and float(lst["north"]) == pytest.approx(north, abs=1e-5)
    assert lst["units"] == "K" and float(lst["valueMin"]) < float(lst["valueMax"])


def test_season_time_lapse_is_an_animation_of_every_pass(client, demo_ctx):
    from firewatch.vision.media import decode_animation

    response = client.get("/api/timelapse", headers=AUTH)
    assert response.status_code == 200 and response.headers["content-type"] == "image/webp"
    frames = decode_animation(response.content)
    assert len(frames) == len(demo_ctx.summaries)
    assert frames[0].shape[1] == 960
    assert client.get("/api/timelapse", headers=AUTH).content == response.content


def test_mine_sites_and_aerial_view(client):
    sites = client.get("/api/sites", headers=AUTH).json()["features"]
    kinds = {f["properties"]["kind"] for f in sites}
    assert {"pit", "washery", "plant"} <= kinds
    assert all(f["properties"]["name"] for f in sites)
    aerial = client.get("/api/scenery", headers=AUTH)
    assert aerial.status_code == 200 and aerial.headers["content-type"] == "image/webp"
    assert client.get("/api/summary", headers=AUTH).json()["scenery_bounds"]


def test_read_only_view_needs_no_login_and_refuses_writes(demo_ctx):
    app = create_app(demo_ctx.settings, demo_ctx.base_dir, ctx=demo_ctx, read_only=True, prefix="/real", other_view={"href": "/", "label": "Simulated demo"})
    with TestClient(app) as c:
        page = c.get("/dashboard")
        assert page.status_code == 200 and 'window.FW_BASE="/real"' in page.text
        assert 'window.FW_BASE="/real"' in c.get("/").text
        summary = c.get("/api/summary").json()
        assert summary["read_only"] and not summary["can_run"] and summary["other_view"]["href"] == "/"
        assert c.get("/api/public/status").json()["read_only"]
        assert c.post("/api/runs/next").status_code == 403
        assert c.post("/api/approvals/anything", json={"approve": True}).status_code == 403

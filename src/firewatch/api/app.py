from __future__ import annotations

import hashlib
import hmac
import html
import json
import logging
import secrets
import threading
from urllib.parse import parse_qs
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from firewatch import __version__
from firewatch.config import Settings
from firewatch.geo.grid import WGS84, transformer
from firewatch.pipeline.timelapse import season_timelapse
from firewatch.util import jsonable, utcnow
from firewatch.vision.media import read_provenance

log = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")
LAYERS = ("lst", "anomaly", "color", "frame", "photo")
security = HTTPBasic(realm="Jharia Fire Watch", auto_error=False)
SESSION_COOKIE = "fw_session"
SESSION_DAYS = 7


class Decision(BaseModel):
    approve: bool
    note: str = ""


class SceneRequest(BaseModel):
    scene_id: str


class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: dict | None = None
        self.last_error: str | None = None


def create_app(settings: Settings, base_dir: str | Path | None = None, ctx=None, read_only: bool = False, prefix: str = "", other_view: dict | None = None) -> FastAPI:
    """The API and pages. A read-only copy (read_only=True) serves an archived run, for example the real Landsat
    season under /real: no sign-in, every write refused. `prefix` is where it is mounted; `other_view` links the
    pages to the other copy."""
    state = {"ctx": ctx}
    runs = RunState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["ctx"] is None:
            from firewatch.pipeline.context import build_context

            state["ctx"] = build_context(settings, Path(base_dir or Path.cwd()))
        yield

    app = FastAPI(title="Jharia Fire Watch", version=__version__, lifespan=lifespan)

    # Sign-in: the login page sets a signed session cookie; scripts can still send HTTP Basic credentials.
    # The signing key comes from the password, so changing the password signs everyone out.
    session_key = hashlib.sha256(f"firewatch-session|{settings.api.user}|{settings.api.password}".encode()).digest()

    def session_token(name: str) -> str:
        return f"{name}:{hmac.new(session_key, name.encode(), hashlib.sha256).hexdigest()}"

    def good_login(name: str, password: str) -> bool:
        good_user = secrets.compare_digest(name.encode(), settings.api.user.encode())
        good_pass = secrets.compare_digest(password.encode(), settings.api.password.encode())
        return good_user and good_pass

    def signed_in(request: Request, credentials: HTTPBasicCredentials | None) -> str | None:
        if credentials and good_login(credentials.username, credentials.password):
            return credentials.username
        token = request.cookies.get(SESSION_COOKIE, "")
        name = token.split(":", 1)[0]
        if name and secrets.compare_digest(token.encode(), session_token(name).encode()) and name == settings.api.user:
            return name
        return None

    def user(request: Request, credentials: HTTPBasicCredentials | None = Depends(security)) -> str:
        if read_only:
            return "viewer"
        name = signed_in(request, credentials)
        if name is None:
            raise HTTPException(status_code=401, detail="sign in at /login")
        return name

    def context():
        if state["ctx"] is None:
            raise HTTPException(503, "starting up")
        return state["ctx"]

    if read_only:
        @app.middleware("http")
        async def refuse_writes(request: Request, call_next):
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                return JSONResponse({"detail": "This is a read-only view; nothing here can be changed."}, status_code=403)
            return await call_next(request)

    # Pages served under a prefix get a one-line script that points their requests at that prefix.
    view_script = (
        f"<script>window.FW_BASE={json.dumps(prefix)};"
        "window.FW_API=function(p){return p.indexOf('/api/')===0?window.FW_BASE+p:p};"
        "window.FW_ASSET=window.FW_API;window.FW_DASH=window.FW_BASE+'/dashboard';"
        "window.FW_EXAMPLES=window.FW_BASE+'/api/public/examples';</script>"
    )

    def page(name: str) -> Response:
        if not prefix:
            return FileResponse(STATIC / name, headers={"Cache-Control": "no-store"})
        text = (STATIC / name).read_text().replace("</head>", view_script + "\n</head>", 1)
        return HTMLResponse(text, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__, "time": utcnow().isoformat()}

    @app.get("/")
    def home():
        return page("home.html")

    @app.get("/help")
    def help_page():
        if prefix:
            return RedirectResponse("/help", status_code=303)
        return FileResponse(STATIC / "help.html", headers={"Cache-Control": "no-store"})

    @app.get("/dashboard")
    def index(request: Request, credentials: HTTPBasicCredentials | None = Depends(security)):
        if not read_only and signed_in(request, credentials) is None:
            return RedirectResponse("/login", status_code=303)
        return page("index.html")

    @app.get("/login")
    def login_page(error: int = 0):
        page = (STATIC / "login.html").read_text()
        demo = settings.api.show_demo_login
        fill = {
            "{{USER}}": html.escape(settings.api.user if demo else "", quote=True),
            "{{PASSWORD}}": html.escape(settings.api.password if demo else "", quote=True),
            "{{DEMO_HIDDEN}}": "" if demo else "hidden",
            "{{ERROR_HIDDEN}}": "" if error else "hidden",
        }
        for key, value in fill.items():
            page = page.replace(key, value)
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.post("/login")
    async def login(request: Request):
        form = parse_qs((await request.body()).decode("utf-8", "replace"))
        name, password = form.get("username", [""])[0], form.get("password", [""])[0]
        if not good_login(name, password):
            return RedirectResponse("/login?error=1", status_code=303)
        res = RedirectResponse("/dashboard", status_code=303)
        res.set_cookie(SESSION_COOKIE, session_token(name), max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax")
        return res

    @app.get("/logout")
    def logout():
        res = RedirectResponse("/", status_code=303)
        res.delete_cookie(SESSION_COOKIE)
        return res

    @app.get("/api/public/status")
    def public_status():
        ctx = context()
        repo = ctx.services.repo
        done = [r for r in repo.list_runs(500) if r.get("status") == "done"]
        latest = max(done, key=lambda r: r["acquired"]) if done else None
        hotspots = repo.list_hotspots(latest["run_id"]) if latest else []
        decisions: dict[str, int] = {}
        for h in hotspots:
            key = h.get("decision") or "log"
            decisions[key] = decisions.get(key, 0) + 1
        closed: dict[str, list[str]] = {}
        for sid, rec in repo.get_closures().items():
            if rec.get("status") == "closed" and sid in ctx.roads.segments:
                closed.setdefault(ctx.roads.segments[sid].name, []).append(sid)
        return jsonable({
            "source": ctx.source.name,
            "read_only": read_only,
            "other_view": other_view,
            "photos": len(done),
            "first_photo": min((r["acquired"] for r in done), default=None),
            "latest_photo": latest["acquired"] if latest else None,
            "platform": latest.get("platform") if latest else None,
            "cloud_fraction": latest.get("cloud_fraction") if latest else None,
            "fires": sum(v for k, v in decisions.items() if k != "exclude"),
            "factories": decisions.get("exclude", 0),
            "watching": decisions.get("log", 0),
            "asked": decisions.get("escalate", 0),
            "pending_decisions": len(repo.list_approvals("pending")),
            "closed_roads": [{"name": name, "sections": sorted(ids)} for name, ids in sorted(closed.items())],
            "fire_list": [public_fire(h, ctx) for h in hotspots],
            "rules": {"close_buffer_m": ctx.settings.risk.close_buffer_m, "auto_close": ctx.settings.roads.auto_close},
            "hot_area_km2": round(sum(float(h.get("area_m2") or 0) for h in hotspots if h.get("decision") != "exclude") / 1e6, 2),
            "places": public_places(ctx),
            "bounds": (ctx.services.blobs.get_json("static/grid.json", {}) or {}).get("lonlat_bounds"),
            "layers": {"run_id": latest["run_id"], "bounds": (latest.get("layers") or {}).get("bounds"), "photo": bool((latest.get("layers") or {}).get("photo"))} if latest else None,
            "scenery": ctx.services.blobs.get_json("static/scenery.json") or None,
        })

    def public_fire(h: dict, ctx) -> dict:
        risk = h.get("risk") or {}
        seg = risk.get("nearest_segment")
        number = int("".join(ch for ch in str(h.get("track_id") or "") if ch.isdigit()) or 0) or None
        return {
            "name": "Factory heat" if h.get("decision") == "exclude" else (f"Fire {number}" if number else "New hot spot"),
            "lon": round(h["centroid_lonlat"][0], 5),
            "lat": round(h["centroid_lonlat"][1], 5),
            "level": "excluded" if h.get("decision") == "exclude" else risk.get("level", "low"),
            "decision": h.get("decision") or "log",
            "distance_m": risk.get("distance_m"),
            "road": ctx.roads.segments[seg].name if seg in ctx.roads.segments else None,
            "heat_k": round(float(h.get("max_anomaly_k") or 0), 1),
        }

    def public_places(ctx) -> list[dict]:
        from shapely.geometry import shape

        blobs = ctx.services.blobs
        out = []
        for feat in (blobs.get_json("static/places.geojson", {}) or {}).get("features", []):
            lon, lat = feat["geometry"]["coordinates"][:2]
            out.append({"name": feat["properties"].get("name"), "kind": "town", "lon": round(lon, 5), "lat": round(lat, 5)})
        for feat in (blobs.get_json("static/mines.geojson", {}) or {}).get("features", []):
            name = (feat.get("properties") or {}).get("name") or ""
            if name and not name.startswith("Open-cast mine "):
                c = shape(feat["geometry"]).centroid
                out.append({"name": name, "kind": "mine", "lon": round(c.x, 5), "lat": round(c.y, 5)})
        if not out:
            to_ll = transformer(ctx.roads.crs, WGS84)
            for route in ctx.routes:
                lon, lat = to_ll.transform(*route.source)
                out.append({"name": route.name.partition(" to ")[0], "kind": "mine", "lon": round(lon, 5), "lat": round(lat, 5)})
        seen = set()
        for seg in ctx.roads.segments.values():
            if seg.name in seen or seg.name in ("Local road", "Service road", "Mine track", "road"):
                continue
            seen.add(seg.name)
            mid = seg.line.interpolate(0.5, normalized=True)
            lon, lat = transformer(ctx.roads.crs, WGS84).transform(mid.x, mid.y)
            out.append({"name": seg.name, "kind": "road", "lon": round(lon, 5), "lat": round(lat, 5)})
        return out

    @app.get("/api/public/layers/{name}")
    def public_layer(name: str):
        ctx = context()
        blobs = ctx.services.blobs
        if name == "scenery":
            data = blobs.get_bytes("static/scenery.webp")
        else:
            if name not in ("photo", "anomaly", "lst"):
                raise HTTPException(404, "unknown layer")
            done = [r for r in ctx.services.repo.list_runs(500) if r.get("status") == "done"]
            if not done:
                raise HTTPException(404, "no photos yet")
            latest = max(done, key=lambda r: r["acquired"])
            key = (latest.get("layers") or {}).get(name)
            data = blobs.get_bytes(key) if key else None
        if data is None:
            raise HTTPException(404, "layer not found")
        return Response(data, media_type="image/webp", headers={"Cache-Control": "max-age=600"})

    @app.get("/api/public/examples")
    def public_examples():
        ctx = context()
        repo = ctx.services.repo
        done = sorted((r for r in repo.list_runs(500) if r.get("status") == "done" and r.get("agent")), key=lambda r: r["acquired"])
        best: dict[str, tuple[float, dict]] = {}
        for run in done:
            for h in repo.list_hotspots(run["run_id"]):
                key = h.get("decision") or "log"
                score = float((h.get("risk") or {}).get("score") or 0)
                if key not in best or score > best[key][0]:
                    best[key] = (score, {**h, "_date": run["acquired"]})
        out = []
        for key in ("close", "escalate", "log", "exclude"):
            if key not in best:
                continue
            h = best[key][1]
            risk = h.get("risk") or {}
            fire = public_fire(h, ctx)
            out.append({
                "decision": key,
                "name": fire["name"],
                "date": h["_date"],
                "heat": max(2, min(30, round(float(h.get("max_anomaly_k") or 0)))),
                "dist": max(0, min(1000, int(round((risk.get("distance_m") or 1000) / 10.0) * 10))),
                "speed": max(0.0, min(8.0, round(float(risk.get("approach_m_per_day") or 0) * 2) / 2)),
                "cloud": int(round(float(h.get("cloud_ring_fraction") or 0) * 20) * 5),
                "seen": bool(h.get("track_id")) and not (risk.get("confidence_notes") and any("first sighting" in n for n in risk.get("confidence_notes"))),
                "factory": key == "exclude",
            })
        return out

    @app.get("/api/summary")
    def summary(_: str = Depends(user)):
        ctx = context()
        repo = ctx.services.repo
        runs_list = repo.list_runs(200)
        done = [r for r in runs_list if r.get("status") == "done"]
        latest = max(done, key=lambda r: r["acquired"]) if done else None
        grid = ctx.services.blobs.get_json("static/grid.json", {})
        scenery = ctx.services.blobs.get_json("static/scenery.json") or {}
        unprocessed = None
        if ctx.source.name == "real":
            seen = {r.get("scene_id") for r in runs_list}
            unprocessed = sum(1 for ref in ctx.source.refs() if ref.scene_id not in seen)
        from firewatch.vision.cvcompat import capabilities

        return jsonable({
            "version": __version__,
            "aoi": settings.aoi.name,
            "source": ctx.source.name,
            "planner": settings.agent.planner,
            "auto_close": settings.roads.auto_close,
            "close_buffer_m": settings.risk.close_buffer_m,
            "latest_run": latest["run_id"] if latest else None,
            "runs": len(done),
            "pending_approvals": len(repo.list_approvals("pending")),
            "closed_segments": sorted(s for s, rec in repo.get_closures().items() if rec.get("status") == "closed"),
            "bounds": grid.get("lonlat_bounds"),
            "scenery_bounds": scenery.get("bounds"),
            "scenery": {"kind": scenery.get("kind", "simulated"), "acquired": scenery.get("acquired"), "platform": scenery.get("platform")} if scenery else None,
            "unprocessed": unprocessed,
            "can_run": settings.api.allow_runs and not read_only,
            "read_only": read_only,
            "other_view": other_view,
            "next_synthetic_index": repo.get_meta("synthetic_next_index") if ctx.source.name == "synthetic" else None,
            "running": runs.active,
            "last_error": runs.last_error,
            "opencv": capabilities().as_dict(),
            "alignment_methods": ctx.registrar.available_methods(),
        })

    @app.get("/api/runs")
    def list_runs(limit: int = 100, _: str = Depends(user)):
        items = context().services.repo.list_runs(limit)
        keep = ("run_id", "scene_id", "acquired", "platform", "status", "agent", "hotspots", "by_level", "decisions", "cloud_fraction", "seconds", "error")
        rows = [{**{k: r.get(k) for k in keep}, "closed": len(r.get("closed_segments") or [])} for r in items]
        return sorted(rows, key=lambda r: r.get("acquired") or "")

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, _: str = Depends(user)):
        run = context().services.repo.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"no run {run_id}")
        return run

    @app.get("/api/runs/{run_id}/hotspots")
    def run_hotspots(run_id: str, _: str = Depends(user)):
        records = context().services.repo.list_hotspots(run_id)
        return sorted(records, key=lambda r: -(r.get("risk") or {}).get("score", 0))

    @app.get("/api/runs/{run_id}/geojson/{name}")
    def run_geojson(run_id: str, name: str, _: str = Depends(user)):
        if name not in ("hotspots", "cones"):
            raise HTTPException(404, "unknown layer")
        data = context().services.blobs.get_json(f"runs/{run_id}/{name}.geojson")
        if data is None:
            raise HTTPException(404, "layer not found")
        return data

    @app.get("/api/runs/{run_id}/layers/{name}")
    def run_layer(run_id: str, name: str, _: str = Depends(user)):
        if name not in LAYERS:
            raise HTTPException(404, "unknown layer")
        data = context().services.blobs.get_bytes(f"runs/{run_id}/{name}.webp")
        if data is None:
            raise HTTPException(404, "layer not found")
        return Response(data, media_type="image/webp", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/runs/{run_id}/layers/{name}/provenance")
    def run_layer_provenance(run_id: str, name: str, _: str = Depends(user)):
        if name not in LAYERS:
            raise HTTPException(404, "unknown layer")
        data = context().services.blobs.get_bytes(f"runs/{run_id}/{name}.webp")
        if data is None:
            raise HTTPException(404, "layer not found")
        return {"layer": name, "xmp": read_provenance(data)}

    @app.get("/api/timelapse")
    def timelapse(_: str = Depends(user)):
        ctx = context()
        with ctx.lock:
            try:
                data = season_timelapse(ctx.services)
            except LookupError as exc:
                raise HTTPException(404, str(exc)) from exc
        return Response(data, media_type="image/webp", headers={"Cache-Control": "no-cache", "Content-Disposition": 'inline; filename="jharia-season.webp"'})

    @app.get("/api/runs/{run_id}/trace")
    def run_trace(run_id: str, hotspot: str | None = None, _: str = Depends(user)):
        raw = context().services.blobs.get_bytes(f"runs/{run_id}/trace.jsonl")
        if raw is None:
            raise HTTPException(404, "trace not found")
        events = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if hotspot:
            events = [e for e in events if e.get("hotspot_id") in (hotspot, None)]
        return events

    @app.get("/api/tracks/{track_id}")
    def get_track(track_id: str, _: str = Depends(user)):
        track = context().services.repo.get_track(track_id)
        if track is None:
            raise HTTPException(404, f"no track {track_id}")
        return track

    @app.get("/api/roads")
    def roads(_: str = Depends(user)):
        ctx = context()
        ctx.apply_persisted_closures()
        repo = ctx.services.repo
        statuses = {sid: {"status": "closed", "reason": rec.get("reason")} for sid, rec in repo.get_closures().items() if rec.get("status") == "closed"}
        for approval in repo.list_approvals("pending"):
            for sid in approval.get("segments") or []:
                if sid not in statuses:
                    statuses[sid] = {"status": "review", "reason": approval.get("reason")}
        return ctx.roads.to_geojson(statuses)

    @app.get("/api/routes")
    def routes(_: str = Depends(user)):
        ctx = context()
        ctx.apply_persisted_closures()
        return {"geojson": ctx.roads.routes_geojson(ctx.routes), "report": ctx.roads.reroute_report(ctx.routes)}

    @app.get("/api/scenery")
    def scenery(_: str = Depends(user)):
        data = context().services.blobs.get_bytes("static/scenery.webp")
        if data is None:
            raise HTTPException(404, "no base photo for this data source")
        return Response(data, media_type="image/webp", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/places")
    def places(_: str = Depends(user)):
        return public_places(context())

    @app.get("/api/sites")
    def sites(_: str = Depends(user)):
        from shapely.geometry import shape

        ctx = context()
        blobs = ctx.services.blobs
        to_ll = transformer(ctx.roads.crs, WGS84)
        features, placed = [], []

        def add(kind: str, name: str, lon: float, lat: float, **extra) -> None:
            features.append({"type": "Feature", "properties": {"kind": kind, "name": name, **extra}, "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]}})
            placed.append((kind, lon, lat))

        def near(lon: float, lat: float, metres: float) -> bool:
            return any(abs(lon - x) * 102000 < metres and abs(lat - y) * 111000 < metres for _k, x, y in placed)

        for route in ctx.routes:
            source, _sep, target = route.name.partition(" to ")
            lon, lat = to_ll.transform(*route.source)
            add("pit", source.strip() or route.name, lon, lat, route=route.name)
        for route in ctx.routes:
            _source, _sep, target = route.name.partition(" to ")
            tlon, tlat = to_ll.transform(*route.target)
            if not near(tlon, tlat, 300):
                name = target.strip() or "depot"
                add("washery", name[:1].upper() + name[1:], tlon, tlat)
        mines = (blobs.get_json("static/mines.geojson", {}) or {}).get("features", [])
        for feat in sorted(mines, key=lambda f: -(f.get("properties") or {}).get("area_ha", 0))[:12]:
            c = shape(feat["geometry"]).centroid
            if not near(c.x, c.y, 700):
                add("pit", (feat.get("properties") or {}).get("name") or "Open-cast mine", c.x, c.y)
        for feat in (blobs.get_json("static/industrial.geojson", {}) or {}).get("features", []):
            props = feat.get("properties") or {}
            c = shape(feat["geometry"]).centroid
            if not near(c.x, c.y, 300):
                add("washery" if props.get("category") == "washery" else "plant", props.get("name") or "Industrial site", c.x, c.y)
        for feat in (blobs.get_json("static/places.geojson", {}) or {}).get("features", []):
            lon, lat = feat["geometry"]["coordinates"][:2]
            add("town", (feat.get("properties") or {}).get("name") or "", lon, lat, place=(feat.get("properties") or {}).get("place"))
        return {"type": "FeatureCollection", "features": features}

    @app.get("/api/approvals")
    def approvals(status: str | None = "pending", _: str = Depends(user)):
        return context().services.repo.list_approvals(None if status in (None, "all") else status)

    @app.post("/api/approvals/{approval_id}")
    def decide(approval_id: str, body: Decision, who: str = Depends(user)):
        from firewatch.agent.approvals import ApprovalError, decide_approval
        from firewatch.agent.guard import Guard

        ctx = context()
        with ctx.lock:
            ctx.apply_persisted_closures()
            try:
                return decide_approval(approval_id, body.approve, who, body.note, repo=ctx.services.repo, roads=ctx.roads, routes=ctx.routes, guard=Guard(ctx.settings), notifier=ctx.services.notifier)
            except ApprovalError as exc:
                raise HTTPException(409, str(exc))

    def _background(job, label: str) -> None:
        def target():
            try:
                job()
                runs.last_error = None
            except Exception as exc:
                log.exception("run %s failed", label)
                runs.last_error = f"{label}: {type(exc).__name__}: {exc}"[:400]
            finally:
                runs.active = None

        with runs.lock:
            if runs.active is not None:
                raise HTTPException(409, f"a pass is already running ({runs.active['label']})")
            runs.active = {"label": label, "started": utcnow().isoformat()}
        threading.Thread(target=target, daemon=True).start()

    @app.post("/api/runs/next", status_code=202)
    def run_next(_: str = Depends(user)):
        if not settings.api.allow_runs:
            raise HTTPException(403, "runs from the dashboard are switched off")
        ctx = context()
        if ctx.source.name not in ("synthetic", "real"):
            raise HTTPException(400, "next pass works on the synthetic demo or downloaded real scenes; post a scene id instead")
        from firewatch.pipeline.runner import run_next_scene

        _background(lambda: run_next_scene(ctx), "next pass")
        return {"accepted": True}

    @app.post("/api/runs", status_code=202)
    def run_scene(body: SceneRequest, _: str = Depends(user)):
        if not settings.api.allow_runs:
            raise HTTPException(403, "runs from the dashboard are switched off")
        ctx = context()
        from firewatch.pipeline.process import run_pass

        _background(lambda: run_pass(ctx, body.scene_id), body.scene_id)
        return {"accepted": True, "scene_id": body.scene_id}

    @app.get("/api/tasks")
    def tasks(_: str = Depends(user)):
        return context().services.repo.list_tasks()

    @app.get("/api/notifications")
    def notifications(limit: int = 50, _: str = Depends(user)):
        notifier = context().services.notifier
        outbox = getattr(notifier, "outbox", None)
        if outbox and Path(outbox).exists():
            lines = Path(outbox).read_text().splitlines()[-limit:]
            return [json.loads(line) for line in reversed(lines)]
        return list(reversed(getattr(notifier, "sent", [])[-limit:]))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"[:300]}, status_code=500)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def app_from_env() -> FastAPI:
    import os

    from firewatch.config import load_settings

    return create_app(load_settings(), Path(os.environ.get("FIREWATCH_BASE_DIR", ".")))

from __future__ import annotations

import copy
import datetime as dt
import math
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import LineString, Point

from firewatch.agent.actions import ActionExecutor
from firewatch.agent.approvals import decide_approval
from firewatch.agent.case import Case
from firewatch.agent.graph import FireWatchAgent
from firewatch.agent.guard import Guard
from firewatch.agent.planner import LLMPlanner, PolicyPlanner
from firewatch.agent.policy import Decision, decide, segments_at_risk
from firewatch.agent.tools import ToolContext, Toolbox
from firewatch.agent.trace import Tracer
from firewatch.config import Settings
from firewatch.evaluation.common import md_table, write_report
from firewatch.models import AlignmentResult, Hotspot, Kinematics, Projection, Track, TrackPoint
from firewatch.risk.roads import RoadNetwork, RoadSegment, Route
from firewatch.risk.scoring import ConfidenceInputs, assess, confidence
from firewatch.services.kv import SQLiteKV
from firewatch.services.notify import ConsoleNotifier
from firewatch.services.repository import Repository
from firewatch.util import jsonable, new_id, parse_time, utcnow
from firewatch.vision.vlm import VLMAnswer, _verdict

SCENARIO_FILE = Path(__file__).with_name("scenarios") / "agent.yaml"
CRS = "EPSG:32645"
PASS_TIME = parse_time("2026-01-08T04:50:00Z")
ROUTES = [Route("West yard to washery", (-2000.0, 0.0), (2000.0, 0.0)), Route("Pit 4 to washery", (1000.0, 2000.0), (2000.0, 0.0))]
HISTORY = {
    "persistent": {"observable": 4, "present": 4, "persistent": True, "verdict": "persistent", "first_seen": "2025-08-28"},
    "new": {"observable": 3, "present": 0, "persistent": False, "verdict": "new"},
    "intermittent": {"observable": 4, "present": 1, "persistent": False, "verdict": "intermittent"},
    "none": {"observable": 0, "present": 0, "persistent": False, "verdict": "unknown", "note": "no earlier passes in the archive"},
}
HIGHRES = {
    "confirmed": {"checked": True, "scene_id": "S2_SCRIPTED", "hot_pixels": 6, "max_nhi": 0.18, "confirmed": True},
    "not_confirmed": {"checked": True, "scene_id": "S2_SCRIPTED", "hot_pixels": 0, "max_nhi": -0.04, "confirmed": False},
    "unavailable": {"checked": False, "reason": "no clear Sentinel-2 image within 6 days"},
}
REALIGN = {
    "ok": {"method": "disk", "window_px": 640, "ok": True, "inliers": 184, "residual_px": 0.31, "agreement_px": 0.22},
    "fail": {"method": "disk", "window_px": 640, "ok": False, "inliers": 9, "residual_px": None, "agreement_px": None, "reason": "disk could not align the window: only 9 inliers"},
    "disagree": {"method": "disk", "window_px": 640, "ok": False, "inliers": 211, "residual_px": 0.29, "agreement_px": 1.31, "agreement_with_check_px": 0.18},
}


def eval_roads() -> RoadNetwork:
    xs = [-2000.0 + 500.0 * i for i in range(9)]
    segments = []
    for i in range(8):
        segments.append(RoadSegment(f"M-{i:02d}", "Main road", False, LineString([(xs[i], 0.0), (xs[i + 1], 0.0)])))
        segments.append(RoadSegment(f"B-{i:02d}", "Bypass", False, LineString([(xs[i], -1500.0), (xs[i + 1], -1500.0)])))
    segments.append(RoadSegment("C-W", "West connector", False, LineString([(-2000.0, 0.0), (-2000.0, -1500.0)])))
    segments.append(RoadSegment("C-E", "East connector", False, LineString([(2000.0, 0.0), (2000.0, -1500.0)])))
    for i in range(4):
        segments.append(RoadSegment(f"H-{i:02d}", "Haul road, pit 4", True, LineString([(1000.0, 500.0 * i), (1000.0, 500.0 * (i + 1))])))
    return RoadNetwork(segments, CRS)


def circle(cx: float, cy: float, r: float, n: int = 24) -> list[tuple[float, float]]:
    pts = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    return pts + [pts[0]]


class ScriptedVLM:
    model = "scripted"

    def __init__(self, answer: str | None):
        self.answer = answer
        self.available = answer is not None
        self.note = "no vision-language model in this scenario"

    def ask(self, image_bgr, question: str) -> VLMAnswer:
        return VLMAnswer(True, self.answer or "", _verdict(self.answer or ""), 0.4, self.model)


class ScriptedLLM:
    available = True
    model_id = "scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies) or [""]
        self.calls = 0

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str | None:
        if "allowed_tools" not in prompt:
            return None
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


class RoguePlanner:
    name = "rogue"

    def __init__(self, tool: str):
        self.tool = tool

    def choose(self, case: Case, options: list[str]) -> tuple[str, str, str]:
        return self.tool, "act now", self.name

    def write_alert(self, facts: dict) -> str | None:
        return None


class ScriptedToolbox(Toolbox):
    def __init__(self, ctx: ToolContext, world: dict):
        super().__init__(ctx)
        self.world = world

    def fetch_history(self, case: Case) -> dict:
        return copy.deepcopy(HISTORY[self.world.get("history", "none")])

    def fetch_highres(self, case: Case) -> dict:
        return copy.deepcopy(HIGHRES[self.world.get("highres", "unavailable")])

    def refine_alignment(self, case: Case) -> dict:
        return copy.deepcopy(REALIGN[self.world.get("realign", "fail")])


def make_planner(spec: dict | None, settings: Settings):
    spec = spec or {"kind": "policy"}
    if spec["kind"] == "llm":
        return LLMPlanner(settings, ScriptedLLM(spec.get("replies", [])))
    if spec["kind"] == "rogue":
        return RoguePlanner(spec.get("tool", "close_road"))
    return PolicyPlanner(settings)


def build_track(spec: dict, hotspot: Hotspot, approach: float) -> Track:
    t = spec.get("track") or {}
    n = int(t.get("points", 1))
    points = []
    for k in range(n):
        back = 16.0 * (n - 1 - k)
        when = PASS_TIME - dt.timedelta(days=back)
        cx, cy = hotspot.centroid
        points.append(TrackPoint(f"EVAL_{k}", when, f"{hotspot.hotspot_id}-p{k}", hotspot.area_m2, hotspot.mean_anomaly_k, hotspot.max_anomaly_k, (cx, cy + approach * back), hotspot.polygon))
    return Track(track_id=hotspot.track_id, points=points, quiet_passes=int(t.get("quiet_passes", 0)), closed_segments=list(t.get("closed_segments", [])), label=t.get("label"))


def build_case(spec: dict, settings: Settings, roads: RoadNetwork) -> Case:
    h = spec["hotspot"]
    radius = float(h.get("radius_m", 90.0))
    gap = float(h.get("gap_m", 200.0))
    cx, cy = float(h.get("x_m", 250.0)), gap + radius
    area = math.pi * radius ** 2
    hotspot = Hotspot(
        hotspot_id=f"{spec['id']}-h01",
        scene_id=f"EVAL_{spec['id']}",
        acquired=PASS_TIME,
        area_m2=area,
        pixel_count=max(1, round(area / 900.0)),
        mean_anomaly_k=float(h["mean_k"]),
        max_anomaly_k=float(h["max_k"]),
        peak_lst_k=305.0 + float(h["max_k"]),
        centroid=(cx, cy),
        centroid_lonlat=(86.30, 23.75),
        polygon=circle(cx, cy, radius),
        bbox_px=(40, 40, 52, 52),
        solidity=float(h.get("solidity", 0.82)),
        rectangularity=float(h.get("rectangularity", 0.62)),
        elongation=float(h.get("elongation", 1.3)),
        cloud_ring_fraction=float(h.get("cloud_edge", 0.0)),
        industrial_overlap=float(h.get("industrial_overlap", 0.0)),
        track_id=spec.get("track_id", f"T-{spec['id']}"),
    )
    t = spec.get("track") or {}
    n = int(t.get("points", 1))
    approach = float(t.get("approach", 0.0)) if n >= 2 else 0.0
    growth = float(t.get("growth_30d", 0.0)) if n >= 2 else 0.0
    kin = Kinematics(n, 16.0 * (n - 1), growth * area / 30.0, growth, 0.0, -approach, abs(approach), 180.0 if approach > 0 else 0.0, 0.0)
    track = build_track(spec, hotspot, approach)
    projection = None
    if n >= 2 and (approach > 0 or growth > 0) and spec.get("projection", "auto") != "none":
        horizon = settings.projection.horizon_days
        grown = radius * math.sqrt(1.0 + max(growth, 0.0) * horizon / 30.0)
        cone = Point(cx, cy).buffer(radius).union(Point(cx, cy - approach * horizon).buffer(grown)).convex_hull
        _, projected = roads.nearest(cone)
        reach = projected is not None and projected <= 0.0
        days = gap / approach if reach and approach > 0 else None
        cone_poly = [tuple(p) for p in cone.exterior.coords]
        projection = Projection(horizon, cone_poly, cone_poly, 1.0 if reach else 0.0, days, None if projected is None else round(projected, 1), settings.projection.ensemble)
    align = spec.get("alignment") or {}
    inputs = ConfidenceInputs(hotspot.cloud_ring_fraction, bool(align.get("ok", True)), align.get("residual_px", 0.3 if align.get("ok", True) else None), n, hotspot.industrial_overlap, float(spec.get("scene_cloud", 0.05)), alignment_agreement_px=align.get("agreement_px"))
    conf, notes = confidence(inputs, settings.hotspots, settings.alignment)
    risk = assess(hotspot, kin, projection, roads, settings.risk, conf, notes)
    return Case(hotspot, track, kin, None, projection, risk, inputs)


def _risk_view(case: Case) -> dict:
    r = case.risk
    return {"risk": r.score, "level": r.level, "confidence": r.confidence, "distance_m": r.distance_m, "projected_distance_m": r.projected_distance_m}


def _pending_review(track_id: str, level: str, segments: list[str]) -> dict:
    return {"approval_id": new_id("A"), "kind": "review_hotspot", "status": "pending", "created": utcnow().isoformat(), "passes": 1, "track_id": track_id, "hotspot_id": f"{track_id}-earlier", "recommended": "close" if segments else "inspect", "segments": segments, "reason": "earlier pass", "risk": {"level": level}}


def classify(expected: str, actual: str | None) -> str:
    if expected == actual:
        return "match"
    if actual == "close":
        return "false closure"
    if actual in ("log", "exclude") and expected in ("escalate", "close"):
        return "missed"
    return "conservative"


def _checks(expect: dict, out: dict) -> list[tuple[str, bool, str]]:
    checks = []
    if "tools" in expect:
        checks.append(("tools", out["tools"] == expect["tools"], f"expected {expect['tools']}, used {out['tools']}"))
    if "max_tools" in expect:
        checks.append(("max_tools", len(out["tools"]) <= expect["max_tools"], f"used {len(out['tools'])}"))
    if "segments_include" in expect:
        missing = sorted(set(expect["segments_include"]) - set(out["segments"]))
        checks.append(("segments", not missing, f"missing {missing}" if missing else "ok"))
    if "closed" in expect:
        checks.append(("closed", sorted(expect["closed"]) == out["closed"], f"closed {out['closed']}"))
    if "notifications" in expect:
        checks.append(("notifications", out["notifications"] == expect["notifications"], f"sent {out['notifications']}"))
    if "drone" in expect:
        checks.append(("drone", out["drone"] == expect["drone"], f"drone requested: {out['drone']}"))
    if "actions_include" in expect:
        missing = sorted(set(expect["actions_include"]) - {a["kind"] for a in out["actions"]})
        checks.append(("actions", not missing, f"missing {missing}" if missing else "ok"))
    if "pending" in expect:
        ok = all(out["pending"].get(k, 0) == v for k, v in expect["pending"].items())
        checks.append(("pending", ok, f"pending {out['pending']}"))
    if "planner_used" in expect:
        checks.append(("planner", expect["planner_used"] in out["planners"], f"planners {sorted(set(out['planners']))}"))
    return checks


def run_scenario(spec: dict, base: Settings) -> dict:
    settings = base.with_overrides(spec.get("settings") or {})
    roads = eval_roads()
    repo = Repository(SQLiteKV(":memory:"))
    notifier = ConsoleNotifier(None)
    run_id = f"EVAL-{spec['id']}"
    tracer = Tracer(run_id)
    case = build_case(spec, settings, roads)
    pre = spec.get("pre_closed")
    if pre == "auto":
        pre = segments_at_risk(case, roads, settings)
    if pre:
        roads.close(pre)
        for sid in pre:
            repo.put_closure(sid, {"status": "closed", "by": "agent", "since": utcnow().isoformat(), "track_id": case.hotspot.track_id})
        case.track.closed_segments = sorted(set(case.track.closed_segments) | set(pre))
    if spec.get("pending_review"):
        repo.put_approval(_pending_review(case.hotspot.track_id, spec["pending_review"], []))
    world = spec.get("world") or {}
    planner = make_planner(spec.get("planner"), settings)
    ctx = ToolContext(settings=settings, source=None, grid=None, registrar=None, alignment=AlignmentResult("aliked_lightglue", bool((spec.get("alignment") or {}).get("ok", True)), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), roads=roads, vlm=ScriptedVLM(world.get("vlm")), repo=repo, run_id=run_id, acquired=PASS_TIME, false_color=np.zeros((64, 64, 3), np.uint8))
    executor = ActionExecutor(settings, roads, ROUTES, repo, notifier, Guard(settings), tracer, run_id, planner, PASS_TIME)
    baseline = decide(case, settings, roads)
    before = _risk_view(case)
    closure_tracks = [case.track] if case.track.closed_segments else []
    FireWatchAgent(settings, {case.case_id: case}, ScriptedToolbox(ctx, world), planner, executor, tracer, roads, closure_tracks).run()
    pending: dict[str, int] = {}
    for a in repo.list_approvals("pending"):
        pending[a["kind"]] = pending.get(a["kind"], 0) + 1
    out = {
        "id": spec["id"],
        "group": spec.get("group", ""),
        "name": spec["name"],
        "expected": spec["expect"]["decision"],
        "decision": case.decision,
        "reason": case.decision_reason,
        "baseline": baseline.kind,
        "tools": list(case.tried),
        "segments": list(case.segments),
        "closed": sorted(roads.closed),
        "actions": [{"kind": r.kind, "actor": r.actor, "target": r.target, "executed": r.executed} for r in executor.records],
        "notifications": len(notifier.sent),
        "drone": any(t.get("status") == "requested" for t in repo.list_tasks()),
        "pending": pending,
        "planners": [e.observation.get("planner") for e in tracer.events if e.node == "plan" and e.observation.get("planner")],
        "before": before,
        "after": _risk_view(case),
        "note": spec.get("note"),
        "trace": tracer.as_dicts(),
    }
    out["outcome"] = classify(out["expected"], out["decision"])
    out["checks"] = [{"check": name, "passed": ok, "detail": detail} for name, ok, detail in _checks(spec["expect"], out)]
    out["passed"] = out["outcome"] == "match" and all(c["passed"] for c in out["checks"])
    return out


def _closure_world(settings: Settings, quiet: int):
    roads = eval_roads()
    repo = Repository(SQLiteKV(":memory:"))
    notifier = ConsoleNotifier(None)
    roads.close(["M-04"])
    repo.put_closure("M-04", {"status": "closed", "by": "agent", "since": utcnow().isoformat(), "track_id": "T-R1"})
    track = Track("T-R1", points=[], quiet_passes=quiet, closed_segments=["M-04"])
    repo.put_track(jsonable(track))
    return roads, repo, notifier, track


def _review_pass(settings: Settings, roads, repo, notifier, track) -> ActionExecutor:
    run_id = new_id("EVAL-R")
    tracer = Tracer(run_id)
    planner = PolicyPlanner(settings)
    executor = ActionExecutor(settings, roads, ROUTES, repo, notifier, Guard(settings), tracer, run_id, planner, PASS_TIME)
    FireWatchAgent(settings, {}, ScriptedToolbox(None, {}), planner, executor, tracer, roads, [track]).run()
    return executor


def sequence_checks(settings: Settings) -> list[dict]:
    rows = []

    def add(cid: str, name: str, passed: bool, detail: str) -> None:
        rows.append({"id": cid, "name": name, "passed": bool(passed), "detail": detail})

    roads, repo, notifier, track = _closure_world(settings, quiet=2)
    _review_pass(settings, roads, repo, notifier, track)
    reopen = [a for a in repo.list_approvals("pending") if a["kind"] == "reopen_road"]
    add("S01", "Two quiet passes: the agent proposes reopening", len(reopen) == 1, f"{len(reopen)} reopen proposal(s)")
    add("S02", "The road stays closed while the proposal waits", "M-04" in roads.closed, f"closed: {sorted(roads.closed)}")
    executor = _review_pass(settings, roads, repo, notifier, track)
    again = [a for a in repo.list_approvals("pending") if a["kind"] == "reopen_road"]
    add("S03", "A later review does not duplicate the proposal", len(again) == 1, f"{len(again)} pending after a second review")
    record = executor.try_agent_reopen(["M-04"], track.track_id)
    add("S04", "The agent cannot reopen a road itself", (not record.executed) and record.kind == "blocked" and "M-04" in roads.closed, f"{record.kind}: {record.reason}")
    decided = decide_approval(again[0]["approval_id"], True, "safety-officer", "site inspected", repo=repo, roads=roads, routes=ROUTES, guard=Guard(settings), notifier=notifier)
    reroute = decided["outcome"].get("reroute") or []
    add("S05", "A person's approval reopens it and restores the normal route", "M-04" not in roads.closed and all(not r.get("extra_m") for r in reroute), f"closed: {sorted(roads.closed)}; reroute {reroute}")

    roads, repo, notifier, track = _closure_world(settings, quiet=1)
    _review_pass(settings, roads, repo, notifier, track)
    pending = [a for a in repo.list_approvals("pending") if a["kind"] == "reopen_road"]
    add("S06", "One quiet pass is not enough to propose reopening", not pending and "M-04" in roads.closed, f"{len(pending)} proposal(s)")

    roads = eval_roads()
    repo = Repository(SQLiteKV(":memory:"))
    notifier = ConsoleNotifier(None)
    review = _pending_review("T-H1", "medium", ["M-04"])
    repo.put_approval(review)
    decide_approval(review["approval_id"], True, "safety-officer", "agree", repo=repo, roads=roads, routes=ROUTES, guard=Guard(settings), notifier=notifier)
    actions = repo.list_actions()
    add("S07", "Approving a review that recommends closure closes the road, recorded as a human action", "M-04" in roads.closed and any(a["actor"] == "human" and a["kind"] == "close_road" for a in actions), f"closed: {sorted(roads.closed)}")

    spec = {"id": "S08", "name": "", "hotspot": {"gap_m": 120, "mean_k": 9.0, "max_k": 16.0}, "track": {"points": 3, "growth_30d": 0.3, "approach": 2.0}, "world": {"highres": "confirmed"}, "pending_review": "medium", "expect": {"decision": "close"}}
    out = run_scenario(spec, settings)
    add("S08", "An automatic closure supersedes the review waiting for that track", out["decision"] == "close" and not out["pending"].get("review_hotspot"), f"decision {out['decision']}, pending {out['pending']}")

    for cid, name, mutate in (
        ("S09", "The guard refuses a closure for a medium-risk case", lambda c: setattr(c.risk, "level", "medium")),
        ("S10", "The guard refuses a closure below the confidence threshold", lambda c: setattr(c.risk, "confidence", 0.5)),
    ):
        roads = eval_roads()
        repo = Repository(SQLiteKV(":memory:"))
        tracer = Tracer("EVAL-G")
        case = build_case({"id": cid, "hotspot": {"gap_m": 120, "mean_k": 9.0, "max_k": 16.0}, "track": {"points": 3, "growth_30d": 0.3, "approach": 2.0}}, settings, roads)
        mutate(case)
        case.decision = "close"
        executor = ActionExecutor(settings, roads, ROUTES, repo, ConsoleNotifier(None), Guard(settings), tracer, "EVAL-G", PolicyPlanner(settings), PASS_TIME)
        executor.execute(case, Decision("close", "forced for the test", ["M-04"]))
        kinds = [r.kind for r in executor.records]
        add(cid, name, not roads.closed and "blocked" in kinds and "escalate" in kinds, f"actions {kinds}")
    return rows


def load_scenarios(path: Path | None = None) -> list[dict]:
    return yaml.safe_load((path or SCENARIO_FILE).read_text())


def perturbations(settings: Settings) -> list[tuple[str, dict]]:
    r = settings.risk
    names = {"anomaly": "Heat weight", "spread": "Spread weight", "proximity": "Proximity weight"}
    out = []
    for key, weight in r.weights.items():
        for factor in (0.8, 1.2):
            out.append((f"{names.get(key, key)} {factor - 1:+.0%}", {"risk": {"weights": {key: round(weight * factor, 4)}}}))
    for key, label in (("high", "High-risk threshold"), ("medium", "Medium-risk threshold"), ("min_confidence_auto", "Confidence needed to close")):
        for delta in (-0.05, 0.05):
            out.append((f"{label} {delta:+.2f}", {"risk": {key: round(getattr(r, key) + delta, 4)}}))
    for factor in (0.75, 1.25):
        out.append((f"Proximity scale d0 {factor - 1:+.0%} ({r.d0_m * factor:g} m)", {"risk": {"d0_m": r.d0_m * factor}}))
    return out


def sensitivity(settings: Settings, scenarios: list[dict], baseline: list[dict]) -> list[dict]:
    rows = []
    for label, override in perturbations(settings):
        tweaked = settings.with_overrides(override)
        results = [run_scenario(spec, tweaked) for spec in scenarios]
        changed = [f"{r['id']} {b['decision']} to {r['decision']}" for r, b in zip(results, baseline) if r["decision"] != b["decision"]]
        rows.append({
            "change": label,
            "changed": changed,
            "matches": sum(r["outcome"] == "match" for r in results),
            "false_closures": sum(r["outcome"] == "false closure" for r in results),
            "missed": sum(r["outcome"] == "missed" for r in results),
        })
    return rows


def evaluate(settings: Settings) -> dict:
    scenarios = load_scenarios()
    results = [run_scenario(spec, settings) for spec in scenarios]
    sequences = sequence_checks(settings)
    n = len(results)
    changed = [r for r in results if r["baseline"] != r["decision"]]
    summary = {
        "scenarios": n,
        "decision_matches": sum(r["outcome"] == "match" for r in results),
        "all_checks_pass": sum(r["passed"] for r in results),
        "false_closures": sum(r["outcome"] == "false closure" for r in results),
        "missed": sum(r["outcome"] == "missed" for r in results),
        "conservative": sum(r["outcome"] == "conservative" for r in results),
        "baseline_matches": sum(r["baseline"] == r["expected"] for r in results),
        "changed_by_evidence": len(changed),
        "changed_for_better": sum(r["decision"] == r["expected"] and r["baseline"] != r["expected"] for r in changed),
        "changed_for_worse": sum(r["baseline"] == r["expected"] and r["decision"] != r["expected"] for r in changed),
        "mean_tools": sum(len(r["tools"]) for r in results) / n if n else 0.0,
        "sequence_checks": len(sequences),
        "sequence_passed": sum(s["passed"] for s in sequences),
    }
    return {"summary": summary, "scenarios": results, "sequences": sequences, "sensitivity": sensitivity(settings, scenarios, results)}


def _trace_table(result: dict) -> str:
    rows = []
    for e in result["trace"]:
        if e["node"] not in ("plan", "tool", "decide", "act"):
            continue
        obs = e.get("observation") or {}
        if e["node"] == "tool":
            b, a = obs.get("before", {}), obs.get("after", {})
            found = ", ".join(f"{k} {v}" for k, v in (e.get("result") or {}).items() if k in ("verdict", "confirmed", "ok", "persistent"))
            what = f"{e['tool']}: {found}"
            detail = f"confidence {b.get('confidence', 0):.2f} to {a.get('confidence', 0):.2f}, risk {a.get('risk', 0):.2f}"
        elif e["node"] == "plan":
            what = f"plan: {e.get('tool') or e.get('decision')}"
            detail = e.get("rationale") or ""
        elif e["node"] == "decide":
            what = f"decide: {e.get('decision')}"
            detail = e.get("rationale") or ""
        else:
            what = "act"
            detail = ", ".join(f"{a['actor']} {a['kind']} {' '.join(a['target'])}".strip() for a in (e.get("result") or {}).get("actions", []))
        rows.append([e["seq"], what, detail])
    return md_table(["#", "Step", "Detail"], rows)


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> Path:
    result = evaluate(settings)
    s = result["summary"]
    rows = []
    for r in result["scenarios"]:
        failed = [c["check"] for c in r["checks"] if not c["passed"]]
        status = "pass" if r["passed"] else ("decision ok, " + ", ".join(failed) + " off" if r["outcome"] == "match" else r["outcome"])
        rows.append([r["id"], r["name"], r["expected"], r["decision"], r["baseline"], ", ".join(r["tools"]) or "-", status])
    md = [
        "# Agent decisions on scripted scenarios",
        "",
        f"{s['scenarios']} single-pass scenarios run through the real LangGraph agent, decision policy, guard and action executor. "
        "Only the tool results are scripted (what older passes, Sentinel-2 and another alignment method would show), so the scenarios isolate the decision logic from image processing. "
        "Expected decisions describe what a careful safety officer would want. \"Without tools\" is the decision the same policy makes on the first measurements, before any evidence gathering.",
        "",
        md_table(["Measure", "Value"], [
            ["Decision matches the expected one", f"{s['decision_matches']} of {s['scenarios']}"],
            ["Matches and every secondary check passes (tools, segments, alerts)", f"{s['all_checks_pass']} of {s['scenarios']}"],
            ["Closures the expected decision did not call for", s["false_closures"]],
            ["Fires dismissed that needed a person or a closure", s["missed"]],
            ["More cautious than expected (sent to a person instead)", s["conservative"]],
            ["Matches without tools (first measurements only)", f"{s['baseline_matches']} of {s['scenarios']}"],
            ["Decisions changed by gathered evidence", f"{s['changed_by_evidence']} ({s['changed_for_better']} corrected, {s['changed_for_worse']} made worse)"],
            ["Tool calls per case", f"{s['mean_tools']:.2f}"],
            ["Safety and memory checks passed", f"{s['sequence_passed']} of {s['sequence_checks']}"],
        ]),
        "",
        "## Scenarios",
        "",
        md_table(["ID", "Situation", "Expected", "Agent", "Without tools", "Tools used", "Result"], rows),
        "",
    ]
    failures = [r for r in result["scenarios"] if not r["passed"]]
    md += ["## Failure cases", ""]
    if failures:
        for r in failures:
            md.append(f"**{r['id']}: {r['name']}.** Expected {r['expected']}, the agent chose {r['decision']} ({r['reason']}).")
            off = [f"{c['check']}: {c['detail']}" for c in r["checks"] if not c["passed"]]
            if off:
                md.append("Checks off: " + "; ".join(off) + ".")
            if r.get("note"):
                md.append(r["note"])
            md.append("")
    else:
        md += ["None in this scenario set.", ""]
    md += ["## Safety and memory checks", "", md_table(["ID", "Check", "Result", "Detail"], [[q["id"], q["name"], "pass" if q["passed"] else "FAIL", q["detail"]] for q in result["sequences"]]), ""]
    sens = result["sensitivity"]
    md += [
        "## How much do the constants matter?",
        "",
        f"Every scenario again, with one risk constant moved at a time: each weight by 20 %, each threshold by 0.05 and the proximity scale by 25 %. A decision that flips under such a small change sits close to a boundary. "
        f"Across the {len(sens)} variations, {sum(1 for x in sens if not x['changed'])} change nothing, and the most any of them changes is {max((len(x['changed']) for x in sens), default=0)} of {s['scenarios']} decisions.",
        "",
        md_table(["Change", "Decisions that change", "Still as expected", "Closures not called for", "Fires missed", "Which"], [[x["change"], len(x["changed"]), f"{x['matches']} of {s['scenarios']}", x["false_closures"], x["missed"], "; ".join(x["changed"]) or "-"] for x in sens]),
        "",
    ]
    sample = next((r for r in result["scenarios"] if r["id"] == "A07"), None)
    if sample:
        md += [
            "## One trace, step by step (A07)",
            "",
            "A first sighting 10 m from the road on a pass where image alignment failed. Each tool result changes the confidence, and the confidence decides whether the agent may act by itself.",
            "",
            _trace_table(sample),
            "",
        ]
    return write_report(out_dir, "agent", result, "\n".join(md))

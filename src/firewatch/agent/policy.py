from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon

from firewatch.agent.case import Case
from firewatch.config import Settings
from firewatch.risk.roads import RoadNetwork, hotspot_geometry

TOOLS = ("check_industrial", "fetch_history", "refine_alignment", "fetch_highres", "request_drone")


def candidate_tools(case: Case, settings: Settings) -> list[str]:
    if case.excluded or case.steps >= settings.agent.max_steps_per_hotspot:
        return []
    f = case.flags(settings)
    options: list[str] = []
    if f["industrial_candidate"] and "check_industrial" not in case.tried:
        options.append("check_industrial")
    if f["new"] and "fetch_history" not in case.tried:
        options.append("fetch_history")
    if f["low_alignment"] and "refine_alignment" not in case.tried:
        options.append("refine_alignment")
    relevant = case.risk.level in ("high", "medium") or f["near_road"]
    if relevant:
        if "fetch_highres" not in case.tried:
            options.append("fetch_highres")
        elif f["approaching"] and f["near_road"] and not case.evidence.get("highres", {}).get("confirmed") and "request_drone" not in case.tried:
            options.append("request_drone")
    return options


@dataclass
class Decision:
    kind: str
    reason: str
    segments: list[str] = field(default_factory=list)


def join_names(names: list[str]) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def segments_at_risk(case: Case, roads: RoadNetwork, settings: Settings) -> list[str]:
    if case.projection and len(case.projection.cone_polygon) >= 4:
        geom = Polygon(case.projection.cone_polygon).buffer(0).union(hotspot_geometry(case.hotspot.polygon, case.hotspot.centroid))
    else:
        geom = hotspot_geometry(case.hotspot.polygon, case.hotspot.centroid)
    return roads.within(geom, settings.risk.close_buffer_m, haul_only=settings.roads.haul_only_closures)


def decide(case: Case, settings: Settings, roads: RoadNetwork) -> Decision:
    r = case.risk
    f = case.flags(settings)
    cfg = settings.risk
    if case.excluded:
        reasons = case.evidence.get("industrial", {}).get("reasons") or ["industrial heat source"]
        return Decision("exclude", "not a coal fire: " + "; ".join(reasons))
    segments = segments_at_risk(case, roads, settings)
    if f["cloud_edge"] and (r.level != "low" or f["near_road"]):
        return Decision("escalate", "partly under cloud, so its true extent is unknown; marked unknown rather than safe", segments)
    if r.level == "high":
        if not settings.agent.allow_auto_close:
            return Decision("escalate", "high risk; automatic closure is switched off", segments)
        if not segments:
            return Decision("escalate", f"high risk ({r.score:.2f}) but no road segment within {cfg.close_buffer_m:.0f} m of the projected spread", segments)
        if r.confidence < cfg.min_confidence_auto:
            return Decision("escalate", f"high risk ({r.score:.2f}) but confidence {r.confidence:.2f} is below {cfg.min_confidence_auto}", segments)
        public = [s for s in segments if s in roads.segments and not roads.segments[s].haul]
        if settings.roads.auto_close == "haul" and public:
            names = sorted({roads.segments[s].name for s in public})
            return Decision("escalate", f"high risk ({r.score:.2f}) next to public {'road' if len(names) == 1 else 'roads'} {join_names(names)}; the agent closes mine haul roads by itself, but closing a public road is a person's call", segments)
        current = roads.within(hotspot_geometry(case.hotspot.polygon, case.hotspot.centroid), cfg.close_buffer_m, haul_only=settings.roads.haul_only_closures)
        if not current and not all(s in roads.closed for s in segments):
            where = f"{r.distance_m:.0f} m" if r.distance_m is not None else "too far"
            return Decision("escalate", f"high risk ({r.score:.2f}) comes from the {settings.projection.horizon_days:.0f}-day projection; the fire itself is {where} from the nearest road, so a pre-emptive closure is a person's call", segments)
        where = f"the fire is within {cfg.close_buffer_m:.0f} m of {join_names(current)}" if current else "the segments it threatens are already closed"
        return Decision("close", f"risk {r.score:.2f} and confidence {r.confidence:.2f} clear the bars of {cfg.high} and {cfg.min_confidence_auto}; {where}, and with its {settings.projection.horizon_days:.0f}-day spread {join_names(segments)} {'is' if len(segments) == 1 else 'are'} at risk", segments)
    if r.level == "medium":
        return Decision("escalate", f"medium risk ({r.score:.2f}); a person should review", segments)
    if r.confidence < 0.5 and f["near_road"]:
        return Decision("escalate", f"low confidence ({r.confidence:.2f}) close to a road", segments)
    return Decision("log", f"risk {r.score:.2f} is below {cfg.medium}; keep watching")

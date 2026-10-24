from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from firewatch.agent.actions import ActionExecutor
from firewatch.agent.case import Case
from firewatch.agent.guard import Guard
from firewatch.agent.policy import Decision, candidate_tools, decide
from firewatch.agent.tools import Toolbox, rescore
from firewatch.agent.trace import Tracer
from firewatch.config import Settings
from firewatch.models import Track
from firewatch.risk.roads import RoadNetwork
from firewatch.util import jsonable

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    queue: list[str]
    current: str | None
    pending_tool: str | None
    done: list[str]


def _risk_view(case: Case) -> dict:
    r = case.risk
    return {"risk": r.score, "level": r.level, "confidence": r.confidence, "distance_m": r.distance_m, "projected_distance_m": r.projected_distance_m}


def _brief(tool: str, result: dict) -> dict:
    if tool == "fetch_history":
        return {k: result.get(k) for k in ("verdict", "observable", "present", "first_seen", "note") if k in result}
    if tool == "fetch_highres":
        return {k: result.get(k) for k in ("checked", "scene_id", "hot_pixels", "max_nhi", "confirmed", "reason") if k in result}
    return {k: v for k, v in result.items() if k not in ("passes",)}


class FireWatchAgent:
    def __init__(self, settings: Settings, cases: dict[str, Case], toolbox: Toolbox, planner, executor: ActionExecutor, tracer: Tracer, roads: RoadNetwork, closure_tracks: list[Track] | None = None, rescore_fn=None):
        self.rescore_fn = rescore_fn or rescore
        self.settings = settings
        self.cases = cases
        self.toolbox = toolbox
        self.planner = planner
        self.executor = executor
        self.tracer = tracer
        self.roads = roads
        self.guard = Guard(settings)
        self.closure_tracks = closure_tracks or []
        self.app = self._build()

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("next_case", self.next_case)
        g.add_node("plan", self.plan)
        g.add_node("use_tool", self.use_tool)
        g.add_node("decide", self.decide)
        g.add_node("act", self.act)
        g.add_node("review_closures", self.review_closures)
        g.add_edge(START, "next_case")
        g.add_conditional_edges("next_case", lambda s: "plan" if s.get("current") else "review_closures", {"plan": "plan", "review_closures": "review_closures"})
        g.add_conditional_edges("plan", lambda s: "use_tool" if s.get("pending_tool") else "decide", {"use_tool": "use_tool", "decide": "decide"})
        g.add_edge("use_tool", "plan")
        g.add_edge("decide", "act")
        g.add_edge("act", "next_case")
        g.add_edge("review_closures", END)
        return g.compile()

    def run(self) -> dict:
        queue = sorted(self.cases, key=lambda cid: -self.cases[cid].risk.score)
        limit = len(queue) * (2 * self.settings.agent.max_steps_per_hotspot + 8) + 20
        final = self.app.invoke({"queue": queue, "done": [], "current": None, "pending_tool": None}, config={"recursion_limit": limit})
        return {"processed": final.get("done", []), "events": len(self.tracer.events)}

    def next_case(self, state: AgentState) -> AgentState:
        queue = list(state.get("queue", []))
        if not queue:
            return {"current": None, "pending_tool": None}
        cid = queue.pop(0)
        case = self.cases[cid]
        self.tracer.emit("observe", cid, observation=case.summary(self.settings))
        return {"queue": queue, "current": cid, "pending_tool": None}

    def plan(self, state: AgentState) -> AgentState:
        case = self.cases[state["current"]]
        options = candidate_tools(case, self.settings)
        if not options:
            self.tracer.emit("plan", case.case_id, observation=_risk_view(case), decision="enough evidence", rationale="no further tool would change the decision" if case.steps else "measurements are sufficient as observed")
            return {"pending_tool": None}
        tool, rationale, by = self.planner.choose(case, options)
        if not self.guard.check_tool(tool, options):
            tool, rationale, by = options[0], f"planner chose an unavailable tool; using {options[0]}", "guard"
        self.tracer.emit("plan", case.case_id, observation={"options": options, "planner": by, **_risk_view(case)}, tool=tool, rationale=rationale)
        return {"pending_tool": tool}

    def use_tool(self, state: AgentState) -> AgentState:
        case = self.cases[state["current"]]
        tool = state["pending_tool"]
        before = _risk_view(case)
        result = self.toolbox.run(tool, case)
        self.rescore_fn(case, self.settings, self.roads)
        self.tracer.emit("tool", case.case_id, tool=tool, result=_brief(tool, result), observation={"before": before, "after": _risk_view(case), "excluded": case.excluded})
        return {"pending_tool": None}

    def decide(self, state: AgentState) -> AgentState:
        case = self.cases[state["current"]]
        decision = decide(case, self.settings, self.roads)
        case.decision = decision.kind
        case.decision_reason = decision.reason
        case.segments = decision.segments
        self.tracer.emit("decide", case.case_id, decision=decision.kind, rationale=decision.reason, observation={**_risk_view(case), "segments": decision.segments, "evidence": list(case.evidence)})
        return {}

    def act(self, state: AgentState) -> AgentState:
        case = self.cases[state["current"]]
        records = self.executor.execute(case, Decision(case.decision or "log", case.decision_reason, case.segments))
        self.tracer.emit("act", case.case_id, decision=case.decision, result={"actions": [{"kind": r.kind, "actor": r.actor, "target": r.target, "executed": r.executed} for r in records]})
        done = list(state.get("done", [])) + [case.case_id]
        return {"done": done, "current": None}

    def review_closures(self, state: AgentState) -> AgentState:
        quiet_needed = self.settings.roads.reopen_quiet_passes
        for track in self.closure_tracks:
            segments = [s for s in track.closed_segments if s in self.roads.closed]
            if not segments:
                continue
            if track.quiet_passes >= quiet_needed:
                reason = f"track {track.track_id} has stayed below medium risk (or cooled) for {track.quiet_passes} passes"
                approval = self.executor.propose_reopen(track, segments, reason)
                self.tracer.emit("review_closures", None, observation={"track_id": track.track_id, "quiet_passes": track.quiet_passes, "segments": segments}, decision="propose_reopen" if approval else "reopen already pending", rationale=reason + "; reopening itself needs a human")
            else:
                self.tracer.emit("review_closures", None, observation={"track_id": track.track_id, "quiet_passes": track.quiet_passes, "segments": segments}, decision="keep closed", rationale=f"needs {quiet_needed} quiet passes before a reopen is proposed")
        return {}


def run_agent(*args, **kwargs) -> dict:
    agent = FireWatchAgent(*args, **kwargs)
    return jsonable(agent.run())

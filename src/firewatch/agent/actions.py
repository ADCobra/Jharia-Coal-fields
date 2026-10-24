from __future__ import annotations

import logging
from urllib.parse import quote

from firewatch.agent.case import Case
from firewatch.agent.guard import Guard
from firewatch.agent.policy import Decision
from firewatch.agent.trace import Tracer
from firewatch.config import Settings
from firewatch.models import ActionRecord, Track
from firewatch.risk.roads import RoadNetwork, Route
from firewatch.services.notify import Notifier
from firewatch.services.repository import Repository
from firewatch.util import jsonable, new_id, utcnow

log = logging.getLogger(__name__)


def alert_facts(case: Case, decision: Decision, reroute: list[dict] | None, settings: Settings, run_id: str) -> dict:
    h, r = case.hotspot, case.risk
    return {
        "hotspot": h.hotspot_id,
        "track": h.track_id,
        "pass": h.scene_id,
        "date": h.acquired.date().isoformat(),
        "lat": round(h.centroid_lonlat[1], 5),
        "lon": round(h.centroid_lonlat[0], 5),
        "risk": r.score,
        "level": r.level,
        "mean_anomaly_k": round(h.mean_anomaly_k, 1),
        "distance_m": r.distance_m,
        "projected_distance_m": r.projected_distance_m,
        "nearest_segment": r.nearest_segment,
        "approach_m_per_day": r.approach_m_per_day,
        "confidence": r.confidence,
        "confidence_notes": r.confidence_notes,
        "decision": decision.kind,
        "decision_reason": decision.reason,
        "segments": decision.segments,
        "reroute": reroute or [],
        "link": f"{settings.agent.dashboard_url.rstrip('/')}/#run={quote(run_id, safe='')}&hotspot={quote(h.hotspot_id, safe='')}",
    }


def template_alert(facts: dict) -> str:
    lines = [f"Hotspot {facts['track'] or facts['hotspot']} at {facts['lat']}, {facts['lon']} (pass {facts['pass']}, {facts['date']})."]
    dist = f"{facts['distance_m']:.0f} m" if facts["distance_m"] is not None else "unknown distance"
    line = f"Risk {facts['risk']:.2f} ({facts['level']}), anomaly {facts['mean_anomaly_k']} K, {dist} from {facts['nearest_segment']}"
    if facts["projected_distance_m"] is not None:
        line += f", projected {facts['projected_distance_m']:.0f} m"
    lines.append(line + ".")
    if facts["approach_m_per_day"] and facts["approach_m_per_day"] > 0.5:
        lines.append(f"Moving toward the road at {facts['approach_m_per_day']:.1f} m/day.")
    if facts["confidence_notes"]:
        lines.append(f"Confidence {facts['confidence']:.2f}: " + "; ".join(facts["confidence_notes"]) + ".")
    if facts["decision"] == "close":
        lines.append(f"Closed by the agent: {', '.join(facts['segments'])}. Reopening needs your approval.")
    elif facts["decision"] == "escalate":
        lines.append(f"Needs your review: {facts['decision_reason']}.")
    for item in facts["reroute"]:
        if item.get("blocked"):
            lines.append(f"Route '{item['route']}' is blocked with these closures.")
        elif item.get("extra_m"):
            lines.append(f"Route '{item['route']}' rerouted: +{item['extra_m']} m.")
    lines.append(f"Evidence and trace: {facts['link']}")
    return "\n".join(lines)


class ActionExecutor:
    def __init__(self, settings: Settings, roads: RoadNetwork, routes: list[Route], repo: Repository, notifier: Notifier, guard: Guard, tracer: Tracer, run_id: str, planner=None, acquired=None):
        self.acquired = acquired
        self.settings = settings
        self.roads = roads
        self.routes = routes
        self.repo = repo
        self.notifier = notifier
        self.guard = guard
        self.tracer = tracer
        self.run_id = run_id
        self.planner = planner
        self.records: list[ActionRecord] = []

    def _record(self, kind: str, actor: str, target: list[str], reason: str, executed: bool, case: Case | None = None, details: dict | None = None, track_id: str | None = None) -> ActionRecord:
        record = ActionRecord(
            action_id=new_id("X"),
            kind=kind,
            actor=actor,
            target=list(target),
            hotspot_id=case.hotspot.hotspot_id if case else None,
            track_id=case.hotspot.track_id if case else track_id,
            reason=reason,
            executed=executed,
            created=utcnow(),
            details=jsonable(details or {}),
        )
        self.records.append(record)
        if case is not None:
            case.actions.append(record)
        self.repo.put_action(self.run_id, jsonable(record))
        return record

    def _message(self, facts: dict) -> str:
        text = self.planner.write_alert(facts) if self.planner is not None else None
        return text.strip() if text else template_alert(facts)

    def execute(self, case: Case, decision: Decision) -> list[ActionRecord]:
        before = len(case.actions)
        if decision.kind == "close":
            self.close_roads(case, decision)
        elif decision.kind == "escalate":
            self.escalate(case, decision)
        elif decision.kind == "exclude":
            self._record("exclude", "agent", [], decision.reason, True, case)
        else:
            self._record("log", "agent", [], decision.reason, True, case)
        return case.actions[before:]

    def close_roads(self, case: Case, decision: Decision) -> None:
        allowed, why = self.guard.check("close_road", "agent", case)
        if not allowed:
            self._record("blocked", "agent", decision.segments, why, False, case, {"attempted": "close_road"})
            self.escalate(case, Decision("escalate", f"closure blocked by guard: {why}", decision.segments))
            return
        new_segments = [s for s in decision.segments if s not in self.roads.closed]
        if not new_segments:
            if case.track is not None:
                case.track.closed_segments = sorted(set(case.track.closed_segments) | set(decision.segments))
                case.track.quiet_passes = 0
            self._record("keep_closed", "agent", decision.segments, f"{decision.reason}; segments already closed, no new alert", True, case)
            return
        self.roads.close(new_segments)
        now = utcnow().isoformat()
        review = self._open_review(case.hotspot.track_id)
        if review is not None:
            review.update({"status": "superseded", "decided_by": "agent", "decided": now, "note": f"the agent closed {', '.join(new_segments)} automatically"})
            self.repo.put_approval(review)
        for sid in new_segments:
            self.repo.put_closure(sid, {"status": "closed", "by": "agent", "since": now, "reason": decision.reason, "hotspot_id": case.hotspot.hotspot_id, "track_id": case.hotspot.track_id, "run_id": self.run_id})
        if case.track is not None:
            case.track.closed_segments = sorted(set(case.track.closed_segments) | set(decision.segments))
            case.track.quiet_passes = 0
        reroute = self.roads.reroute_report(self.routes)
        self._record("close_road", "agent", decision.segments, decision.reason, True, case, {"newly_closed": new_segments, "reroute": reroute})
        facts = alert_facts(case, decision, reroute, self.settings, self.run_id)
        sent = self.notifier.send(f"Road closure near {case.hotspot.track_id or case.hotspot.hotspot_id}: {', '.join(decision.segments[:4])}", self._message(facts), "critical", facts)
        self._record("notify", "agent", ["safety-officer"], "closure alert", True, case, {"channel": sent.get("channel"), "subject": sent.get("subject")})

    def _open_review(self, track_id: str | None) -> dict | None:
        if not track_id:
            return None
        for approval in self.repo.list_approvals("pending"):
            if approval.get("kind") == "review_hotspot" and approval.get("track_id") == track_id:
                return approval
        return None

    def escalate(self, case: Case, decision: Decision) -> None:
        rank = {"low": 0, "medium": 1, "high": 2}
        existing = self._open_review(case.hotspot.track_id)
        fields = {
            "run_id": self.run_id,
            "pass_date": case.hotspot.acquired.isoformat(),
            "hotspot_id": case.hotspot.hotspot_id,
            "track_id": case.hotspot.track_id,
            "recommended": "close" if decision.segments else "inspect",
            "segments": decision.segments,
            "reason": decision.reason,
            "risk": jsonable(case.risk),
        }
        if existing is not None:
            previous_level = (existing.get("risk") or {}).get("level", "low")
            existing.update(fields)
            existing["updated"] = utcnow().isoformat()
            existing["passes"] = int(existing.get("passes", 1)) + 1
            self.repo.put_approval(existing)
            self._record("escalate", "agent", decision.segments, decision.reason, True, case, {"approval_id": existing["approval_id"], "refreshed": True})
            if rank.get(case.risk.level, 0) <= rank.get(previous_level, 0):
                return
            approval = existing
        else:
            approval = {"approval_id": new_id("A"), "kind": "review_hotspot", "status": "pending", "created": utcnow().isoformat(), "passes": 1, **fields}
            self.repo.put_approval(approval)
            self._record("escalate", "agent", decision.segments, decision.reason, True, case, {"approval_id": approval["approval_id"]})
        facts = alert_facts(case, decision, None, self.settings, self.run_id)
        sent = self.notifier.send(f"Review needed: hotspot {case.hotspot.track_id or case.hotspot.hotspot_id}", self._message(facts), "warning", facts)
        self._record("notify", "agent", ["safety-officer"], "review request", True, case, {"channel": sent.get("channel"), "approval_id": approval["approval_id"]})

    def propose_reopen(self, track: Track, segments: list[str], reason: str) -> dict | None:
        pending = [a for a in self.repo.list_approvals("pending") if a.get("kind") == "reopen_road" and a.get("track_id") == track.track_id]
        if pending:
            return None
        approval = {
            "approval_id": new_id("A"),
            "kind": "reopen_road",
            "status": "pending",
            "created": utcnow().isoformat(),
            "pass_date": self.acquired.isoformat() if self.acquired else None,
            "run_id": self.run_id,
            "track_id": track.track_id,
            "segments": segments,
            "reason": reason,
        }
        self.repo.put_approval(approval)
        self._record("propose_reopen", "agent", segments, reason, True, None, {"approval_id": approval["approval_id"]}, track_id=track.track_id)
        self.notifier.send(f"Reopen proposed: {', '.join(segments[:4])}", f"{reason}\nApprove or reject in the dashboard: {self.settings.agent.dashboard_url}", "info", approval)
        return approval

    def try_agent_reopen(self, segments: list[str], track_id: str | None = None) -> ActionRecord:
        allowed, why = self.guard.check("reopen_road", "agent")
        if allowed:
            self.roads.reopen(segments)
        return self._record("reopen_road" if allowed else "blocked", "agent", segments, why or "reopen", allowed, None, {"attempted": "reopen_road"}, track_id=track_id)

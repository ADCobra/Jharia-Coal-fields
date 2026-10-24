from __future__ import annotations

from firewatch.agent.guard import Guard
from firewatch.risk.roads import RoadNetwork, Route
from firewatch.services.notify import Notifier
from firewatch.services.repository import Repository
from firewatch.util import new_id, utcnow


class ApprovalError(Exception):
    pass


def decide_approval(approval_id: str, approve: bool, user: str, note: str, *, repo: Repository, roads: RoadNetwork, routes: list[Route], guard: Guard, notifier: Notifier) -> dict:
    approval = repo.get_approval(approval_id)
    if approval is None:
        raise ApprovalError(f"unknown approval {approval_id}")
    if approval.get("status") != "pending":
        raise ApprovalError(f"approval {approval_id} is already {approval.get('status')}")
    now = utcnow().isoformat()
    segments = list(approval.get("segments") or [])
    outcome = {"executed": False}
    if approve:
        if approval["kind"] == "reopen_road":
            allowed, why = guard.check("reopen_road", "human")
            if not allowed:
                raise ApprovalError(why)
            roads.reopen(segments)
            for sid in segments:
                repo.put_closure(sid, {"status": "open", "by": user, "since": now, "reason": f"reopened after approval {approval_id}: {note}".strip(), "track_id": approval.get("track_id")})
            if approval.get("track_id"):
                track = repo.get_track(approval["track_id"])
                if track:
                    track["closed_segments"] = [s for s in track.get("closed_segments", []) if s not in segments]
                    repo.put_track(track)
            outcome = {"executed": True, "action": "reopen_road", "reroute": roads.reroute_report(routes)}
        elif approval["kind"] == "review_hotspot" and approval.get("recommended") == "close" and segments:
            allowed, why = guard.check("close_road", "human")
            if not allowed:
                raise ApprovalError(why)
            roads.close(segments)
            for sid in segments:
                repo.put_closure(sid, {"status": "closed", "by": user, "since": now, "reason": f"closed after review {approval_id}: {note}".strip(), "hotspot_id": approval.get("hotspot_id"), "track_id": approval.get("track_id")})
            if approval.get("track_id"):
                track = repo.get_track(approval["track_id"])
                if track:
                    track["closed_segments"] = sorted(set(track.get("closed_segments", [])) | set(segments))
                    track["quiet_passes"] = 0
                    repo.put_track(track)
            outcome = {"executed": True, "action": "close_road", "reroute": roads.reroute_report(routes)}
        else:
            outcome = {"executed": True, "action": "acknowledged"}
    approval.update({"status": "approved" if approve else "rejected", "decided_by": user, "decided": now, "note": note, "outcome": outcome})
    repo.put_approval(approval)
    repo.put_action(approval.get("run_id") or "manual", {
        "action_id": new_id("X"),
        "kind": outcome.get("action", "reject") if approve else "reject",
        "actor": "human",
        "target": segments,
        "hotspot_id": approval.get("hotspot_id"),
        "track_id": approval.get("track_id"),
        "reason": note or ("approved" if approve else "rejected"),
        "executed": bool(outcome.get("executed")),
        "created": now,
        "details": {"approval_id": approval_id, "user": user},
    })
    notifier.send(f"{'Approved' if approve else 'Rejected'}: {approval['kind']} {', '.join(segments[:4])}", f"{user} {'approved' if approve else 'rejected'} {approval_id}. {note}".strip(), "info", {"approval_id": approval_id})
    return approval

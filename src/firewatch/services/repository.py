from __future__ import annotations

from typing import Any

from firewatch.services.kv import KVStore


class Repository:
    def __init__(self, kv: KVStore):
        self.kv = kv

    def put_run(self, run: dict) -> None:
        self.kv.put("run", run["run_id"], run)

    def get_run(self, run_id: str) -> dict | None:
        return self.kv.get("run", run_id)

    def list_runs(self, limit: int = 50) -> list[dict]:
        return self.kv.list("run", limit=limit, newest_first=True)

    def put_hotspot(self, run_id: str, hotspot: dict) -> None:
        self.kv.put("hotspot", f"{run_id}#{hotspot['hotspot_id']}", {**hotspot, "run_id": run_id})

    def list_hotspots(self, run_id: str) -> list[dict]:
        return self.kv.list("hotspot", prefix=f"{run_id}#")

    def put_track(self, track: dict) -> None:
        self.kv.put("track", track["track_id"], track)

    def get_track(self, track_id: str) -> dict | None:
        return self.kv.get("track", track_id)

    def list_tracks(self, status: str | None = None) -> list[dict]:
        tracks = self.kv.list("track")
        return [t for t in tracks if status is None or t.get("status") == status]

    def put_closure(self, segment_id: str, record: dict) -> None:
        self.kv.put("closure", segment_id, {**record, "segment_id": segment_id})

    def get_closures(self) -> dict[str, dict]:
        return {c["segment_id"]: c for c in self.kv.list("closure")}

    def put_approval(self, approval: dict) -> None:
        self.kv.put("approval", approval["approval_id"], approval)

    def get_approval(self, approval_id: str) -> dict | None:
        return self.kv.get("approval", approval_id)

    def list_approvals(self, status: str | None = None) -> list[dict]:
        items = self.kv.list("approval", newest_first=True)
        return [a for a in items if status is None or a.get("status") == status]

    def put_task(self, task: dict) -> None:
        self.kv.put("task", task["task_id"], task)

    def list_tasks(self) -> list[dict]:
        return self.kv.list("task", newest_first=True)

    def put_action(self, run_id: str, action: dict) -> None:
        self.kv.put("action", f"{run_id}#{action['action_id']}", {**action, "run_id": run_id})

    def list_actions(self, run_id: str | None = None) -> list[dict]:
        return self.kv.list("action", prefix=f"{run_id}#" if run_id else "")

    def get_meta(self, key: str, default: Any = None) -> Any:
        item = self.kv.get("meta", key)
        return default if item is None else item.get("value", default)

    def put_meta(self, key: str, value: Any) -> None:
        self.kv.put("meta", key, {"value": value})

    def next_track_id(self) -> str:
        seq = int(self.get_meta("track_seq", 0)) + 1
        self.put_meta("track_seq", seq)
        return f"T{seq:04d}"

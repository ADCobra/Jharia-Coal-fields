from __future__ import annotations

import cv2
import numpy as np

from firewatch.services.factory import Services
from firewatch.vision.media import encode_animation

TIMELAPSE_KEY = "timelapse/season.webp"


def frame_key(run_id: str) -> str:
    return f"runs/{run_id}/frame.webp"


def build_timelapse(services: Services, frame_ms: int = 650, hold_ms: int = 2200, ext: str = ".webp") -> tuple[bytes, dict]:
    runs = sorted((r for r in services.repo.list_runs(limit=2000) if r.get("status") == "done"), key=lambda r: r.get("acquired", ""))
    frames, used = [], []
    for run in runs:
        data = services.blobs.get_bytes(frame_key(run["run_id"]))
        if not data:
            continue
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None or (frames and image.shape != frames[0].shape):
            continue
        frames.append(image)
        used.append({"run_id": run["run_id"], "scene_id": run.get("scene_id"), "acquired": run.get("acquired")})
    if not frames:
        raise LookupError("no pass has a time-lapse frame yet")
    durations = [frame_ms] * len(frames)
    durations[-1] = hold_ms
    return encode_animation(frames, durations, ext), {"frames": len(frames), "first": used[0], "last": used[-1]}


def season_timelapse(services: Services) -> bytes:
    last = services.repo.get_meta("last_run_id")
    meta = services.repo.get_meta("timelapse") or {}
    if last and meta.get("last_run_id") == last:
        cached = services.blobs.get_bytes(TIMELAPSE_KEY)
        if cached:
            return cached
    data, info = build_timelapse(services)
    services.blobs.put_bytes(TIMELAPSE_KEY, data, "image/webp")
    services.repo.put_meta("timelapse", {"last_run_id": last, **info})
    return data

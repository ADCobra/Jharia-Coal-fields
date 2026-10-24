from __future__ import annotations

import numpy as np
from firewatch.models import Track
from firewatch.risk.roads import RoadNetwork, hotspot_geometry
from firewatch.util import days_between
from firewatch.vision.similarity import AnalogueIndex, signature
from firewatch.vision.tracking import kinematics


def analogue_rows(tracks: list[Track], roads: RoadNetwork, d0_m: float, horizon_days: float = 64.0) -> list[tuple[np.ndarray, dict]]:
    rows = []
    for track in tracks:
        pts = track.points
        if len(pts) < 3 or track.label == "industrial":
            continue
        for i in range(1, len(pts) - 1):
            future = [p for p in pts[i + 1:] if 0 < days_between(pts[i].acquired, p.acquired) <= horizon_days]
            if not future:
                continue
            kin = kinematics(Track(track.track_id, pts[: i + 1]))
            now_d = roads.nearest(hotspot_geometry(pts[i].polygon, pts[i].centroid))[1] if len(roads) else None
            fut_d = min(roads.nearest(hotspot_geometry(p.polygon, p.centroid))[1] for p in future) if len(roads) else None
            meta = {
                "track_id": track.track_id,
                "date": pts[i].acquired.date().isoformat(),
                "distance_m": None if now_d is None else round(now_d, 1),
                "grew": max(p.area_m2 for p in future) >= 1.3 * pts[i].area_m2,
                "reached_road": bool(now_d is not None and now_d > d0_m and fut_d is not None and fut_d <= d0_m),
            }
            rows.append((signature(pts[i].area_m2, pts[i].mean_anomaly_k, pts[i].max_anomaly_k, kin, now_d), meta))
    return rows


def build_analogue_index(tracks: list[Track], roads: RoadNetwork, d0_m: float, horizon_days: float = 64.0) -> AnalogueIndex:
    index = AnalogueIndex()
    for vector, meta in analogue_rows(tracks, roads, d0_m, horizon_days):
        index.add(vector, meta)
    return index.build()

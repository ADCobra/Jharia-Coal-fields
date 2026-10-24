from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from shapely.geometry import Polygon

from firewatch.config import TrackingConfig
from firewatch.models import Hotspot, Kinematics, Track, TrackPoint
from firewatch.util import bearing_deg, days_between


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2 or np.ptp(t) == 0:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def _shrink(v: float, t: np.ndarray, noise: float) -> float:
    if noise <= 0 or v == 0:
        return v
    spread = float(np.sqrt(np.sum((t - t.mean()) ** 2)))
    if spread <= 0:
        return 0.0
    se = noise / spread
    return v * max(0.0, 1.0 - (se / abs(v)) ** 2)


def kinematics(track: Track, last_n: int = 8, noise_m: float = 0.0) -> Kinematics:
    pts = track.points[-last_n:]
    if len(pts) < 2:
        p = pts[0] if pts else None
        return Kinematics(len(pts), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if p else Kinematics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    t = np.array([days_between(pts[0].acquired, p.acquired) for p in pts])
    area = np.array([p.area_m2 for p in pts])
    xs = np.array([p.centroid[0] for p in pts])
    ys = np.array([p.centroid[1] for p in pts])
    anom = np.array([p.mean_anomaly_k for p in pts])
    growth = _slope(t, area)
    vx, vy = _slope(t, xs), _slope(t, ys)
    if noise_m > 0:
        speed = math.hypot(vx, vy)
        kept = _shrink(speed, t, noise_m)
        scale = kept / speed if speed > 0 else 0.0
        vx, vy = vx * scale, vy * scale
    return Kinematics(
        n_points=len(pts),
        span_days=float(np.ptp(t)),
        growth_m2_per_day=growth,
        growth_frac_per_30d=growth * 30.0 / max(float(area.mean()), 1.0),
        vx_m_per_day=vx,
        vy_m_per_day=vy,
        speed_m_per_day=math.hypot(vx, vy),
        bearing_deg=bearing_deg(vx, vy),
        anomaly_trend_k_per_day=_slope(t, anom),
    )


def _iou(a: list, b: list) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
    if pa.is_empty or pb.is_empty:
        return 0.0
    union = pa.union(pb).area
    return float(pa.intersection(pb).area / union) if union > 0 else 0.0


@dataclass
class Association:
    matched: dict[str, str]
    new: list[str]
    unmatched_tracks: list[str]


def associate(tracks: list[Track], hotspots: list[Hotspot], cfg: TrackingConfig) -> Association:
    candidates = []
    for track in tracks:
        if track.status == "inactive" or not track.points:
            continue
        last = track.points[-1]
        kin = kinematics(track, cfg.history_points, cfg.position_noise_m)
        for h in hotspots:
            dt_days = max(0.0, days_between(last.acquired, h.acquired))
            px = last.centroid[0] + kin.vx_m_per_day * dt_days
            py = last.centroid[1] + kin.vy_m_per_day * dt_days
            dist = math.hypot(h.centroid[0] - px, h.centroid[1] - py)
            iou = _iou(last.polygon, h.polygon)
            if dist <= cfg.max_match_distance_m or iou > 0.0:
                candidates.append((dist - 300.0 * iou, track.track_id, h.hotspot_id))
    candidates.sort()
    matched: dict[str, str] = {}
    used_tracks: set[str] = set()
    for _, tid, hid in candidates:
        if hid in matched or tid in used_tracks:
            continue
        matched[hid] = tid
        used_tracks.add(tid)
    new = [h.hotspot_id for h in hotspots if h.hotspot_id not in matched]
    unmatched = [t.track_id for t in tracks if t.status != "inactive" and t.track_id not in used_tracks]
    return Association(matched, new, unmatched)


def point_from_hotspot(h: Hotspot) -> TrackPoint:
    return TrackPoint(scene_id=h.scene_id, acquired=h.acquired, hotspot_id=h.hotspot_id, area_m2=h.area_m2, mean_anomaly_k=h.mean_anomaly_k, max_anomaly_k=h.max_anomaly_k, centroid=h.centroid, polygon=h.polygon)


def update_tracks(
    tracks: list[Track],
    hotspots: list[Hotspot],
    assoc: Association,
    cfg: TrackingConfig,
    new_track_id: Callable[[], str],
    observable: Callable[[float, float], bool],
) -> list[Track]:
    by_id = {t.track_id: t for t in tracks}
    by_hotspot = {h.hotspot_id: h for h in hotspots}
    touched: list[Track] = []
    for hid, tid in assoc.matched.items():
        track = by_id[tid]
        track.points.append(point_from_hotspot(by_hotspot[hid]))
        track.points = track.points[-max(cfg.history_points * 3, 12):]
        track.status = "active"
        track.missed = 0
        by_hotspot[hid].track_id = tid
        touched.append(track)
    for hid in assoc.new:
        track = Track(track_id=new_track_id(), points=[point_from_hotspot(by_hotspot[hid])])
        by_hotspot[hid].track_id = track.track_id
        touched.append(track)
    for tid in assoc.unmatched_tracks:
        track = by_id[tid]
        last = track.points[-1]
        if observable(*last.centroid):
            track.missed += 1
            track.status = "inactive" if track.missed > cfg.max_missed_passes else "cooling"
        touched.append(track)
    return touched

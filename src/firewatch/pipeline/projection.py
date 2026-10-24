from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

from firewatch.config import ProjectionConfig
from firewatch.geo.grid import GridSpec
from firewatch.models import Hotspot, Projection, SpreadEstimate
from firewatch.physics.heat_spread import project_cone
from firewatch.risk.roads import RoadNetwork


def _largest_polygon(mask: np.ndarray, grid: GridSpec) -> list[tuple[float, float]]:
    if not mask.any():
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if len(contour) > 80:
        contour = cv2.approxPolyDP(contour, 0.8, True)
    poly = grid.pixel_polygon_to_map(contour.reshape(-1, 2).astype(np.float64))
    if len(poly) >= 3:
        poly.append(poly[0])
    return poly


def project_hotspot(hotspot: Hotspot, label_id: int, labels: np.ndarray, anomaly: np.ndarray, spread: SpreadEstimate, grid: GridSpec, roads: RoadNetwork, cfg: ProjectionConfig) -> Projection:
    cx, cy = hotspot.centroid
    r0, c0, h, w = grid.window_around(cx, cy, cfg.crop_px // 2)
    crop = grid.crop(r0, c0, h, w)
    a = anomaly[r0:r0 + h, c0:c0 + w]
    m = labels[r0:r0 + h, c0:c0 + w] == label_id
    road_mask = roads.rasterize(crop, 1.5 * grid.res) if len(roads) else None
    cone = project_cone(a, m, spread, grid.res, cfg, road_mask)
    cone_poly = _largest_polygon(cone.probability >= cfg.cone_probability, crop)
    core_poly = _largest_polygon(cone.probability >= 0.5, crop)
    projected = None
    if len(cone_poly) >= 4 and len(roads):
        geom = Polygon(cone_poly).buffer(0)
        if not geom.is_empty:
            projected = roads.nearest(geom)[1]
    return Projection(
        horizon_days=cfg.horizon_days,
        cone_polygon=cone_poly,
        core_polygon=core_poly,
        reach_probability=round(cone.reach_probability, 3),
        median_days_to_road=cone.median_days_to_road,
        projected_distance_m=None if projected is None else round(projected, 1),
        ensemble=cfg.ensemble,
    )

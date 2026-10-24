from __future__ import annotations

import cv2
import numpy as np

from firewatch.data.sources import HighResPatch


def normalized_hotspot_index(swir16: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((swir22 - swir16) / (swir22 + swir16)).astype(np.float32)


def polygon_mask(patch: HighResPatch, polygon: list[tuple[float, float]], buffer_m: float) -> np.ndarray:
    mask = np.zeros(patch.grid.shape, np.uint8)
    if len(polygon) >= 3:
        pix = patch.grid.map_polygon_to_pixel(polygon)
        cv2.fillPoly(mask, [np.round(pix).astype(np.int32).reshape(-1, 1, 2)], 1)
    radius = int(round(buffer_m / patch.grid.res))
    if radius > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    return mask.astype(bool)


def hot_pixel_check(patch: HighResPatch, polygon: list[tuple[float, float]], buffer_m: float = 60.0, nhi_min: float = 0.0, swir22_min: float = 0.12) -> dict:
    region = polygon_mask(patch, polygon, buffer_m) & patch.valid
    if not region.any():
        return {"checked": False, "reason": "hotspot outside the high-resolution patch"}
    nhi = normalized_hotspot_index(patch.swir16, patch.swir22)
    hot = region & (nhi > nhi_min) & (patch.swir22 > swir22_min)
    count = int(hot.sum())
    return {
        "checked": True,
        "scene_id": patch.scene_id,
        "acquired": patch.acquired.isoformat(),
        "hot_pixels": count,
        "hot_fraction": round(count / int(region.sum()), 4),
        "max_nhi": round(float(np.nanmax(np.where(region, nhi, np.nan))), 3),
        "confirmed": count >= 2,
    }

from __future__ import annotations

import datetime as dt

import cv2
import numpy as np
from matplotlib import colormaps

from firewatch.geo.grid import GridSpec
from firewatch.geo.raster import reproject_to_wgs84
from firewatch.risk.roads import RoadNetwork
from firewatch.util import jsonable
from firewatch.vision.media import encode_image

LEVEL_BGR = {"high": (40, 40, 225), "medium": (0, 140, 255), "low": (40, 210, 250)}
CLOSED_BGR = (60, 20, 200)
ROAD_BGR = (175, 175, 170)
CONE_BGR = (235, 235, 235)
EXCLUDED_BGR = (150, 150, 150)


def colorize(array: np.ndarray, vmin: float, vmax: float, cmap: str = "inferno", alpha: int = 210) -> np.ndarray:
    finite = np.isfinite(array)
    scaled = np.clip((np.nan_to_num(array, nan=vmin) - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgba = (colormaps[cmap](scaled) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(finite, alpha, 0)
    return rgba


def image_layer(array: np.ndarray, grid: GridSpec, vmin: float, vmax: float, cmap: str, alpha: int = 210, fmt: str = ".webp", provenance: dict | None = None) -> tuple[bytes, tuple[float, float, float, float]]:
    rgba = colorize(array, vmin, vmax, cmap, alpha)
    bands = np.moveaxis(rgba, -1, 0)
    warped, bounds = reproject_to_wgs84(bands, grid, "nearest")
    bgra = np.ascontiguousarray(np.moveaxis(warped, 0, -1)[..., [2, 1, 0, 3]])
    params = [cv2.IMWRITE_WEBP_QUALITY, 88] if fmt == ".webp" else [cv2.IMWRITE_PNG_COMPRESSION, 9]
    if provenance is not None:
        west, south, east, north = bounds
        provenance = {**provenance, "crs": "EPSG:4326", "west": f"{west:.6f}", "south": f"{south:.6f}", "east": f"{east:.6f}", "north": f"{north:.6f}", "valueMin": f"{vmin:.2f}", "valueMax": f"{vmax:.2f}", "colormap": cmap, "opencv": cv2.__version__}
    return encode_image(fmt, bgra, params, provenance), bounds


def stretch(band: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0, gamma: float = 0.9) -> np.ndarray:
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros(band.shape, np.uint8)
    lo, hi = np.percentile(band[finite], [lo_pct, hi_pct])
    scaled = np.clip((np.nan_to_num(band, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    return (scaled * 255).astype(np.uint8)


def color_layer(reflectance: dict[str, np.ndarray], matrix, grid: GridSpec, provenance: dict | None = None) -> bytes:
    from firewatch.vision.registration import warp_to_reference

    bgr = np.dstack([stretch(reflectance["red"]), stretch(reflectance["nir08"]), stretch(reflectance["swir16"])])
    bgr = warp_to_reference(bgr, matrix, grid.shape, fill=0)
    bgr = cv2.addWeighted(bgr, 0.88, cv2.GaussianBlur(bgr, (0, 0), 0.6), 0.12, 0)
    alpha = np.full(grid.shape, 255, np.uint8)
    bands = np.moveaxis(np.dstack([bgr[..., 2], bgr[..., 1], bgr[..., 0], alpha]), -1, 0)
    warped, bounds = reproject_to_wgs84(bands, grid, "bilinear")
    bgra = np.ascontiguousarray(np.moveaxis(warped, 0, -1)[..., [2, 1, 0, 3]])
    if provenance is not None:
        west, south, east, north = bounds
        provenance = {**provenance, "crs": "EPSG:4326", "west": f"{west:.6f}", "south": f"{south:.6f}", "east": f"{east:.6f}", "north": f"{north:.6f}", "bands": "SWIR1, NIR, red (Landsat 6-5-4)", "opencv": cv2.__version__}
    return encode_image(".webp", bgra, [cv2.IMWRITE_WEBP_QUALITY, 85], provenance)


def true_colour(reflectance: dict[str, np.ndarray], lo_pct: float = 1.0, hi_pct: float = 99.0, gamma: float = 0.8) -> np.ndarray:
    rgb = np.dstack([reflectance["red"], reflectance["green"], reflectance["blue"]]).astype(np.float32)
    finite = np.isfinite(rgb).all(axis=2)
    if not finite.any():
        return np.zeros(rgb.shape, np.uint8)
    lum = rgb[finite].mean(axis=1)
    lo, hi = np.percentile(lum, [lo_pct, hi_pct])
    scaled = np.clip((np.nan_to_num(rgb, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    grey = scaled.mean(axis=2, keepdims=True)
    scaled = np.clip(grey + 1.15 * (scaled - grey), 0, 1)
    out = (scaled * 255).astype(np.uint8)
    out[~finite] = 0
    return out


def photo_layer(reflectance: dict[str, np.ndarray], grid: GridSpec, provenance: dict | None = None, matrix=None) -> tuple[bytes, tuple[float, float, float, float]]:
    rgb = true_colour(reflectance)
    if matrix is not None:
        from firewatch.vision.registration import warp_to_reference

        rgb = warp_to_reference(rgb, matrix, grid.shape, fill=0)
    alpha = np.where(rgb.any(axis=2), 255, 0).astype(np.uint8)
    bands = np.moveaxis(np.dstack([rgb, alpha]), -1, 0)
    warped, bounds = reproject_to_wgs84(bands, grid, "bilinear")
    rgba = np.moveaxis(warped, 0, -1)
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])
    if provenance is not None:
        west, south, east, north = bounds
        provenance = {**provenance, "crs": "EPSG:4326", "west": f"{west:.6f}", "south": f"{south:.6f}", "east": f"{east:.6f}", "north": f"{north:.6f}", "bands": "red, green, blue surface reflectance (Landsat 4-3-2)", "opencv": cv2.__version__}
    return encode_image(".webp", bgra, [cv2.IMWRITE_WEBP_QUALITY, 86], provenance), bounds


def _pixels(grid: GridSpec, coords, scale: float) -> np.ndarray:
    pix = grid.map_polygon_to_pixel(list(coords))
    return np.round((np.asarray(pix, np.float64) + 0.5) * scale).astype(np.int32).reshape(-1, 1, 2)


def render_frame(base: np.ndarray | None, anomaly: np.ndarray, records: list[dict], roads: RoadNetwork, grid: GridSpec, when: dt.datetime, width: int = 960) -> np.ndarray:
    h, w = anomaly.shape
    gray = np.full((h, w), 110, np.uint8) if base is None else base
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.42 + 70.0
    a = np.nan_to_num(anomaly, nan=0.0)
    heat = colorize(np.where(a > 1.0, a, np.nan), 1.0, 15.0, "YlOrRd", 255)[..., [2, 1, 0]].astype(np.float32)
    weight = (np.clip((a - 1.0) / 4.0, 0.0, 1.0) * 0.9)[..., None]
    canvas = canvas * (1.0 - weight) + heat * weight
    scale = width / float(w)
    frame = cv2.resize(np.clip(canvas, 0, 255).astype(np.uint8), (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    for sid, seg in roads.segments.items():
        closed = sid in roads.closed
        cv2.polylines(frame, [_pixels(grid, seg.line.coords, scale)], False, CLOSED_BGR if closed else ROAD_BGR, 4 if closed else (1 if seg.haul else 2), cv2.LINE_AA)
    for rec in records:
        cone = (rec.get("projection") or {}).get("cone_polygon") or []
        if len(cone) >= 4:
            cv2.polylines(frame, [_pixels(grid, cone, scale)], True, CONE_BGR, 1, cv2.LINE_AA)
    for rec in records:
        poly = rec.get("polygon") or []
        if len(poly) < 3:
            continue
        pts = _pixels(grid, poly, scale)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        if rec.get("decision") == "exclude":
            cv2.circle(frame, (int(round(cx)), int(round(cy))), int(round(radius)) + 7, EXCLUDED_BGR, 1, cv2.LINE_AA)
            continue
        color = LEVEL_BGR.get((rec.get("risk") or {}).get("level", "low"), LEVEL_BGR["low"])
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), int(round(radius)) + 7, color, 2, cv2.LINE_AA)
    shown = [r for r in records if r.get("decision") != "exclude"]
    high = sum(1 for r in shown if (r.get("risk") or {}).get("level") == "high")
    closed = sorted(roads.closed)
    lines = [f"{when.day} {when:%b %Y}", f"{len(shown)} hotspot{'s' if len(shown) != 1 else ''}, {high} high risk" + (f", closed: {', '.join(closed[:4])}{' and more' if len(closed) > 4 else ''}" if closed else "")]
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, 0.8 if i == 0 else 0.5, 1)[0][0] for i, t in enumerate(lines)) + 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, 62), (20, 18, 16), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.putText(frame, lines[0], (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(frame, lines[1], (12, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (215, 215, 215), 1, cv2.LINE_AA)
    return frame


def polygon_feature(coords_map: list, grid: GridSpec, properties: dict) -> dict | None:
    if len(coords_map) < 4:
        return None
    ring = grid.map_to_lonlat_coords(coords_map)
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Feature", "properties": jsonable(properties), "geometry": {"type": "Polygon", "coordinates": [ring]}}


def hotspot_features(records: list[dict], grid: GridSpec) -> dict:
    features = []
    for rec in records:
        props = {k: rec.get(k) for k in ("hotspot_id", "track_id", "area_m2", "mean_anomaly_k", "max_anomaly_k", "decision", "decision_reason")}
        risk = rec.get("risk") or {}
        props.update({"risk": risk.get("score"), "level": risk.get("level"), "confidence": risk.get("confidence"), "distance_m": risk.get("distance_m"), "nearest_segment": risk.get("nearest_segment")})
        feat = polygon_feature(rec.get("polygon", []), grid, props)
        if feat is None:
            lon, lat = rec["centroid_lonlat"]
            feat = {"type": "Feature", "properties": jsonable(props), "geometry": {"type": "Point", "coordinates": [lon, lat]}}
        features.append(feat)
    return {"type": "FeatureCollection", "features": features}


def cone_features(records: list[dict], grid: GridSpec) -> dict:
    features = []
    for rec in records:
        proj = rec.get("projection") or {}
        for kind, key in (("cone", "cone_polygon"), ("core", "core_polygon")):
            feat = polygon_feature(proj.get(key) or [], grid, {"hotspot_id": rec["hotspot_id"], "track_id": rec.get("track_id"), "kind": kind, "reach_probability": proj.get("reach_probability"), "median_days_to_road": proj.get("median_days_to_road"), "horizon_days": proj.get("horizon_days")})
            if feat:
                features.append(feat)
    return {"type": "FeatureCollection", "features": features}

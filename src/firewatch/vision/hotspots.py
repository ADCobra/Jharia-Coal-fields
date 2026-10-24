from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from firewatch.config import HotspotConfig
from firewatch.geo.grid import GridSpec
from firewatch.models import Hotspot, SceneRef
from firewatch.vision.anomaly import AnomalyResult


@dataclass
class Detection:
    hotspots: list[Hotspot]
    labels: np.ndarray
    mask: np.ndarray
    candidates: np.ndarray | None = None


def _simplify(contour: np.ndarray, max_vertices: int) -> np.ndarray:
    if len(contour) <= max_vertices:
        return contour
    perimeter = cv2.arcLength(contour, True)
    eps = 0.002 * perimeter
    approx = contour
    while len(approx) > max_vertices and eps < perimeter:
        approx = cv2.approxPolyDP(contour, eps, True)
        eps *= 1.6
    return approx


def _open2(mask: np.ndarray) -> np.ndarray:
    k = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), k, anchor=(1, 1), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.dilate(eroded, k, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def hysteresis_mask(anomaly: AnomalyResult, valid: np.ndarray, cfg: HotspotConfig) -> tuple[np.ndarray, np.ndarray]:
    a = np.nan_to_num(anomaly.anomaly, nan=-1e3)
    z = np.nan_to_num(anomaly.z, nan=-1e3)
    strong = valid & (z >= cfg.sigma_high) & (a >= cfg.min_contrast_k)
    weak = valid & (z >= cfg.sigma_low) & (a >= cfg.min_weak_contrast_k)
    weak = _open2(weak) | strong
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    seeds = np.zeros(count, bool)
    seeds[np.unique(labels[strong])] = True
    seeds[0] = False
    keep = seeds[labels]
    return keep, strong


def candidate_mask(anomaly: AnomalyResult, valid: np.ndarray, cfg: HotspotConfig) -> np.ndarray:
    a = np.nan_to_num(anomaly.anomaly, nan=-1e3)
    z = np.nan_to_num(anomaly.z, nan=-1e3)
    weak = valid & (z >= cfg.persist_sigma) & (a >= cfg.persist_contrast_k)
    return _open2(weak)


def persistent_mask(candidates: np.ndarray, prior: np.ndarray, keep: np.ndarray, cfg: HotspotConfig) -> np.ndarray:
    reach = cv2.dilate(prior.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.persist_reach_px + 1, 2 * cfg.persist_reach_px + 1))).astype(bool)
    fresh = candidates & ~keep
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fresh.astype(np.uint8), connectivity=8)
    hit = np.zeros(count, bool)
    hit[np.unique(labels[reach & fresh])] = True
    hit[0] = False
    hit &= stats[:, cv2.CC_STAT_AREA] >= cfg.persist_min_area_px
    return hit[labels]


def detect_hotspots(
    anomaly: AnomalyResult,
    lst: np.ndarray,
    valid: np.ndarray,
    grid: GridSpec,
    ref: SceneRef,
    cfg: HotspotConfig,
    industrial_mask: np.ndarray | None = None,
    prior: np.ndarray | None = None,
) -> Detection:
    keep, _ = hysteresis_mask(anomaly, valid, cfg)
    candidates = candidate_mask(anomaly, valid, cfg)
    promoted = np.zeros_like(keep)
    if cfg.persistence and prior is not None and prior.any():
        promoted = persistent_mask(candidates, prior, keep, cfg)
        keep = keep | promoted
    count, labels, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), connectivity=8)
    out_labels = np.zeros(labels.shape, np.int32)
    hotspots: list[Hotspot] = []
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.cloud_ring_px + 1, 2 * cfg.cloud_ring_px + 1))
    pad = cfg.cloud_ring_px + 2
    h, w = labels.shape
    order = sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    for lab in order:
        area_px = int(stats[lab, cv2.CC_STAT_AREA])
        x, y, bw, bh = (int(v) for v in stats[lab, :4])
        r0, c0 = max(0, y - pad), max(0, x - pad)
        r1, c1 = min(h, y + bh + pad), min(w, x + bw + pad)
        comp = labels[r0:r1, c0:c1] == lab
        is_promoted = bool((promoted[r0:r1, c0:c1] & comp).sum() * 2 >= area_px)
        if area_px < (cfg.persist_min_area_px if is_promoted else cfg.min_area_px):
            continue
        comp_u8 = comp.astype(np.uint8)
        contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(c0, r0))
        if not contours:
            continue
        contour = max(contours, key=len)
        simple = _simplify(contour, cfg.max_vertices).reshape(-1, 2)
        polygon = grid.pixel_polygon_to_map(simple.astype(np.float64))
        if len(polygon) >= 3 and polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        rows, cols = np.nonzero(comp)
        rows = rows + r0
        cols = cols + c0
        anom_vals = anomaly.anomaly[rows, cols]
        weights = np.clip(anom_vals, 1e-3, None)
        crow = float(np.sum(rows * weights) / np.sum(weights))
        ccol = float(np.sum(cols * weights) / np.sum(weights))
        cx, cy = grid.xy(crow, ccol)
        lon, lat = grid.to_lonlat(float(cx), float(cy))
        hull = cv2.convexHull(contour)
        hull_mask = np.zeros_like(comp_u8)
        cv2.fillConvexPoly(hull_mask, hull.reshape(-1, 2) - np.array([c0, r0]), 1)
        solidity = area_px / max(int(hull_mask.sum()), area_px)
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour.reshape(-1, 2).astype(np.float32))
        rect_px = (rw + 1.0) * (rh + 1.0)
        rectangularity = float(min(1.0, area_px / max(rect_px, 1.0)))
        elongation = float(max(rw, rh) + 1.0) / float(min(rw, rh) + 1.0)
        ring = cv2.dilate(comp_u8, ring_kernel).astype(bool) & ~comp
        ring_valid = valid[r0:r1, c0:c1][ring]
        cloud_ring_fraction = float(1.0 - ring_valid.mean()) if ring_valid.size else 0.0
        industrial = float(industrial_mask[rows, cols].mean()) if industrial_mask is not None else 0.0
        hotspot_id = f"{ref.scene_id}-h{len(hotspots) + 1:02d}"
        hotspots.append(
            Hotspot(
                hotspot_id=hotspot_id,
                scene_id=ref.scene_id,
                acquired=ref.acquired,
                area_m2=area_px * grid.pixel_area,
                pixel_count=area_px,
                mean_anomaly_k=float(np.mean(anom_vals)),
                max_anomaly_k=float(np.max(anom_vals)),
                peak_lst_k=float(np.nanmax(lst[rows, cols])),
                centroid=(float(cx), float(cy)),
                centroid_lonlat=(float(lon), float(lat)),
                polygon=polygon,
                bbox_px=(r0, c0, r1, c1),
                solidity=float(solidity),
                rectangularity=rectangularity,
                elongation=elongation,
                cloud_ring_fraction=cloud_ring_fraction,
                industrial_overlap=industrial,
                persistent=is_promoted,
            )
        )
        out_labels[rows, cols] = len(hotspots)
    return Detection(hotspots=hotspots, labels=out_labels, mask=out_labels > 0, candidates=candidates | (out_labels > 0))

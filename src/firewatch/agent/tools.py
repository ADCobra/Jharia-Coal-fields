from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon

from firewatch.agent.case import Case
from firewatch.config import Settings
from firewatch.data.sources import SceneSource
from firewatch.geo.grid import GridSpec
from firewatch.models import AlignmentResult
from firewatch.physics.thermal import derive_thermal
from firewatch.risk.roads import RoadNetwork
from firewatch.risk.scoring import assess, confidence
from firewatch.services.repository import Repository
from firewatch.util import new_id, utcnow
from firewatch.vision.anomaly import compute_anomaly
from firewatch.vision.false_heat import industrial_assessment
from firewatch.vision.registration import FeatureRegistrar, to_window, transform_error_px
from firewatch.vision.swir import hot_pixel_check
from firewatch.vision.vlm import VisionLanguageChecker

log = logging.getLogger(__name__)

EVIDENCE_KEY = {
    "fetch_history": "history",
    "refine_alignment": "realign",
    "check_industrial": "industrial",
    "fetch_highres": "highres",
    "request_drone": "drone",
}


@dataclass
class ToolContext:
    settings: Settings
    source: SceneSource
    grid: GridSpec
    registrar: FeatureRegistrar
    alignment: AlignmentResult
    roads: RoadNetwork
    vlm: VisionLanguageChecker
    repo: Repository
    run_id: str
    acquired: dt.datetime
    reference_image: np.ndarray | None = None
    scene_image: np.ndarray | None = None
    scene_mask: np.ndarray | None = None
    false_color: np.ndarray | None = None


def rescore(case: Case, settings: Settings, roads: RoadNetwork) -> None:
    inputs = dataclasses.replace(case.conf_inputs, evidence=case.evidence)
    conf, notes = confidence(inputs, settings.hotspots, settings.alignment)
    case.risk = assess(case.hotspot, case.kin, case.projection, roads, settings.risk, conf, notes, case.evidence.get("analogues"))


def polygon_mask(grid: GridSpec, polygon: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(grid.shape, np.uint8)
    if len(polygon) >= 3:
        pix = grid.map_polygon_to_pixel(polygon)
        cv2.fillPoly(mask, [np.round(pix).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


class Toolbox:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self._tiebreaks: dict = {}

    def run(self, name: str, case: Case) -> dict:
        try:
            result = getattr(self, name)(case)
        except Exception as exc:
            log.exception("tool %s failed for %s", name, case.case_id)
            result = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        case.evidence[EVIDENCE_KEY[name]] = result
        case.tried.append(name)
        case.steps += 1
        return result

    def fetch_history(self, case: Case) -> dict:
        s = self.ctx.settings
        refs = self.ctx.source.history(case.hotspot.acquired, s.agent.history_passes)
        if not refs:
            return {"passes": [], "observable": 0, "present": 0, "persistent": False, "verdict": "unknown", "note": "no earlier passes in the archive"}
        cx, cy = case.hotspot.centroid
        r0, c0, h, w = self.ctx.grid.window_around(cx, cy, 48)
        crop = self.ctx.grid.crop(r0, c0, h, w)
        mask = polygon_mask(crop, case.hotspot.polygon)
        if not mask.any():
            return {"passes": [], "observable": 0, "present": 0, "persistent": False, "verdict": "unknown", "note": "hotspot outside the crop"}
        passes = []
        for ref in refs:
            try:
                raw = self.ctx.source.load(ref, crop)
            except Exception as exc:
                passes.append({"scene_id": ref.scene_id, "date": ref.acquired.date().isoformat(), "error": str(exc)[:200]})
                continue
            thermal = derive_thermal(raw, s.thermal)
            anomaly = compute_anomaly(thermal.lst, thermal.valid, thermal.albedo, thermal.ndvi, s.hotspots, s.thermal, thermal.water)
            inside = mask & thermal.valid
            valid_fraction = float(inside.sum()) / float(mask.sum())
            contrast = float(np.nanmean(anomaly.anomaly[inside])) if inside.sum() >= 3 else None
            present = bool(contrast is not None and valid_fraction >= 0.6 and contrast >= s.hotspots.min_contrast_k)
            passes.append({"scene_id": ref.scene_id, "date": ref.acquired.date().isoformat(), "valid_fraction": round(valid_fraction, 2), "contrast_k": None if contrast is None else round(contrast, 2), "present": present})
        observable = [p for p in passes if p.get("valid_fraction", 0.0) >= 0.6]
        n_present = sum(1 for p in observable if p["present"])
        persistent = len(observable) >= 2 and n_present * 2 >= len(observable)
        first = next((p["date"] for p in passes if p.get("present")), None)
        verdict = "persistent" if persistent else ("new" if observable and n_present == 0 else ("intermittent" if n_present else "unknown"))
        return {"passes": passes, "observable": len(observable), "present": n_present, "persistent": persistent, "first_seen": first, "verdict": verdict}

    def refine_alignment(self, case: Case) -> dict:
        ctx = self.ctx
        if ctx.reference_image is None or ctx.scene_image is None:
            return {"ok": False, "reason": "no reference image available"}
        cfg = ctx.settings.alignment
        used = ctx.alignment
        method = ctx.registrar.pick(cfg.tiebreak_preference, {used.method, used.check_method})
        if method is None:
            return {"ok": False, "reason": "no third matcher available"}
        h, w = ctx.reference_image.shape
        size = min(cfg.tiebreak_window_px, h, w)
        r0, c0, r1, c1 = case.hotspot.bbox_px
        step = max(1, size // 4)
        top = int(np.clip(((r0 + r1) // 2 - size // 2) // step * step, 0, h - size))
        left = int(np.clip(((c0 + c1) // 2 - size // 2) // step * step, 0, w - size))
        window = (slice(top, top + size), slice(left, left + size))
        key = (method, top, left)
        if key not in self._tiebreaks:
            mask = ctx.scene_mask[window] if ctx.scene_mask is not None else None
            self._tiebreaks[key] = ctx.registrar.register(ctx.reference_image[window], ctx.scene_image[window], None, mask, method=method)
        result = self._tiebreaks[key]
        out = {"method": method, "window_px": size, "ok": False, "inliers": result.n_inliers, "residual_px": result.residual_px, "agreement_px": None, "seconds": round(result.seconds, 2)}
        if not result.ok:
            out["reason"] = f"{method} could not align the window: {result.note}"
            return out
        out["agreement_px"] = round(transform_error_px(result.matrix, to_window(used.matrix, left, top), (size, size)), 3)
        if used.check_matrix is not None:
            out["agreement_with_check_px"] = round(transform_error_px(result.matrix, to_window(used.check_matrix, left, top), (size, size)), 3)
        out["ok"] = out["agreement_px"] <= cfg.max_disagreement_px
        return out

    def check_industrial(self, case: Case) -> dict:
        history = case.evidence.get("history") or {}
        persistence = history["present"] / history["observable"] if history.get("observable") else None
        result = industrial_assessment(case.hotspot, persistence)
        vlm = self.ctx.vlm
        if result["verdict"] == "uncertain":
            if vlm.available and self.ctx.false_color is not None:
                r0, c0, r1, c1 = case.hotspot.bbox_px
                pad = 16
                img = self.ctx.false_color[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
                answer = vlm.ask(img, "Is there a factory, chimney or industrial building in this image?")
                result["vlm"] = {"model": answer.model, "answer": answer.answer, "verdict": answer.verdict, "seconds": round(answer.seconds, 2)}
                if answer.verdict == "yes":
                    result["verdict"] = "industrial"
                elif answer.verdict == "no":
                    result["verdict"] = "fire"
            else:
                result["vlm"] = {"available": False, "note": getattr(vlm, "note", "not configured")}
        if result["verdict"] == "industrial":
            case.excluded = True
            if case.track is not None:
                case.track.label = "industrial"
        return result

    def fetch_highres(self, case: Case) -> dict:
        s = self.ctx.settings
        geom = Polygon(case.hotspot.polygon).buffer(s.agent.highres_buffer_m) if len(case.hotspot.polygon) >= 4 else None
        if geom is None or geom.is_empty:
            cx, cy = case.hotspot.centroid
            b = s.agent.highres_buffer_m
            bounds = (cx - b, cy - b, cx + b, cy + b)
        else:
            bounds = geom.bounds
        patch = self.ctx.source.highres(bounds, self.ctx.grid.crs, case.hotspot.acquired)
        if patch is None:
            return {"checked": False, "reason": f"no clear Sentinel-2 image within {s.data.sentinel_window_days} days"}
        return hot_pixel_check(patch, case.hotspot.polygon)

    def request_drone(self, case: Case) -> dict:
        for task in self.ctx.repo.list_tasks():
            if task.get("status") == "requested" and task.get("track_id") and task.get("track_id") == case.hotspot.track_id:
                return {"requested": False, "existing_task": task["task_id"], "simulated": True, "note": "a survey for this track is already pending"}
        task = {
            "task_id": new_id("D"),
            "kind": "drone_survey",
            "status": "requested",
            "simulated": True,
            "hotspot_id": case.hotspot.hotspot_id,
            "track_id": case.hotspot.track_id,
            "lonlat": case.hotspot.centroid_lonlat,
            "run_id": self.ctx.run_id,
            "requested": utcnow().isoformat(),
        }
        self.ctx.repo.put_task(task)
        return {"requested": True, "task_id": task["task_id"], "simulated": True}

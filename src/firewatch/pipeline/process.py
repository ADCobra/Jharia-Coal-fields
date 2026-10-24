from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from firewatch.agent.actions import ActionExecutor
from firewatch.agent.case import Case
from firewatch.agent.graph import FireWatchAgent
from firewatch.agent.guard import Guard
from firewatch.agent.planner import build_planner
from firewatch.agent.tools import ToolContext, Toolbox
from firewatch.agent.trace import Tracer
from firewatch.models import AlignmentResult, RawScene, SceneRef, ThermalProduct, Track
from firewatch.physics.heat_spread import estimate_spread
from firewatch.physics.thermal import derive_thermal
from firewatch.pipeline.analogues import build_analogue_index
from firewatch.pipeline.context import Context
from firewatch import __version__
from firewatch.pipeline.products import color_layer, cone_features, hotspot_features, image_layer, photo_layer, render_frame
from firewatch.pipeline.projection import project_hotspot
from firewatch.pipeline.timelapse import frame_key
from firewatch.risk.roads import hotspot_geometry
from firewatch.risk.scoring import ConfidenceInputs, assess, confidence
from firewatch.util import days_between, jsonable, new_id, utcnow
from firewatch.vision.anomaly import AnomalyResult, compute_anomaly
from firewatch.vision.flow import front_motion
from firewatch.vision.hotspots import Detection, detect_hotspots
from firewatch.vision.media import encode_image
from firewatch.vision.registration import IDENTITY, to_uint8, warp_to_reference
from firewatch.vision.similarity import signature, summarise
from firewatch.vision.tracking import associate, kinematics, update_tracks

log = logging.getLogger(__name__)
REFERENCE_KEY = "reference/registration.npz"


class Timer:
    def __init__(self) -> None:
        self.marks: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.marks[name] = round(self.marks.get(name, 0.0) + time.perf_counter() - start, 3)


@dataclass
class PassResult:
    run_id: str
    ref: SceneRef
    summary: dict
    hotspots: list[dict]
    cases: dict[str, Case]
    alignment: AlignmentResult
    thermal: ThermalProduct
    anomaly: AnomalyResult
    detection: Detection
    raw: RawScene
    events: list[dict]


def _reference(ctx: Context, image: np.ndarray, mask: np.ndarray, ref: SceneRef, cloud_fraction: float):
    stored = ctx.services.blobs.get_arrays(REFERENCE_KEY)
    if stored is not None:
        return stored["image"], stored["mask"].astype(bool), False
    if cloud_fraction <= 0.3:
        ctx.services.blobs.put_arrays(REFERENCE_KEY, image=image, mask=mask.astype(np.uint8))
        ctx.services.repo.put_meta("reference_scene", ref.scene_id)
        return image, mask, True
    return None, None, False


def align_scene(ctx: Context, raw: RawScene, thermal: ThermalProduct, timer: Timer):
    band = raw.reflectance[ctx.settings.alignment.band]
    mask = thermal.valid & np.isfinite(band)
    moving = to_uint8(band, mask)
    ref_img, ref_mask, is_new = _reference(ctx, moving, mask, raw.ref, thermal.cloud_fraction)
    if ref_img is None:
        return AlignmentResult(method="none", ok=False, matrix=IDENTITY, note="no clear reference scene yet"), None, moving, mask
    if is_new:
        return AlignmentResult(method="reference", ok=True, matrix=IDENTITY, residual_px=0.0, shift_before_px=0.0, shift_after_px=0.0, note="this pass is the registration reference"), ref_img, moving, mask
    with timer("align"):
        result = ctx.registrar.register_checked(ref_img, moving, ref_mask, mask)
    return result, ref_img, moving, mask


def warp_thermal(thermal: ThermalProduct, matrix, shape: tuple[int, int]) -> ThermalProduct:
    warp = lambda a: warp_to_reference(a, matrix, shape)
    lst = warp(thermal.lst)
    ndvi = warp(thermal.ndvi)
    albedo = warp(thermal.albedo)
    valid = warp_to_reference(thermal.valid, matrix, shape) & np.isfinite(lst) & np.isfinite(ndvi) & np.isfinite(albedo)
    water = warp_to_reference(thermal.water, matrix, shape) if thermal.water is not None else None
    return ThermalProduct(lst=lst, brightness_temp=warp(thermal.brightness_temp), emissivity=warp(thermal.emissivity), ndvi=ndvi, albedo=albedo, valid=valid, cloud_fraction=thermal.cloud_fraction, water=water)


def false_color(raw: RawScene, matrix, shape: tuple[int, int]) -> np.ndarray:
    channels = [to_uint8(raw.reflectance[b], None, clahe=False) for b in ("red", "nir08", "swir22")]
    bgr = np.dstack(channels)
    return warp_to_reference(bgr, matrix, shape, fill=0)


def _previous(ctx: Context, current: SceneRef):
    repo = ctx.services.repo
    last = repo.get_meta("last_run_id")
    if not last:
        return None
    run = repo.get_run(last)
    arrays = ctx.services.blobs.get_arrays(f"runs/{last}/maps.npz")
    if not run or not arrays:
        return None
    days = days_between(run["acquired"], current.acquired)
    if days <= 0:
        return None
    prior = arrays["candidates"].astype(bool) if "candidates" in arrays else arrays["labels"] > 0
    return {"anomaly": arrays["anomaly"].astype(np.float32), "days": days, "scene_id": run["scene_id"], "prior": prior}


def _observable_fn(grid, valid: np.ndarray):
    def observable(x: float, y: float) -> bool:
        col, row = grid.colrow(x, y)
        r, c = int(round(float(row))), int(round(float(col)))
        return 0 <= r < valid.shape[0] and 0 <= c < valid.shape[1] and bool(valid[r, c])
    return observable


def run_pass(ctx: Context, ref: SceneRef | str, *, agent: bool = True, run_id: str | None = None) -> PassResult:
    with ctx.lock:
        return _run_pass(ctx, ref, agent=agent, run_id=run_id)


def _run_pass(ctx: Context, ref: SceneRef | str, *, agent: bool, run_id: str | None) -> PassResult:
    s = ctx.settings
    services = ctx.services
    repo, blobs = services.repo, services.blobs
    grid = ctx.grid
    timer = Timer()
    started = utcnow()
    if isinstance(ref, str):
        ref = ctx.source.get_ref(ref)
    run_id = run_id or new_id("R", started)
    repo.put_run({"run_id": run_id, "scene_id": ref.scene_id, "acquired": ref.acquired.isoformat(), "source": ref.source, "status": "running", "started": started.isoformat(), "agent": agent})
    try:
        with timer("load"):
            raw = ctx.source.load(ref, grid)
        with timer("thermal"):
            native = derive_thermal(raw, s.thermal)
        alignment, ref_img, moving, moving_mask = align_scene(ctx, raw, native, timer)
        with timer("warp"):
            thermal = warp_thermal(native, alignment.matrix, grid.shape)
        with timer("anomaly"):
            anomaly = compute_anomaly(thermal.lst, thermal.valid, thermal.albedo, thermal.ndvi, s.hotspots, s.thermal, thermal.water)
        previous = _previous(ctx, ref)
        with timer("hotspots"):
            detection = detect_hotspots(anomaly, thermal.lst, thermal.valid, grid, ref, s.hotspots, ctx.industrial_mask, previous["prior"] if previous else None)
        observable = _observable_fn(grid, thermal.valid)
        with timer("tracking"):
            all_tracks = [Track.from_dict(t) for t in repo.list_tracks()]
            active = [t for t in all_tracks if t.status != "inactive"]
            assoc = associate(active, detection.hotspots, s.tracking)
            touched = update_tracks(active, detection.hotspots, assoc, s.tracking, repo.next_track_id, observable)
            tracks = {t.track_id: t for t in all_tracks}
            tracks.update({t.track_id: t for t in touched})
        analogue_index = build_analogue_index([t for t in tracks.values() if len(t.points) >= 3], ctx.roads, s.risk.d0_m)
        cases: dict[str, Case] = {}
        with timer("assess"):
            for label_id, h in enumerate(detection.hotspots, start=1):
                track = tracks.get(h.track_id)
                kin = kinematics(track, s.tracking.history_points, s.tracking.position_noise_m) if track else None
                flow = None
                if previous is not None:
                    r0, c0, r1, c1 = h.bbox_px
                    pad = 12
                    sl = (slice(max(0, r0 - pad), r1 + pad), slice(max(0, c0 - pad), c1 + pad))
                    flow = front_motion(previous["anomaly"][sl], anomaly.anomaly[sl], detection.labels[sl] == label_id, days=previous["days"], res_m=grid.res)
                spread = estimate_spread(kin if kin and kin.n_points >= 2 else None, flow, h.mean_anomaly_k, s.projection)
                _, distance = ctx.roads.nearest(hotspot_geometry(h.polygon, h.centroid))
                projection = None
                if spread.source != "none" and distance is not None and distance <= 4 * s.risk.d0_m:
                    projection = project_hotspot(h, label_id, detection.labels, anomaly.anomaly, spread, grid, ctx.roads, s.projection)
                conf_inputs = ConfidenceInputs(h.cloud_ring_fraction, alignment.ok, alignment.residual_px, kin.n_points if kin else 1, h.industrial_overlap, thermal.cloud_fraction, alignment_agreement_px=alignment.agreement_px)
                conf, notes = confidence(conf_inputs, s.hotspots, s.alignment)
                neighbours = analogue_index.query(signature(h.area_m2, h.mean_anomaly_k, h.max_anomaly_k, kin, distance), k=5, exclude_track=h.track_id)
                precedent = summarise(neighbours) if neighbours else None
                risk = assess(h, kin, projection, ctx.roads, s.risk, conf, notes, precedent)
                case = Case(h, track, kin, spread, projection, risk, conf_inputs)
                if precedent:
                    case.evidence["analogues"] = precedent
                if track is not None and track.label == "industrial":
                    case.excluded = True
                    case.evidence["industrial"] = {"verdict": "industrial", "reasons": [f"track {track.track_id} was classified earlier as an industrial heat source"], "cached": True}
                cases[h.hotspot_id] = case
        best_by_track: dict[str, float] = {}
        for case in cases.values():
            tid = case.hotspot.track_id
            if tid:
                best_by_track[tid] = max(best_by_track.get(tid, 0.0), case.risk.score)
        for track in tracks.values():
            if not track.closed_segments:
                continue
            if track.track_id in best_by_track:
                track.quiet_passes = track.quiet_passes + 1 if best_by_track[track.track_id] < s.risk.medium else 0
            elif track.points and observable(*track.points[-1].centroid):
                track.quiet_passes += 1
        tracer = Tracer(run_id, services.trace_sinks)
        tracer.emit("perceive", None, observation={
            "scene_id": ref.scene_id,
            "acquired": ref.acquired.isoformat(),
            "cloud_fraction": round(thermal.cloud_fraction, 3),
            "alignment": {"method": alignment.method, "ok": alignment.ok, "inliers": alignment.n_inliers, "residual_px": alignment.residual_px, "shift_before_px": alignment.shift_before_px, "shift_after_px": alignment.shift_after_px, "check_method": alignment.check_method, "agreement_px": alignment.agreement_px},
            "sigma_k": round(anomaly.sigma_k, 3),
            "trend": anomaly.trend,
            "hotspots": len(detection.hotspots),
        })
        flagged = ("near_road", "cloud_edge", "industrial_candidate")
        agent_cases = {cid: c for cid, c in cases.items() if c.risk.score >= s.risk.consider or any(c.flags(s)[k] for k in flagged)}
        for cid, case in cases.items():
            if cid in agent_cases and agent:
                continue
            case.decision = "log"
            case.decision_reason = "agent disabled for this pass" if not agent else f"risk {case.risk.score:.2f} is below {s.risk.consider} and it is away from roads"
            tracer.emit("skip", cid, observation={"risk": case.risk.score, "level": case.risk.level, "distance_m": case.risk.distance_m}, decision="log", rationale=case.decision_reason)
        executor = None
        if agent:
            with timer("agent"):
                planner = build_planner(s, services.llm)
                executor = ActionExecutor(s, ctx.roads, ctx.routes, repo, services.notifier, Guard(s), tracer, run_id, planner, ref.acquired)
                tool_ctx = ToolContext(
                    settings=s,
                    source=ctx.source,
                    grid=grid,
                    registrar=ctx.registrar,
                    alignment=alignment,
                    roads=ctx.roads,
                    vlm=ctx.vlm,
                    repo=repo,
                    run_id=run_id,
                    acquired=ref.acquired,
                    reference_image=ref_img,
                    scene_image=moving,
                    scene_mask=moving_mask,
                    false_color=false_color(raw, alignment.matrix, grid.shape) if ctx.vlm.available else None,
                )
                closure_tracks = [t for t in tracks.values() if t.closed_segments]
                FireWatchAgent(s, agent_cases, Toolbox(tool_ctx), planner, executor, tracer, ctx.roads, closure_tracks).run()
        with timer("persist"):
            records = []
            for cid, case in cases.items():
                rec = jsonable(case.hotspot)
                rec.update({
                    "risk": jsonable(case.risk),
                    "decision": case.decision,
                    "decision_reason": case.decision_reason,
                    "segments": case.segments,
                    "projection": jsonable(case.projection) if case.projection else None,
                    "spread": jsonable(case.spread),
                    "kinematics": jsonable(case.kin) if case.kin else None,
                    "evidence": jsonable(case.evidence),
                    "tools": case.tried,
                    "actions": [a.kind for a in case.actions],
                })
                repo.put_hotspot(run_id, rec)
                records.append(rec)
                if case.track and case.track.points and case.track.points[-1].hotspot_id == cid:
                    case.track.points[-1].risk = case.risk.score
            touched_ids = {t.track_id for t in touched} | {c.hotspot.track_id for c in cases.values() if c.hotspot.track_id}
            for track in tracks.values():
                if track.track_id in touched_ids or track.closed_segments:
                    repo.put_track(jsonable(track))
            valid_lst = thermal.lst[thermal.valid]
            lo, hi = (np.percentile(valid_lst, [2, 99.5]) if valid_lst.size else (280.0, 330.0))
            prov = {"runId": run_id, "sceneId": ref.scene_id, "acquired": ref.acquired.isoformat(), "source": ref.source, "platform": ref.platform, "alignment": alignment.method, "software": f"Jharia Fire Watch {__version__}"}
            lst_img, bounds = image_layer(thermal.lst, grid, float(lo), float(hi), "inferno", 200, provenance={**prov, "layer": "land surface temperature", "units": "K"})
            anom = np.where(np.nan_to_num(anomaly.anomaly, nan=0.0) > 1.0, anomaly.anomaly, np.nan)
            anom_img, _ = image_layer(anom, grid, 1.0, 15.0, "YlOrRd", 235, provenance={**prov, "layer": "heat above the local background", "units": "K"})
            blobs.put_bytes(f"runs/{run_id}/lst.webp", lst_img, "image/webp")
            blobs.put_bytes(f"runs/{run_id}/anomaly.webp", anom_img, "image/webp")
            if all(b in raw.reflectance for b in ("red", "nir08", "swir16")):
                blobs.put_bytes(f"runs/{run_id}/color.webp", color_layer(raw.reflectance, alignment.matrix, grid, {**prov, "layer": "satellite image"}), "image/webp")
            has_photo = all(b in raw.reflectance for b in ("red", "green", "blue"))
            if has_photo:
                photo_img, _ = photo_layer(raw.reflectance, grid, {**prov, "layer": "Landsat photo"}, alignment.matrix)
                blobs.put_bytes(f"runs/{run_id}/photo.webp", photo_img, "image/webp")
            base = warp_to_reference(to_uint8(raw.reflectance[s.alignment.band], None, clahe=False), alignment.matrix, grid.shape)
            frame = render_frame(base, anomaly.anomaly, records, ctx.roads, grid, ref.acquired)
            blobs.put_bytes(frame_key(run_id), encode_image(".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, 80], {**prov, "layer": "time-lapse frame"}), "image/webp")
            blobs.put_json(f"runs/{run_id}/hotspots.geojson", hotspot_features(records, grid))
            blobs.put_json(f"runs/{run_id}/cones.geojson", cone_features(records, grid))
            blobs.put_arrays(f"runs/{run_id}/maps.npz", anomaly=anomaly.anomaly.astype(np.float16), lst=thermal.lst.astype(np.float16), valid=thermal.valid.astype(np.uint8), labels=detection.labels.astype(np.int16), candidates=detection.candidates.astype(np.uint8))
            tracer.flush(blobs, f"runs/{run_id}/trace.jsonl")
        decisions = Counter(c.decision or "none" for c in cases.values())
        levels = Counter(c.risk.level for c in cases.values())
        actions = executor.records if executor else []
        finished = utcnow()
        summary = {
            "run_id": run_id,
            "scene_id": ref.scene_id,
            "acquired": ref.acquired.isoformat(),
            "source": ref.source,
            "platform": ref.platform,
            "status": "done",
            "agent": agent,
            "started": started.isoformat(),
            "finished": finished.isoformat(),
            "seconds": round((finished - started).total_seconds(), 2),
            "cloud_fraction": round(thermal.cloud_fraction, 4),
            "alignment": jsonable(alignment),
            "sigma_k": round(anomaly.sigma_k, 3),
            "trend": anomaly.trend,
            "hotspots": len(detection.hotspots),
            "by_level": dict(levels),
            "decisions": dict(decisions),
            "actions": [{"kind": a.kind, "actor": a.actor, "target": a.target, "executed": a.executed, "track_id": a.track_id} for a in actions],
            "closed_segments": sorted(ctx.roads.closed),
            "planner": build_planner(s, services.llm).name,
            "layers": {"lst": f"runs/{run_id}/lst.webp", "anomaly": f"runs/{run_id}/anomaly.webp", "color": f"runs/{run_id}/color.webp", "frame": frame_key(run_id), "photo": f"runs/{run_id}/photo.webp" if has_photo else None, "bounds": bounds, "lst_range_k": [round(float(lo), 1), round(float(hi), 1)]},
            "timings": timer.marks,
            "trace_events": len(tracer.events),
        }
        repo.put_run(summary)
        repo.put_meta("last_run_id", run_id)
        metrics = services.metrics
        dims = {"aoi": s.aoi.name}
        metrics.put("Hotspots", len(detection.hotspots), "Count", dims)
        metrics.put("HighRiskHotspots", levels.get("high", 0), "Count", dims)
        metrics.put("RoadsClosedByAgent", sum(1 for a in actions if a.kind == "close_road"), "Count", dims)
        metrics.put("PassSeconds", summary["seconds"], "Seconds", dims)
        if alignment.residual_px is not None:
            metrics.put("AlignmentResidualMeters", alignment.residual_px * grid.res, "None", dims)
        return PassResult(run_id, ref, summary, records, cases, alignment, thermal, anomaly, detection, raw, tracer.as_dicts())
    except Exception as exc:
        log.exception("pass %s failed", ref.scene_id if isinstance(ref, SceneRef) else ref)
        repo.put_run({"run_id": run_id, "scene_id": getattr(ref, "scene_id", str(ref)), "acquired": getattr(ref, "acquired", started).isoformat(), "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500], "started": started.isoformat()})
        raise

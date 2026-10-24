from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from firewatch.config import Settings
from firewatch.data.synthetic import SyntheticWorld
from firewatch.evaluation.common import INK, INK_2, SERIES, md_table, save, styled_axes, write_report
from firewatch.physics.thermal import derive_thermal
from firewatch.vision.anomaly import compute_anomaly
from firewatch.vision.hotspots import detect_hotspots

VARIANTS = (
    ("detrended", {"detrend": True}, {"persistence": False}, True),
    ("detrended + persistence", {"detrend": True}, {"persistence": True}, True),
    ("detrended, sharp inputs, water kept", {"detrend": True, "psf_sigma_px": 0.0}, {"persistence": False}, False),
    ("raw temperature", {"detrend": False}, {"persistence": False}, True),
)
PLAIN = "detrended, sharp inputs, water kept"
WORLDS = (("tuned", {}), ("held-out", {"profile": "holdout", "seed": 23}))


def score_scene(labels: np.ndarray, truth: dict, valid: np.ndarray, min_area_px: int) -> dict:
    fires = {name: mask & valid for name, mask in truth["fire_masks"].items()}
    present = {name: m for name, m in fires.items() if int(m.sum()) >= min_area_px}
    all_truth = np.zeros(labels.shape, bool)
    for m in fires.values():
        all_truth |= m
    industrial = truth["industrial"]
    tp = fp = ind = 0
    for lab in range(1, int(labels.max()) + 1):
        det = labels == lab
        if not det.any():
            continue
        if (det & all_truth).any():
            tp += 1
        elif (det & industrial).any():
            ind += 1
        else:
            fp += 1
    found = [name for name, m in present.items() if (m & (labels > 0)).any()]
    det_mask = (labels > 0) & ~industrial
    union = (det_mask | all_truth).sum()
    iou = float((det_mask & all_truth).sum() / union) if union else 1.0
    seen = cv2.GaussianBlur(truth["fire_anomaly"].astype(np.float32), (0, 0), 1.3)
    missed = [{"fire": name, "true_px": int(present[name].sum()), "peak_seen_k": round(float(seen[present[name]].max()), 1)} for name in sorted(set(present) - set(found))]
    return {"fires_present": len(present), "fires_found": len(found), "missed": missed, "true_detections": tp, "false_detections": fp, "industrial_detections": ind, "pixel_iou": iou}


def evaluate_world(settings: Settings, world: SyntheticWorld, first: int = -6, last: int = 9) -> dict:
    out = {}
    raws = [world.raw_scene(i) for i in range(first, last + 1)]
    for label, thermal_update, hot_update, use_water in VARIANTS:
        thermal_cfg = settings.thermal.model_copy(update=thermal_update)
        hot_cfg = settings.hotspots.model_copy(update=hot_update)
        prior = None
        per_pass = []
        for raw in raws:
            thermal = derive_thermal(raw, thermal_cfg)
            anomaly = compute_anomaly(thermal.lst, thermal.valid, thermal.albedo, thermal.ndvi, hot_cfg, thermal_cfg, thermal.water if use_water else None)
            detection = detect_hotspots(anomaly, thermal.lst, thermal.valid, world.grid, raw.ref, hot_cfg, world.industrial_mask, prior)
            prior = detection.candidates
            row = score_scene(detection.labels, raw.truth, thermal.valid, settings.hotspots.min_area_px)
            row.update({"scene_id": raw.ref.scene_id, "promoted": sum(h.persistent for h in detection.hotspots)})
            per_pass.append(row)
        tp = sum(p["true_detections"] for p in per_pass)
        fp = sum(p["false_detections"] for p in per_pass)
        present = sum(p["fires_present"] for p in per_pass)
        found = sum(p["fires_found"] for p in per_pass)
        out[label] = {
            "passes": per_pass,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": found / present if present else None,
            "found": found,
            "present": present,
            "false_per_pass": fp / len(per_pass),
            "mean_pixel_iou": float(np.mean([p["pixel_iou"] for p in per_pass])),
            "promoted": sum(p["promoted"] for p in per_pass),
        }
    return out


def evaluate(settings: Settings, scale: float | None = None) -> dict:
    return {name: evaluate_world(settings, SyntheticWorld.from_settings(settings, scale=scale, misregister=False, **kw)) for name, kw in WORLDS}


def chart(result: dict, path: Path) -> Path:
    fig, axes = styled_axes(7.6, 5.2, rows=2)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("mean_pixel_iou", "Pixel IoU")]
    variants = [v[0] for v in VARIANTS]
    width = 0.8 / len(variants)
    for ax, (world, data) in zip(axes, result.items()):
        for i, variant in enumerate(variants):
            values = [(data[variant][key] or 0) * 100 for key, _ in metrics]
            xs = [k + (i - (len(variants) - 1) / 2) * width for k in range(len(metrics))]
            bars = ax.bar(xs, values, width=width * 0.9, color=SERIES[i], label=variant[0].upper() + variant[1:], linewidth=0)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_ylim(0, 115)
        ax.set_ylabel(f"{world.capitalize()} world (%)", color=INK_2, fontsize=8.5)
    axes[-1].set_xticks(range(len(metrics)))
    axes[-1].set_xticklabels([name for _, name in metrics], color=INK, fontsize=9)
    fig.suptitle("Hotspot detection on synthetic passes", color=INK, fontsize=10.5, x=0.02, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="lower center")
    return save(fig, path, legend_space=0.09)


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> Path:
    result = evaluate(settings, scale)
    chart_path = chart(result, out_dir / "detection.png")
    plain = result["held-out"][PLAIN]
    rows = []
    for world, data in result.items():
        for variant, v in data.items():
            rows.append([world, variant, v["precision"], f"{v['found']} of {v['present']} ({v['recall']:.3f})", v["false_per_pass"], v["mean_pixel_iou"], v["promoted"]])
    md = [
        "# Hotspot detection against synthetic ground truth",
        "",
        "Sixteen passes of two synthetic worlds. The tuned world is the one the thresholds were set on: five fires (one starts mid-season, one goes out by itself in October, one is put out in January), a mapped coke-oven battery and occasional cloud. "
        "The held-out world was built afterwards and never used for tuning: a different seed, seven irregular multi-lobed fires (three moving toward roads, two starting late, two going out), weaker heat, 1.7 times the sensor noise, more cloud, and a second coke plant that is missing from the industrial map.",
        "",
        "A fire counts as present when at least the minimum hotspot area of its true heat footprint is cloud-free. Detections on coke plants are counted separately, because excluding them is the agent's job. "
        "Persistence keeps a weak blob (2.5 K or more, at least 4 pixels) only if it overlaps a candidate or hotspot from the previous pass.",
        "",
        md_table(["World", "Variant", "Precision", "Fires found", "False hotspots per pass", "Mean pixel IoU", "Kept by persistence"], rows),
        "",
        f"![Detection on the tuned and held-out worlds]({chart_path.name})",
        "",
        "Detrending fits temperature against albedo and NDVI before looking for local anomalies. Without it, dark coal surfaces heated by the sun look like fires.",
        "",
        "## What the held-out world caught",
        "",
        f"Its first run gave false hotspots on every pass, always in the same place: the shore of a pond next to a coal pit. The row \"{PLAIN}\" is the detector as it was then: {plain['false_per_pass']:.1f} false hotspots per pass and a precision of {plain['precision']:.2f} on this world. Two things were wrong. "
        "The albedo and NDVI used for detrending were sharp 30 m values while the thermal band is 100 m resampled to 30 m, so every sharp edge between dark coal and a brighter surface left a warm fringe. "
        "And open water, about 20 K colder than the land around it, dragged the local background down. "
        f"The detrending inputs are now smoothed to the thermal footprint (Gaussian, sigma {settings.thermal.psf_sigma_px:g} px), and water pixels from the QA band plus a {settings.hotspots.water_buffer_px}-pixel ring are left out of the background and of detection. "
        "That removed every false hotspot in both worlds without losing a fire. The held-out world is no longer untouched after that fix, so the next unseen test has to be real scenes.",
    ]
    for world, data in result.items():
        missed = [[p["scene_id"], m["fire"], m["true_px"], m["peak_seen_k"]] for p in data["detrended + persistence"]["passes"] for m in p["missed"]]
        md += ["", f"## Fires missed in the {world} world (detrended + persistence)", ""]
        if missed:
            md += [f"Peak contrast is the true fire heat after the sensor blur. A new hotspot needs {settings.hotspots.min_contrast_k:g} K; a persistent one {settings.hotspots.persist_contrast_k:g} K.", "", md_table(["Scene", "Fire", "True footprint (px)", "Peak contrast seen (K)"], missed)]
        else:
            md.append("None.")
    return write_report(out_dir, "detection", result, "\n".join(md))

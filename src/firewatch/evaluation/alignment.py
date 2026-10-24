from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from firewatch.config import Settings
from firewatch.data.synthetic import SyntheticWorld, _smooth_noise
from firewatch.evaluation.common import INK, INK_2, SERIES, md_table, save, styled_axes, write_report
from firewatch.vision.registration import ESTIMATORS, FeatureRegistrar, invert_affine, to_uint8, transform_error_px

BASE = {"gamma": 1.0, "haze": 0.0, "season": 0.0, "dry": 0.0, "changes": 0, "blur": 0.0, "noise": 0.004}
CONDITIONS = {
    "Geolocation only": {**BASE},
    "Haze and low sun": {**BASE, "gamma": 0.7, "haze": 0.07, "noise": 0.006},
    "Greener vegetation": {**BASE, "season": 0.45, "noise": 0.006},
    "Cross-sensor": {**BASE, "gamma": 0.85, "haze": 0.03, "season": 0.15, "blur": 1.2, "noise": 0.008},
    "Dry season, new dumps": {**BASE, "dry": 1.2, "changes": 30},
}
METHOD_NAMES = {"orb": "ORB", "sift": "SIFT", "disk": "DISK", "aliked_lightglue": "ALIKED + LightGlue"}
METHOD_ORDER = ("orb", "sift", "disk", "aliked_lightglue")
SUCCESS_PX = 0.5
ESTIMATOR_NAMES = {"ransac": "RANSAC", "magsac": "MAGSAC++", "accurate": "USAC_ACCURATE"}
CONTAMINATION = (0.5, 0.8, 0.9, 0.95)


def make_pair(world: SyntheticWorld, crop: tuple[slice, slice], cond: dict, rng: np.random.Generator):
    nir = world.reflectance["nir08"][crop].astype(np.float32)
    veg = world.f_veg[crop]
    h, w = nir.shape
    mov = nir * (1.0 + cond["season"] * (veg - float(veg.mean())) * 2.0)
    if cond["dry"]:
        soil = np.clip(1.0 - veg - world.w_coal[crop] - world.w_water[crop], 0.0, 1.0)
        mov = mov + cond["dry"] * (veg * (0.20 - 0.38) + soil * (0.30 - 0.20))
    for _ in range(cond["changes"]):
        patch = np.zeros((h, w), np.uint8)
        cv2.ellipse(patch, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), (int(rng.uniform(6, 22)), int(rng.uniform(4, 14))), rng.uniform(0, 180), 0, 360, 1, -1)
        level = float(rng.choice([0.05, 0.30]))
        mov = np.where(patch > 0, level * (1.0 + 0.07 * rng.standard_normal((h, w))), mov).astype(np.float32)
    if cond["haze"]:
        mov = mov + cond["haze"] * (0.6 + 0.4 * _smooth_noise(rng, (h, w), 40))
    if cond["gamma"] != 1.0:
        lo, hi = float(mov.min()), float(mov.max())
        mov = lo + (hi - lo) * ((mov - lo) / max(hi - lo, 1e-6)) ** cond["gamma"]
    if cond["blur"]:
        mov = cv2.GaussianBlur(mov, (0, 0), cond["blur"])
    mov = mov + cond["noise"] * rng.standard_normal((h, w)).astype(np.float32)
    angle = rng.uniform(-0.15, 0.15)
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0 + rng.uniform(-0.001, 0.001))
    m[0, 2] += rng.uniform(-2.0, 2.0)
    m[1, 2] += rng.uniform(-2.0, 2.0)
    warped = cv2.warpAffine(mov.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return to_uint8(nir), to_uint8(warped), invert_affine(m)


def evaluate(settings: Settings, base_dir: Path, pairs: int = 5, crop_px: int = 640, scale: float | None = None) -> dict:
    world = SyntheticWorld.from_settings(settings, scale=scale, misregister=False)
    h, w = world.shape
    size = min(crop_px, h, w)
    r0, c0 = (h - size) // 2, (w - size) // 2
    crop = (slice(r0, r0 + size), slice(c0, c0 + size))
    models_dir = Path(settings.alignment.models_dir)
    registrar = FeatureRegistrar(settings.alignment, models_dir if models_dir.is_absolute() else base_dir / models_dir)
    methods = [m for m in METHOD_ORDER if m in registrar.available_methods()]
    rows = []
    for c_index, (name, cond) in enumerate(CONDITIONS.items()):
        rng = np.random.default_rng(1000 + c_index)
        samples = [make_pair(world, crop, cond, rng) for _ in range(pairs)]
        for method in methods:
            errors, inliers, seconds, ok_count = [], [], [], 0
            for ref, mov, truth in samples:
                result = registrar.register(ref, mov, method=method)
                seconds.append(result.seconds)
                inliers.append(result.n_inliers)
                if result.ok:
                    ok_count += 1
                    errors.append(transform_error_px(result.matrix, truth, ref.shape))
                else:
                    errors.append(math.inf)
            finite = [e for e in errors if math.isfinite(e)]
            rows.append({
                "condition": name,
                "method": method,
                "pairs": pairs,
                "registered": ok_count,
                "success_rate": sum(1 for e in errors if e <= SUCCESS_PX) / pairs,
                "median_error_px": float(np.median(finite)) if finite else None,
                "p90_error_px": float(np.percentile(finite, 90)) if finite else None,
                "mean_inliers": float(np.mean(inliers)),
                "mean_seconds": float(np.mean(seconds)),
            })
    estimators = estimator_study(registrar, world, crop, pairs) if "sift" in methods else None
    return {"methods": methods, "rows": rows, "success_threshold_px": SUCCESS_PX, "crop_px": size, "pairs_per_condition": pairs, "estimators": estimators}


def _fit_rows(registrar: FeatureRegistrar, corr: list, label: str, extra: dict) -> list[dict]:
    rows = []
    for est in ESTIMATORS:
        errors, inliers, seconds = [], [], []
        for src, dst, truth, shape in corr:
            start = time.perf_counter()
            matrix, mask = registrar.fit(src, dst, est) if len(src) >= 6 else (None, None)
            seconds.append(time.perf_counter() - start)
            errors.append(transform_error_px(matrix, truth, shape) if matrix is not None else math.inf)
            inliers.append(int(mask.sum()) if mask is not None else 0)
        finite = [e for e in errors if math.isfinite(e)]
        rows.append({**extra, "case": label, "estimator": est, "success_rate": sum(e <= SUCCESS_PX for e in errors) / len(errors), "median_error_px": float(np.median(finite)) if finite else None, "mean_inliers": float(np.mean(inliers)), "mean_ms": 1000 * float(np.mean(seconds))})
    return rows


def estimator_study(registrar: FeatureRegistrar, world: SyntheticWorld, crop: tuple[slice, slice], pairs: int, method: str = "sift", keep: int = 200) -> dict:
    rows = []
    for c_index, (name, cond) in enumerate(CONDITIONS.items()):
        rng = np.random.default_rng(1000 + c_index)
        corr = []
        for ref, mov, truth in (make_pair(world, crop, cond, rng) for _ in range(pairs)):
            src, dst, _, _ = registrar.correspondences(method, ref, mov)
            corr.append((src, dst, truth, ref.shape))
        rows += _fit_rows(registrar, corr, name, {"study": "condition"})
    rng = np.random.default_rng(77)
    clean = []
    for ref, mov, truth in (make_pair(world, crop, CONDITIONS["Geolocation only"], rng) for _ in range(pairs)):
        src, dst, _, _ = registrar.correspondences(method, ref, mov)
        pick = rng.choice(len(src), min(keep, len(src)), replace=False)
        clean.append((src[pick], dst[pick], truth, ref.shape))
    for share in CONTAMINATION:
        corr = []
        for src, dst, truth, shape in clean:
            n_bad = int(round(len(src) * share / (1.0 - share)))
            h, w = shape
            bad_src = np.column_stack([rng.uniform(0, w, n_bad), rng.uniform(0, h, n_bad)]).astype(np.float32)
            bad_dst = np.column_stack([rng.uniform(0, w, n_bad), rng.uniform(0, h, n_bad)]).astype(np.float32)
            order = rng.permutation(len(src) + n_bad)
            corr.append((np.vstack([src, bad_src])[order], np.vstack([dst, bad_dst])[order], truth, shape))
        rows += _fit_rows(registrar, corr, f"{share:.0%} outliers", {"study": "contamination", "outlier_share": share})
    return {"method": method, "true_matches": keep, "rows": rows}


def chart(result: dict, path: Path) -> Path:
    conditions = list(CONDITIONS)
    methods = result["methods"]
    fig, (top, bottom) = styled_axes(7.6, 5.4, rows=2)
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        rows = {r["condition"]: r for r in result["rows"] if r["method"] == method}
        xs = [k + (i - (len(methods) - 1) / 2) * width for k in range(len(conditions))]
        color = SERIES[i % len(SERIES)]
        top.bar(xs, [rows[c]["success_rate"] * 100 for c in conditions], width=width * 0.9, color=color, label=METHOD_NAMES.get(method, method), linewidth=0)
        errors = [rows[c]["median_error_px"] for c in conditions]
        bottom.bar([x for x, e in zip(xs, errors) if e is not None], [e for e in errors if e is not None], width=width * 0.9, color=color, linewidth=0)
        for x, e in zip(xs, errors):
            if e is None:
                bottom.text(x, 0.006, "failed", rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK_2)
    top.set_ylim(0, 105)
    top.set_ylabel(f"Within {result['success_threshold_px']} px (%)", color=INK_2, fontsize=8.5)
    fig.suptitle("Image alignment by condition", color=INK, fontsize=10.5, x=0.02, ha="left")
    bottom.set_yscale("log")
    bottom.set_ylim(0.005, 2.0)
    bottom.set_ylabel("Median error (px, log)", color=INK_2, fontsize=8.5)
    bottom.set_xticks(range(len(conditions)))
    bottom.set_xticklabels(conditions, color=INK, fontsize=8)
    handles, labels = top.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(methods), loc="lower center")
    return save(fig, path, legend_space=0.05)


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> Path:
    result = evaluate(settings, base_dir, scale=scale)
    chart_path = chart(result, out_dir / "alignment.png") if result["rows"] else None
    rows = [[r["condition"], METHOD_NAMES.get(r["method"], r["method"]), f"{r['registered']}/{r['pairs']}", f"{r['success_rate']:.0%}", r["median_error_px"], r["p90_error_px"], round(r["mean_inliers"]), round(r["mean_seconds"], 2)] for r in result["rows"]]
    md = [
        "# Alignment: classic features vs OpenCV 5 learned features",
        "",
        f"Synthetic NIR image pairs ({result['crop_px']} px square, {result['pairs_per_condition']} per condition) with a known shift and rotation of up to 2 px and 0.15 degrees. "
        "Error is the RMS distance between the estimated and true transform over a grid of points. "
        f"Every method fits the transform with {ESTIMATOR_NAMES[settings.alignment.estimator]} (cv2.estimateAffine2D); the estimator comparison further down is why. "
        "The dry-season pair is the hard one: vegetation darkens and bare soil brightens in the near infrared, which reverses local contrast, and thirty new dumps and pits appear.",
        "",
        md_table(["Condition", "Method", "Registered", f"Within {SUCCESS_PX} px", "Median error (px)", "90th pct error (px)", "Inliers", "Seconds"], rows),
        "",
    ]
    if chart_path:
        md.append(f"![Alignment success by condition]({chart_path.name})")
    md.append("")
    est = result.get("estimators")
    if est:
        by = {(r["case"], r["estimator"]): r for r in est["rows"]}
        cell = lambda r: f"{r['success_rate']:.0%}, {r['median_error_px']:.3f} px" if r["median_error_px"] is not None else f"{r['success_rate']:.0%}, failed"
        cases = [c for c in CONDITIONS] + [f"{x:.0%} outliers" for x in CONTAMINATION]
        md += [
            "## RANSAC, MAGSAC++ or USAC_ACCURATE",
            "",
            f"The same SIFT matches, fitted with each robust estimator that cv2.estimateAffine2D offers. The last rows keep {est['true_matches']} true matches from a clean pair and add random false ones until they make up the stated share. "
            f"Each cell gives the share of pairs within {SUCCESS_PX} px and the median error.",
            "",
            md_table(["Case"] + [ESTIMATOR_NAMES[e] for e in ESTIMATORS], [[c] + [cell(by[(c, e)]) for e in ESTIMATORS] for c in cases]),
            "",
            md_table(["Case", "Estimator", "Inliers", "Milliseconds"], [[r["case"], ESTIMATOR_NAMES[r["estimator"]], round(r["mean_inliers"]), round(r["mean_ms"], 1)] for r in est["rows"] if r["study"] == "contamination"]),
            "",
        ]
    md.append("These pairs are synthetic, so the numbers rank the methods under controlled conditions; they are not accuracy figures for real Landsat scenes.")
    return write_report(out_dir, "alignment", result, "\n".join(md))

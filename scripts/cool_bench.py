"""Time the core OpenCV work of the Fire Watch pipeline with whichever cv2 is importable.

Runs unchanged under COOL (Cloud-Optimized OpenCV Library) and under the opencv-contrib wheel from PyPI, so
the two can be compared on the same instance. Only numpy and cv2 are needed: COOL's Python environments do not
have the rest of Fire Watch installed.

    python scripts/cool_bench.py --label cool --out cool.json
    python scripts/cool_bench.py --label pip --out pip.json
    python scripts/cool_compare.py cool.json pip.json --out reports

The workload follows the pipeline on its real grid (1364 x 857 px at 30 m): SIFT alignment of two passes with a
known misregistration, warping the products onto the reference grid, the anomaly filters, and the hotspot
outlines. Every kernel also records a checksum of its result, so the comparison can show that both builds
compute the same thing, not only how fast.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import time

import cv2
import numpy as np

H, W = 857, 1364
SHIFT = (1.3, -0.8)
ROTATION_DEG = 0.1


def scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A near-infrared-like 8-bit image and a temperature field with fires, both on the pipeline grid."""
    rng = np.random.default_rng(seed)
    nir = np.zeros((H, W), np.float32)
    for sigma, weight in ((40, 0.5), (12, 0.3), (3, 0.2)):
        layer = cv2.GaussianBlur(rng.random((H, W), dtype=np.float32), (0, 0), sigma)
        layer = (layer - layer.min()) / (np.ptp(layer) + 1e-6)
        nir += weight * layer
    for _ in range(60):
        cx, cy = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        axes = (int(rng.integers(8, 40)), int(rng.integers(6, 30)))
        cv2.ellipse(nir, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(0.1, 0.9)), -1)
    for _ in range(25):
        p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        p2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.line(nir, p1, p2, float(rng.uniform(0.6, 1.0)), int(rng.integers(1, 3)), cv2.LINE_AA)
    nir8 = cv2.normalize(nir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    temp = (300 + 8 * cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 6)).astype(np.float32)
    for _ in range(7):
        cx, cy = int(rng.integers(80, W - 80)), int(rng.integers(80, H - 80))
        blob = np.zeros((H, W), np.float32)
        cv2.ellipse(blob, (cx, cy), (int(rng.integers(5, 18)), int(rng.integers(4, 14))), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        temp += float(rng.uniform(6, 20)) * cv2.GaussianBlur(blob, (0, 0), 2.0)
    return nir8, temp


def known_transform() -> np.ndarray:
    m = cv2.getRotationMatrix2D((W / 2, H / 2), ROTATION_DEG, 1.0)
    m[0, 2] += SHIFT[0]
    m[1, 2] += SHIFT[1]
    return m


def timed(fn, repeats: int, warmup: int = 2, min_sample_s: float = 0.025):
    """Median seconds per call. Fast kernels are looped inside each sample so that timer noise does not dominate."""
    for _ in range(warmup):
        result = fn()
    t0 = time.perf_counter()
    result = fn()
    inner = max(1, int(min_sample_s / max(time.perf_counter() - t0, 1e-7)))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            result = fn()
        times.append((time.perf_counter() - t0) / inner)
    times.sort()
    return result, {"median_ms": round(statistics.median(times) * 1000, 4), "min_ms": round(times[0] * 1000, 4), "p90_ms": round(times[min(len(times) - 1, int(0.9 * len(times)))] * 1000, 4), "runs": repeats, "calls_per_run": inner}


def checksum(a) -> float:
    a = np.asarray(a, np.float64)
    return round(float(np.nansum(a)), 3)


def build_info() -> dict:
    info = cv2.getBuildInformation()
    keep = {}
    for key in ("Baseline", "Dispatched code generation", "Parallel framework", "Version control", "KleidiCV", "HAL", "ONNX", "Lapack"):
        m = re.search(rf"^\s*{re.escape(key)}[^:]*:\s*(.+)$", info, re.M)
        if m:
            keep[key] = m.group(1).strip()
    keep["mentions_cool"] = bool(re.search(r"\bCOOL\b|cloud.optimi[sz]ed", info, re.I))
    keep["mentions_kleidicv"] = "kleidicv" in info.lower()
    keep["contrib_xfeatures2d"] = hasattr(cv2, "xfeatures2d")
    keep["dnn_new_engine"] = hasattr(cv2.dnn, "ENGINE_NEW") if hasattr(cv2, "dnn") else False
    return keep


def run(repeats: int, threads: int | None) -> dict:
    if threads:
        cv2.setNumThreads(threads)
    ref8, temp = scene(7)
    m_true = known_transform()
    moving8 = cv2.warpAffine(ref8, m_true, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    moving_temp = cv2.warpAffine(temp, m_true, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    kernels = {}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    (a8, b8), k = timed(lambda: (clahe.apply(ref8), clahe.apply(moving8)), repeats)
    kernels["clahe"] = {**k, "checksum": checksum(a8)}

    sift = cv2.SIFT_create(nfeatures=4000)
    (kpa, da), k = timed(lambda: sift.detectAndCompute(a8, None), repeats)
    kpb, db = sift.detectAndCompute(b8, None)
    kernels["sift_detect"] = {**k, "checksum": len(kpa)}

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match():
        pairs = matcher.knnMatch(da, db, k=2)
        return [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]

    good, k = timed(match, repeats)
    kernels["sift_match"] = {**k, "checksum": len(good)}

    src = np.float32([kpa[g.queryIdx].pt for g in good])
    dst = np.float32([kpb[g.trainIdx].pt for g in good])
    (m_est, inliers), k = timed(lambda: cv2.estimateAffine2D(src, dst, method=cv2.USAC_ACCURATE, ransacReprojThreshold=1.5, maxIters=5000, confidence=0.999), repeats)
    kernels["estimate_affine_usac"] = {**k, "checksum": int(inliers.sum()) if inliers is not None else 0}

    def align():
        ka, dsa = sift.detectAndCompute(clahe.apply(ref8), None)
        kb, dsb = sift.detectAndCompute(clahe.apply(moving8), None)
        pairs = matcher.knnMatch(dsa, dsb, k=2)
        g = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
        s = np.float32([ka[x.queryIdx].pt for x in g])
        d = np.float32([kb[x.trainIdx].pt for x in g])
        return cv2.estimateAffine2D(s, d, method=cv2.USAC_ACCURATE, ransacReprojThreshold=1.5, maxIters=5000, confidence=0.999)[0]

    m_align, k = timed(align, repeats)
    grid = np.stack(np.meshgrid(np.linspace(0, W - 1, 20), np.linspace(0, H - 1, 20)), -1).reshape(-1, 2)
    ones = np.ones((len(grid), 1))
    err = np.hypot(*((np.hstack([grid, ones]) @ (m_align - m_true).T).T)) if m_align is not None else np.array([np.inf])
    kernels["sift_alignment_total"] = {**k, "checksum": round(float(np.sqrt(np.mean(err ** 2))), 4), "rms_error_px": round(float(np.sqrt(np.mean(err ** 2))), 4)}

    inv = cv2.invertAffineTransform(m_align if m_align is not None else m_true)
    warped, k = timed(lambda: cv2.warpAffine(moving_temp, inv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan), repeats)
    kernels["warp_affine_f32"] = {**k, "checksum": checksum(warped)}

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x = (inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]).astype(np.float32)
    map_y = (inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]).astype(np.float32)
    remapped, k = timed(lambda: cv2.remap(moving_temp, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan), repeats)
    kernels["remap_f32"] = {**k, "checksum": checksum(remapped)}

    blurred, k = timed(lambda: cv2.GaussianBlur(temp, (0, 0), 1.4, borderType=cv2.BORDER_REFLECT), repeats)
    kernels["gaussian_blur_f32"] = {**k, "checksum": checksum(blurred)}

    background, k = timed(lambda: cv2.boxFilter(temp, -1, (61, 61), borderType=cv2.BORDER_REFLECT), repeats)
    kernels["box_filter_61"] = {**k, "checksum": checksum(background)}

    gx, k = timed(lambda: cv2.Sobel(temp, cv2.CV_32F, 1, 0, ksize=3), repeats)
    kernels["sobel_f32"] = {**k, "checksum": checksum(gx)}

    anomaly = temp - background
    mask = (anomaly > 3.0).astype(np.uint8)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated, k = timed(lambda: cv2.dilate(mask, ellipse), repeats)
    kernels["dilate"] = {**k, "checksum": checksum(dilated)}
    eroded, k = timed(lambda: cv2.erode(dilated, ellipse), repeats)
    kernels["erode"] = {**k, "checksum": checksum(eroded)}

    (n, labels, stats, _), k = timed(lambda: cv2.connectedComponentsWithStats(eroded, connectivity=8), repeats)
    kernels["connected_components"] = {**k, "checksum": int(n)}

    def outlines():
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        for c in contours:
            shapes.append((cv2.approxPolyDP(c, 1.0, True), cv2.contourArea(cv2.convexHull(c)), cv2.minAreaRect(c)))
        return contours, shapes

    (contours, shapes), k = timed(outlines, repeats)
    kernels["contours_and_shape"] = {**k, "checksum": len(contours)}

    small, k = timed(lambda: cv2.resize(temp, (W // 4, H // 4), interpolation=cv2.INTER_AREA), repeats)
    kernels["resize_area"] = {**k, "checksum": checksum(small)}

    if hasattr(cv2, "DISOpticalFlow_create"):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        a = cv2.normalize(anomaly, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        b = cv2.warpAffine(a, m_true, (W, H))
        flow, k = timed(lambda: dis.calc(a, b, None), repeats)
        kernels["dis_optical_flow"] = {**k, "checksum": round(float(np.median(flow[..., 0])), 3)}

    pipeline_ms = sum(v["median_ms"] for key, v in kernels.items() if key not in ("sift_detect", "sift_match", "estimate_affine_usac", "clahe"))
    return {"kernels": kernels, "core_pipeline_median_ms": round(pipeline_ms, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True, help="name for this build, for example cool or pip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0, help="cv2.setNumThreads; 0 keeps the build's default")
    args = ap.parse_args()
    result = {
        "label": args.label,
        "opencv": cv2.__version__,
        "cv2_path": os.path.dirname(cv2.__file__),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "opencv_threads": cv2.getNumThreads(),
        "build": build_info(),
        "grid": f"{W} x {H} px",
        **run(args.repeats, args.threads),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"label": result["label"], "opencv": result["opencv"], "core_pipeline_median_ms": result["core_pipeline_median_ms"], "align_ms": result["kernels"]["sift_alignment_total"]["median_ms"], "rms_px": result["kernels"]["sift_alignment_total"]["rms_error_px"]}))


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        part = re.search(r"CPU part\s*:\s*(\S+)", text)
        name = re.search(r"model name\s*:\s*(.+)", text)
        if name:
            return name.group(1).strip()
        parts = {"0xd0c": "Neoverse N1 (Graviton2)", "0xd40": "Neoverse V1 (Graviton3)", "0xd4f": "Neoverse V2 (Graviton4)"}
        if part:
            return parts.get(part.group(1), part.group(1))
    except OSError:
        pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    main()

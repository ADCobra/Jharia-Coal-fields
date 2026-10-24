"""Compare two cool_bench.py results (COOL against the PyPI wheel) and write reports/cool.md and cool.json.

    python scripts/cool_compare.py --cool cool-1.json cool-2.json cool-3.json --pip pip-1.json pip-2.json pip-3.json --out reports
        [--pipeline-cool bench_cool.json --pipeline-pip bench_pip.json]

Give several rounds per build, run alternately (COOL, wheel, COOL, wheel, ...), so that any drift in the
machine's speed hits both builds alike; each kernel's time is the median over the rounds.

The optional pipeline files are the bench.json written by `firewatch eval bench` under each build, which adds
the full per-pass timings of the real pipeline to the kernel table.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LABELS = {
    "clahe": "CLAHE on both passes",
    "sift_detect": "SIFT keypoints and descriptors (4000)",
    "sift_match": "Brute-force matching, 0.78 ratio test",
    "estimate_affine_usac": "estimateAffine2D, USAC_ACCURATE",
    "sift_alignment_total": "Whole SIFT alignment of a pass",
    "warp_affine_f32": "warpAffine, float32 temperature",
    "remap_f32": "remap, float32",
    "gaussian_blur_f32": "GaussianBlur, sigma 1.4 (thermal footprint)",
    "box_filter_61": "boxFilter 61 x 61 (local background)",
    "sobel_f32": "Sobel, float32",
    "dilate": "dilate, 5 x 5 ellipse",
    "erode": "erode, 5 x 5 ellipse",
    "connected_components": "connectedComponentsWithStats",
    "contours_and_shape": "findContours + approxPolyDP + convexHull + minAreaRect",
    "resize_area": "resize, INTER_AREA",
    "dis_optical_flow": "DIS optical flow (front motion)",
}


def same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= max(1e-3, 1e-4 * max(abs(a), abs(b)))
    return a == b


def merge(paths: list[str]) -> dict:
    """One result whose kernel times are the median over rounds; checksums must agree across rounds."""
    runs = [json.loads(Path(p).read_text()) for p in paths]
    merged = {**runs[0], "rounds": len(runs), "kernels": {}}
    for key in runs[0]["kernels"]:
        values = [r["kernels"][key] for r in runs if key in r["kernels"]]
        merged["kernels"][key] = {**values[0], "median_ms": round(statistics.median(v["median_ms"] for v in values), 4),
                                  "round_ms": [v["median_ms"] for v in values], "stable_checksum": all(same(v["checksum"], values[0]["checksum"]) for v in values)}
    merged["core_pipeline_median_ms"] = round(statistics.median(r["core_pipeline_median_ms"] for r in runs), 3)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cool", nargs="+", required=True)
    ap.add_argument("--pip", nargs="+", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--pipeline-cool")
    ap.add_argument("--pipeline-pip")
    args = ap.parse_args()
    cool, pip = merge(args.cool), merge(args.pip)
    rows, out = [], {"cool": cool, "pip": pip, "kernels": {}}
    for key, label in LABELS.items():
        c, p = cool["kernels"].get(key), pip["kernels"].get(key)
        if not c or not p:
            continue
        speedup = p["median_ms"] / c["median_ms"] if c["median_ms"] else None
        agree = same(c["checksum"], p["checksum"])
        out["kernels"][key] = {"cool_ms": c["median_ms"], "pip_ms": p["median_ms"], "speedup": round(speedup, 2) if speedup else None, "same_result": agree}
        rows.append(f"| {label} | {p['median_ms']:.2f} | {c['median_ms']:.2f} | {speedup:.2f}x | {'yes' if agree else 'differs'} |")
    total = pip["core_pipeline_median_ms"] / cool["core_pipeline_median_ms"]
    out["core_speedup"] = round(total, 2)
    differs = [k for k, v in out["kernels"].items() if not v["same_result"]]

    lines = [
        "# COOL against the PyPI wheel on Graviton",
        "",
        f"Both runs on the same instance: {cool['machine']}, {cool['cpus']} vCPUs, {cool.get('cpu_model', '')}. "
        f"COOL: OpenCV {cool['opencv']} from `{cool['cv2_path']}`, Python {cool['python']}, {cool['opencv_threads']} threads. "
        f"PyPI: OpenCV {pip['opencv']} from `{pip['cv2_path']}`, Python {pip['python']}, {pip['opencv_threads']} threads.",
        "",
        f"The workload is the core OpenCV work of one pass on the pipeline's grid ({cool['grid']}), produced by `scripts/cool_bench.py`: "
        "SIFT alignment of two passes with a known 1.5 px, 0.1° misregistration, warping the temperature onto the reference grid, "
        "the anomaly filters and the hotspot outlines. "
        f"Each kernel is timed as the median over repeated runs after a warm-up, in {cool['rounds']} rounds per build run alternately; the table gives the median over rounds.",
        "",
        f"**Core OpenCV work per pass: {pip['core_pipeline_median_ms']:.0f} ms with the wheel, {cool['core_pipeline_median_ms']:.0f} ms with COOL ({total:.2f}x).**",
        "",
        "| Kernel | PyPI wheel (ms) | COOL (ms) | Speed-up | Same result |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"SIFT alignment error against the known transform: {pip['kernels']['sift_alignment_total']['rms_error_px']} px with the wheel, "
        f"{cool['kernels']['sift_alignment_total']['rms_error_px']} px with COOL.",
        "",
        "Every kernel also records a checksum of its output (keypoint and match counts, inliers, sums of the filtered images, contour counts). "
        + ("Both builds produce the same result in every kernel." if not differs else
           f"These differ between the builds: {', '.join(differs)}. Small floating-point differences between optimised kernels are expected; "
           "check the alignment error above, which is what the pipeline depends on."),
    ]
    if args.pipeline_cool and args.pipeline_pip:
        pc, pp = json.loads(Path(args.pipeline_cool).read_text()), json.loads(Path(args.pipeline_pip).read_text())
        out["pipeline"] = {"cool": pc, "pip": pp}
        stages_c, stages_p = pc.get("pipeline", pc).get("stages", {}), pp.get("pipeline", pp).get("stages", {})
        lines += ["", "## The whole pipeline", "", "`firewatch eval bench` run once under each build: the full pass on the synthetic season, including the agent and the second alignment with ALIKED + LightGlue.", "",
                  "| Stage | PyPI wheel (s) | COOL (s) |", "| --- | --- | --- |"]
        for stage in stages_p:
            if stage in stages_c:
                lines.append(f"| {stage} | {stages_p[stage]:.3f} | {stages_c[stage]:.3f} |")
        mp, mc = pp.get("pipeline", pp).get("median_pass_seconds"), pc.get("pipeline", pc).get("median_pass_seconds")
        if mp and mc:
            lines += ["", f"Median time per pass: {mp:.2f} s with the wheel, {mc:.2f} s with COOL ({mp / mc:.2f}x)."]
    lines += ["", "Build details:", "", "| | PyPI wheel | COOL |", "| --- | --- | --- |"]
    for key in sorted(set(cool["build"]) | set(pip["build"])):
        lines.append(f"| {key} | {pip['build'].get(key, '')} | {cool['build'].get(key, '')} |")
    lines += ["", "Reproduce with `infra/cool_bench.sh` (see the README)."]
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cool.md").write_text("\n".join(lines) + "\n")
    (dest / "cool.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import platform
import re
import statistics
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from firewatch.config import Settings
from firewatch.evaluation.alignment import CONDITIONS, METHOD_NAMES, make_pair
from firewatch.evaluation.common import md_table, write_report
from firewatch.vision.cvcompat import model_path
from firewatch.vision.registration import FeatureRegistrar

# FIREWATCH_BENCH_SKIP=disk leaves out the DISK network, whose 1024 px forward pass does not fit in 2 GB of memory.
SKIP = {name.strip().lower() for name in os.environ.get("FIREWATCH_BENCH_SKIP", "").split(",") if name.strip()}
STAGES = ("load", "thermal", "align", "warp", "anomaly", "hotspots", "tracking", "assess", "agent", "persist")


def machine() -> dict:
    return {"machine": platform.machine(), "processor": platform.processor() or platform.machine(), "cpus": os.cpu_count(), "system": platform.system(), "opencv_threads": cv2.getNumThreads(), "opencv": cv2.__version__}


def pipeline_timings(settings: Settings, base_dir: Path, scale: float | None, passes: int = 4) -> dict:
    from firewatch.pipeline.context import build_context
    from firewatch.pipeline.runner import run_demo

    models = Path(settings.alignment.models_dir)
    models = models if models.is_absolute() else (base_dir / models).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        s = settings.with_overrides({
            "data": {"source": "synthetic", "synthetic": {"scale": scale or settings.data.synthetic.scale}},
            "storage": {"backend": "local", "root": f"{tmp}/blobs"},
            "state": {"backend": "sqlite", "sqlite_path": f"{tmp}/state.db"},
            "notify": {"backend": "console", "outbox": f"{tmp}/outbox.jsonl"},
            "alignment": {"models_dir": str(models)},
            "observability": {"log_group": None, "metrics_namespace": None},
            "agent": {"planner": "policy"},
        })
        ctx = build_context(s, tmp)
        summaries = run_demo(ctx, passes=passes, archive=2)
        grid = f"{ctx.grid.width} x {ctx.grid.height} px"
    agent_passes = [p for p in summaries if p.get("agent")]
    stages = {}
    for stage in STAGES:
        values = [p["timings"][stage] for p in agent_passes if stage in p["timings"]]
        if values:
            stages[stage] = round(statistics.median(values), 3)
    return {"grid": grid, "passes": len(agent_passes), "median_pass_seconds": round(statistics.median(p["seconds"] for p in agent_passes), 2), "stages": stages, "method": agent_passes[-1]["alignment"]["method"] if agent_passes else None}


def matcher_timings(settings: Settings, base_dir: Path, size: int = 768, repeats: int = 3) -> list[dict]:
    from firewatch.data.synthetic import SyntheticWorld

    world = SyntheticWorld.from_settings(settings, misregister=False)
    h, w = world.shape
    size = min(size, h, w)
    r0, c0 = (h - size) // 2, (w - size) // 2
    ref, mov, _ = make_pair(world, (slice(r0, r0 + size), slice(c0, c0 + size)), CONDITIONS["Geolocation only"], np.random.default_rng(5))
    models = Path(settings.alignment.models_dir)
    registrar = FeatureRegistrar(settings.alignment, models if models.is_absolute() else base_dir / models)
    rows = []
    for method in registrar.available_methods():
        if method in SKIP:
            continue
        registrar.register(ref, mov, method=method)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            registrar.register(ref, mov, method=method)
            times.append(time.perf_counter() - start)
        rows.append({"method": method, "image_px": size, "median_seconds": round(statistics.median(times), 3)})
    return rows


ENGINES = ("ENGINE_NEW", "ENGINE_ORT", "ENGINE_CLASSIC", "ENGINE_AUTO")


def _short_error(exc: Exception) -> str:
    text = str(exc)
    node = re.findall(r"Node \[(\w+)@", text)
    detail = re.findall(r"error: \(-?\d+:[^)]*\) (.*?) in function", text)
    if node and detail:
        return f"{node[-1]} node: {detail[-1]}"
    return text.strip().splitlines()[-1][:160] if text.strip() else type(exc).__name__


def build_flags() -> dict:
    flags = {}
    for line in cv2.getBuildInformation().splitlines():
        key, _, value = line.strip().partition(":")
        if key in ("ONNX Runtime", "DNN MLAS", "Lapack"):
            flags[key] = value.strip()
    return flags


def engine_timings(settings: Settings, base_dir: Path, repeats: int = 2) -> list[dict]:
    models = Path(settings.alignment.models_dir)
    models = models if models.is_absolute() else base_dir / models
    rng = np.random.default_rng(0)
    rows = []
    for label, filename, size in (("ALIKED", settings.alignment.aliked_model, 640), ("DISK", settings.alignment.disk_model, 1024)):
        path = model_path(models, filename)
        if path is None or label.lower() in SKIP:
            continue
        blob = rng.random((1, 3, size, size), dtype=np.float32)
        for engine in (e for e in ENGINES if hasattr(cv2.dnn, e)):
            row = {"model": label, "engine": engine, "input_px": size}
            try:
                start = time.perf_counter()
                net = cv2.dnn.readNetFromONNX(str(path), getattr(cv2.dnn, engine))
                row["load_seconds"] = round(time.perf_counter() - start, 3)
                names = net.getUnconnectedOutLayersNames()
                net.setInput(blob)
                net.forward(names)
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    net.setInput(blob)
                    net.forward(names)
                    times.append(time.perf_counter() - start)
                row.update({"ok": True, "forward_seconds": round(statistics.median(times), 3)})
            except cv2.error as exc:
                row.update({"ok": False, "error": _short_error(exc)})
            rows.append(row)
    return rows


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> Path:
    data = {"machine": machine(), "pipeline": pipeline_timings(settings, base_dir, scale), "matchers": matcher_timings(settings, base_dir), "engines": engine_timings(settings, base_dir), "build": build_flags()}
    m, p = data["machine"], data["pipeline"]
    md = [
        "# Run time",
        "",
        f"{m['system']} {m['machine']}, {m['cpus']} CPUs, OpenCV {m['opencv']} with {m['opencv_threads']} threads. "
        f"Synthetic grid of {p['grid']} at 30 m, {p['passes']} passes with the agent after two archive passes. Alignment used {METHOD_NAMES.get(p['method'], p['method'])}.",
        "",
        f"Median time per pass: **{p['median_pass_seconds']} s**.",
        "",
        md_table(["Stage", "Median seconds"], [[k, v] for k, v in p["stages"].items()]),
        "",
        "## Image registration alone",
        "",
        md_table(["Method", "Image (px)", "Median seconds"], [[METHOD_NAMES.get(r["method"], r["method"]), f"{r['image_px']} x {r['image_px']}", r["median_seconds"]] for r in data["matchers"]]),
        "",
    ]
    if data["engines"]:
        flags = data["build"]
        md += [
            "",
            "## OpenCV 5 DNN engines",
            "",
            f"The ALIKED and DISK networks loaded with `cv2.dnn.readNetFromONNX(path, engine)` and run on a random image. This wheel reports ONNX Runtime: {flags.get('ONNX Runtime', 'unknown')} and DNN MLAS: {flags.get('DNN MLAS', 'unknown')}.",
            "",
            md_table(["Model", "Engine", "Input (px)", "Loads", "Load seconds", "Forward seconds", "Why not"], [[r["model"], r["engine"], f"{r['input_px']} x {r['input_px']}", "yes" if r["ok"] else "no", r.get("load_seconds"), r.get("forward_seconds"), r.get("error", "")] for r in data["engines"]]),
            "",
            "The classic engine cannot import either network, which is why the matchers ask for the new engine explicitly. Without ONNX Runtime in the build, ENGINE_ORT gives the same speed as the new engine.",
        ]
    if SKIP:
        md += ["", f"Left out on this machine (`FIREWATCH_BENCH_SKIP`): {', '.join(sorted(SKIP))}."]
    md += ["", "Run `firewatch eval bench` on the Graviton instance to get the same tables for arm64."]
    return write_report(out_dir, "bench", data, "\n".join(md))

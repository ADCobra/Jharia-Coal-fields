from __future__ import annotations

import statistics
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from firewatch.config import Settings
from firewatch.evaluation.common import md_table, write_report
from firewatch.services.factory import build_services
from firewatch.services.repository import Repository

SOURCE_URL = "https://www.bcclweb.in/masterplan/FireDrilling.pdf"
REFERENCES = [
    {"label": "2017", "area_km2": 3.28, "start": "2017-01-01", "end": "2017-06-30", "note": "NRSC coal fire study, 2017 baseline (whole coalfield)"},
    {"label": "2020", "area_km2": 1.89, "start": "2020-01-01", "end": "2021-05-31", "note": "NRSC coal fire study, Landsat-8 band 10, 2020 period (the summary cites May 2021 data)"},
]


def fire_area_km2(hotspots: list[dict]) -> float:
    polys = []
    for h in hotspots:
        if h.get("decision") == "exclude" or len(h.get("polygon") or []) < 4:
            continue
        poly = Polygon(h["polygon"]).buffer(0)
        if not poly.is_empty:
            polys.append(poly)
    return float(unary_union(polys).area) / 1e6 if polys else 0.0


def run_areas(repo: Repository, sources: tuple[str, ...] = ("landsat", "real"), max_cloud: float = 0.2) -> list[dict]:
    rows = []
    for run in repo.list_runs(limit=100000):
        if run.get("status") != "done" or run.get("source") not in sources:
            continue
        cloud = run.get("cloud_fraction")
        if cloud is not None and cloud > max_cloud:
            continue
        hotspots = repo.list_hotspots(run["run_id"])
        rows.append({
            "run_id": run["run_id"],
            "scene_id": run["scene_id"],
            "date": str(run["acquired"])[:10],
            "cloud_fraction": cloud,
            "hotspots": sum(1 for h in hotspots if h.get("decision") != "exclude"),
            "area_km2": round(fire_area_km2(hotspots), 3),
        })
    return sorted(rows, key=lambda r: r["date"])


def _summary(window: list[dict], ref_km2: float) -> dict:
    areas = [r["area_km2"] for r in window]
    return {"median_km2": round(statistics.median(areas), 3), "min_km2": min(areas), "max_km2": max(areas), "ratio": round(statistics.median(areas) / ref_km2, 2)}


def compare(rows: list[dict]) -> list[dict]:
    out = []
    for ref in REFERENCES:
        window = [r for r in rows if ref["start"] <= r["date"] <= ref["end"]]
        item = {**ref, "passes": len(window)}
        if window:
            item.update(_summary(window, ref["area_km2"]))
        out.append(item)
    later = [r for r in rows if r["date"] > REFERENCES[-1]["end"]]
    if later:
        latest = REFERENCES[-1]
        span = f"{later[0]['date'][:7]} to {later[-1]['date'][:7]}"
        out.append({"label": f"{span} against {latest['label']}", "area_km2": latest["area_km2"], "start": later[0]["date"], "end": later[-1]["date"], "note": f"processed season compared with the latest published NRSC figure ({latest['label']}); years apart, so only a range check", "passes": len(later), **_summary(later, latest["area_km2"])})
    return out


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> Path:
    repo = build_services(settings, base_dir).repo
    rows = run_areas(repo)
    comparisons = compare(rows)
    md = [
        "# Fire area against the NRSC coal fire mapping",
        "",
        f"NRSC (ISRO) mapped the Jharia coalfield fire area as about 3.28 km² in 2017 and about 1.89 km² in 2020, using Landsat-8 band 10 ([BCCL fire report]({SOURCE_URL})). "
        "This compares the total area of non-industrial hotspots per clear pass (cloud at most 20 %) with those figures. Thresholds and methods differ, so a ratio between about 0.5 and 2 means the detector is in the right range; it is not a pixel-level validation.",
        "",
    ]
    if not rows:
        md += [
            "**Not run here:** the state store has no processed Landsat passes. To run it:",
            "",
            "```",
            "firewatch -c config/landsat.yaml backfill --start 2020-01-01 --end 2021-05-31",
            "firewatch -c config/landsat.yaml backfill --start 2017-01-01 --end 2017-06-30",
            "firewatch -c config/landsat.yaml eval nrsc",
            "```",
        ]
    else:
        md += [md_table(["Period", "NRSC area (km²)", "Clear passes", "Our median (km²)", "Range (km²)", "Ratio"], [[c["label"], c["area_km2"], c["passes"], c.get("median_km2"), f"{c['min_km2']}–{c['max_km2']}" if c.get("passes") else "", c.get("ratio")] for c in comparisons]), ""]
        md += ["## Passes", "", md_table(["Date", "Scene", "Cloud", "Hotspots", "Area (km²)"], [[r["date"], r["scene_id"], r["cloud_fraction"], r["hotspots"], r["area_km2"]] for r in rows])]
    return write_report(out_dir, "nrsc", {"references": comparisons, "passes": rows, "source": SOURCE_URL}, "\n".join(md))

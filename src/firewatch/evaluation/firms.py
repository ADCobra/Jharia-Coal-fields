from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path

from shapely.geometry import Point, Polygon

from firewatch.config import Settings
from firewatch.evaluation.common import md_table, write_report
from firewatch.geo.grid import WGS84, transformer
from firewatch.services.factory import build_services
from firewatch.services.repository import Repository

RADIUS_M = {"VIIRS": 400.0, "MODIS": 1000.0}


def read_firms_csv(path: str | Path, bbox: tuple[float, float, float, float]) -> list[dict]:
    minx, miny, maxx, maxy = bbox
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
            instrument = (row.get("instrument") or "VIIRS").upper()
            rows.append({
                "lat": lat,
                "lon": lon,
                "date": dt.date.fromisoformat(row["acq_date"]),
                "time": row.get("acq_time", ""),
                "instrument": "MODIS" if "MODIS" in instrument else "VIIRS",
                "confidence": row.get("confidence", ""),
                "frp": float(row["frp"]) if row.get("frp") else None,
                "daynight": row.get("daynight", ""),
            })
    return rows


def _geometry(h: dict):
    if len(h.get("polygon") or []) >= 4:
        poly = Polygon(h["polygon"]).buffer(0)
        if not poly.is_empty:
            return poly
    return Point(h["centroid"])


def match_run(hotspots: list[dict], detections: list[dict], crs: str) -> dict:
    to_map = transformer(WGS84, crs)
    geoms = [(h, _geometry(h)) for h in hotspots]
    det_rows = []
    for d in detections:
        x, y = to_map.transform(d["lon"], d["lat"])
        radius = RADIUS_M[d["instrument"]]
        near = [h for h, g in geoms if g.distance(Point(x, y)) <= radius]
        det_rows.append({**d, "near_fire": any(h.get("decision") != "exclude" for h in near), "near_industrial": any(h.get("decision") == "exclude" for h in near)})
    confirmed = 0
    for h, g in geoms:
        if h.get("decision") == "exclude":
            continue
        for d in detections:
            x, y = to_map.transform(d["lon"], d["lat"])
            if g.distance(Point(x, y)) <= RADIUS_M[d["instrument"]]:
                confirmed += 1
                break
    fires = sum(1 for h in hotspots if h.get("decision") != "exclude")
    return {
        "detections": len(detections),
        "near_fire": sum(r["near_fire"] for r in det_rows),
        "near_industrial": sum(r["near_industrial"] and not r["near_fire"] for r in det_rows),
        "unmatched": sum(not r["near_fire"] and not r["near_industrial"] for r in det_rows),
        "fire_hotspots": fires,
        "fire_hotspots_with_firms": confirmed,
    }


def evaluate(repo: Repository, detections: list[dict], crs: str, max_days: int = 1, source: str = "landsat") -> list[dict]:
    rows = []
    for run in sorted(repo.list_runs(limit=100000), key=lambda r: r["acquired"]):
        if run.get("status") != "done" or run.get("source") != source:
            continue
        day = dt.date.fromisoformat(str(run["acquired"])[:10])
        window = [d for d in detections if abs((d["date"] - day).days) <= max_days]
        if not window:
            continue
        rows.append({"date": day.isoformat(), "scene_id": run["scene_id"], **match_run(repo.list_hotspots(run["run_id"]), window, crs)})
    return rows


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None, csv_path: str | None = None) -> Path:
    csv_path = csv_path or os.environ.get("FIREWATCH_FIRMS_CSV")
    md = [
        "# Hotspots against NASA FIRMS active-fire detections",
        "",
        "FIRMS (VIIRS 375 m, MODIS 1 km) flags flaming or very hot pixels. Subsurface coal fires are often too cool for it, and it also flags coke ovens and flares, so only partial agreement is expected. "
        "A detection counts as matched when one of our hotspots lies within 400 m (VIIRS) or 1 km (MODIS) of it, within one day of the Landsat pass.",
        "",
    ]
    data: dict = {"csv": csv_path}
    if not csv_path or not Path(csv_path).exists():
        md += [
            "**Not run here:** no FIRMS file was given. Download an archive CSV for the area of interest from https://firms.modaps.eosdis.nasa.gov/download/ and run:",
            "",
            "```",
            "FIREWATCH_FIRMS_CSV=fire_archive_SV-C2.csv firewatch -c config/landsat.yaml eval firms",
            "```",
        ]
        return write_report(out_dir, "firms", data, "\n".join(md))
    detections = read_firms_csv(csv_path, settings.aoi.bbox)
    rows = evaluate(build_services(settings, base_dir).repo, detections, settings.aoi.crs)
    data.update({"detections_in_aoi": len(detections), "passes": rows})
    if not rows:
        md.append(f"{len(detections)} FIRMS detections fall inside the area, but none is within a day of a processed Landsat pass.")
        return write_report(out_dir, "firms", data, "\n".join(md))
    total = {k: sum(r[k] for r in rows) for k in ("detections", "near_fire", "near_industrial", "unmatched", "fire_hotspots", "fire_hotspots_with_firms")}
    data["total"] = total
    md += [
        md_table(["Measure", "Value"], [
            ["FIRMS detections within a day of a pass", total["detections"]],
            ["Next to one of our fire hotspots", total["near_fire"]],
            ["Next to a hotspot we excluded as industrial", total["near_industrial"]],
            ["Not near any hotspot", total["unmatched"]],
            ["Our fire hotspots with a FIRMS detection nearby", f"{total['fire_hotspots_with_firms']} of {total['fire_hotspots']}"],
        ]),
        "",
        md_table(["Date", "Scene", "FIRMS", "Near fire", "Near industrial", "Unmatched", "Our fires", "With FIRMS"], [[r["date"], r["scene_id"], r["detections"], r["near_fire"], r["near_industrial"], r["unmatched"], r["fire_hotspots"], r["fire_hotspots_with_firms"]] for r in rows]),
    ]
    return write_report(out_dir, "firms", data, "\n".join(md))

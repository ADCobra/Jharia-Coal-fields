"""Download roads and industrial sites for the area of interest from OpenStreetMap.

Writes data/jharia_roads.geojson and data/jharia_industrial.geojson, the two files
config/landsat.yaml expects. OpenStreetMap data is (c) OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

OVERPASS = "https://overpass-api.de/api/interpreter"
ROAD_CLASSES = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track"
HAUL_CLASSES = {"track", "service"}


def query(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    area = f"({south},{west},{north},{east})"
    return f"""
[out:json][timeout:180];
(
  way["highway"~"^({ROAD_CLASSES})$"]{area};
  way["landuse"="industrial"]{area};
  way["man_made"="works"]{area};
  way["industrial"]{area};
);
out tags geom;
"""


def to_geojson(elements: list[dict], haul_near_mines: bool = True) -> tuple[dict, dict]:
    roads, sites = [], []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        coords = [[round(p["lon"], 6), round(p["lat"], 6)] for p in el["geometry"]]
        if "highway" in tags:
            if len(coords) < 2:
                continue
            haul = tags["highway"] in HAUL_CLASSES and haul_near_mines
            roads.append({"type": "Feature", "properties": {"id": f"osm-{el['id']}", "name": tags.get("name") or tags.get("ref") or tags["highway"], "highway": tags["highway"], "haul": haul}, "geometry": {"type": "LineString", "coordinates": coords}})
        elif len(coords) >= 4 and coords[0] == coords[-1]:
            sites.append({"type": "Feature", "properties": {"id": f"osm-{el['id']}", "name": tags.get("name") or tags.get("industrial") or "industrial site"}, "geometry": {"type": "Polygon", "coordinates": [coords]}})
    return {"type": "FeatureCollection", "features": roads}, {"type": "FeatureCollection", "features": sites}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bbox", type=float, nargs=4, default=[86.10, 23.63, 86.50, 23.86], metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--out", default="data")
    parser.add_argument("--no-haul", action="store_true", help="do not mark service roads and tracks as haul roads")
    args = parser.parse_args()
    response = httpx.post(OVERPASS, data={"data": query(tuple(args.bbox))}, timeout=240)
    response.raise_for_status()
    roads, sites = to_geojson(response.json().get("elements", []), not args.no_haul)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jharia_roads.geojson").write_text(json.dumps(roads))
    (out / "jharia_industrial.geojson").write_text(json.dumps(sites))
    print(f"{len(roads['features'])} road ways and {len(sites['features'])} industrial polygons written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

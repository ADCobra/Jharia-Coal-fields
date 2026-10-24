from __future__ import annotations

import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import polygonize, transform, unary_union
from shapely.strtree import STRtree

from firewatch.geo.grid import WGS84, transformer

MAJOR = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4, "motorway_link": 5, "trunk_link": 5, "primary_link": 5, "secondary_link": 5, "tertiary_link": 5, "unclassified": 6, "road": 6}
MINOR = {"residential": 7, "living_street": 7, "service": 8, "track": 9}
MINOR_NAMES = {"residential": "Local road", "living_street": "Local road", "service": "Service road", "track": "Mine track"}
SECTION_M = 400.0
MINE_ROAD_REACH_M = 1500.0
LANE_REACH_M = 250.0
HAUL_REACH_M = 300.0
MIN_MINE_M2 = 30_000.0
MAX_INDUSTRIAL_M2 = 3_000_000.0
TOWN_PLACES = ("city", "town", "suburb")


def _name(tags: dict) -> str | None:
    return tags.get("name:en") or tags.get("name") or tags.get("official_name")


def _line(el: dict) -> list[tuple[float, float]]:
    return [(p["lon"], p["lat"]) for p in el.get("geometry") or [] if p]


def _polygons(el: dict):
    if el["type"] == "way":
        coords = _line(el)
        if len(coords) >= 4 and coords[0] == coords[-1]:
            poly = Polygon(coords).buffer(0)
            return poly if not poly.is_empty else None
        return None
    if el["type"] == "relation":
        outer = [LineString(_line_of(m)) for m in el.get("members", []) if m.get("role") in ("outer", "") and len(_line_of(m)) >= 2]
        polys = [p.buffer(0) for p in polygonize(unary_union(outer))] if outer else []
        polys = [p for p in polys if not p.is_empty]
        if not polys:
            return None
        merged = unary_union(polys)
        return merged if not merged.is_empty else None
    return None


def _line_of(member: dict) -> list[tuple[float, float]]:
    return [(p["lon"], p["lat"]) for p in member.get("geometry") or [] if p]


MINE_WORDS = ("mine", "colliery", "coal project", "cluster", "ocp", "ugp", "quarry")


def _is_mine(tags: dict) -> bool:
    if tags.get("industrial") == "mine" or tags.get("landuse") == "quarry":
        return True
    name = (_name(tags) or "").lower()
    return tags.get("landuse") == "industrial" and any(w in name for w in MINE_WORDS)


def _category(tags: dict, name: str | None) -> str:
    text = " ".join(str(v) for v in (name, tags.get("industrial"), tags.get("product"), tags.get("works"))).lower()
    if "brick" in text:
        return "brickyard"
    if "wash" in text:
        return "washery"
    if "coke" in text or "coking" in text:
        return "coke"
    if tags.get("power") == "plant" or "power" in text or "thermal" in text:
        return "power"
    if "steel" in text or "iron" in text:
        return "steel"
    if "fertil" in text:
        return "fertiliser"
    return "factory"


def _feature(geom, props: dict) -> dict:
    return {"type": "Feature", "properties": props, "geometry": mapping(geom)}


def _round(geojson: dict) -> dict:
    def fix(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(float(c[0]), 6), round(float(c[1]), 6)]
        return [fix(x) for x in c]

    for f in geojson["features"]:
        f["geometry"]["coordinates"] = fix(f["geometry"]["coordinates"])
    return geojson


def _sections(line_m: LineString, length: float) -> list[LineString]:
    from shapely.ops import substring

    count = max(1, int(math.ceil(line_m.length / length)))
    step = line_m.length / count
    return [substring(line_m, i * step, (i + 1) * step) for i in range(count)]


def derive(osm: dict, crs: str = "EPSG:32645") -> dict:
    to_m = transformer(WGS84, crs).transform
    to_ll = transformer(crs, WGS84).transform
    elements = osm.get("elements", [])

    mines, plants, places, water = [], [], [], []
    for el in elements:
        tags = el.get("tags") or {}
        if el["type"] == "node":
            if tags.get("place") in TOWN_PLACES and _name(tags):
                places.append(_feature(Point(el["lon"], el["lat"]), {"name": _name(tags), "place": tags["place"]}))
            elif tags.get("man_made") in ("mineshaft", "adit"):
                mines.append({"geom_m": Point(to_m(el["lon"], el["lat"])).buffer(60), "name": _name(tags) or "Mine shaft", "kind": "shaft"})
            continue
        poly = _polygons(el)
        if poly is None:
            continue
        poly_m = transform(to_m, poly)
        if tags.get("landuse") == "quarry":
            if poly_m.area >= MIN_MINE_M2:
                mines.append({"geom_m": poly_m, "name": _name(tags) or "", "kind": "open-cast", "resource": tags.get("resource")})
        elif tags.get("natural") == "water":
            water.append(poly_m)
        elif _is_mine(tags):
            if poly_m.area >= MIN_MINE_M2:
                mines.append({"geom_m": poly_m, "name": _name(tags) or "", "kind": "open-cast", "resource": tags.get("resource")})
        elif tags.get("landuse") in ("industrial",) or tags.get("man_made") == "works" or "industrial" in tags or tags.get("power") == "plant":
            plants.append({"geom_m": poly_m, "name": _name(tags), "category": _category(tags, _name(tags)), "tags": tags})

    mine_union = unary_union([m["geom_m"] for m in mines]) if mines else None
    mines.sort(key=lambda m: -m["geom_m"].area)
    unnamed = 0
    for m in mines:
        if not m["name"]:
            unnamed += 1
            m["name"] = f"Open-cast mine {unnamed}"

    industrial = []
    for p in plants:
        area = p["geom_m"].area
        if area > MAX_INDUSTRIAL_M2 and p["category"] in ("factory", "fertiliser"):
            continue
        if mine_union is not None and p["geom_m"].intersection(mine_union).area > 0.3 * area and p["category"] == "factory":
            continue
        name = p["name"] or {"washery": "Coal washery", "coke": "Coke plant", "power": "Power plant", "steel": "Steel plant", "fertiliser": "Fertiliser plant", "brickyard": "Brick kiln"}.get(p["category"], "Industrial site")
        industrial.append({"geom_m": p["geom_m"], "name": name, "category": p["category"]})

    reach = mine_union.buffer(MINE_ROAD_REACH_M) if mine_union is not None else None
    lane_reach = mine_union.buffer(LANE_REACH_M) if mine_union is not None else None
    haul_zone = mine_union.buffer(HAUL_REACH_M) if mine_union is not None else None
    ways = []
    for el in elements:
        tags = el.get("tags") or {}
        hw = tags.get("highway")
        if el["type"] != "way" or not hw or tags.get("area") == "yes":
            continue
        rank = MAJOR.get(hw, MINOR.get(hw))
        if rank is None:
            continue
        coords = _line(el)
        if len(coords) < 2:
            continue
        line_m = LineString([to_m(*c) for c in coords])
        if line_m.length < 1:
            continue
        if hw in MINOR and (reach is None or not line_m.intersects(reach)):
            continue
        if hw in ("residential", "living_street") and (lane_reach is None or not line_m.intersects(lane_reach)):
            continue
        name = _name(tags) or tags.get("ref")
        haul = hw in ("track", "service", "unclassified") and haul_zone is not None and line_m.intersects(haul_zone)
        ways.append({"rank": rank, "hw": hw, "name": name, "line_m": line_m, "haul": haul, "osm": el.get("id")})

    named_order: dict[str, int] = {}
    for w in sorted(ways, key=lambda w: (w["rank"], -w["line_m"].length)):
        if w["name"] and w["name"] not in named_order:
            named_order[w["name"]] = len(named_order) + 1
    counters = {"T": 0, "L": 0}
    sections_by_code: dict[str, int] = {}
    roads = []
    for w in sorted(ways, key=lambda w: (w["rank"], w["name"] or "~", w["osm"] or 0)):
        if w["name"]:
            code = f"R{named_order[w['name']]}"
            label = w["name"]
        else:
            prefix = "T" if w["hw"] == "track" else "L"
            counters[prefix] += 1
            code = f"{prefix}{counters[prefix]}"
            label = MINOR_NAMES.get(w["hw"], "Local road")
        for piece in _sections(w["line_m"], SECTION_M):
            n = sections_by_code.get(code, 0) + 1
            sections_by_code[code] = n
            roads.append(_feature(transform(to_ll, piece), {"id": f"{code}-{n:02d}", "name": label, "highway": w["hw"], "haul": w["haul"], "osm_id": w["osm"]}))

    mine_features = [_feature(transform(to_ll, m["geom_m"]), {"name": m["name"], "kind": m["kind"], "area_ha": round(m["geom_m"].area / 1e4, 1)}) for m in mines]
    industrial_features = [_feature(transform(to_ll, p["geom_m"]), {"name": p["name"], "category": p["category"], "area_ha": round(p["geom_m"].area / 1e4, 1)}) for p in industrial]

    routes = _routes(mines, industrial, places, roads, to_m, to_ll)
    return {
        "roads": _round({"type": "FeatureCollection", "features": roads}),
        "mines": _round({"type": "FeatureCollection", "features": mine_features}),
        "industrial": _round({"type": "FeatureCollection", "features": industrial_features}),
        "places": _round({"type": "FeatureCollection", "features": places}),
        "routes": routes,
    }


def _routes(mines: list[dict], industrial: list[dict], places: list[dict], roads: list[dict], to_m, to_ll, limit: int = 6) -> list[dict]:
    if not roads:
        return []
    road_lines = [transform(to_m, LineString(f["geometry"]["coordinates"])) for f in roads]
    tree = STRtree(road_lines)
    targets = [(p["name"], p["geom_m"].centroid) for p in industrial if p["category"] == "washery"]
    if not targets:
        targets = [(p["name"], p["geom_m"].centroid) for p in industrial if p["category"] in ("coke", "power", "steel")]
    if not targets:
        targets = [(f["properties"]["name"], Point(to_m(*f["geometry"]["coordinates"]))) for f in places if f["properties"]["place"] in ("city", "town")]
    if not targets:
        return []
    out, used = [], set()
    for m in mines:
        if m["kind"] != "open-cast":
            continue
        c = m["geom_m"].centroid
        near = road_lines[int(tree.nearest(c))]
        if near.distance(c) > 2000:
            continue
        name, target = min(targets, key=lambda t: t[1].distance(c))
        if target.distance(c) > 20000 or target.distance(c) < 800 or m["name"] in used:
            continue
        used.add(m["name"])
        src = to_ll(c.x, c.y)
        dst = to_ll(target.x, target.y)
        out.append({"name": f"{m['name']} to {name}", "from": [round(src[0], 6), round(src[1], 6)], "to": [round(dst[0], 6), round(dst[1], 6)]})
        if len(out) >= limit:
            break
    return out


def ensure_derived(real_dir: Path, crs: str = "EPSG:32645", force: bool = False) -> Path:
    import yaml

    src = real_dir / "osm.json"
    out = real_dir / "derived"
    marker = out / "roads.geojson"
    if not src.exists():
        if marker.exists():
            return out
        raise FileNotFoundError(f"{src} is missing: run scripts/fetch_real_data.py first")
    if not force and marker.exists() and marker.stat().st_mtime >= src.stat().st_mtime:
        return out
    derived = derive(json.loads(src.read_text()), crs)
    out.mkdir(parents=True, exist_ok=True)
    for key in ("roads", "mines", "industrial", "places"):
        (out / f"{key}.geojson").write_text(json.dumps(derived[key]))
    (out / "routes.yaml").write_text(yaml.safe_dump(derived["routes"], sort_keys=False, allow_unicode=True))
    return out

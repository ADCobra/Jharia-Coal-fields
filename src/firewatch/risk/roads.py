from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from firewatch.geo.grid import WGS84, GridSpec, transformer


@dataclass
class RoadSegment:
    segment_id: str
    name: str
    haul: bool
    line: LineString

    @property
    def length_m(self) -> float:
        return float(self.line.length)


@dataclass
class Route:
    name: str
    source: tuple[float, float]
    target: tuple[float, float]


def _node(pt: Iterable[float]) -> tuple[float, float]:
    x, y = pt
    return (round(float(x) * 2) / 2, round(float(y) * 2) / 2)


class RoadNetwork:
    def __init__(self, segments: list[RoadSegment], crs: str):
        self.crs = crs
        self.segments = {s.segment_id: s for s in segments}
        self._order = list(self.segments)
        self._tree = STRtree([self.segments[i].line for i in self._order]) if segments else None
        self.graph = nx.Graph()
        for seg in segments:
            coords = list(seg.line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                na, nb = _node(a), _node(b)
                if na != nb:
                    self.graph.add_edge(na, nb, weight=math.dist(na, nb), segment=seg.segment_id)
        self.closed: set[str] = set()

    @classmethod
    def from_geojson(cls, data: dict | str | Path, crs: str) -> "RoadNetwork":
        if not isinstance(data, dict):
            data = json.loads(Path(data).read_text())
        to_map = transformer(WGS84, crs)
        segments = []
        for i, feat in enumerate(data.get("features", [])):
            geom = shape(feat["geometry"])
            props = feat.get("properties") or {}
            lines = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
            for j, line in enumerate(lines):
                xs, ys = to_map.transform(*zip(*line.coords))
                sid = str(props.get("id") or props.get("osm_id") or f"road-{i}")
                if len(lines) > 1:
                    sid = f"{sid}-{j}"
                haul = props.get("haul")
                if isinstance(haul, str):
                    haul = haul.lower() in ("yes", "true", "1")
                segments.append(RoadSegment(sid, str(props.get("name") or props.get("highway") or "road"), bool(haul), LineString(list(zip(xs, ys)))))
        return cls(segments, crs)

    def __len__(self) -> int:
        return len(self.segments)

    def _candidates(self, geom: BaseGeometry, radius: float | None = None) -> list[str]:
        if self._tree is None:
            return []
        if radius is None:
            return [self._order[int(self._tree.nearest(geom))]]
        return [self._order[int(i)] for i in self._tree.query(geom.buffer(radius))]

    def nearest(self, geom: BaseGeometry, haul_only: bool = False) -> tuple[str | None, float | None]:
        if not self.segments:
            return None, None
        if haul_only:
            pool = [s for s in self.segments.values() if s.haul]
            if not pool:
                return None, None
            best = min(pool, key=lambda s: s.line.distance(geom))
            return best.segment_id, float(best.line.distance(geom))
        sid = self._candidates(geom)[0]
        return sid, float(self.segments[sid].line.distance(geom))

    def within(self, geom: BaseGeometry, distance: float, haul_only: bool = False) -> list[str]:
        out = []
        for sid in self._candidates(geom, distance):
            seg = self.segments[sid]
            if haul_only and not seg.haul:
                continue
            if seg.line.distance(geom) <= distance:
                out.append(sid)
        return sorted(set(out))

    def approach_direction(self, segment_id: str, point: tuple[float, float]) -> tuple[float, float]:
        p = Point(point)
        on_road, _ = nearest_points(self.segments[segment_id].line, p)
        dx, dy = on_road.x - p.x, on_road.y - p.y
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 0 else (0.0, 0.0)

    def rasterize(self, grid: GridSpec, buffer_m: float = 0.0, ids: Iterable[str] | None = None) -> np.ndarray:
        mask = np.zeros(grid.shape, np.uint8)
        thickness = max(1, int(round(2 * buffer_m / grid.res)) + 1)
        selected = self.segments.values() if ids is None else [self.segments[i] for i in ids]
        for seg in selected:
            pix = grid.map_polygon_to_pixel(seg.line.coords)
            cv2.polylines(mask, [np.round(pix).astype(np.int32).reshape(-1, 1, 2)], False, 1, thickness)
        return mask.astype(bool)

    def close(self, ids: Iterable[str]) -> None:
        self.closed.update(i for i in ids if i in self.segments)

    def reopen(self, ids: Iterable[str]) -> None:
        self.closed.difference_update(ids)

    def nearest_node(self, xy: tuple[float, float]) -> tuple[float, float] | None:
        if self.graph.number_of_nodes() == 0:
            return None
        return min(self.graph.nodes, key=lambda n: (n[0] - xy[0]) ** 2 + (n[1] - xy[1]) ** 2)

    def _view(self, closed: set[str]):
        return nx.subgraph_view(self.graph, filter_edge=lambda a, b: self.graph[a][b]["segment"] not in closed)

    def route_length(self, route: Route, closed: set[str] | None = None) -> float | None:
        src, dst = self.nearest_node(route.source), self.nearest_node(route.target)
        if src is None or dst is None:
            return None
        try:
            return float(nx.shortest_path_length(self._view(closed or set()), src, dst, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def route_path(self, route: Route, closed: set[str] | None = None) -> list[tuple[float, float]]:
        src, dst = self.nearest_node(route.source), self.nearest_node(route.target)
        try:
            return list(nx.shortest_path(self._view(closed or set()), src, dst, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def reroute_report(self, routes: list[Route], closed: set[str] | None = None) -> list[dict]:
        closed = self.closed if closed is None else closed
        report = []
        for route in routes:
            before = self.route_length(route, set())
            after = self.route_length(route, closed)
            report.append({
                "route": route.name,
                "before_m": None if before is None else round(before),
                "after_m": None if after is None else round(after),
                "extra_m": None if before is None or after is None else round(after - before),
                "blocked": after is None and before is not None,
            })
        return report

    def to_geojson(self, statuses: dict[str, dict] | None = None) -> dict:
        to_ll = transformer(self.crs, WGS84)
        statuses = statuses or {}
        features = []
        for sid, seg in self.segments.items():
            lon, lat = to_ll.transform(*zip(*seg.line.coords))
            status = statuses.get(sid, {})
            features.append({
                "type": "Feature",
                "properties": {"id": sid, "name": seg.name, "haul": seg.haul, "closed": sid in self.closed, "status": status.get("status", "closed" if sid in self.closed else "open"), "reason": status.get("reason")},
                "geometry": {"type": "LineString", "coordinates": [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]},
            })
        return {"type": "FeatureCollection", "features": features}

    def routes_geojson(self, routes: list[Route]) -> dict:
        to_ll = transformer(self.crs, WGS84)
        features = []
        for route in routes:
            path = self.route_path(route, self.closed)
            normal = self.route_path(route, set())
            for kind, nodes in (("current", path), ("normal", normal)):
                if len(nodes) < 2:
                    continue
                lon, lat = to_ll.transform(*zip(*nodes))
                features.append({"type": "Feature", "properties": {"route": route.name, "kind": kind}, "geometry": {"type": "LineString", "coordinates": [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]}})
        return {"type": "FeatureCollection", "features": features}


def load_routes(spec, network: RoadNetwork) -> list[Route]:
    to_map = transformer(WGS84, network.crs)
    if isinstance(spec, (str, Path)) and str(spec) != "auto" and Path(spec).exists():
        import yaml

        spec = yaml.safe_load(Path(spec).read_text())
    if isinstance(spec, list):
        routes = []
        for item in spec:
            sx, sy = to_map.transform(*item["from"])
            tx, ty = to_map.transform(*item["to"])
            routes.append(Route(item["name"], (sx, sy), (tx, ty)))
        return routes
    return auto_routes(network)


def auto_routes(network: RoadNetwork, limit: int = 6) -> list[Route]:
    g = network.graph
    if g.number_of_nodes() < 2:
        return []
    depot = min(g.nodes, key=lambda n: n[0])
    haul_ends = []
    for sid, seg in network.segments.items():
        if not seg.haul:
            continue
        for end in (seg.line.coords[0], seg.line.coords[-1]):
            node = _node(end)
            if node in g and g.degree(node) == 1:
                haul_ends.append((sid, node))
    routes = [Route(f"{sid.split('-')[0]} to depot", node, depot) for sid, node in haul_ends[:limit]]
    if not routes:
        far = max(g.nodes, key=lambda n: n[0])
        routes.append(Route("west to east", depot, far))
    return routes


def hotspot_geometry(polygon: list[tuple[float, float]], centroid: tuple[float, float]) -> BaseGeometry:
    if len(polygon) >= 4:
        geom = Polygon(polygon).buffer(0)
        if not geom.is_empty:
            return geom
    return Point(centroid)

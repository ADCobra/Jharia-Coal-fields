from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import shape

from firewatch.config import Settings
from firewatch.data.sources import SceneSource
from firewatch.data.synthetic import SyntheticSource, SyntheticWorld
from firewatch.geo.grid import WGS84, GridSpec, transformer
from firewatch.risk.roads import RoadNetwork, Route, load_routes
from firewatch.services.factory import Services, build_services
from firewatch.vision.registration import FeatureRegistrar
from firewatch.vision.vlm import VisionLanguageChecker, build_vlm

log = logging.getLogger(__name__)
SCENERY_KEY = "static/scenery.webp"
SCENERY_META = "static/scenery.json"


@dataclass
class Context:
    settings: Settings
    services: Services
    source: SceneSource
    grid: GridSpec
    roads: RoadNetwork
    routes: list[Route]
    registrar: FeatureRegistrar
    vlm: VisionLanguageChecker
    industrial_mask: np.ndarray | None
    base_dir: Path
    world: SyntheticWorld | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def apply_persisted_closures(self) -> None:
        self.roads.closed.clear()
        for sid, record in self.services.repo.get_closures().items():
            if record.get("status") == "closed":
                self.roads.close([sid])


def rasterize_polygons(geojson: dict, grid: GridSpec) -> np.ndarray:
    mask = np.zeros(grid.shape, np.uint8)
    to_map = transformer(WGS84, grid.crs)
    for feat in geojson.get("features", []):
        geom = shape(feat["geometry"])
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in polys:
            xs, ys = to_map.transform(*zip(*poly.exterior.coords))
            pix = grid.map_polygon_to_pixel(list(zip(xs, ys)))
            cv2.fillPoly(mask, [np.round(pix).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def _load_geojson(value: str | dict | None, base: Path) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    path = Path(value)
    path = path if path.is_absolute() else base / path
    return json.loads(path.read_text())


def _real_photo(source, grid: GridSpec, blobs) -> None:
    refs = [r for r in source.refs() if r.properties.get("aoi_cloud") is not None]
    if not refs:
        return
    best = min(refs, key=lambda r: (round(float(r.properties["aoi_cloud"]), 2), -r.acquired.timestamp()))
    key = {"scene": best.scene_id, "width": grid.width, "height": grid.height, "kind": "photo"}
    meta = blobs.get_json(SCENERY_META)
    if meta and meta.get("key") == key and blobs.exists(SCENERY_KEY):
        return
    from firewatch.pipeline.products import photo_layer

    raw = source.load(best, grid)
    prov = {"layer": "Landsat photo", "sceneId": best.scene_id, "acquired": best.acquired.isoformat(), "platform": best.platform, "source": str(best.properties.get("archive") or "Landsat Collection 2 Level-2")}
    data, bounds = photo_layer(raw.reflectance, grid, prov)
    blobs.put_bytes(SCENERY_KEY, data, "image/webp")
    blobs.put_json(SCENERY_META, {"key": key, "bounds": list(bounds), "kind": "photo", "scene_id": best.scene_id, "acquired": best.acquired.isoformat(), "platform": best.platform})


def build_context(settings: Settings, base_dir: str | Path | None = None, *, source: SceneSource | None = None, world: SyntheticWorld | None = None) -> Context:
    base = Path(base_dir or Path.cwd())
    services = build_services(settings, base)
    if settings.data.source == "synthetic" and source is None:
        world = world or SyntheticWorld.from_settings(settings)
        source = SyntheticSource(world)
    if world is not None:
        grid = world.grid
        roads_geo = world.roads_geojson()
        industrial_geo = world.industrial_geojson()
        routes_spec = world.routes()
    else:
        if settings.data.source == "real":
            from firewatch.data.osm import ensure_derived
            from firewatch.data.real import RealSceneSource, real_dir

            ensure_derived(real_dir(settings, base), settings.aoi.crs)
            source = source or RealSceneSource(settings, base)
        else:
            from firewatch.data.landsat import LandsatSource

            source = source or LandsatSource(settings)
        grid = GridSpec.from_bbox(settings.aoi.bbox, settings.aoi.crs, settings.aoi.resolution_m)
        roads_geo = _load_geojson(settings.roads.path, base)
        if roads_geo is None:
            raise ValueError("roads.path must point to a GeoJSON of roads for real data (see scripts/fetch_osm.py)")
        industrial_geo = _load_geojson(settings.masks.industrial, base)
        routes_spec = settings.roads.routes
    roads = RoadNetwork.from_geojson(roads_geo, grid.crs)
    routes = load_routes(routes_spec if not isinstance(routes_spec, str) or routes_spec == "auto" else str(base / routes_spec), roads)
    models_dir = Path(settings.alignment.models_dir)
    registrar = FeatureRegistrar(settings.alignment, models_dir if models_dir.is_absolute() else base / models_dir)
    industrial_mask = rasterize_polygons(industrial_geo, grid) if industrial_geo else None
    ctx = Context(
        settings=settings,
        services=services,
        source=source,
        grid=grid,
        roads=roads,
        routes=routes,
        registrar=registrar,
        vlm=build_vlm(settings.vlm),
        industrial_mask=industrial_mask,
        base_dir=base,
        world=world,
    )
    ctx.apply_persisted_closures()
    blobs = services.blobs
    blobs.put_json("static/grid.json", {**grid.to_dict(), "lonlat_bounds": grid.lonlat_bounds()})
    blobs.put_json("static/roads.geojson", roads_geo)
    blobs.put_json("static/industrial.geojson", industrial_geo or {"type": "FeatureCollection", "features": []})
    blobs.put_json("static/routes.json", [{"name": r.name} for r in routes])
    if settings.data.source == "real":
        from firewatch.data.real import real_dir

        derived = real_dir(settings, base) / "derived"
        for name in ("mines", "places"):
            path = derived / f"{name}.geojson"
            blobs.put_json(f"static/{name}.geojson", json.loads(path.read_text()) if path.exists() else {"type": "FeatureCollection", "features": []})
        if settings.api.scenery:
            _real_photo(source, grid, blobs)
    if world is not None and settings.api.scenery:
        key = {"seed": world.seed, "profile": world.profile, "width": grid.width, "height": grid.height}
        meta = blobs.get_json(SCENERY_META)
        if not meta or meta.get("key") != key or not blobs.exists(SCENERY_KEY):
            from firewatch.data.scenery import scenery_layer

            data, bounds = scenery_layer(world)
            blobs.put_bytes(SCENERY_KEY, data, "image/webp")
            blobs.put_json(SCENERY_META, {"key": key, "bounds": list(bounds)})
    return ctx

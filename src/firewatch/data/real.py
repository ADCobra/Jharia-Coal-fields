from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

from firewatch.config import Settings
from firewatch.data.sentinel2 import Sentinel2Reader
from firewatch.data.sources import HighResPatch
from firewatch.geo.grid import GridSpec
from firewatch.geo.raster import read_to_grid
from firewatch.models import RawScene, SceneRef
from firewatch.util import parse_time

log = logging.getLogger(__name__)
REFLECTIVE = ("blue", "green", "red", "nir08", "swir16", "swir22")
PLATFORMS = {"landsat-8": "LANDSAT_8", "landsat-9": "LANDSAT_9"}
SR_SCALE = (2.75e-05, -0.2)
ST_SCALE = (0.00341802, 149.0)


def real_dir(settings: Settings, base: Path) -> Path:
    root = Path(settings.data.real_dir)
    return root if root.is_absolute() else base / root


class RealSceneSource:
    name = "real"

    def __init__(self, settings: Settings, base: Path):
        self.settings = settings
        self.root = real_dir(settings, base)
        self.scene_dir = self.root / "scenes"
        self.sentinel = Sentinel2Reader(settings)

    def refs(self) -> list[SceneRef]:
        refs = []
        for meta_path in sorted(self.scene_dir.glob("*.json")):
            tif = meta_path.with_suffix(".tif")
            if not tif.exists():
                continue
            meta = json.loads(meta_path.read_text())
            platform = meta.get("platform") or ""
            path_row = None
            if meta.get("wrs_path") is not None and meta.get("wrs_row") is not None:
                path_row = f"{int(meta['wrs_path']):03d}{int(meta['wrs_row']):03d}"
            refs.append(SceneRef(
                scene_id=meta["id"],
                acquired=parse_time(meta["datetime"]),
                platform=PLATFORMS.get(platform, platform.upper().replace("-", "_")),
                source="landsat",
                cloud_cover=meta.get("cloud_cover"),
                path_row=path_row,
                assets={"stack": str(tif)},
                properties={
                    "bands": meta["bands"],
                    "scale": meta.get("scale", {}),
                    "aoi_cloud": meta.get("aoi_cloud"),
                    "view:sun_elevation": meta.get("sun_elevation"),
                    "stac_item": meta.get("stac_item"),
                    "archive": meta.get("source"),
                    "level": "L2SP",
                },
            ))
        return sorted(refs, key=lambda r: r.acquired)

    def list_scenes(self, start: dt.datetime, end: dt.datetime) -> list[SceneRef]:
        start, end = parse_time(start), parse_time(end)
        return [r for r in self.refs() if start <= r.acquired <= end]

    def get_ref(self, scene_id: str) -> SceneRef:
        for ref in self.refs():
            if ref.scene_id == scene_id:
                return ref
        raise KeyError(f"scene {scene_id} is not in {self.scene_dir}")

    def history(self, before: dt.datetime, n: int) -> list[SceneRef]:
        before = parse_time(before)
        earlier = [r for r in self.refs() if r.acquired < before - dt.timedelta(hours=1)]
        return earlier[-n:] if n > 0 else []

    def _band(self, ref: SceneRef, grid: GridSpec, name: str, **kwargs) -> np.ndarray:
        index = ref.properties["bands"].index(name) + 1
        return read_to_grid(ref.assets["stack"], grid, band=index, **kwargs)

    def load(self, ref: SceneRef, grid: GridSpec) -> RawScene:
        scale = ref.properties.get("scale") or {}
        sr_mult, sr_add = scale.get("reflectance", SR_SCALE)
        st_mult, st_add = scale.get("surface_temperature", ST_SCALE)
        reflectance = {}
        for band in REFLECTIVE:
            if band in ref.properties["bands"]:
                dn = self._band(ref, grid, band, resampling="bilinear", nodata=0)
                reflectance[band] = (dn * sr_mult + sr_add).astype(np.float32)
        st_dn = self._band(ref, grid, "lwir11", resampling="bilinear", nodata=0)
        surface_temp = (st_dn * st_mult + st_add).astype(np.float32)
        qa_pixel = self._band(ref, grid, "qa_pixel", resampling="nearest", nodata=None, dtype="uint16", fill=1)
        return RawScene(ref=ref, grid=grid, thermal_dn=np.zeros(grid.shape, np.float32), reflectance=reflectance, qa_pixel=qa_pixel, mtl={}, l2_surface_temp=surface_temp)

    def highres(self, bounds: tuple[float, float, float, float], crs: str, when: dt.datetime) -> HighResPatch | None:
        try:
            return self.sentinel.patch(bounds, crs, parse_time(when))
        except Exception as exc:
            log.warning("Sentinel-2 check unavailable: %s", exc)
            return None

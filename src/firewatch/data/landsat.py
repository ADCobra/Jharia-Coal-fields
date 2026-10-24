from __future__ import annotations

import datetime as dt
import logging

import numpy as np

from firewatch.config import Settings
from firewatch.data import stac
from firewatch.data.mtl import OLI_BANDS, parse_mtl, reflectance_rescaling, sun_elevation
from firewatch.data.sentinel2 import Sentinel2Reader
from firewatch.data.sources import HighResPatch
from firewatch.geo.grid import GridSpec
from firewatch.geo.raster import read_to_grid
from firewatch.models import RawScene, SceneRef
from firewatch.physics.radiometry import l2_surface_temp_to_kelvin, toa_reflectance
from firewatch.util import parse_time

log = logging.getLogger(__name__)
REFLECTIVE = ("blue", "red", "nir08", "swir16", "swir22")


class LandsatSource:
    name = "landsat"

    def __init__(self, settings: Settings, with_l2: bool = False):
        self.settings = settings
        self.with_l2 = with_l2
        self.region = settings.aws.region
        self.sentinel = Sentinel2Reader(settings)
        self._refs: dict[str, SceneRef] = {}

    def _remember(self, refs: list[SceneRef]) -> list[SceneRef]:
        for ref in refs:
            self._refs[ref.scene_id] = ref
        return refs

    def list_scenes(self, start: dt.datetime, end: dt.datetime) -> list[SceneRef]:
        return self._remember(stac.search_landsat(self.settings, parse_time(start), parse_time(end)))

    def get_ref(self, scene_id: str) -> SceneRef:
        if scene_id not in self._refs:
            self._remember([stac.get_landsat_ref(self.settings, scene_id)])
        return self._refs[scene_id]

    def history(self, before: dt.datetime, n: int) -> list[SceneRef]:
        before = parse_time(before)
        start = before - dt.timedelta(days=self.settings.data.history_lookback_days)
        refs = [r for r in self.list_scenes(start, before - dt.timedelta(hours=1)) if r.acquired < before]
        return refs[-n:] if n > 0 else []

    def _requester_pays(self, href: str) -> bool:
        return self.settings.data.requester_pays and href.startswith("s3://")

    def _read(self, href: str, grid: GridSpec, **kwargs) -> np.ndarray:
        return read_to_grid(href, grid, requester_pays=self._requester_pays(href), region=self.region, **kwargs)

    def load(self, ref: SceneRef, grid: GridSpec) -> RawScene:
        assets = ref.assets
        mtl_href = assets["MTL.json"]
        mtl = parse_mtl(stac.fetch_bytes(mtl_href, self._requester_pays(mtl_href), self.region))
        sun = sun_elevation(mtl)
        thermal = self._read(assets["lwir11"], grid, resampling="bilinear", nodata=0)
        reflectance = {}
        for band in REFLECTIVE:
            dn = self._read(assets[band], grid, resampling="bilinear", nodata=0)
            mult, add = reflectance_rescaling(mtl, OLI_BANDS[band])
            reflectance[band] = toa_reflectance(dn, mult, add, sun)
        qa_pixel = self._read(assets["qa_pixel"], grid, resampling="nearest", nodata=1, dtype="uint16", fill=1)
        l2 = None
        if self.with_l2:
            l2_ref = stac.find_l2_match(self.settings, ref)
            if l2_ref is not None:
                st_dn = self._read(l2_ref.assets["lwir11"], grid, resampling="bilinear", nodata=0, fill=0)
                l2 = l2_surface_temp_to_kelvin(st_dn)
            else:
                log.info("no Level-2 surface temperature found for %s", ref.scene_id)
        return RawScene(ref=ref, grid=grid, thermal_dn=thermal, reflectance=reflectance, qa_pixel=qa_pixel, mtl=mtl, l2_surface_temp=l2)

    def highres(self, bounds: tuple[float, float, float, float], crs: str, when: dt.datetime) -> HighResPatch | None:
        return self.sentinel.patch(bounds, crs, parse_time(when))

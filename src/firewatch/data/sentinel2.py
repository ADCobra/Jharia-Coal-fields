from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np

from firewatch.config import Settings
from firewatch.data import stac
from firewatch.data.sources import HighResPatch
from firewatch.geo.grid import WGS84, GridSpec, transformer
from firewatch.geo.raster import read_to_grid
from firewatch.util import parse_time
from firewatch.vision.clouds import scl_clear_mask

log = logging.getLogger(__name__)
BAND_KEYS = {"nir": "nir", "swir16": "swir16", "swir22": "swir22"}


def _scale_offset(asset: dict) -> tuple[float, float]:
    bands = asset.get("raster:bands") or [{}]
    first = bands[0] if bands else {}
    return float(first.get("scale", 1e-4)), float(first.get("offset", 0.0))


class Sentinel2Reader:
    def __init__(self, settings: Settings, resolution_m: float = 20.0):
        self.settings = settings
        self.resolution_m = resolution_m

    def _patch_grid(self, bounds: tuple[float, float, float, float], crs: str) -> GridSpec:
        minx, miny, maxx, maxy = bounds
        res = self.resolution_m
        width = max(4, int(math.ceil((maxx - minx) / res)))
        height = max(4, int(math.ceil((maxy - miny) / res)))
        return GridSpec(crs, (res, 0.0, minx, 0.0, -res, maxy), width, height)

    def find_item(self, bounds: tuple[float, float, float, float], crs: str, when: dt.datetime) -> dict | None:
        minx, miny, maxx, maxy = bounds
        lon, lat = transformer(crs, WGS84).transform([minx, maxx], [miny, maxy])
        bbox = [min(lon), min(lat), max(lon), max(lat)]
        window = dt.timedelta(days=self.settings.data.sentinel_window_days)
        items = stac.search_items(self.settings.data.sentinel_stac_url, [self.settings.data.sentinel_collection], bbox, when - window, when + window, max_items=50)
        if not items:
            return None
        def rank(item: dict) -> tuple[float, float]:
            props = item["properties"]
            gap = abs((parse_time(props["datetime"]) - when).total_seconds()) / 86400.0
            return (props.get("eo:cloud_cover", 100.0) / 20.0 + gap, gap)
        return min(items, key=rank)

    def patch(self, bounds: tuple[float, float, float, float], crs: str, when: dt.datetime) -> HighResPatch | None:
        try:
            item = self.find_item(bounds, crs, when)
        except Exception as exc:
            log.warning("Sentinel-2 search failed: %s", exc)
            return None
        if item is None:
            return None
        grid = self._patch_grid(bounds, crs)
        assets = item["assets"]
        bands = {}
        for name, key in BAND_KEYS.items():
            scale, offset = _scale_offset(assets[key])
            dn = read_to_grid(assets[key]["href"], grid, resampling="bilinear", nodata=0)
            bands[name] = (dn * scale + offset).astype(np.float32)
        scl = read_to_grid(assets["scl"]["href"], grid, resampling="nearest", nodata=0, dtype="uint8", fill=0)
        valid = scl_clear_mask(scl) & np.isfinite(bands["swir22"])
        return HighResPatch(item["id"], parse_time(item["properties"]["datetime"]), grid, bands["nir"], bands["swir16"], bands["swir22"], valid, "sentinel-2-l2a")

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine

WGS84 = "EPSG:4326"


@lru_cache(maxsize=64)
def transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


@dataclass(frozen=True)
class GridSpec:
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int

    @classmethod
    def from_bbox(cls, bbox_lonlat: Sequence[float], crs: str, resolution: float) -> "GridSpec":
        lon0, lat0, lon1, lat1 = bbox_lonlat
        lons = np.concatenate([np.linspace(lon0, lon1, 21), np.full(21, lon1), np.linspace(lon1, lon0, 21), np.full(21, lon0)])
        lats = np.concatenate([np.full(21, lat0), np.linspace(lat0, lat1, 21), np.full(21, lat1), np.linspace(lat1, lat0, 21)])
        xs, ys = transformer(WGS84, crs).transform(lons, lats)
        res = float(resolution)
        minx = math.floor(min(xs) / res) * res
        maxx = math.ceil(max(xs) / res) * res
        miny = math.floor(min(ys) / res) * res
        maxy = math.ceil(max(ys) / res) * res
        width = int(round((maxx - minx) / res))
        height = int(round((maxy - miny) / res))
        return cls(crs, (res, 0.0, minx, 0.0, -res, maxy), width, height)

    @classmethod
    def from_dict(cls, data: dict) -> "GridSpec":
        return cls(data["crs"], tuple(data["transform"]), int(data["width"]), int(data["height"]))

    def to_dict(self) -> dict:
        return {"crs": self.crs, "transform": list(self.transform), "width": self.width, "height": self.height}

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)

    @property
    def res(self) -> float:
        return abs(self.transform[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def origin(self) -> tuple[float, float]:
        return (self.transform[2], self.transform[5])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x0, y1 = self.origin
        return (x0, y1 - self.height * self.res, x0 + self.width * self.res, y1)

    @property
    def pixel_area(self) -> float:
        return self.res * self.res

    def xy(self, rows, cols):
        x0, y1 = self.origin
        rows = np.asarray(rows, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64)
        return x0 + (cols + 0.5) * self.res, y1 - (rows + 0.5) * self.res

    def colrow(self, x, y):
        x0, y1 = self.origin
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return (x - x0) / self.res - 0.5, (y1 - y) / self.res - 0.5

    def to_lonlat(self, x, y):
        return transformer(self.crs, WGS84).transform(x, y)

    def from_lonlat(self, lon, lat):
        return transformer(WGS84, self.crs).transform(lon, lat)

    def lonlat_bounds(self) -> tuple[float, float, float, float]:
        minx, miny, maxx, maxy = self.bounds
        lon, lat = self.to_lonlat(np.array([minx, minx, maxx, maxx]), np.array([miny, maxy, miny, maxy]))
        return (float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat)))

    def crop(self, row0: int, col0: int, height: int, width: int) -> "GridSpec":
        r0 = max(0, int(row0))
        c0 = max(0, int(col0))
        r1 = min(self.height, int(row0) + int(height))
        c1 = min(self.width, int(col0) + int(width))
        x0, y1 = self.origin
        return GridSpec(self.crs, (self.res, 0.0, x0 + c0 * self.res, 0.0, -self.res, y1 - r0 * self.res), c1 - c0, r1 - r0)

    def offset_in(self, parent: "GridSpec") -> tuple[int, int]:
        col = int(round((self.origin[0] - parent.origin[0]) / parent.res))
        row = int(round((parent.origin[1] - self.origin[1]) / parent.res))
        return row, col

    def window_around(self, x: float, y: float, half_px: int) -> tuple[int, int, int, int]:
        col, row = self.colrow(x, y)
        r0 = max(0, int(round(float(row))) - half_px)
        c0 = max(0, int(round(float(col))) - half_px)
        r1 = min(self.height, int(round(float(row))) + half_px)
        c1 = min(self.width, int(round(float(col))) + half_px)
        return r0, c0, r1 - r0, c1 - c0

    def pixel_polygon_to_map(self, points_colrow: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
        pts = np.asarray(list(points_colrow), dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return []
        xs, ys = self.xy(pts[:, 1], pts[:, 0])
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def map_polygon_to_pixel(self, coords: Iterable[Sequence[float]]) -> np.ndarray:
        pts = np.asarray(list(coords), dtype=np.float64).reshape(-1, 2)
        cols, rows = self.colrow(pts[:, 0], pts[:, 1])
        return np.stack([cols, rows], axis=1)

    def map_to_lonlat_coords(self, coords: Iterable[Sequence[float]]) -> list[list[float]]:
        pts = np.asarray(list(coords), dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return []
        lon, lat = self.to_lonlat(pts[:, 0], pts[:, 1])
        return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]


def scaled_bbox(bbox: Sequence[float], scale: float) -> tuple[float, float, float, float]:
    if scale >= 1.0:
        return tuple(float(v) for v in bbox)
    lon0, lat0, lon1, lat1 = bbox
    cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    hw, hh = (lon1 - lon0) * scale / 2, (lat1 - lat0) * scale / 2
    return (cx - hw, cy - hh, cx + hw, cy + hh)

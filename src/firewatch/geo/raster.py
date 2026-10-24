from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject

from firewatch.geo.grid import WGS84, GridSpec

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF,.tif,.tiff,.TIFF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
}


@contextlib.contextmanager
def raster_env(requester_pays: bool = False, region: str | None = None):
    options = dict(GDAL_ENV)
    session = None
    if requester_pays:
        options["AWS_REQUEST_PAYER"] = "requester"
        try:
            import boto3
            from rasterio.session import AWSSession

            session = AWSSession(boto3.Session(region_name=region), requester_pays=True)
        except Exception:
            session = None
    env = rasterio.Env(session=session, **options) if session is not None else rasterio.Env(**options)
    with env:
        yield


def read_to_grid(
    href: str,
    grid: GridSpec,
    *,
    band: int = 1,
    resampling: str = "bilinear",
    nodata: float | None = None,
    dtype: str = "float32",
    fill: float | int = np.nan,
    requester_pays: bool = False,
    region: str | None = None,
) -> np.ndarray:
    with raster_env(requester_pays, region):
        with rasterio.open(href) as src:
            src_nodata = src.nodata if nodata is None else nodata
            vrt_options = dict(
                crs=grid.crs,
                transform=grid.affine,
                width=grid.width,
                height=grid.height,
                resampling=Resampling[resampling],
            )
            if src_nodata is not None:
                vrt_options.update(src_nodata=src_nodata, nodata=src_nodata)
            with WarpedVRT(src, **vrt_options) as vrt:
                data = vrt.read(band, masked=True)
    return np.ma.filled(data.astype(dtype), fill)


def write_geotiff(path: str | Path, array: np.ndarray, grid: GridSpec, nodata: float | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=1,
        dtype=str(array.dtype),
        crs=grid.crs,
        transform=grid.affine,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return path


def reproject_to_wgs84(array: np.ndarray, grid: GridSpec, resampling: str = "nearest") -> tuple[np.ndarray, tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = grid.bounds
    dst_transform, width, height = calculate_default_transform(grid.crs, WGS84, grid.width, grid.height, minx, miny, maxx, maxy)
    if array.ndim == 2:
        out = np.zeros((height, width), dtype=array.dtype)
        reproject(array, out, src_transform=grid.affine, src_crs=grid.crs, dst_transform=dst_transform, dst_crs=WGS84, resampling=Resampling[resampling])
    else:
        out = np.zeros((array.shape[0], height, width), dtype=array.dtype)
        for i in range(array.shape[0]):
            reproject(array[i], out[i], src_transform=grid.affine, src_crs=grid.crs, dst_transform=dst_transform, dst_crs=WGS84, resampling=Resampling[resampling])
    west = dst_transform.c
    north = dst_transform.f
    east = west + dst_transform.a * width
    south = north + dst_transform.e * height
    return out, (west, south, east, north)

from __future__ import annotations

import numpy as np

from firewatch.config import ThermalConfig

LIANG_OLI = {"blue": 0.356, "red": 0.130, "nir08": 0.373, "swir16": 0.085, "swir22": 0.072}
LIANG_INTERCEPT = -0.0018


def ndvi(red, nir):
    red = np.asarray(red, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (nir - red) / (nir + red)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def vegetation_fraction(ndvi_values, ndvi_soil: float, ndvi_veg: float):
    pv = (np.asarray(ndvi_values, dtype=np.float32) - ndvi_soil) / (ndvi_veg - ndvi_soil)
    return np.clip(pv, 0.0, 1.0) ** 2


def emissivity_from_ndvi(ndvi_values, cfg: ThermalConfig):
    v = np.asarray(ndvi_values, dtype=np.float32)
    pv = vegetation_fraction(v, cfg.ndvi_soil, cfg.ndvi_vegetation)
    es, ev, f = cfg.emissivity_soil, cfg.emissivity_vegetation, cfg.cavity_shape_factor
    mixed = ev * pv + es * (1.0 - pv) + (1.0 - es) * (1.0 - pv) * f * ev
    eps = np.where(v < cfg.ndvi_soil, es, mixed)
    eps = np.where(v > cfg.ndvi_vegetation, ev, eps)
    eps = np.where(v < 0.0, cfg.emissivity_water, eps)
    return np.where(np.isfinite(v), eps, np.nan).astype(np.float32)


def broadband_albedo(reflectance: dict[str, np.ndarray]):
    out = np.full_like(next(iter(reflectance.values())), LIANG_INTERCEPT, dtype=np.float32)
    for band, weight in LIANG_OLI.items():
        out = out + weight * reflectance[band]
    return np.clip(out, 0.0, 1.0).astype(np.float32)

from __future__ import annotations

import math

import numpy as np

RHO_UM_K = 14388.0
L2_ST_SCALE = 0.00341802
L2_ST_OFFSET = 149.0


def dn_to_radiance(dn, mult: float, add: float):
    return mult * np.asarray(dn, dtype=np.float32) + add


def radiance_to_dn(radiance, mult: float, add: float):
    return (np.asarray(radiance, dtype=np.float32) - add) / mult


def radiance_to_brightness_temp(radiance, k1: float, k2: float):
    radiance = np.asarray(radiance, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = k2 / np.log(k1 / radiance + 1.0)
    return np.where(radiance > 0, bt, np.nan).astype(np.float32)


def brightness_temp_to_radiance(bt, k1: float, k2: float):
    bt = np.asarray(bt, dtype=np.float64)
    return (k1 / (np.exp(k2 / bt) - 1.0)).astype(np.float32)


def lst_from_brightness_temp(bt, emissivity, wavelength_um: float):
    bt = np.asarray(bt, dtype=np.float32)
    emissivity = np.asarray(emissivity, dtype=np.float32)
    return (bt / (1.0 + (wavelength_um * bt / RHO_UM_K) * np.log(emissivity))).astype(np.float32)


def brightness_temp_from_lst(lst, emissivity, wavelength_um: float):
    lst = np.asarray(lst, dtype=np.float32)
    emissivity = np.asarray(emissivity, dtype=np.float32)
    return (lst / (1.0 - (wavelength_um * lst / RHO_UM_K) * np.log(emissivity))).astype(np.float32)


def toa_reflectance(dn, mult: float, add: float, sun_elevation_deg: float | None = None):
    refl = mult * np.asarray(dn, dtype=np.float32) + add
    if sun_elevation_deg:
        refl = refl / math.sin(math.radians(sun_elevation_deg))
    return refl.astype(np.float32)


def reflectance_to_dn(refl, mult: float, add: float, sun_elevation_deg: float | None = None):
    refl = np.asarray(refl, dtype=np.float32)
    if sun_elevation_deg:
        refl = refl * math.sin(math.radians(sun_elevation_deg))
    return (refl - add) / mult


def l2_surface_temp_to_kelvin(dn, nodata: int = 0):
    dn = np.asarray(dn, dtype=np.float32)
    out = dn * L2_ST_SCALE + L2_ST_OFFSET
    return np.where(dn == nodata, np.nan, out).astype(np.float32)


def thermal_dn_to_lst(dn, emissivity, *, mult: float, add: float, k1: float, k2: float, wavelength_um: float):
    radiance = dn_to_radiance(dn, mult, add)
    bt = radiance_to_brightness_temp(radiance, k1, k2)
    return bt, lst_from_brightness_temp(bt, emissivity, wavelength_um)

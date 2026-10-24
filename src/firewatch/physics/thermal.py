from __future__ import annotations

import numpy as np

from firewatch.config import ThermalConfig
from firewatch.data.mtl import thermal_constants
from firewatch.models import RawScene, ThermalProduct
from firewatch.physics.emissivity import LIANG_OLI, broadband_albedo, emissivity_from_ndvi, ndvi
from firewatch.physics.radiometry import thermal_dn_to_lst
from firewatch.vision.clouds import QA_WATER, cloud_fraction, landsat_masks


def derive_thermal(raw: RawScene, cfg: ThermalConfig) -> ThermalProduct:
    clear, cloudy = landsat_masks(raw.qa_pixel)
    refl = raw.reflectance
    veg_index = ndvi(refl["red"], refl["nir08"])
    eps = emissivity_from_ndvi(veg_index, cfg)
    if raw.mtl:
        consts = thermal_constants(raw.mtl, 10)
        dn = np.where(raw.thermal_dn > 0, raw.thermal_dn, np.nan).astype(np.float32)
        bt, lst = thermal_dn_to_lst(dn, eps, wavelength_um=cfg.wavelength_um, **consts)
    else:
        lst = np.asarray(raw.l2_surface_temp, np.float32)
        bt = lst.copy()
    albedo = broadband_albedo({band: refl[band] for band in LIANG_OLI})
    valid = clear & np.isfinite(lst) & np.isfinite(veg_index) & np.isfinite(albedo) & (lst > 220) & (lst < 420)
    lst = np.where(valid, lst, np.nan).astype(np.float32)
    return ThermalProduct(
        lst=lst,
        brightness_temp=bt,
        emissivity=eps,
        ndvi=veg_index,
        albedo=albedo,
        valid=valid,
        cloud_fraction=cloud_fraction(clear, cloudy),
        water=(raw.qa_pixel.astype(np.uint16) & QA_WATER) > 0,
    )

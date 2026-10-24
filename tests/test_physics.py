import numpy as np
import pytest

from firewatch.data.mtl import parse_mtl, reflectance_rescaling, sun_elevation, thermal_constants
from firewatch.physics import radiometry as rad
from firewatch.physics.emissivity import broadband_albedo, emissivity_from_ndvi, ndvi

L8 = {"mult": 3.342e-4, "add": 0.1, "k1": 774.8853, "k2": 1321.0789}

MTL_TEXT = """GROUP = LANDSAT_METADATA_FILE
  GROUP = IMAGE_ATTRIBUTES
    SPACECRAFT_ID = "LANDSAT_8"
    SUN_ELEVATION = 47.12345678
  END_GROUP = IMAGE_ATTRIBUTES
  GROUP = LEVEL1_RADIOMETRIC_RESCALING
    RADIANCE_MULT_BAND_10 = 3.3420E-04
    RADIANCE_ADD_BAND_10 = 0.10000
    REFLECTANCE_MULT_BAND_4 = 2.0000E-05
    REFLECTANCE_ADD_BAND_4 = -0.100000
  END_GROUP = LEVEL1_RADIOMETRIC_RESCALING
  GROUP = LEVEL1_THERMAL_CONSTANTS
    K1_CONSTANT_BAND_10 = 774.8853
    K2_CONSTANT_BAND_10 = 1321.0789
  END_GROUP = LEVEL1_THERMAL_CONSTANTS
END_GROUP = LANDSAT_METADATA_FILE
END
"""


def test_radiance_brightness_round_trip():
    bt = np.array([260.0, 300.0, 340.0, 400.0], np.float32)
    radiance = rad.brightness_temp_to_radiance(bt, L8["k1"], L8["k2"])
    back = rad.radiance_to_brightness_temp(radiance, L8["k1"], L8["k2"])
    np.testing.assert_allclose(back, bt, atol=1e-3)


def test_dn_round_trip_through_lst():
    lst = np.array([[295.0, 310.0], [330.0, 360.0]], np.float32)
    eps = np.array([[0.97, 0.985], [0.96, 0.99]], np.float32)
    bt = rad.brightness_temp_from_lst(lst, eps, 10.895)
    dn = rad.radiance_to_dn(rad.brightness_temp_to_radiance(bt, L8["k1"], L8["k2"]), L8["mult"], L8["add"])
    _, recovered = rad.thermal_dn_to_lst(dn, eps, wavelength_um=10.895, **L8)
    np.testing.assert_allclose(recovered, lst, atol=0.01)


def test_emissivity_correction_raises_temperature():
    bt = np.float32(300.0)
    assert rad.lst_from_brightness_temp(bt, 0.97, 10.895) > bt
    assert rad.lst_from_brightness_temp(bt, 1.0, 10.895) == pytest.approx(300.0)


def test_level2_scaling():
    k = rad.l2_surface_temp_to_kelvin(np.array([0, 44177], np.uint16))
    assert np.isnan(k[0])
    assert k[1] == pytest.approx(44177 * 0.00341802 + 149.0, abs=1e-3)


def test_emissivity_follows_ndvi_thresholds(settings):
    cfg = settings.thermal
    eps = emissivity_from_ndvi(np.array([-0.2, 0.1, 0.35, 0.8]), cfg)
    assert eps[0] == pytest.approx(cfg.emissivity_water)
    assert eps[1] == pytest.approx(cfg.emissivity_soil)
    assert cfg.emissivity_soil < eps[2] < 1.0
    assert eps[3] == pytest.approx(cfg.emissivity_vegetation)


def test_ndvi_and_albedo_ranges():
    red = np.array([0.05, 0.2], np.float32)
    nir = np.array([0.4, 0.2], np.float32)
    v = ndvi(red, nir)
    assert v[0] > 0.7 and v[1] == pytest.approx(0.0)
    bands = {b: np.full(2, 0.2, np.float32) for b in ("blue", "red", "nir08", "swir16", "swir22")}
    albedo = broadband_albedo(bands)
    assert np.all((albedo > 0.15) & (albedo < 0.25))


def test_mtl_text_and_json_agree():
    text = parse_mtl(MTL_TEXT)
    as_json = parse_mtl({"LANDSAT_METADATA_FILE": {"LEVEL1_THERMAL_CONSTANTS": {"K1_CONSTANT_BAND_10": "774.8853", "K2_CONSTANT_BAND_10": "1321.0789"}, "LEVEL1_RADIOMETRIC_RESCALING": {"RADIANCE_MULT_BAND_10": "3.3420E-04", "RADIANCE_ADD_BAND_10": "0.10000"}}})
    assert thermal_constants(text) == pytest.approx(thermal_constants(as_json))
    assert thermal_constants(text)["k2"] == pytest.approx(1321.0789)
    assert reflectance_rescaling(text, 4) == pytest.approx((2e-5, -0.1))
    assert sun_elevation(text) == pytest.approx(47.12345678)


def test_missing_mtl_key_is_reported():
    with pytest.raises(KeyError, match="RADIANCE_MULT_BAND_10"):
        thermal_constants({})

import datetime as dt

import numpy as np
import pytest

from firewatch.data import stac
from firewatch.data.landsat import LandsatSource
from firewatch.data.sentinel2 import Sentinel2Reader
from firewatch.data.synthetic import SyntheticWorld, export_landsat_files
from firewatch.geo.grid import GridSpec
from firewatch.geo.raster import write_geotiff
from firewatch.physics.thermal import derive_thermal
from firewatch.util import parse_time
from firewatch.vision.swir import hot_pixel_check

SCENE = "LC09_L1TP_140044_20251121_20251121_02_T1"


def stac_item(**overrides):
    base = "collection02/level-1/standard/oli-tirs/2025/140/044/" + SCENE
    assets = {k: {"href": f"https://landsatlook.usgs.gov/data/{base}/{SCENE}_{k}.TIF", "alternate": {"s3": {"href": f"s3://usgs-landsat/{base}/{SCENE}_{k}.TIF"}}} for k in stac.LANDSAT_ASSETS + ("thumbnail",)}
    props = {"datetime": "2025-11-21T04:50:12.345Z", "platform": "LANDSAT_9", "eo:cloud_cover": 3.2, "view:sun_elevation": 41.2, "landsat:wrs_path": "140", "landsat:wrs_row": "044", "landsat:correction": "L1TP", "proj:epsg": 32645}
    props.update(overrides)
    return {"id": SCENE, "bbox": [85.6, 22.9, 87.9, 25.0], "properties": props, "assets": assets}


def test_stac_item_becomes_a_scene_ref(settings):
    ref = stac.landsat_ref(stac_item(), prefer_s3=True)
    assert ref.path_row == "140044" and ref.platform == "LANDSAT_9"
    assert ref.assets["lwir11"].startswith("s3://usgs-landsat/")
    assert "thumbnail" not in ref.assets
    assert stac.keep_landsat(ref, settings) == (True, "")
    assert stac.intersects_aoi(ref, settings.aoi.bbox)
    assert not stac.intersects_aoi(ref, (80.0, 10.0, 81.0, 11.0))
    assert stac.landsat_ref(stac_item(), prefer_s3=False).assets["lwir11"].startswith("https://")


@pytest.mark.parametrize("change, reason", [
    ({"datetime": "2025-07-21T04:50:12Z"}, "month"),
    ({"eo:cloud_cover": 85.0}, "cloud"),
    ({"view:sun_elevation": -20.0}, "night or low sun"),
    ({"landsat:correction": "L1GT"}, "not terrain corrected"),
    ({"platform": "LANDSAT_7"}, "platform"),
])
def test_unusable_scenes_are_skipped(settings, change, reason):
    ok, why = stac.keep_landsat(stac.landsat_ref(stac_item(**change), prefer_s3=True), settings)
    assert not ok and why == reason


def test_landsat_reader_on_files_written_like_usgs(settings, tmp_path):
    world = SyntheticWorld.from_settings(settings, scale=0.3, misregister=False)
    synthetic = world.raw_scene(3)
    ref = export_landsat_files(synthetic, tmp_path / "scene")
    loaded = LandsatSource(settings).load(ref, world.grid)
    np.testing.assert_allclose(loaded.thermal_dn, synthetic.thermal_dn, atol=0.51)
    for band in ("red", "nir08", "swir22"):
        np.testing.assert_allclose(loaded.reflectance[band], synthetic.reflectance[band], atol=1e-4)
    np.testing.assert_array_equal(loaded.qa_pixel, synthetic.qa_pixel)
    a, b = derive_thermal(loaded, settings.thermal), derive_thermal(synthetic, settings.thermal)
    np.testing.assert_array_equal(a.valid, b.valid)
    agree = np.abs(a.lst - b.lst) < 0.01
    on_threshold = np.isclose(b.ndvi, settings.thermal.ndvi_soil, atol=1e-6) | np.isclose(b.ndvi, settings.thermal.ndvi_vegetation, atol=1e-6)
    assert np.all(agree | on_threshold | ~b.valid)


def test_fetch_bytes_reads_local_files(tmp_path):
    path = tmp_path / "MTL.json"
    path.write_text("{}")
    assert stac.fetch_bytes(str(path)) == b"{}"
    assert stac.fetch_bytes(path.as_uri()) == b"{}"


def test_sentinel2_reader_finds_hot_pixels(settings, tmp_path, monkeypatch):
    crs = "EPSG:32645"
    src = GridSpec(crs, (10.0, 0.0, 420000.0, 0.0, -10.0, 2621200.0), 120, 120)
    nir = np.full((120, 120), 2900, np.uint16)
    swir16 = np.full((120, 120), 2000, np.uint16)
    swir22 = np.full((120, 120), 1800, np.uint16)
    swir16[54:66, 54:66] = 2600
    swir22[54:66, 54:66] = 4600
    scl = np.full((120, 120), 5, np.uint8)
    scl[:20, :20] = 9
    paths = {name: str(write_geotiff(tmp_path / f"{name}.tif", arr, src, nodata=0)) for name, arr in (("nir", nir), ("swir16", swir16), ("swir22", swir22), ("scl", scl))}
    bands = [{"scale": 0.0001, "offset": -0.1}]
    item = {"id": "S2B_45QXF_20251120_0_L2A", "properties": {"datetime": "2025-11-20T05:00:00Z", "eo:cloud_cover": 4.0}, "assets": {k: {"href": v, "raster:bands": bands} for k, v in paths.items()}}
    monkeypatch.setattr(stac, "search_items", lambda *a, **k: [item])
    when = parse_time("2025-11-21T04:50:00Z")
    patch = Sentinel2Reader(settings).patch((420000.0, 2620000.0, 421200.0, 2621200.0), crs, when)
    assert patch.grid.shape == (60, 60) and patch.grid.res == 20.0
    assert not patch.valid[:5, :5].any() and patch.valid[40:, 40:].all()
    assert patch.swir22[30, 30] == pytest.approx(0.36, abs=1e-3)
    hot = [(420550.0, 2620650.0), (420650.0, 2620650.0), (420650.0, 2620550.0), (420550.0, 2620550.0), (420550.0, 2620650.0)]
    cool = [(420950.0, 2620250.0), (421050.0, 2620250.0), (421050.0, 2620150.0), (420950.0, 2620150.0), (420950.0, 2620250.0)]
    found = hot_pixel_check(patch, hot)
    assert found["confirmed"] and found["hot_pixels"] >= 2
    assert not hot_pixel_check(patch, cool, buffer_m=20.0)["confirmed"]
    assert patch.acquired == dt.datetime(2025, 11, 20, 5, 0, tzinfo=dt.timezone.utc)

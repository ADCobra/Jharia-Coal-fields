# Data sources

Every input is open data. Nothing needs a commercial licence, and the AWS-hosted copies avoid downloading whole scenes.

## Landsat 8 and 9, Collection 2 Level-1

The main input. Band 10 (TIRS, 10.6–11.19 µm) is recorded at 100 m and delivered resampled to 30 m. The two satellites together pass over Jharia every 8 days in clear weather, at about 10:30 local time.

| What | Where |
| --- | --- |
| Scene search | USGS STAC server, `https://landsatlook.usgs.gov/stac-server`, collection `landsat-c2l1` |
| Pixels | Cloud-optimised GeoTIFFs in `s3://usgs-landsat` (us-west-2, requester pays) |
| Assets read | `lwir11` (band 10), `blue`, `red`, `nir08`, `swir16`, `swir22`, `qa_pixel`, `MTL.json` |
| New-scene notifications | SNS topic `arn:aws:sns:us-west-2:673253540267:public-c2-notify-v2` |

Only the window covering the area of interest is read (`rasterio` + `WarpedVRT`, see `geo/raster.py`), reprojected straight onto the 30 m UTM 45N grid. A scene is skipped when it is from another platform, outside the configured months, cloudier than `data.max_cloud`, taken with the sun below 15°, not terrain corrected (L1TP), or missing an asset (`data/stac.py: keep_landsat`).

Calibration constants come from each scene's `MTL.json`, never from code: `RADIANCE_MULT_BAND_10`, `RADIANCE_ADD_BAND_10`, `K1_CONSTANT_BAND_10`, `K2_CONSTANT_BAND_10`, the reflectance rescaling for bands 2, 4, 5, 6, 7, and `SUN_ELEVATION`.

`QA_PIXEL` bits used (`vision/clouds.py`): fill (0), dilated cloud (1), cirrus (2), cloud (3), cloud shadow (4), clear (6), water (7), and the high-confidence pairs for cloud (8–9), shadow (10–11) and cirrus (14–15). Cloud and shadow are dilated by 2 pixels before anything else runs.

## Landsat Collection 2 Level-2 surface temperature

Used only to check our own temperatures (`firewatch eval lst`). Collection `landsat-c2l2-st`, asset `lwir11`, scaled as `DN × 0.00341802 + 149.0` K. Our retrieval skips atmospheric correction, so expect a bias of a few kelvin against it on humid days. Hotspot detection works on local contrast, so a scene-wide bias barely matters there.

## Sentinel-2 Level-2A

The agent's `fetch_highres` tool looks for the closest clear Sentinel-2 image within `data.sentinel_window_days` (6) of the Landsat pass and checks the hotspot for hot pixels in shortwave infrared.

| What | Where |
| --- | --- |
| Search | Earth Search v1, `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a` |
| Assets | `nir`, `swir16` (B11), `swir22` (B12), `scl` |
| Test | normalised hotspot index `(B12 − B11) / (B12 + B11) > 0` and B12 reflectance above 0.12, at least 2 pixels |

Scene classification values 0, 1, 3, 8, 9, 10 and 11 (no data, saturated, cloud shadow, clouds, cirrus, snow) are masked. Sentinel-2 only sees fires hot enough to glow in SWIR, so a negative result does not count against a hotspot.

## OpenStreetMap

Roads and industrial sites for real runs. `python scripts/fetch_osm.py` writes `data/jharia_roads.geojson` and `data/jharia_industrial.geojson` from the Overpass API. Service roads and tracks are marked as haul roads, which is a rough rule. Data © OpenStreetMap contributors, ODbL.

## Reference data for evaluation

| Dataset | Used by | Notes |
| --- | --- | --- |
| NRSC coal fire mapping, as reported by BCCL ([FireDrilling.pdf](https://www.bcclweb.in/masterplan/FireDrilling.pdf)) | `firewatch eval nrsc` | About 3.28 km² of fire in 2017 and 1.89 km² in 2020, mapped from Landsat-8 band 10. The same document gives 8.9 km² for the 2009 Master Plan. |
| NASA FIRMS active fires ([download](https://firms.modaps.eosdis.nasa.gov/download/)) | `firewatch eval firms` | VIIRS 375 m and MODIS 1 km. Flags flaming fires, flares and coke ovens; misses most smouldering ground. |

## The synthetic coalfield

The demo, the tests and most evaluations run on `data/synthetic.py`, a coalfield built from scratch so every number can be checked against a known truth.

It has the real area of interest and grid (1364 × 857 pixels at 30 m) and five fires that behave differently: A creeps toward a haul road until it is put out in mid-January, B burns far from any road, C starts beside the main road in mid-November, D sits by the link road and hides under cloud on one pass, and E goes out by itself in late October. There is also a coke-oven battery hotter than any fire, pits, dumps, water, villages, seasonal temperature and vegetation, random cloud with shadows, and a misregistration of up to 1.6 pixels in each direction and 0.12° of rotation on every pass except the reference.

Each pass is written out the way Landsat delivers it: band 10 as digital numbers through the real L8 calibration constants and the emissivity model, reflective bands as DNs with sun-elevation scaling, a `QA_PIXEL` array with real bit values, and an `MTL` dictionary. The pipeline reads synthetic and real scenes through the same code; only the source class differs.

A second, held-out world (`SyntheticWorld.from_settings(..., profile="holdout", seed=23)`) was added after the detector was tuned, to see what the tuning had overfitted. It has a different seed and layout, seven irregular fires with up to three lobes each (three of them heading for a road fast enough to reach it during the season, two starting late, two going out), weaker heat, 1.7 times the sensor noise, cloud on half the passes, up to 2.4 pixels of misregistration, and a second coke plant that is not on the industrial map. `firewatch eval detection` and `firewatch eval forecast` report both worlds.

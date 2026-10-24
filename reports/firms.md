# Hotspots against NASA FIRMS active-fire detections

FIRMS (VIIRS 375 m, MODIS 1 km) flags flaming or very hot pixels. Subsurface coal fires are often too cool for it, and it also flags coke ovens and flares, so only partial agreement is expected. A detection counts as matched when one of our hotspots lies within 400 m (VIIRS) or 1 km (MODIS) of it, within one day of the Landsat pass.

**Not run here:** no FIRMS file was given. Download an archive CSV for the area of interest from https://firms.modaps.eosdis.nasa.gov/download/ and run:

```
FIREWATCH_FIRMS_CSV=fire_archive_SV-C2.csv firewatch -c config/landsat.yaml eval firms
```

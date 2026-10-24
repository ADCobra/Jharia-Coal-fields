Real-data inputs live here. Nothing in this folder is needed for the synthetic demo.

- `jharia_roads.geojson` and `jharia_industrial.geojson`: run `python scripts/fetch_osm.py` to download them from OpenStreetMap (ODbL). Service roads and tracks are marked as haul roads, which is a rough rule; edit the `haul` property where you know better.
- FIRMS archives for `firewatch eval firms`: download from https://firms.modaps.eosdis.nasa.gov/download/ for the bounding box 86.10, 23.63, 86.50, 23.86.

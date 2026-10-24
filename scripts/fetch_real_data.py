"""Download real Landsat 8 and 9 scenes and OpenStreetMap data for the Jharia coalfield.

Scenes come from the Landsat Collection 2 Level-2 archive on Microsoft Planetary Computer,
which is free and needs no account. Only the coalfield window of each band is read, so a
scene costs about 10 MB. Map data is (c) OpenStreetMap contributors, ODbL.

    python scripts/fetch_real_data.py                     # last 12 months
    python scripts/fetch_real_data.py --start 2025-10-01 --end 2026-05-31

Needs numpy and rasterio. Writes data/real/scenes/<scene>.tif and .json, and data/real/osm.json.
Run it again to resume; finished scenes are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
COLLECTION = "landsat-c2-l2"
BBOX = (86.10, 23.63, 86.50, 23.86)
BANDS = (
    ("qa_pixel", "QA_PIXEL"),
    ("lwir11", "ST_B10"),
    ("blue", "SR_B2"),
    ("green", "SR_B3"),
    ("red", "SR_B4"),
    ("nir08", "SR_B5"),
    ("swir16", "SR_B6"),
    ("swir22", "SR_B7"),
)
CLOUDY_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
FILL_BIT = 1
USER_AGENT = "jharia-fire-watch/1.0 (OpenCV AI Competition 2026)"


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = ssl_context()


def http(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 120, tries: int = 4) -> bytes:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=request_headers)
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404):
                raise
            if attempt == tries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  network hiccup ({exc}), retrying in {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def post_json(url: str, body: dict) -> dict:
    return json.loads(http(url, json.dumps(body).encode(), {"Content-Type": "application/json"}))


def search(start: dt.date, end: dt.date, max_cloud: float) -> list[dict]:
    body = {
        "collections": [COLLECTION],
        "bbox": list(BBOX),
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": ["landsat-8", "landsat-9"]}},
        "limit": 100,
    }
    url, items = STAC_SEARCH, []
    while url:
        page = post_json(url, body) if body is not None else json.loads(http(url))
        items.extend(page.get("features", []))
        nxt = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
        if not nxt:
            break
        url = nxt["href"]
        if nxt.get("method", "GET").upper() == "POST":
            body = {**body, **nxt.get("body", {})} if nxt.get("merge") else nxt.get("body", body)
        else:
            body = None
    return items


def rings(geometry: dict) -> list[list[list[float]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    return []


def inside(point: tuple[float, float], ring: list[list[float]]) -> bool:
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            hit = not hit
    return hit


def covers_aoi(item: dict) -> bool:
    w, s, e, n = BBOX
    corners = [(w, s), (w, n), (e, s), (e, n), ((w + e) / 2, (s + n) / 2)]
    return any(all(inside(c, ring) for c in corners) for ring in rings(item["geometry"]))


class Signer:
    def __init__(self):
        self.tokens: dict[str, tuple[str, float]] = {}

    def __call__(self, href: str) -> str:
        parts = urllib.parse.urlparse(href)
        if not parts.netloc.endswith(".blob.core.windows.net"):
            return href
        account = parts.netloc.split(".")[0]
        container = parts.path.lstrip("/").split("/")[0]
        key = f"{account}/{container}"
        token, expires = self.tokens.get(key, ("", 0.0))
        if time.time() > expires - 300:
            data = json.loads(http(f"{SAS_TOKEN}/{key}"))
            token = data["token"]
            expiry = data.get("msft:expiry")
            try:
                expires = dt.datetime.fromisoformat(expiry.replace("Z", "+00:00")).timestamp() if expiry else time.time() + 1800
            except ValueError:
                expires = time.time() + 1800
            self.tokens[key] = (token, expires)
        return f"{href}?{token}"


def band_href(item: dict, key: str, suffix: str) -> str:
    assets = item["assets"]
    if key in assets:
        return assets[key]["href"]
    for asset in assets.values():
        if asset.get("href", "").upper().endswith(f"_{suffix}.TIF"):
            return asset["href"]
    raise KeyError(f"{item['id']} has no {key} band")


def download_scene(item: dict, out: Path, sign: Signer, max_aoi_cloud: float) -> str:
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    tif, meta_path = out / f"{item['id']}.tif", out / f"{item['id']}.json"
    if tif.exists() and meta_path.exists():
        return "already downloaded"
    arrays, profile, window = [], None, None
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE")
    with env:
        for key, suffix in BANDS:
            with rasterio.open(sign(band_href(item, key, suffix))) as src:
                if window is None:
                    left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *BBOX, densify_pts=21)
                    pad = 600
                    window = from_bounds(left - pad, bottom - pad, right + pad, top + pad, transform=src.transform)
                    window = window.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
                    profile = {"crs": src.crs, "transform": src.window_transform(window), "width": int(window.width), "height": int(window.height)}
                data = src.read(1, window=window)
            if key == "qa_pixel":
                qa = data.astype(np.uint16)
                filled = (qa & FILL_BIT) > 0
                cloudy = ((qa & CLOUDY_BITS) > 0) & ~filled
                share_fill = float(filled.mean())
                share_cloud = float(cloudy.sum()) / max(1, int((~filled).sum()))
                if share_fill > 0.2:
                    return f"skipped, {share_fill:.0%} of the coalfield is outside the image"
                if share_cloud > max_aoi_cloud:
                    meta_path.with_suffix(".skip").write_text(json.dumps({"id": item["id"], "aoi_cloud": share_cloud}))
                    return f"skipped, {share_cloud:.0%} cloud over the coalfield"
                aoi_cloud = share_cloud
            arrays.append(data.astype(np.uint16))
    stack = np.stack(arrays)
    part = tif.with_suffix(".part")
    with rasterio.open(part, "w", driver="GTiff", count=len(BANDS), dtype="uint16", nodata=0, compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256, **profile) as dst:
        dst.write(stack)
        for i, (key, _) in enumerate(BANDS, start=1):
            dst.set_band_description(i, key)
    part.replace(tif)
    props = item["properties"]
    meta = {
        "id": item["id"],
        "datetime": props["datetime"],
        "platform": props.get("platform"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "aoi_cloud": round(aoi_cloud, 4),
        "wrs_path": props.get("landsat:wrs_path"),
        "wrs_row": props.get("landsat:wrs_row"),
        "sun_elevation": props.get("view:sun_elevation"),
        "sun_azimuth": props.get("view:sun_azimuth"),
        "collection_category": props.get("landsat:collection_category"),
        "bands": [key for key, _ in BANDS],
        "scale": {"reflectance": [2.75e-05, -0.2], "surface_temperature": [0.00341802, 149.0]},
        "source": f"Microsoft Planetary Computer, {COLLECTION}",
        "stac_item": f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/{COLLECTION}/items/{item['id']}",
        "downloaded": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return f"saved, {aoi_cloud:.0%} cloud over the coalfield, {tif.stat().st_size / 1e6:.1f} MB"


def overpass_query() -> str:
    w, s, e, n = BBOX
    area = f"({s},{w},{n},{e})"
    skip = "footway|path|steps|cycleway|pedestrian|construction|proposed|bridleway|corridor|platform|raceway|elevator|bus_stop"
    return f"""
[out:json][timeout:300];
(
  way["highway"]["highway"!~"^({skip})$"]{area};
  way["landuse"~"^(quarry|industrial|landfill)$"]{area};
  relation["landuse"~"^(quarry|industrial)$"]{area};
  way["man_made"~"^(works|mineshaft|adit)$"]{area};
  node["man_made"~"^(mineshaft|adit)$"]{area};
  way["industrial"]{area};
  way["power"="plant"]{area};
  way["natural"="water"]{area};
  node["place"~"^(city|town|suburb|village|hamlet|neighbourhood)$"]{area};
);
out tags geom;
"""


def fetch_osm(out: Path) -> str:
    target = out / "osm.json"
    if target.exists() and target.stat().st_size > 1000:
        return "already downloaded"
    data = urllib.parse.urlencode({"data": overpass_query()}).encode()
    last = None
    for url in OVERPASS:
        try:
            raw = http(url, data, {"Content-Type": "application/x-www-form-urlencoded"}, timeout=360, tries=2)
            payload = json.loads(raw)
            payload["fetched"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            payload["bbox"] = list(BBOX)
            payload["source"] = url
            target.write_text(json.dumps(payload))
            return f"{len(payload.get('elements', []))} map features from {urllib.parse.urlparse(url).netloc}"
        except Exception as exc:
            last = exc
            print(f"  {urllib.parse.urlparse(url).netloc} failed ({exc}), trying the next server", flush=True)
    raise RuntimeError(f"OpenStreetMap download failed: {last}")


def main() -> int:
    today = dt.date.today()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=dt.date.fromisoformat, default=today - dt.timedelta(days=365))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=today)
    parser.add_argument("--max-cloud", type=float, default=60, help="skip scenes with more cloud than this over the whole image, in percent")
    parser.add_argument("--max-aoi-cloud", type=float, default=0.5, help="skip scenes with more than this share of cloud over the coalfield")
    parser.add_argument("--max-scenes", type=int, default=40, help="keep at most this many scenes, the most recent")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "real"))
    parser.add_argument("--skip-osm", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_osm:
        print("Downloading roads, mines and plants from OpenStreetMap...", flush=True)
        print(f"  {fetch_osm(out)}", flush=True)

    print(f"Searching Landsat 8 and 9 scenes over Jharia from {args.start} to {args.end}...", flush=True)
    items = search(args.start, args.end, args.max_cloud)
    covering = [it for it in items if covers_aoi(it)]
    by_day: dict[str, dict] = {}
    for it in sorted(covering, key=lambda it: it["properties"].get("eo:cloud_cover", 100)):
        by_day.setdefault(it["properties"]["datetime"][:10], it)
    chosen = sorted(by_day.values(), key=lambda it: it["properties"]["datetime"])[-args.max_scenes:]
    print(f"  {len(items)} found, {len(covering)} cover the whole coalfield, {len(chosen)} to fetch", flush=True)
    if not chosen:
        print("No scenes to download. Try a wider date range or a higher --max-cloud.", file=sys.stderr)
        return 1

    sign = Signer()
    saved = 0
    for n, item in enumerate(chosen, start=1):
        if (scenes_dir / f"{item['id']}.skip").exists():
            print(f"[{n}/{len(chosen)}] {item['id']}: skipped earlier (cloudy)", flush=True)
            continue
        try:
            result = download_scene(item, scenes_dir, sign, args.max_aoi_cloud)
        except Exception as exc:
            result = f"failed ({exc}); run the command again to retry"
        if result.startswith(("saved", "already")):
            saved += 1
        print(f"[{n}/{len(chosen)}] {item['id']} ({item['properties']['datetime'][:10]}): {result}", flush=True)

    print(f"\nDone: {saved} usable scenes in {scenes_dir}", flush=True)
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

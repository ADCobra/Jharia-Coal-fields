from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from urllib.parse import urlparse

from firewatch.config import Settings
from firewatch.models import SceneRef
from firewatch.util import parse_time

log = logging.getLogger(__name__)

LANDSAT_ASSETS = ("lwir11", "blue", "red", "nir08", "swir16", "swir22", "qa_pixel", "MTL.json")
L2_ASSETS = ("lwir11", "qa_pixel")
SENTINEL_ASSETS = ("nir", "swir16", "swir22", "scl")


def search_items(url: str, collections: list[str], bbox: list[float], start: dt.datetime, end: dt.datetime, max_items: int = 1000, ids: list[str] | None = None) -> list[dict]:
    from pystac_client import Client

    client = Client.open(url)
    kwargs = {"collections": collections, "max_items": max_items}
    if ids:
        kwargs["ids"] = ids
    else:
        kwargs["bbox"] = bbox
        kwargs["datetime"] = f"{parse_time(start).isoformat()}/{parse_time(end).isoformat()}"
    return list(client.search(**kwargs).items_as_dicts())


def asset_href(asset: dict, prefer_s3: bool) -> str:
    alternate = (asset.get("alternate") or {}).get("s3", {}).get("href")
    return alternate if prefer_s3 and alternate else asset["href"]


def landsat_ref(item: dict, prefer_s3: bool, wanted: tuple[str, ...] = LANDSAT_ASSETS) -> SceneRef:
    props = item.get("properties", {})
    assets = {k: asset_href(v, prefer_s3) for k, v in item.get("assets", {}).items() if k in wanted}
    try:
        path_row = f"{int(props.get('landsat:wrs_path')):03d}{int(props.get('landsat:wrs_row')):03d}"
    except (TypeError, ValueError):
        path_row = None
    keep = ("view:sun_elevation", "landsat:cloud_cover_land", "landsat:collection_category", "landsat:correction", "proj:epsg")
    properties = {k: props.get(k) for k in keep if k in props}
    if item.get("bbox"):
        properties["bbox"] = list(item["bbox"])
    return SceneRef(
        scene_id=item["id"],
        acquired=parse_time(props["datetime"]),
        platform=props.get("platform", "LANDSAT"),
        source="landsat",
        cloud_cover=props.get("eo:cloud_cover"),
        path_row=path_row,
        assets=assets,
        properties=properties,
    )


def intersects_aoi(ref: SceneRef, bbox: tuple[float, float, float, float]) -> bool:
    b = ref.properties.get("bbox")
    if not b:
        return True
    return not (b[2] < bbox[0] or b[0] > bbox[2] or b[3] < bbox[1] or b[1] > bbox[3])


def keep_landsat(ref: SceneRef, settings: Settings) -> tuple[bool, str]:
    d = settings.data
    if ref.platform not in d.platforms:
        return False, "platform"
    if ref.acquired.month not in d.months:
        return False, "month"
    if ref.cloud_cover is not None and float(ref.cloud_cover) > d.max_cloud:
        return False, "cloud"
    sun = ref.properties.get("view:sun_elevation")
    if sun is not None and float(sun) < d.min_sun_elevation:
        return False, "night or low sun"
    if ref.properties.get("landsat:correction") not in (None, "L1TP"):
        return False, "not terrain corrected"
    missing = [a for a in LANDSAT_ASSETS if a not in ref.assets]
    if missing:
        return False, f"missing assets {missing}"
    return True, ""


def search_landsat(settings: Settings, start: dt.datetime, end: dt.datetime) -> list[SceneRef]:
    items = search_items(settings.data.landsat_stac_url, [settings.data.landsat_l1_collection], list(settings.aoi.bbox), start, end)
    refs = []
    for item in items:
        ref = landsat_ref(item, settings.data.prefer_s3)
        ok, why = keep_landsat(ref, settings)
        if ok:
            refs.append(ref)
        else:
            log.debug("skipping %s: %s", ref.scene_id, why)
    return sorted(refs, key=lambda r: r.acquired)


def get_landsat_ref(settings: Settings, scene_id: str) -> SceneRef:
    items = search_items(settings.data.landsat_stac_url, [settings.data.landsat_l1_collection], list(settings.aoi.bbox), dt.datetime.min, dt.datetime.max, ids=[scene_id])
    if not items:
        raise KeyError(f"scene {scene_id} not found in {settings.data.landsat_l1_collection}")
    return landsat_ref(items[0], settings.data.prefer_s3)


def find_l2_match(settings: Settings, ref: SceneRef) -> SceneRef | None:
    start = ref.acquired - dt.timedelta(hours=12)
    end = ref.acquired + dt.timedelta(hours=12)
    items = search_items(settings.data.landsat_stac_url, [settings.data.landsat_l2_collection], list(settings.aoi.bbox), start, end)
    for item in items:
        cand = landsat_ref(item, settings.data.prefer_s3, L2_ASSETS)
        if cand.path_row == ref.path_row and "lwir11" in cand.assets:
            return cand
    return None


def fetch_bytes(href: str, requester_pays: bool = False, region: str | None = None) -> bytes:
    parsed = urlparse(href)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path if parsed.scheme == "file" else href).read_bytes()
    if parsed.scheme == "s3":
        import boto3

        extra = {"RequestPayer": "requester"} if requester_pays else {}
        obj = boto3.client("s3", region_name=region).get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), **extra)
        return obj["Body"].read()
    import httpx

    response = httpx.get(href, follow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.content

from __future__ import annotations

import math

import cv2
import numpy as np

SOIL = (104, 128, 146)
SCRUB = (74, 100, 94)
FIELDS = ((82, 124, 108), (90, 128, 122), (104, 136, 146), (76, 112, 98), (96, 124, 136), (86, 120, 118))
TREE = (46, 66, 52)
COAL = (30, 31, 33)
BENCH_LIGHT = (86, 88, 90)
BENCH_DARK = (58, 59, 61)
DUMP = (112, 118, 124)
DISTURBED = (72, 76, 80)
WATER = (62, 60, 44)
VILLAGE = (120, 128, 136)
ROOFS = ((76, 92, 150), (168, 174, 180), (98, 110, 140), (140, 148, 156))
ASPHALT = (86, 88, 90)
HAUL = (92, 104, 116)
PLANT = (120, 124, 128)


def _blend(base: np.ndarray, color, weight: np.ndarray) -> np.ndarray:
    w = np.clip(weight, 0.0, 1.0)[..., None]
    return base * (1.0 - w) + np.asarray(color, np.float32) * w


def _up(mask: np.ndarray, k: int, sigma: float = 0.0) -> np.ndarray:
    out = cv2.resize(mask.astype(np.float32), (mask.shape[1] * k, mask.shape[0] * k), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(out, (0, 0), sigma) if sigma > 0 else out


def render_scenery(world, k: int = 3) -> np.ndarray:
    rng = np.random.default_rng(world.seed + 4242)
    h, w = world.shape
    H, W = h * k, w * k
    veg = _up(world.f_veg, k, 1.5)
    noise = _up(world._fine_noise, k, 1.0)
    coarse = _up(world._veg_noise, k)
    img = _blend(np.zeros((H, W, 3), np.float32) + np.asarray(SOIL, np.float32), SCRUB, np.clip((veg - 0.2) / 0.5, 0, 1))
    img *= (1.0 + 0.06 * noise + 0.04 * coarse)[..., None]

    fields = np.zeros((H, W, 3), np.float32)
    field_mask = np.zeros((H, W), np.float32)
    for _ in range(int(7000 * (h * w) / (850 * 1360))):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        if veg[int(cy), int(cx)] < 0.35:
            continue
        size = (rng.uniform(8, 24), rng.uniform(6, 18))
        box = cv2.boxPoints(((cx, cy), size, rng.uniform(-12, 12) + rng.choice([0, 90])))
        color = np.asarray(FIELDS[rng.integers(len(FIELDS))], np.float32) * rng.uniform(0.9, 1.1)
        cv2.fillConvexPoly(fields, np.round(box).astype(np.int32), color.tolist())
        cv2.fillConvexPoly(field_mask, np.round(box).astype(np.int32), 1.0)
        cv2.polylines(field_mask, [np.round(box).astype(np.int32)], True, 0.55, 1)
    fm = 0.8 * field_mask[..., None]
    img = img * (1 - fm) + fields * fm

    trees = np.zeros((H, W), np.float32)
    for _ in range(int(26000 * (h * w) / (850 * 1360))):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
        if rng.random() < veg[y, x] * 0.9:
            cv2.circle(trees, (x, y), int(rng.integers(1, 4)), 1.0, -1)
    img = _blend(img, TREE, cv2.GaussianBlur(trees, (0, 0), 0.7) * 0.85)

    m = world._masks
    disturbed = np.zeros((h, w), np.float32)
    for c, r, a, b, ang in world.pits:
        cv2.ellipse(disturbed, (int(c), int(r)), (int(a * 2.2), int(b * 2.6)), ang, 0, 360, 1.0, -1)
    road_px = np.zeros((h, w), np.float32)
    for road in world.roads:
        if road.haul:
            cv2.polylines(road_px, [np.round(np.array(road.coords)).astype(np.int32).reshape(-1, 1, 2)], False, 1.0, 3)
    for site in world.sites:
        cv2.circle(disturbed, (int(site.col), int(site.row)), int(max(site.half_w, site.half_h) * 3), 0.8, -1)
    rough = _up(world._shape_noise, k, 1.0)
    dist = _up(cv2.GaussianBlur(disturbed, (0, 0), 4.0), k, 2.0) * (0.85 + 0.25 * rough)
    dust = _up(cv2.GaussianBlur(road_px, (0, 0), 2.5), k, 1.5)
    img = _blend(img, DISTURBED, np.clip(dist * 1.3, 0, 0.92))
    img = _blend(img, DUMP, np.clip(dust * 0.9, 0, 0.6))
    dump = _up(m["dump"], k, 2.0)
    streaks = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 1.2)
    streaks = cv2.blur(streaks, (1, 9))
    img = _blend(img, DUMP, dump * 0.95)
    img *= (1.0 + 0.08 * dump * streaks)[..., None]

    for c, r, a, b, ang in world.pits:
        center = (int(c * k), int(r * k))
        cv2.ellipse(img, center, (int(a * k * 1.25), int(b * k * 1.3)), ang, 0, 360, tuple(float(v) for v in DISTURBED), -1)
        cv2.ellipse(img, center, (int(a * k * 1.08), int(b * k * 1.12)), ang, 0, 360, tuple(float(v) for v in DUMP), -1)
        steps = 6
        for i in range(steps):
            f = 1.0 - i / (steps + 1.5)
            color = BENCH_LIGHT if i % 2 == 0 else BENCH_DARK
            cv2.ellipse(img, center, (max(2, int(a * k * f)), max(2, int(b * k * f))), ang, 0, 360, tuple(float(v) for v in color), -1)
        cv2.ellipse(img, center, (max(2, int(a * k * 0.32)), max(2, int(b * k * 0.3))), ang, 0, 360, tuple(float(v) for v in COAL), -1)
        t = math.radians(ang)
        rim = (int(center[0] + a * k * math.cos(t)), int(center[1] + a * k * math.sin(t)))
        cv2.line(img, rim, center, tuple(float(v) for v in HAUL), max(2, k), cv2.LINE_AA)

    water = _up(m["water"], k, 1.2)
    img = _blend(img, WATER, water)

    count, labels, stats, cents = cv2.connectedComponentsWithStats(m["urban"].astype(np.uint8))
    urban = _up(m["urban"], k, 2.5)
    img = _blend(img, VILLAGE, urban * 0.75)
    for i in range(1, count):
        cx, cy = cents[i] * k
        radius = math.sqrt(stats[i, cv2.CC_STAT_AREA] / math.pi) * k
        for _ in range(int(radius * radius * 0.35)):
            ang, dist = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.uniform(0, 1))
            x, y = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
            box = cv2.boxPoints(((x, y), (rng.uniform(2, 4.5), rng.uniform(2, 4)), rng.uniform(0, 90)))
            cv2.fillConvexPoly(img, np.round(box).astype(np.int32), tuple(float(v) for v in ROOFS[rng.integers(len(ROOFS))]))

    for site in world.sites:
        x0, y0 = int((site.col - site.half_w * 1.6) * k), int((site.row - site.half_h * 1.8) * k)
        x1, y1 = int((site.col + site.half_w * 1.6) * k), int((site.row + site.half_h * 1.8) * k)
        cv2.rectangle(img, (x0, y0), (x1, y1), tuple(float(v) for v in PLANT), -1)
        rows = max(2, int(site.half_h))
        for j in range(rows):
            y = y0 + int((j + 0.6) * (y1 - y0) / (rows + 0.2))
            cv2.rectangle(img, (x0 + k, y - k), (x1 - k, y + k // 2), tuple(float(v) for v in BENCH_DARK), -1)
        cv2.circle(img, (x1 - 2 * k, y0 + 2 * k), max(2, k), tuple(float(v) for v in COAL), -1)

    for road in world.roads:
        pts = np.round(np.array(road.coords) * k).astype(np.int32).reshape(-1, 1, 2)
        if road.haul:
            cv2.polylines(img, [pts], False, tuple(float(v) for v in HAUL), max(2, int(k * 1.3)), cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], False, (70, 72, 74), k * 2 + 1, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, tuple(float(v) for v in ASPHALT), k * 2 - 1, cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    img += rng.normal(0, 2.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def scenery_layer(world, k: int = 3) -> tuple[bytes, tuple[float, float, float, float]]:
    from firewatch.geo.grid import GridSpec
    from firewatch.geo.raster import reproject_to_wgs84
    from firewatch.vision.media import encode_image

    g = world.grid
    a, _, x0, _, e, y0 = g.transform
    fine = GridSpec(g.crs, (a / k, 0.0, x0, 0.0, e / k, y0), g.width * k, g.height * k)
    bgr = render_scenery(world, k)
    bands = np.moveaxis(np.dstack([bgr, np.full(bgr.shape[:2], 255, np.uint8)]), -1, 0)
    warped, bounds = reproject_to_wgs84(bands, fine, "bilinear")
    bgra = np.ascontiguousarray(np.moveaxis(warped, 0, -1))
    west, south, east, north = bounds
    provenance = {"layer": "simulated aerial view of the synthetic coalfield", "seed": world.seed, "profile": world.profile, "crs": "EPSG:4326", "west": f"{west:.6f}", "south": f"{south:.6f}", "east": f"{east:.6f}", "north": f"{north:.6f}"}
    return encode_image(".webp", bgra, [cv2.IMWRITE_WEBP_QUALITY, 82], provenance), bounds

from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from firewatch.config import Settings, ThermalConfig
from firewatch.data.sources import HighResPatch
from firewatch.geo.grid import GridSpec, scaled_bbox
from firewatch.models import RawScene, SceneRef
from firewatch.physics import radiometry as rad
from firewatch.physics.emissivity import broadband_albedo, emissivity_from_ndvi, ndvi
from firewatch.util import parse_time
from firewatch.vision import clouds as qa

THERMAL_L8 = {"mult": 3.342e-4, "add": 0.1, "k1": 774.8853, "k2": 1321.0789}
REFL_MULT = 2.0e-5
REFL_ADD = -0.1
BANDS = ("blue", "red", "nir08", "swir16", "swir22")
BAND_NUMBERS = {"blue": 2, "red": 4, "nir08": 5, "swir16": 6, "swir22": 7}
SURFACES = {
    "soil": (0.09, 0.14, 0.20, 0.27, 0.22),
    "veg": (0.03, 0.04, 0.38, 0.19, 0.09),
    "coal": (0.045, 0.05, 0.055, 0.06, 0.055),
    "dump": (0.14, 0.20, 0.25, 0.31, 0.27),
    "water": (0.07, 0.05, 0.02, 0.01, 0.008),
    "urban": (0.12, 0.15, 0.18, 0.22, 0.20),
}
CLOUD_REFL = (0.55, 0.55, 0.58, 0.45, 0.35)
SCENE_RE = re.compile(r"^SYN_([+-]\d{3})_\d{8}$")
TRUTH_THRESHOLD_K = 2.5
PROFILES = {
    "default": {"noise_k": 0.12, "cloud_prob": 0.35, "misreg_px": 1.6, "misreg_deg": 0.12},
    "holdout": {"noise_k": 0.2, "cloud_prob": 0.5, "misreg_px": 2.4, "misreg_deg": 0.2},
}


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    a = rng.standard_normal(shape).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), sigma)
    a -= a.mean()
    s = float(a.std())
    return a / s if s > 0 else a


def _blur_mask(mask: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)


@dataclass
class FireSpec:
    name: str
    col: float
    row: float
    sigma_px: float
    amplitude_k: float
    vcol: float = 0.0
    vrow: float = 0.0
    growth_px2_per_day: float = 0.0
    amp_rate_k_per_day: float = 0.0
    start_day: float = -1e9
    end_day: float = 1e9
    elongation: float = 1.25
    angle_deg: float = 0.0
    lobes: list[tuple[float, float, float, float]] = field(default_factory=list)

    def state(self, day: float):
        if day < self.start_day:
            return None
        col = self.col + self.vcol * day
        row = self.row + self.vrow * day
        sigma = math.sqrt(max(1.0, self.sigma_px**2 + self.growth_px2_per_day * day))
        amp = self.amplitude_k + self.amp_rate_k_per_day * day
        if day > self.end_day:
            amp *= math.exp(-(day - self.end_day) / 10.0)
        return col, row, sigma, max(0.0, amp)


@dataclass
class RoadSpec:
    road_id: str
    name: str
    haul: bool
    coords: list[tuple[float, float]]


@dataclass
class IndustrialSite:
    name: str
    col: float
    row: float
    half_w: float
    half_h: float
    delta_k: float
    mapped: bool = True


@dataclass
class SyntheticWorld:
    grid: GridSpec
    thermal: ThermalConfig
    seed: int = 7
    start_date: str = "2025-10-04"
    interval_days: int = 16
    misregister: bool = True
    reference_index: int | None = None
    fires: list[FireSpec] = field(default_factory=list)
    roads: list[RoadSpec] = field(default_factory=list)
    sites: list[IndustrialSite] = field(default_factory=list)
    routes_px: list[dict] = field(default_factory=list)
    cloud_plan: dict[int, list[tuple[float, float, float]]] = field(default_factory=dict)
    profile: str = "default"

    def __post_init__(self) -> None:
        self.params = PROFILES[self.profile]
        self.start = parse_time(self.start_date)
        self.rng = np.random.default_rng(self.seed)
        self._build_surface()
        self._build_roads()
        self._place_fires_and_sites()
        self._finish_surface()

    @classmethod
    def from_settings(cls, settings: Settings, scale: float | None = None, misregister: bool = True, profile: str = "default", seed: int | None = None) -> "SyntheticWorld":
        syn = settings.data.synthetic
        bbox = scaled_bbox(settings.aoi.bbox, syn.scale if scale is None else scale)
        grid = GridSpec.from_bbox(bbox, settings.aoi.crs, settings.aoi.resolution_m)
        return cls(grid=grid, thermal=settings.thermal, seed=syn.seed if seed is None else seed, start_date=syn.start_date, interval_days=syn.pass_interval_days, misregister=misregister, profile=profile)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def _build_surface(self) -> None:
        h, w = self.shape
        rng = self.rng
        self._veg_noise = _smooth_noise(rng, (h, w), 26)
        self._fine_noise = _smooth_noise(rng, (h, w), 1.2)
        self._temp_noise = _smooth_noise(rng, (h, w), 18)
        self._shape_noise = _smooth_noise(rng, (h, w), 3.0)
        scale_area = (h * w) / (850 * 1360)
        coal = np.zeros((h, w), np.uint8)
        dump = np.zeros((h, w), np.uint8)
        water = np.zeros((h, w), np.uint8)
        urban = np.zeros((h, w), np.uint8)
        self.pits: list[tuple[float, float, float, float, float]] = []
        for _ in range(max(3, int(round(9 * scale_area)))):
            c = rng.uniform(0.1 * w, 0.9 * w)
            r = rng.uniform(0.1 * h, 0.9 * h)
            a, b, ang = rng.uniform(10, 24), rng.uniform(6, 12), rng.uniform(0, 180)
            cv2.ellipse(coal, (int(c), int(r)), (int(a), int(b)), ang, 0, 360, 1, -1)
            t = math.radians(ang)
            dc, dr = 1.4 * a * math.cos(t), 1.4 * a * math.sin(t)
            cv2.ellipse(dump, (int(c + dc), int(r + dr)), (int(a * 0.6), int(b * 0.9)), ang, 0, 360, 1, -1)
            self.pits.append((c, r, a, b, ang))
        for _ in range(max(2, int(round(5 * scale_area)))):
            cv2.ellipse(water, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), (int(rng.uniform(3, 8)), int(rng.uniform(2, 6))), rng.uniform(0, 180), 0, 360, 1, -1)
        for _ in range(max(2, int(round(6 * scale_area)))):
            cv2.circle(urban, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(5, 13)), 1, -1)
        self._masks = {"coal": coal, "dump": dump, "water": water, "urban": urban}

    def _finish_surface(self) -> None:
        h, w = self.shape
        road_mask = np.zeros((h, w), np.uint8)
        for road in self.roads:
            pts = np.round(np.array(road.coords)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(road_mask, [pts], False, 1, 2 if not road.haul else 1)
        m = self._masks
        w_water = _blur_mask(m["water"])
        w_coal = _blur_mask(m["coal"]) * (1 - w_water)
        w_dump = np.clip(_blur_mask(m["dump"]) * (1 - w_water - w_coal), 0, 1)
        w_urban = np.clip(_blur_mask(np.maximum(m["urban"], road_mask)) * (1 - w_water - w_coal - w_dump), 0, 1)
        rest = np.clip(1 - (w_water + w_coal + w_dump + w_urban), 0, 1)
        veg_frac = np.clip(0.42 + 0.28 * self._veg_noise, 0.02, 0.95)
        weights = {"water": w_water, "coal": w_coal, "dump": w_dump, "urban": w_urban, "veg": rest * veg_frac, "soil": rest * (1 - veg_frac)}
        texture = 1.0 + 0.07 * self._fine_noise
        self.reflectance: dict[str, np.ndarray] = {}
        band_rng = np.random.default_rng(self.seed + 99)
        for i, band in enumerate(BANDS):
            acc = np.zeros((h, w), np.float32)
            for surface, weight in weights.items():
                acc += weight * SURFACES[surface][i]
            acc = acc * texture + 0.002 * band_rng.standard_normal((h, w)).astype(np.float32)
            self.reflectance[band] = np.clip(acc, 0.005, 0.9).astype(np.float32)
        self.w_water = w_water
        self.w_coal = w_coal
        self.f_veg = weights["veg"]
        self.true_ndvi = ndvi(self.reflectance["red"], self.reflectance["nir08"])
        self.albedo = broadband_albedo(self.reflectance)
        self.industrial_mask = np.zeros((h, w), bool)
        self.plant_mask = np.zeros((h, w), bool)
        self._industrial = np.zeros((h, w), np.float32)
        for site in self.sites:
            r0, r1 = int(site.row - site.half_h), int(site.row + site.half_h) + 1
            c0, c1 = int(site.col - site.half_w), int(site.col + site.half_w) + 1
            core = np.zeros((h, w), np.float32)
            core[max(0, r0):r1, max(0, c0):c1] = 1.0
            self.plant_mask |= core > 0
            if site.mapped:
                self.industrial_mask |= core > 0
            self._industrial += site.delta_k * core + 6.0 * cv2.GaussianBlur(core, (0, 0), 3.0)

    def _build_roads(self) -> None:
        h, w = self.shape
        cols = np.linspace(0, w - 1, max(6, int(w / 40) + 1))
        m1 = [(float(c), float(0.56 * h + 0.03 * h * math.sin(c / (0.07 * w)))) for c in cols]
        k = int(np.argmin(np.abs(cols - 0.33 * w)))
        cx, cy = m1[k]
        rows = list(np.linspace(0, h - 1, max(5, int(h / 40) + 1)))
        j = int(np.argmin(np.abs(np.array(rows) - cy)))
        rows[j] = cy
        m2 = [(float(cx + 0.02 * w * math.sin((r - cy) / (0.08 * h))), float(r)) for r in rows]
        m2[j] = (cx, cy)
        self._main = {"M1": m1, "M2": m2}
        roads: list[RoadSpec] = []
        for rid, name, pts in (("M1", "Dhanbad-Jharia main road", m1), ("M2", "Katras link road", m2)):
            for i in range(len(pts) - 1):
                roads.append(RoadSpec(f"{rid}-{i:02d}", name, False, [pts[i], pts[i + 1]]))
        vertices = m1 + m2
        depot = m1[0]
        self.routes_px = []
        for n, (pc, pr, *_rest) in enumerate(self.pits, start=1):
            end = min(vertices, key=lambda v: (v[0] - pc) ** 2 + (v[1] - pr) ** 2)
            dx, dy = end[0] - pc, end[1] - pr
            length = math.hypot(dx, dy)
            if length < 8:
                continue
            nx, ny = -dy / length, dx / length
            bend = (pc + dx * 0.5 + nx * 0.12 * length, pr + dy * 0.5 + ny * 0.12 * length)
            pts = _subdivide([(pc, pr), bend, end], 30.0)
            rid = f"H{n}"
            for i in range(len(pts) - 1):
                roads.append(RoadSpec(f"{rid}-{i:02d}", f"Haul road, pit {n}", True, [pts[i], pts[i + 1]]))
            self.routes_px.append({"name": f"Pit {n} to washery", "from": (pc, pr), "to": depot, "haul": rid})
        self.roads = roads

    def _interior_score(self, c: float, r: float) -> float:
        h, w = self.shape
        return min(c, w - 1 - c, r, h - 1 - r)

    def _distance_to_roads(self, c: float, r: float) -> float:
        best = float("inf")
        for road in self.roads:
            (x0, y0), (x1, y1) = road.coords[0], road.coords[-1]
            best = min(best, _point_segment_distance(c, r, x0, y0, x1, y1))
        return best

    def _place_fires_and_sites(self) -> None:
        h, w = self.shape
        rng = np.random.default_rng(self.seed + 1)
        haul_ids = sorted({r.road_id.split("-")[0] for r in self.roads if r.haul})
        best = None
        for hid in haul_ids:
            segs = [r for r in self.roads if r.road_id.startswith(hid + "-")]
            seg = segs[len(segs) // 2]
            (x0, y0), (x1, y1) = seg.coords
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            score = self._interior_score(mx, my)
            if best is None or score > best[0]:
                best = (score, mx, my, x1 - x0, y1 - y0)
        _, mx, my, dx, dy = best
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        offset = 30.0
        if self._interior_score(mx + nx * offset, my + ny * offset) < self._interior_score(mx - nx * offset, my - ny * offset):
            nx, ny = -nx, -ny
        ax, ay = mx + nx * offset, my + ny * offset
        interval = self.interval_days
        fires = [
            FireSpec("A", ax, ay, 3.5, 11.0, vcol=-nx * 0.11, vrow=-ny * 0.11, growth_px2_per_day=0.22, amp_rate_k_per_day=0.02, end_day=6 * interval + 6, angle_deg=math.degrees(math.atan2(ny, nx))),
        ]
        candidates = [(rng.uniform(0.12 * w, 0.88 * w), rng.uniform(0.12 * h, 0.88 * h)) for _ in range(400)]
        far = max(candidates, key=lambda p: min(self._distance_to_roads(*p), math.hypot(p[0] - ax, p[1] - ay)))
        fires.append(FireSpec("B", far[0], far[1], 4.0, 10.0))
        m1 = self._main["M1"]
        c_vertex = m1[min(len(m1) - 2, int(len(m1) * 0.72))]
        side = 1.0 if c_vertex[1] < 0.5 * h else -1.0
        fires.append(FireSpec("C", c_vertex[0], c_vertex[1] + side * 20.0, 3.0, 6.5, vrow=-side * 0.03, growth_px2_per_day=0.15, start_day=2.5 * interval))
        m2 = self._main["M2"]
        d_vertex = m2[max(1, int(len(m2) * 0.25))]
        dside = 1.0 if d_vertex[0] < 0.5 * w else -1.0
        dx_, dy_ = d_vertex[0] + dside * 22.0, d_vertex[1]
        fires.append(FireSpec("D", dx_, dy_, 3.2, 9.0))
        others = [p for p in candidates if self._distance_to_roads(*p) > 40 and all(math.hypot(p[0] - f.col, p[1] - f.row) > 60 for f in fires)]
        e = others[0] if others else candidates[1]
        fires.append(FireSpec("E", e[0], e[1], 3.0, 12.0, end_day=20.0))
        self.fires = fires
        self.cloud_plan = {5: [(dx_ + 7.0, dy_ - 2.0, 9.0)]}
        s_vertex = m1[max(1, int(len(m1) * 0.5))]
        sside = -1.0 if s_vertex[1] > 0.5 * h else 1.0
        self.sites = [IndustrialSite("Coke oven battery", s_vertex[0] + 8.0, s_vertex[1] + sside * 14.0, 5.0, 3.0, 28.0)]
        if self.profile == "holdout":
            self._holdout_fires_and_sites()

    def _nearest_road_point(self, c: float, r: float) -> tuple[float, float]:
        best, point = float("inf"), (c, r)
        for road in self.roads:
            for x, y in road.coords:
                d = math.hypot(x - c, y - r)
                if d < best:
                    best, point = d, (x, y)
        return point

    def _holdout_fires_and_sites(self) -> None:
        h, w = self.shape
        rng = np.random.default_rng(self.seed + 101)
        fires: list[FireSpec] = []
        for n in range(7):
            moving = n < 3
            for _attempt in range(400):
                c, r = rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.9 * h)
                d = self._distance_to_roads(c, r)
                if all(math.hypot(c - f.col, r - f.row) > 45 for f in fires) and (15 <= d <= 35 if moving else d > 8):
                    break
            tx, ty = self._nearest_road_point(c, r)
            dist = max(math.hypot(tx - c, ty - r), 1.0)
            speed = rng.uniform(0.12, 0.3) if moving else 0.0
            lobes = [(rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(0.3, 0.8), rng.uniform(0.5, 0.9)) for _ in range(int(rng.integers(1, 4)))]
            fires.append(FireSpec(
                name=f"H{n + 1}", col=c, row=r, sigma_px=rng.uniform(1.8, 3.8), amplitude_k=rng.uniform(4.0, 12.0),
                vcol=speed * (tx - c) / dist, vrow=speed * (ty - r) / dist, growth_px2_per_day=rng.uniform(0.0, 0.25),
                start_day=float(rng.uniform(-90, 60)) if n in (3, 4) else -1e9, end_day=float(rng.uniform(20, 120)) if n in (5, 6) else 1e9,
                elongation=rng.uniform(1.0, 1.8), angle_deg=rng.uniform(0, 180), lobes=lobes,
            ))
        self.fires = fires
        self.cloud_plan = {}
        far = max(((rng.uniform(0.15 * w, 0.85 * w), rng.uniform(0.15 * h, 0.85 * h)) for _ in range(200)), key=lambda p: min(math.hypot(p[0] - f.col, p[1] - f.row) for f in fires))
        self.sites.append(IndustrialSite("Unmapped coke plant", far[0], far[1], 4.0, 4.0, 24.0, mapped=False))

    def day_of(self, index: int) -> float:
        return float(index * self.interval_days)

    def date_of(self, index: int) -> dt.datetime:
        return self.start + dt.timedelta(days=self.day_of(index), hours=4, minutes=50)

    def scene_id(self, index: int) -> str:
        return f"SYN_{index:+04d}_{self.date_of(index):%Y%m%d}"

    @staticmethod
    def index_of(scene_id: str) -> int:
        match = SCENE_RE.match(scene_id)
        if not match:
            raise KeyError(f"not a synthetic scene id: {scene_id}")
        return int(match.group(1))

    def ref(self, index: int) -> SceneRef:
        return SceneRef(scene_id=self.scene_id(index), acquired=self.date_of(index), platform="LANDSAT_8", source="synthetic", cloud_cover=None, path_row="140044")

    def fire_fields(self, day: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        h, w = self.shape
        total = np.zeros((h, w), np.float32)
        per_fire: dict[str, np.ndarray] = {}
        for fire in self.fires:
            state = fire.state(day)
            field_ = np.zeros((h, w), np.float32)
            if state is not None and state[3] > 0.05:
                col, row, sigma, amp = state
                radius = int(math.ceil(4.5 * sigma * fire.elongation + max([math.hypot(dx, dy) for dx, dy, _, _ in fire.lobes], default=0.0)))
                r0, r1 = max(0, int(row) - radius), min(h, int(row) + radius + 1)
                c0, c1 = max(0, int(col) - radius), min(w, int(col) + radius + 1)
                if r1 > r0 and c1 > c0:
                    rr, cc = np.mgrid[r0:r1, c0:c1].astype(np.float32)
                    t = math.radians(fire.angle_deg)
                    u = (cc - col) * math.cos(t) + (rr - row) * math.sin(t)
                    v = -(cc - col) * math.sin(t) + (rr - row) * math.cos(t)
                    g = np.exp(-0.5 * ((u / (sigma * fire.elongation)) ** 2 + (v / (sigma / fire.elongation)) ** 2))
                    for dx, dy, weight, scale in fire.lobes:
                        g += weight * np.exp(-0.5 * (((cc - col - dx) ** 2 + (rr - row - dy) ** 2) / (sigma * scale) ** 2))
                    if fire.lobes:
                        g /= max(float(g.max()), 1e-6)
                    g *= 1.0 + 0.22 * self._shape_noise[r0:r1, c0:c1]
                    field_[r0:r1, c0:c1] = np.clip(amp * g, 0, None)
            per_fire[fire.name] = field_
            total += field_
        return total, per_fire

    def _clouds(self, index: int, fire_positions: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.shape
        rng = np.random.default_rng(self.seed * 1000 + index + 500)
        mask = np.zeros((h, w), np.uint8)
        for c, r, radius in self.cloud_plan.get(index, []):
            cv2.circle(mask, (int(c), int(r)), int(radius), 1, -1)
        if index != 0 and rng.random() < self.params["cloud_prob"]:
            for _ in range(int(rng.integers(1, 4))):
                for _attempt in range(20):
                    c, r = rng.uniform(0, w), rng.uniform(0, h)
                    radius = rng.uniform(8, 20)
                    if all(math.hypot(c - fc, r - fr) > radius + 30 for fc, fr in fire_positions):
                        cv2.circle(mask, (int(c), int(r)), int(radius), 1, -1)
                        break
        if not mask.any():
            return np.zeros((h, w), bool), np.zeros((h, w), bool)
        soft = _blur_mask(mask, 2.0) + 0.12 * _smooth_noise(rng, (h, w), 2.0)
        cloud = soft > 0.5
        shifted = np.zeros_like(cloud)
        shifted[7:, 10:] = cloud[:-7, :-10]
        return cloud, shifted & ~cloud

    def true_lst(self, day: float) -> np.ndarray:
        date = self.start + dt.timedelta(days=day)
        doy = date.timetuple().tm_yday
        base = 303.0 + 8.0 * math.cos(2 * math.pi * (doy - 130) / 365.0)
        fires, _ = self.fire_fields(day)
        lst = base + 26.0 * (1.0 - self.albedo) - 9.0 * self.f_veg - 22.0 * self.w_water + 0.6 * self._temp_noise + fires + self._industrial
        return lst.astype(np.float32)

    def random_transform(self, index: int) -> np.ndarray:
        h, w = self.shape
        if index == 0 or index == self.reference_index or not self.misregister:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rng = np.random.default_rng(self.seed * 7919 + index)
        angle = rng.uniform(-self.params["misreg_deg"], self.params["misreg_deg"])
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        m[0, 2] += rng.uniform(-self.params["misreg_px"], self.params["misreg_px"])
        m[1, 2] += rng.uniform(-self.params["misreg_px"], self.params["misreg_px"])
        return m

    def raw_scene(self, index: int, crop: GridSpec | None = None) -> RawScene:
        h, w = self.shape
        day = self.day_of(index)
        date = self.date_of(index)
        doy = date.timetuple().tm_yday
        sun_elevation = 52.0 + 14.0 * math.cos(2 * math.pi * (doy - 172) / 365.0)
        fire_total, per_fire = self.fire_fields(day)
        positions = [(s[0], s[1]) for s in (f.state(day) for f in self.fires) if s]
        cloud, shadow = self._clouds(index, positions)
        lst = self.true_lst(day)
        noise_rng = np.random.default_rng(self.seed * 31 + index + 1000)
        lst = np.where(shadow, lst - 3.0, lst)
        lst = np.where(cloud, 268.0 + 3.0 * self._temp_noise, lst).astype(np.float32)
        sensor = cv2.GaussianBlur(lst, (0, 0), 1.3) + self.params["noise_k"] * noise_rng.standard_normal((h, w)).astype(np.float32)
        eps = emissivity_from_ndvi(self.true_ndvi, self.thermal) - 0.012 * self.w_coal
        eps = np.where(cloud, 0.99, eps).astype(np.float32)
        bt = rad.brightness_temp_from_lst(sensor, eps, self.thermal.wavelength_um)
        radiance = rad.brightness_temp_to_radiance(bt, THERMAL_L8["k1"], THERMAL_L8["k2"])
        thermal_dn = np.clip(np.round(rad.radiance_to_dn(radiance, THERMAL_L8["mult"], THERMAL_L8["add"])), 1, 65535).astype(np.float32)
        refl = {}
        for i, band in enumerate(BANDS):
            values = self.reflectance[band].copy()
            values = np.where(shadow, values * 0.6, values)
            values = np.where(cloud, CLOUD_REFL[i] + 0.03 * self._fine_noise, values)
            dn = np.clip(np.round(rad.reflectance_to_dn(values, REFL_MULT, REFL_ADD, sun_elevation)), 1, 65535)
            refl[band] = rad.toa_reflectance(dn, REFL_MULT, REFL_ADD, sun_elevation)
        qa_pixel = np.full((h, w), qa.QA_CLEAR, np.uint16)
        qa_pixel[self.w_water > 0.5] |= qa.QA_WATER
        ring = cv2.dilate(cloud.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~cloud
        qa_pixel[cloud] = qa.QA_CLOUD | qa.QA_CLOUD_CONF_HIGH
        qa_pixel[ring] = (qa_pixel[ring] & ~np.uint16(qa.QA_CLEAR)) | qa.QA_DILATED_CLOUD
        qa_pixel[shadow & ~cloud] = qa.QA_SHADOW | qa.QA_SHADOW_CONF_HIGH
        l2 = cv2.GaussianBlur(lst, (0, 0), 1.3) + 0.25 * noise_rng.standard_normal((h, w)).astype(np.float32)
        l2 = np.where(cloud, np.nan, l2).astype(np.float32)
        m = self.random_transform(index)
        warp = lambda a, interp=cv2.INTER_LINEAR: cv2.warpAffine(a, m, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)
        thermal_dn = warp(thermal_dn)
        refl = {b: warp(v) for b, v in refl.items()}
        qa_pixel = warp(qa_pixel, cv2.INTER_NEAREST)
        l2 = warp(l2)
        mtl = {
            "SPACECRAFT_ID": "LANDSAT_8",
            "SUN_ELEVATION": f"{sun_elevation:.4f}",
            "CLOUD_COVER": f"{100.0 * cloud.mean():.2f}",
            "RADIANCE_MULT_BAND_10": f"{THERMAL_L8['mult']:.4E}",
            "RADIANCE_ADD_BAND_10": f"{THERMAL_L8['add']:.5f}",
            "K1_CONSTANT_BAND_10": f"{THERMAL_L8['k1']:.4f}",
            "K2_CONSTANT_BAND_10": f"{THERMAL_L8['k2']:.4f}",
        }
        for band, num in BAND_NUMBERS.items():
            mtl[f"REFLECTANCE_MULT_BAND_{num}"] = f"{REFL_MULT:.4E}"
            mtl[f"REFLECTANCE_ADD_BAND_{num}"] = f"{REFL_ADD:.6f}"
        truth = {
            "index": index,
            "day": day,
            "transform": m.tolist(),
            "fire_anomaly": fire_total,
            "fire_masks": {name: arr > TRUTH_THRESHOLD_K for name, arr in per_fire.items()},
            "cloud": cloud,
            "industrial": self.plant_mask,
            "lst_true": lst,
        }
        ref = self.ref(index)
        ref.cloud_cover = float(100.0 * cloud.mean())
        scene = RawScene(ref=ref, grid=self.grid, thermal_dn=thermal_dn, reflectance=refl, qa_pixel=qa_pixel, mtl=mtl, l2_surface_temp=l2, truth=truth)
        return crop_scene(scene, crop) if crop is not None and crop != self.grid else scene

    def highres(self, bounds: tuple[float, float, float, float], crs: str, when: dt.datetime) -> HighResPatch:
        minx, miny, maxx, maxy = bounds
        res = 20.0
        width = max(4, int(math.ceil((maxx - minx) / res)))
        height = max(4, int(math.ceil((maxy - miny) / res)))
        grid = GridSpec(crs, (res, 0.0, minx, 0.0, -res, maxy), width, height)
        rows, cols = np.mgrid[0:height, 0:width]
        xs, ys = grid.xy(rows, cols)
        wc, wr = self.grid.colrow(xs, ys)
        map_x, map_y = wc.astype(np.float32), wr.astype(np.float32)
        sample = lambda a: cv2.remap(a.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        day = (when - self.start).total_seconds() / 86400.0
        fires, _ = self.fire_fields(day)
        heat = sample(fires + self._industrial)
        emission = np.clip((heat - 8.0) / 10.0, 0.0, 1.5)
        nir = sample(self.reflectance["nir08"])
        swir16 = sample(self.reflectance["swir16"]) + 0.05 * emission
        swir22 = sample(self.reflectance["swir22"]) + 0.22 * emission
        valid = np.ones((height, width), bool)
        return HighResPatch(f"SYN_S2_{when:%Y%m%d}", when, grid, nir, swir16, swir22, valid, "synthetic")

    def roads_geojson(self) -> dict:
        features = []
        for road in self.roads:
            coords = self.grid.map_to_lonlat_coords(self.grid.pixel_polygon_to_map(road.coords))
            features.append({"type": "Feature", "properties": {"id": road.road_id, "name": road.name, "haul": road.haul}, "geometry": {"type": "LineString", "coordinates": coords}})
        return {"type": "FeatureCollection", "features": features}

    def industrial_geojson(self) -> dict:
        features = []
        for site in (x for x in self.sites if x.mapped):
            ring = [(site.col - site.half_w, site.row - site.half_h), (site.col + site.half_w, site.row - site.half_h), (site.col + site.half_w, site.row + site.half_h), (site.col - site.half_w, site.row + site.half_h), (site.col - site.half_w, site.row - site.half_h)]
            coords = self.grid.map_to_lonlat_coords(self.grid.pixel_polygon_to_map(ring))
            features.append({"type": "Feature", "properties": {"name": site.name}, "geometry": {"type": "Polygon", "coordinates": [coords]}})
        return {"type": "FeatureCollection", "features": features}

    def routes(self) -> list[dict]:
        out = []
        for route in self.routes_px:
            (fx, fy), (tx, ty) = self.grid.pixel_polygon_to_map([route["from"], route["to"]])
            out.append({"name": route["name"], "from": self.grid.map_to_lonlat_coords([(fx, fy)])[0], "to": self.grid.map_to_lonlat_coords([(tx, ty)])[0]})
        return out


def crop_scene(scene: RawScene, grid: GridSpec) -> RawScene:
    row, col = grid.offset_in(scene.grid)
    sl = (slice(row, row + grid.height), slice(col, col + grid.width))
    truth = None
    if scene.truth is not None:
        truth = dict(scene.truth)
        for key in ("fire_anomaly", "cloud", "industrial", "lst_true"):
            truth[key] = scene.truth[key][sl]
        truth["fire_masks"] = {k: v[sl] for k, v in scene.truth["fire_masks"].items()}
        truth["crop_offset"] = (row, col)
    return RawScene(
        ref=scene.ref,
        grid=grid,
        thermal_dn=scene.thermal_dn[sl],
        reflectance={k: v[sl] for k, v in scene.reflectance.items()},
        qa_pixel=scene.qa_pixel[sl],
        mtl=scene.mtl,
        l2_surface_temp=None if scene.l2_surface_temp is None else scene.l2_surface_temp[sl],
        truth=truth,
    )


def export_landsat_files(scene: RawScene, out_dir: str | Path, pad_px: int = 24) -> SceneRef:
    from firewatch.geo.raster import write_geotiff

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = scene.grid
    a, b, c, d, e, f = g.transform
    big = GridSpec(g.crs, (a, b, c - pad_px * a, d, e, f - pad_px * e), g.width + 2 * pad_px, g.height + 2 * pad_px)

    def padded(arr: np.ndarray, fill: int) -> np.ndarray:
        canvas = np.full((big.height, big.width), fill, dtype=np.uint16)
        canvas[pad_px:pad_px + g.height, pad_px:pad_px + g.width] = arr
        return canvas

    sid = scene.ref.scene_id
    sun = float(scene.mtl["SUN_ELEVATION"])
    assets = {"lwir11": write_geotiff(out / f"{sid}_B10.TIF", padded(np.clip(np.round(scene.thermal_dn), 1, 65535).astype(np.uint16), 0), big, nodata=0)}
    for band, num in BAND_NUMBERS.items():
        dn = np.clip(np.round(rad.reflectance_to_dn(scene.reflectance[band], REFL_MULT, REFL_ADD, sun)), 1, 65535).astype(np.uint16)
        assets[band] = write_geotiff(out / f"{sid}_B{num}.TIF", padded(dn, 0), big, nodata=0)
    assets["qa_pixel"] = write_geotiff(out / f"{sid}_QA_PIXEL.TIF", padded(scene.qa_pixel.astype(np.uint16), qa.QA_FILL), big, nodata=qa.QA_FILL)
    groups = {"IMAGE_ATTRIBUTES": {}, "LEVEL1_RADIOMETRIC_RESCALING": {}, "LEVEL1_THERMAL_CONSTANTS": {}}
    for key, value in scene.mtl.items():
        group = "LEVEL1_THERMAL_CONSTANTS" if key.startswith("K") else "LEVEL1_RADIOMETRIC_RESCALING" if "_BAND_" in key else "IMAGE_ATTRIBUTES"
        groups[group][key] = value
    mtl_path = out / f"{sid}_MTL.json"
    mtl_path.write_text(json.dumps({"LANDSAT_METADATA_FILE": groups}, indent=1))
    hrefs = {k: str(v) for k, v in assets.items()}
    hrefs["MTL.json"] = str(mtl_path)
    return SceneRef(scene_id=sid, acquired=scene.ref.acquired, platform=scene.ref.platform, source="landsat", cloud_cover=scene.ref.cloud_cover, path_row=scene.ref.path_row, assets=hrefs)


def _subdivide(points: list[tuple[float, float]], max_len: float) -> list[tuple[float, float]]:
    out = [points[0]]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / max_len)))
        for k in range(1, n + 1):
            out.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    out[-1] = points[-1]
    return out


def _point_segment_distance(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


class SyntheticSource:
    name = "synthetic"

    def __init__(self, world: SyntheticWorld, first_index: int = -6, last_index: int = 12):
        self.world = world
        self.first_index = first_index
        self.last_index = last_index
        if world.reference_index is None:
            world.reference_index = first_index

    def _indices(self) -> range:
        return range(self.first_index, self.last_index + 1)

    def list_scenes(self, start: dt.datetime, end: dt.datetime) -> list[SceneRef]:
        start, end = parse_time(start), parse_time(end)
        return [self.world.ref(i) for i in self._indices() if start <= self.world.date_of(i) <= end]

    def get_ref(self, scene_id: str) -> SceneRef:
        return self.world.ref(self.world.index_of(scene_id))

    def ref_for_index(self, index: int) -> SceneRef:
        return self.world.ref(index)

    def load(self, ref: SceneRef, grid: GridSpec) -> RawScene:
        return self.world.raw_scene(self.world.index_of(ref.scene_id), crop=grid)

    def history(self, before: dt.datetime, n: int) -> list[SceneRef]:
        before = parse_time(before)
        earlier = [self.world.ref(i) for i in self._indices() if self.world.date_of(i) < before]
        return earlier[-n:] if n > 0 else []

    def highres(self, bounds: tuple[float, float, float, float], crs: str, when: dt.datetime) -> HighResPatch:
        return self.world.highres(bounds, crs, parse_time(when))

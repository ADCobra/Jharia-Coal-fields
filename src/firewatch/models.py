from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from firewatch.geo.grid import GridSpec
from firewatch.util import parse_time

Coord = tuple[float, float]


@dataclass
class SceneRef:
    scene_id: str
    acquired: dt.datetime
    platform: str
    source: str
    cloud_cover: float | None = None
    path_row: str | None = None
    assets: dict[str, str] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneRef":
        values = dict(data)
        values["acquired"] = parse_time(values["acquired"])
        return cls(**values)


@dataclass
class RawScene:
    ref: SceneRef
    grid: GridSpec
    thermal_dn: np.ndarray
    reflectance: dict[str, np.ndarray]
    qa_pixel: np.ndarray
    mtl: dict[str, str]
    l2_surface_temp: np.ndarray | None = None
    truth: dict[str, Any] | None = None


@dataclass
class ThermalProduct:
    lst: np.ndarray
    brightness_temp: np.ndarray
    emissivity: np.ndarray
    ndvi: np.ndarray
    albedo: np.ndarray
    valid: np.ndarray
    cloud_fraction: float
    water: np.ndarray | None = None


@dataclass
class Hotspot:
    hotspot_id: str
    scene_id: str
    acquired: dt.datetime
    area_m2: float
    pixel_count: int
    mean_anomaly_k: float
    max_anomaly_k: float
    peak_lst_k: float
    centroid: Coord
    centroid_lonlat: Coord
    polygon: list[Coord]
    bbox_px: tuple[int, int, int, int]
    solidity: float
    rectangularity: float
    elongation: float
    cloud_ring_fraction: float
    industrial_overlap: float
    track_id: str | None = None
    persistent: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Hotspot":
        values = dict(data)
        values["acquired"] = parse_time(values["acquired"])
        values["centroid"] = tuple(values["centroid"])
        values["centroid_lonlat"] = tuple(values["centroid_lonlat"])
        values["polygon"] = [tuple(p) for p in values["polygon"]]
        values["bbox_px"] = tuple(values["bbox_px"])
        return cls(**values)


@dataclass
class TrackPoint:
    scene_id: str
    acquired: dt.datetime
    hotspot_id: str
    area_m2: float
    mean_anomaly_k: float
    max_anomaly_k: float
    centroid: Coord
    polygon: list[Coord]
    risk: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "TrackPoint":
        values = dict(data)
        values["acquired"] = parse_time(values["acquired"])
        values["centroid"] = tuple(values["centroid"])
        values["polygon"] = [tuple(p) for p in values["polygon"]]
        return cls(**values)


@dataclass
class Track:
    track_id: str
    points: list[TrackPoint] = field(default_factory=list)
    status: str = "active"
    missed: int = 0
    quiet_passes: int = 0
    closed_segments: list[str] = field(default_factory=list)
    label: str | None = None

    @property
    def first_seen(self) -> dt.datetime | None:
        return self.points[0].acquired if self.points else None

    @property
    def last(self) -> TrackPoint | None:
        return self.points[-1] if self.points else None

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        values = dict(data)
        values["points"] = [TrackPoint.from_dict(p) for p in values.get("points", [])]
        return cls(**values)


@dataclass
class Kinematics:
    n_points: int
    span_days: float
    growth_m2_per_day: float
    growth_frac_per_30d: float
    vx_m_per_day: float
    vy_m_per_day: float
    speed_m_per_day: float
    bearing_deg: float
    anomaly_trend_k_per_day: float


@dataclass
class SpreadEstimate:
    vx_m_per_day: float
    vy_m_per_day: float
    diffusivity_m2_per_day: float
    decay_per_day: float
    source: str
    flow_vx_m_per_day: float | None = None
    flow_vy_m_per_day: float | None = None


@dataclass
class Projection:
    horizon_days: float
    cone_polygon: list[Coord]
    core_polygon: list[Coord]
    reach_probability: float
    median_days_to_road: float | None
    projected_distance_m: float | None
    ensemble: int


@dataclass
class RiskAssessment:
    hotspot_id: str
    track_id: str | None
    score: float
    level: str
    components: dict[str, float]
    distance_m: float | None
    projected_distance_m: float | None
    nearest_segment: str | None
    approach_m_per_day: float
    confidence: float
    confidence_notes: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
    method: str
    ok: bool
    matrix: list[list[float]]
    n_ref: int = 0
    n_mov: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    residual_px: float | None = None
    shift_before_px: float | None = None
    shift_after_px: float | None = None
    seconds: float = 0.0
    note: str = ""
    estimator: str | None = None
    check_method: str | None = None
    check_ok: bool | None = None
    check_matrix: list[list[float]] | None = None
    agreement_px: float | None = None


@dataclass
class ActionRecord:
    action_id: str
    kind: str
    actor: str
    target: list[str]
    hotspot_id: str | None
    track_id: str | None
    reason: str
    executed: bool
    created: dt.datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    run_id: str
    seq: int
    ts: dt.datetime
    node: str
    hotspot_id: str | None
    observation: dict[str, Any] = field(default_factory=dict)
    decision: str | None = None
    tool: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None

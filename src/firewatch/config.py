from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULTS_FILE = Path(__file__).with_name("defaults.yaml")
ENV_PREFIX = "FIREWATCH__"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AOIConfig(_Section):
    name: str
    bbox: tuple[float, float, float, float]
    crs: str
    resolution_m: float


class SyntheticConfig(_Section):
    seed: int
    scale: float
    pass_interval_days: int
    start_date: str


class DataConfig(_Section):
    source: Literal["landsat", "synthetic", "real"]
    landsat_stac_url: str
    landsat_l1_collection: str
    landsat_l2_collection: str
    sentinel_stac_url: str
    sentinel_collection: str
    platforms: list[str]
    max_cloud: float
    months: list[int]
    min_sun_elevation: float
    requester_pays: bool
    prefer_s3: bool
    history_lookback_days: int
    sentinel_window_days: int
    real_dir: str
    synthetic: SyntheticConfig


class ThermalConfig(_Section):
    wavelength_um: float
    emissivity_soil: float
    emissivity_vegetation: float
    emissivity_water: float
    ndvi_soil: float
    ndvi_vegetation: float
    cavity_shape_factor: float
    detrend: bool
    detrend_sample: int
    psf_sigma_px: float


class HotspotConfig(_Section):
    background_window_px: int
    sigma_high: float
    sigma_low: float
    min_contrast_k: float
    min_weak_contrast_k: float
    min_area_px: int
    cloud_ring_px: int
    cloud_edge_fraction: float
    water_buffer_px: int
    max_vertices: int
    persistence: bool
    persist_sigma: float
    persist_contrast_k: float
    persist_min_area_px: int
    persist_reach_px: int


Matcher = Literal["aliked_lightglue", "disk", "sift", "orb"]


class AlignmentConfig(_Section):
    method: Literal["auto", "aliked_lightglue", "disk", "sift", "orb", "none"]
    preference: list[Matcher]
    estimator: Literal["ransac", "magsac", "accurate"]
    cross_check: bool
    check_preference: list[Matcher]
    max_disagreement_px: float
    tiebreak_preference: list[Matcher]
    tiebreak_window_px: int
    band: str
    models_dir: str
    aliked_model: str
    lightglue_model: str
    disk_model: str
    max_features: int
    ransac_threshold_px: float
    min_inliers: int
    max_residual_px: float
    ratio_test: float


class TrackingConfig(_Section):
    max_match_distance_m: float
    max_missed_passes: int
    history_points: int
    position_noise_m: float = 0.0


class ProjectionConfig(_Section):
    horizon_days: float
    ensemble: int
    crop_px: int
    velocity_sigma_frac: float
    diffusivity_sigma_frac: float
    max_diffusivity_m2_per_day: float
    footprint_contrast_k: float
    cone_probability: float
    seed: int


class RiskConfig(_Section):
    weights: dict[str, float]
    anomaly_ref_k: float
    growth_ref_frac_per_30d: float
    approach_ref_m_per_day: float
    d0_m: float
    high: float
    medium: float
    consider: float
    min_confidence_auto: float
    close_buffer_m: float
    precedent_min_analogues: int
    precedent_min_tracks: int


class RoadsConfig(_Section):
    path: str | None
    routes: Any
    haul_only_closures: bool
    auto_close: Literal["all", "haul"] = "all"
    reopen_quiet_passes: int


class AgentConfig(_Section):
    planner: Literal["policy", "bedrock"]
    max_steps_per_hotspot: int
    allow_auto_close: bool
    bedrock_model_id: str
    history_passes: int
    highres_buffer_m: float
    dashboard_url: str


class VLMConfig(_Section):
    enabled: bool
    siglip_model: str | None
    embedding_model: str | None
    gemma_model: str | None
    tokenizer_config: str | None
    max_new_tokens: int


class MaskConfig(_Section):
    industrial: str | None


class StorageConfig(_Section):
    backend: Literal["local", "s3"]
    root: str
    bucket: str | None
    prefix: str


class StateConfig(_Section):
    backend: Literal["sqlite", "dynamodb"]
    sqlite_path: str
    table: str | None


class NotifyConfig(_Section):
    backend: Literal["console", "sns"]
    topic_arn: str | None
    outbox: str


class ObservabilityConfig(_Section):
    log_group: str | None
    metrics_namespace: str | None


class APIConfig(_Section):
    user: str
    password: str
    allow_runs: bool
    scenery: bool


class AWSConfig(_Section):
    region: str


class WorkerConfig(_Section):
    queue_url: str | None
    wait_seconds: int
    path_rows: list[str]


class Settings(_Section):
    aoi: AOIConfig
    data: DataConfig
    thermal: ThermalConfig
    hotspots: HotspotConfig
    alignment: AlignmentConfig
    tracking: TrackingConfig
    projection: ProjectionConfig
    risk: RiskConfig
    roads: RoadsConfig
    agent: AgentConfig
    vlm: VLMConfig
    masks: MaskConfig
    storage: StorageConfig
    state: StateConfig
    notify: NotifyConfig
    observability: ObservabilityConfig
    api: APIConfig
    aws: AWSConfig
    worker: WorkerConfig

    def with_overrides(self, overrides: dict[str, Any]) -> "Settings":
        data = self.model_dump()
        deep_merge(data, overrides)
        return Settings.model_validate(data)


def deep_merge(base: dict, extra: dict) -> dict:
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _env_overrides(environ: dict[str, str]) -> dict:
    out: dict = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX):].split("__") if p]
        if not path:
            continue
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = yaml.safe_load(raw) if raw != "" else None
    return out


def load_settings(*paths: str | Path, environ: dict[str, str] | None = None) -> Settings:
    environ = dict(os.environ if environ is None else environ)
    data = yaml.safe_load(DEFAULTS_FILE.read_text())
    extra_paths = list(paths)
    if not extra_paths and environ.get("FIREWATCH_CONFIG"):
        extra_paths = [p.strip() for p in environ["FIREWATCH_CONFIG"].split(",") if p.strip()]
    for path in extra_paths:
        deep_merge(data, yaml.safe_load(Path(path).read_text()) or {})
    deep_merge(data, _env_overrides(environ))
    return Settings.model_validate(data)

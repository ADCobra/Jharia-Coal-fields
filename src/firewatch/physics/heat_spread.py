from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from firewatch.config import ProjectionConfig
from firewatch.models import Kinematics, SpreadEstimate

MAX_SPEED_M_PER_DAY = 30.0


def estimate_spread(kin: Kinematics | None, flow: tuple[float, float] | None, mean_anomaly_k: float, cfg: ProjectionConfig) -> SpreadEstimate:
    track_v = (kin.vx_m_per_day, kin.vy_m_per_day) if kin and kin.n_points >= 2 else None
    if track_v and flow:
        w = 0.6 if kin.n_points >= 3 else 0.5
        vx, vy = w * track_v[0] + (1 - w) * flow[0], w * track_v[1] + (1 - w) * flow[1]
        source = "track+flow"
    elif track_v:
        vx, vy, source = track_v[0], track_v[1], "track"
    elif flow:
        vx, vy, source = flow[0], flow[1], "flow"
    else:
        vx, vy, source = 0.0, 0.0, "none"
    speed = math.hypot(vx, vy)
    if speed > MAX_SPEED_M_PER_DAY:
        vx, vy = vx * MAX_SPEED_M_PER_DAY / speed, vy * MAX_SPEED_M_PER_DAY / speed
    growth = kin.growth_m2_per_day if kin else 0.0
    diffusivity = min(max(growth, 0.0) / (4.0 * math.pi), cfg.max_diffusivity_m2_per_day)
    decay = 0.0
    if kin and kin.n_points >= 2 and mean_anomaly_k > 0:
        decay = min(0.1, max(0.0, -kin.anomaly_trend_k_per_day / mean_anomaly_k))
    return SpreadEstimate(vx, vy, diffusivity, decay, source, flow[0] if flow else None, flow[1] if flow else None)


def advect_diffuse(u: np.ndarray, vx_px: float, vy_px: float, d_px2: float, decay: float, days: float) -> np.ndarray:
    limits = [days]
    if d_px2 > 0:
        limits.append(0.24 / d_px2)
    speed = abs(vx_px) + abs(vy_px)
    if speed > 0:
        limits.append(0.9 / speed)
    steps = max(1, int(math.ceil(days / min(limits))))
    dt = days / steps
    u = u.astype(np.float32).copy()
    for _ in range(steps):
        if vx_px >= 0:
            dudx = u - np.roll(u, 1, axis=1)
        else:
            dudx = np.roll(u, -1, axis=1) - u
        if vy_px >= 0:
            dudy = u - np.roll(u, 1, axis=0)
        else:
            dudy = np.roll(u, -1, axis=0) - u
        lap = cv2.Laplacian(u, cv2.CV_32F, ksize=1) if d_px2 > 0 else 0.0
        u = u + dt * (-vx_px * dudx - vy_px * dudy + d_px2 * lap - decay * u)
        u[0, :] = u[-1, :] = 0.0
        u[:, 0] = u[:, -1] = 0.0
        np.maximum(u, 0.0, out=u)
    return u


@dataclass
class ConeResult:
    probability: np.ndarray
    reach_probability: float
    median_days_to_road: float | None
    member_reach_days: list[float]


def project_cone(anomaly: np.ndarray, mask: np.ndarray, spread: SpreadEstimate, res_m: float, cfg: ProjectionConfig, road_mask: np.ndarray | None = None) -> ConeResult:
    seed_region = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    u0 = np.where(seed_region, np.clip(np.nan_to_num(anomaly), 0, None), 0.0).astype(np.float32)
    rng = np.random.default_rng(cfg.seed)
    n_steps = 12
    step_days = cfg.horizon_days / n_steps
    speed = math.hypot(spread.vx_m_per_day, spread.vy_m_per_day)
    heading = math.atan2(spread.vy_m_per_day, spread.vx_m_per_day)
    counts = np.zeros(u0.shape, np.float32)
    reach_days: list[float] = []
    for _ in range(cfg.ensemble):
        s = max(0.0, speed * (1.0 + cfg.velocity_sigma_frac * rng.standard_normal()))
        theta = heading + math.radians(15.0) * rng.standard_normal()
        vx_m, vy_m = s * math.cos(theta), s * math.sin(theta)
        d_m = spread.diffusivity_m2_per_day * math.exp(cfg.diffusivity_sigma_frac * rng.standard_normal())
        vx_px, vy_px, d_px = vx_m / res_m, -vy_m / res_m, d_m / (res_m * res_m)
        u = u0
        covered = u0 > cfg.footprint_contrast_k
        reached = None
        if road_mask is not None and (covered & road_mask).any():
            reached = 0.0
        for k in range(n_steps):
            u = advect_diffuse(u, vx_px, vy_px, d_px, spread.decay_per_day, step_days)
            footprint = u > cfg.footprint_contrast_k
            covered |= footprint
            if reached is None and road_mask is not None and (footprint & road_mask).any():
                reached = (k + 1) * step_days
        counts += covered
        if reached is not None:
            reach_days.append(reached)
    probability = counts / float(cfg.ensemble)
    reach_p = len(reach_days) / float(cfg.ensemble)
    median = float(np.median(reach_days)) if reach_days and reach_p >= 0.5 else None
    return ConeResult(probability, reach_p, median, reach_days)

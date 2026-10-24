from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from firewatch.config import HotspotConfig, ThermalConfig


@dataclass
class AnomalyResult:
    residual: np.ndarray
    background: np.ndarray
    anomaly: np.ndarray
    z: np.ndarray
    sigma_k: float
    trend: dict[str, float] = field(default_factory=dict)


def masked_box_mean(values: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    m = mask.astype(np.float32)
    v = np.where(mask, values, 0.0).astype(np.float32)
    num = cv2.boxFilter(v, -1, (window, window), normalize=False, borderType=cv2.BORDER_REFLECT)
    den = cv2.boxFilter(m, -1, (window, window), normalize=False, borderType=cv2.BORDER_REFLECT)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0.5, out, np.nan).astype(np.float32)


def masked_gaussian(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values
    m = mask.astype(np.float32)
    v = np.where(mask, values, 0.0).astype(np.float32)
    num = cv2.GaussianBlur(v, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(m, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(mask & (den > 1e-3), out, values).astype(np.float32)


def water_ring(water: np.ndarray | None, radius: int) -> np.ndarray | None:
    if water is None or not water.any():
        return None
    if radius <= 0:
        return water
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(water.astype(np.uint8), kernel).astype(bool)


def robust_sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def detrend_temperature(lst: np.ndarray, valid: np.ndarray, albedo: np.ndarray, ndvi: np.ndarray, sample: int = 200_000, seed: int = 0) -> tuple[np.ndarray, dict[str, float]]:
    valid = valid & np.isfinite(lst) & np.isfinite(albedo) & np.isfinite(ndvi)
    idx = np.flatnonzero(valid.ravel())
    if idx.size < 50:
        med = float(np.nanmedian(lst)) if np.isfinite(lst).any() else 0.0
        return (lst - med).astype(np.float32), {"intercept": med}
    rng = np.random.default_rng(seed)
    if idx.size > sample:
        idx = rng.choice(idx, sample, replace=False)
    a = albedo.ravel()[idx].astype(np.float64)
    n = ndvi.ravel()[idx].astype(np.float64)
    y = lst.ravel()[idx].astype(np.float64)
    x = np.stack([np.ones_like(a), a, n, n * n], axis=1)
    w = np.ones_like(y)
    beta = np.zeros(4)
    for _ in range(6):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
        r = y - x @ beta
        s = max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r))))
        c = 1.345 * s
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
    fitted = beta[0] + beta[1] * albedo + beta[2] * ndvi + beta[3] * ndvi * ndvi
    residual = np.where(valid, lst - fitted, np.nan).astype(np.float32)
    return residual, {"intercept": float(beta[0]), "albedo": float(beta[1]), "ndvi": float(beta[2]), "ndvi2": float(beta[3])}


def compute_anomaly(lst: np.ndarray, valid: np.ndarray, albedo: np.ndarray, ndvi: np.ndarray, hot: HotspotConfig, thermal: ThermalConfig, water: np.ndarray | None = None) -> AnomalyResult:
    valid = valid & np.isfinite(lst) & np.isfinite(albedo) & np.isfinite(ndvi)
    ring = water_ring(water, hot.water_buffer_px)
    if ring is not None:
        valid = valid & ~ring
    if thermal.detrend:
        albedo = masked_gaussian(albedo, valid, thermal.psf_sigma_px)
        ndvi = masked_gaussian(ndvi, valid, thermal.psf_sigma_px)
        residual, trend = detrend_temperature(lst, valid, albedo, ndvi, thermal.detrend_sample)
    else:
        med = float(np.nanmedian(lst[valid])) if valid.any() else 0.0
        residual, trend = np.where(valid, lst - med, np.nan).astype(np.float32), {"intercept": med}
    window = hot.background_window_px
    background = masked_box_mean(residual, valid, window)
    for _ in range(2):
        sigma = robust_sigma(residual - background)
        hot_px = (residual - background) > 2.0 * max(sigma, 0.1)
        background = masked_box_mean(residual, valid & ~hot_px, window)
    background = np.where(np.isfinite(background), background, 0.0).astype(np.float32)
    anomaly = np.where(valid, residual - background, np.nan).astype(np.float32)
    sigma = max(robust_sigma(anomaly), 0.05)
    z = (anomaly / sigma).astype(np.float32)
    return AnomalyResult(residual=residual, background=background, anomaly=anomaly, z=z, sigma_k=sigma, trend=trend)

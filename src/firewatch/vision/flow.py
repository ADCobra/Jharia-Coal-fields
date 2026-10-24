from __future__ import annotations

import cv2
import numpy as np

from firewatch.vision.cvcompat import dis_preset_medium


def _to_u8(anomaly: np.ndarray, clip_k: float) -> np.ndarray:
    a = np.nan_to_num(anomaly, nan=0.0)
    return (np.clip(a, 0.0, clip_k) / clip_k * 255.0).astype(np.uint8)


def front_motion(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray, *, days: float, res_m: float, clip_k: float = 15.0) -> tuple[float, float] | None:
    if days <= 0 or not mask.any():
        return None
    a = _to_u8(prev, clip_k)
    b = _to_u8(cur, clip_k)
    if a.max() == 0 or b.max() == 0:
        return None
    dis = cv2.DISOpticalFlow_create(dis_preset_medium())
    flow = dis.calc(a, b, None)
    region = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    gx = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    weight = np.hypot(gx, gy) * region
    total = float(weight.sum())
    if total <= 0:
        return None
    dx = float((flow[..., 0] * weight).sum() / total)
    dy = float((flow[..., 1] * weight).sum() / total)
    return dx * res_m / days, -dy * res_m / days

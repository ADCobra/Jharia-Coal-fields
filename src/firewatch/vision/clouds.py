from __future__ import annotations

import cv2
import numpy as np

QA_FILL = 1 << 0
QA_DILATED_CLOUD = 1 << 1
QA_CIRRUS = 1 << 2
QA_CLOUD = 1 << 3
QA_SHADOW = 1 << 4
QA_SNOW = 1 << 5
QA_CLEAR = 1 << 6
QA_WATER = 1 << 7
QA_CLOUD_CONF_HIGH = 3 << 8
QA_SHADOW_CONF_HIGH = 3 << 10
QA_CIRRUS_CONF_HIGH = 3 << 14

SCL_BAD = (0, 1, 3, 8, 9, 10, 11)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def landsat_masks(qa: np.ndarray, dilate_px: int = 2) -> tuple[np.ndarray, np.ndarray]:
    qa = qa.astype(np.uint16)
    fill = (qa & QA_FILL) > 0
    cloudy = ((qa & (QA_DILATED_CLOUD | QA_CIRRUS | QA_CLOUD | QA_SHADOW)) > 0)
    cloudy |= (qa & QA_CLOUD_CONF_HIGH) == QA_CLOUD_CONF_HIGH
    cloudy |= (qa & QA_CIRRUS_CONF_HIGH) == QA_CIRRUS_CONF_HIGH
    cloudy &= ~fill
    cloudy = _dilate(cloudy, dilate_px) & ~fill
    valid = ~fill & ~cloudy
    return valid, cloudy


def landsat_clear_mask(qa: np.ndarray, dilate_px: int = 2) -> np.ndarray:
    return landsat_masks(qa, dilate_px)[0]


def scl_clear_mask(scl: np.ndarray, dilate_px: int = 1) -> np.ndarray:
    bad = np.isin(scl, SCL_BAD)
    return ~_dilate(bad, dilate_px)


def cloud_fraction(valid: np.ndarray, cloudy: np.ndarray) -> float:
    footprint = valid | cloudy
    total = int(footprint.sum())
    return float(cloudy.sum()) / total if total else 1.0

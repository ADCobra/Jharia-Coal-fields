from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from firewatch.config import AlignmentConfig
from firewatch.models import AlignmentResult
from firewatch.vision.cvcompat import capabilities, dnn_engine, model_path

log = logging.getLogger(__name__)
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
LEARNED = ("aliked_lightglue", "disk")
ESTIMATORS = {"ransac": cv2.RANSAC, "magsac": cv2.USAC_MAGSAC, "accurate": cv2.USAC_ACCURATE}


def to_uint8(band: np.ndarray, valid: np.ndarray | None = None, clahe: bool = True) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    ok = np.isfinite(band) if valid is None else (valid & np.isfinite(band))
    if not ok.any():
        return np.zeros(band.shape, np.uint8)
    lo, hi = np.percentile(band[ok], [2, 98])
    scaled = (band - lo) / max(hi - lo, 1e-6)
    fill = float(np.median(scaled[ok]))
    scaled = np.where(ok, scaled, fill)
    img = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    if clahe:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return img


def warp_to_reference(array: np.ndarray, matrix, shape: tuple[int, int], nearest: bool = False, fill: float = np.nan) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if np.allclose(m, IDENTITY):
        return array
    h, w = shape
    if array.dtype == bool:
        out = cv2.warpAffine(array.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out.astype(bool)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    src = array
    if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any() and not nearest:
        finite = np.isfinite(array)
        filled = np.where(finite, array, 0).astype(np.float32)
        warped = cv2.warpAffine(filled, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        weight = cv2.warpAffine(finite.astype(np.float32), m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.where(weight > 0.999, warped, fill).astype(array.dtype)
    border_value = float(fill) if np.issubdtype(array.dtype, np.floating) else 0
    return cv2.warpAffine(src, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def phase_shift(ref: np.ndarray, mov: np.ndarray) -> float:
    a = np.nan_to_num(ref.astype(np.float32))
    b = np.nan_to_num(mov.astype(np.float32))
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, window)
    return float(math.hypot(dx, dy))


def _keypoint_array(kps) -> np.ndarray:
    return np.float32([k.pt for k in kps]).reshape(-1, 2)


def _filter_by_mask(kps, desc, mask: np.ndarray | None):
    if mask is None or not len(kps):
        return kps, desc
    h, w = mask.shape
    keep = []
    for i, k in enumerate(kps):
        x, y = int(round(k.pt[0])), int(round(k.pt[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            keep.append(i)
    return [kps[i] for i in keep], (desc[keep] if desc is not None and len(keep) else None)


class FeatureRegistrar:
    def __init__(self, cfg: AlignmentConfig, models_dir: str | Path | None = None):
        self.cfg = cfg
        self.models_dir = Path(models_dir or cfg.models_dir)
        self._aliked = None
        self._lightglue = None
        self._disk = None
        self._sift = None
        self._orb = None

    def available_methods(self) -> list[str]:
        caps = capabilities()
        usable = {"sift", "orb"}
        if caps.aliked and caps.lightglue and model_path(self.models_dir, self.cfg.aliked_model) and model_path(self.models_dir, self.cfg.lightglue_model):
            usable.add("aliked_lightglue")
        if caps.disk and model_path(self.models_dir, self.cfg.disk_model):
            usable.add("disk")
        ordered = [m for m in self.cfg.preference if m in usable]
        return ordered + sorted(usable - set(ordered))

    def resolve(self, method: str | None = None) -> str:
        method = method or self.cfg.method
        available = self.available_methods()
        if method == "auto":
            return available[0]
        if method in ("none",) or method in available:
            return method
        log.warning("alignment method %s unavailable, falling back to %s", method, available[0])
        return available[0]

    def pick(self, preference: list[str], exclude: set[str | None]) -> str | None:
        available = self.available_methods()
        for method in list(preference) + available:
            if method in available and method not in exclude:
                return method
        return None

    def _get_aliked(self):
        if self._aliked is None:
            params = cv2.ALIKED_Params()
            params.engine = dnn_engine("new")
            self._aliked = cv2.ALIKED_create(str(model_path(self.models_dir, self.cfg.aliked_model)), params)
        return self._aliked

    def _get_lightglue(self):
        if self._lightglue is None:
            self._lightglue = cv2.LightGlueMatcher_create(str(model_path(self.models_dir, self.cfg.lightglue_model)), 0.0)
        return self._lightglue

    def _get_disk(self):
        if self._disk is None:
            self._disk = cv2.DISK_create(str(model_path(self.models_dir, self.cfg.disk_model)), 2048, 0.0)
        return self._disk

    def detect(self, method: str, img: np.ndarray, mask: np.ndarray | None):
        if method == "aliked_lightglue":
            kps, desc = self._get_aliked().detectAndCompute(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None)
        elif method == "disk":
            kps, desc = self._get_disk().detectAndCompute(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None)
        elif method == "sift":
            if self._sift is None:
                self._sift = cv2.SIFT_create(self.cfg.max_features)
            kps, desc = self._sift.detectAndCompute(img, None)
        else:
            if self._orb is None:
                self._orb = cv2.ORB_create(self.cfg.max_features)
            kps, desc = self._orb.detectAndCompute(img, None)
        return _filter_by_mask(list(kps), desc, mask)

    def match(self, method: str, kq, dq, kt, dt, size_q: tuple[int, int], size_t: tuple[int, int]) -> list:
        if dq is None or dt is None or len(kq) < 3 or len(kt) < 3:
            return []
        if method == "aliked_lightglue":
            matcher = self._get_lightglue()
            matcher.setPairInfo(_keypoint_array(kq), _keypoint_array(kt), size_q, size_t)
            return list(matcher.match(dq, dt))
        norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
        knn = cv2.BFMatcher(norm).knnMatch(dq, dt, k=2)
        ratio = self.cfg.ratio_test
        return [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]

    def correspondences(self, method: str, ref_img: np.ndarray, mov_img: np.ndarray, ref_mask: np.ndarray | None = None, mov_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
        k_ref, d_ref = self.detect(method, ref_img, ref_mask)
        k_mov, d_mov = self.detect(method, mov_img, mov_mask)
        matches = self.match(method, k_mov, d_mov, k_ref, d_ref, (mov_img.shape[1], mov_img.shape[0]), (ref_img.shape[1], ref_img.shape[0]))
        src = np.float32([k_mov[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst = np.float32([k_ref[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return src, dst, len(k_ref), len(k_mov)

    def fit(self, src: np.ndarray, dst: np.ndarray, estimator: str | None = None):
        return cv2.estimateAffine2D(src, dst, method=ESTIMATORS[estimator or self.cfg.estimator], ransacReprojThreshold=self.cfg.ransac_threshold_px, maxIters=10000, confidence=0.999)

    def register_checked(self, ref_img: np.ndarray, mov_img: np.ndarray, ref_mask: np.ndarray | None = None, mov_mask: np.ndarray | None = None) -> AlignmentResult:
        primary = self.register(ref_img, mov_img, ref_mask, mov_mask)
        if not self.cfg.cross_check or primary.method == "none":
            return primary
        check_method = self.pick(self.cfg.check_preference, {primary.method})
        if check_method is None:
            return primary
        check = self.register(ref_img, mov_img, ref_mask, mov_mask, method=check_method)
        if not primary.ok and check.ok:
            check.note = f"{primary.method} failed ({primary.note}); {check_method} used instead"
            check.check_method, check.check_ok, check.check_matrix = primary.method, False, None
            check.seconds += primary.seconds
            return check
        primary.check_method, primary.check_ok = check_method, check.ok
        primary.check_matrix = check.matrix if check.ok else None
        primary.seconds += check.seconds
        if primary.ok and check.ok:
            primary.agreement_px = transform_error_px(primary.matrix, check.matrix, ref_img.shape)
        return primary

    def register(self, ref_img: np.ndarray, mov_img: np.ndarray, ref_mask: np.ndarray | None = None, mov_mask: np.ndarray | None = None, method: str | None = None, estimator: str | None = None) -> AlignmentResult:
        method = self.resolve(method)
        estimator = estimator or self.cfg.estimator
        start = time.perf_counter()
        before = phase_shift(ref_img, mov_img)
        if method == "none":
            return AlignmentResult(method=method, ok=True, matrix=IDENTITY, shift_before_px=before, shift_after_px=before, note="alignment disabled")
        try:
            src, dst, n_ref, n_mov = self.correspondences(method, ref_img, mov_img, ref_mask, mov_mask)
        except cv2.error as exc:
            log.warning("feature matching failed with %s: %s", method, exc)
            return AlignmentResult(method=method, ok=False, matrix=IDENTITY, shift_before_px=before, seconds=time.perf_counter() - start, note=f"opencv error: {exc}"[:300])
        result = AlignmentResult(method=method, ok=False, matrix=IDENTITY, n_ref=n_ref, n_mov=n_mov, n_matches=len(src), shift_before_px=before, estimator=estimator)
        if len(src) >= 6:
            matrix, inliers = self.fit(src, dst, estimator)
            if matrix is not None and inliers is not None:
                inl = inliers.ravel().astype(bool)
                proj = src[inl] @ matrix[:, :2].T + matrix[:, 2]
                residual = float(np.sqrt(np.mean(np.sum((proj - dst[inl]) ** 2, axis=1)))) if inl.any() else None
                scale = math.sqrt(abs(np.linalg.det(matrix[:, :2])))
                plausible = 0.98 < scale < 1.02 and float(np.hypot(*matrix[:, 2])) < 0.05 * max(ref_img.shape) + 30
                result.matrix = matrix.tolist()
                result.n_inliers = int(inl.sum())
                result.residual_px = residual
                result.ok = bool(plausible and result.n_inliers >= self.cfg.min_inliers)
                if not plausible:
                    result.note = f"implausible transform (scale {scale:.4f})"
                elif not result.ok:
                    result.note = f"only {result.n_inliers} inliers"
        else:
            result.note = "too few matches"
        if result.ok:
            warped = warp_to_reference(mov_img.astype(np.float32), result.matrix, ref_img.shape, fill=0.0)
            result.shift_after_px = phase_shift(ref_img, warped)
        else:
            result.matrix = IDENTITY
            result.shift_after_px = before
        result.seconds = time.perf_counter() - start
        return result


def transform_error_px(estimated, truth_inverse, shape: tuple[int, int], step: int = 32) -> float:
    h, w = shape
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    a = np.asarray(estimated, dtype=np.float64)
    b = np.asarray(truth_inverse, dtype=np.float64)
    pa = pts @ a[:, :2].T + a[:, 2]
    pb = pts @ b[:, :2].T + b[:, 2]
    return float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))


def to_window(matrix, left: int, top: int) -> list[list[float]]:
    m = np.asarray(matrix, dtype=np.float64)
    offset = np.array([left, top], dtype=np.float64)
    out = m.copy()
    out[:, 2] = m[:, :2] @ offset + m[:, 2] - offset
    return out.tolist()


def invert_affine(matrix) -> list[list[float]]:
    return cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float64)).tolist()

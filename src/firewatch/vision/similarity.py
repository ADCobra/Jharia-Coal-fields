from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from firewatch.models import Kinematics

FEATURES = ("log_area", "mean_anomaly", "max_anomaly", "growth_30d", "speed", "history", "road_distance")


def signature(area_m2: float, mean_anomaly_k: float, max_anomaly_k: float, kin: Kinematics | None, distance_m: float | None = None) -> np.ndarray:
    growth = kin.growth_frac_per_30d if kin and kin.n_points >= 2 else 0.0
    speed = kin.speed_m_per_day if kin and kin.n_points >= 2 else 0.0
    points = kin.n_points if kin else 1
    return np.array(
        [
            math.log1p(area_m2 / 1e4),
            mean_anomaly_k / 10.0,
            max_anomaly_k / 20.0,
            max(-2.0, min(2.0, growth)),
            min(3.0, speed / 5.0),
            min(1.0, points / 8.0),
            2.0 if distance_m is None else min(2.0, distance_m / 1000.0),
        ],
        dtype=np.float32,
    )


@dataclass
class Analogue:
    meta: dict
    distance: float


@dataclass
class AnalogueIndex:
    dim: int = len(FEATURES)
    vectors: list[np.ndarray] = field(default_factory=list)
    metas: list[dict] = field(default_factory=list)
    backend: str = "empty"
    _index: object = None

    def add(self, vector: np.ndarray, meta: dict) -> None:
        self.vectors.append(np.asarray(vector, np.float32))
        self.metas.append(meta)

    def build(self, trees: int = 10) -> "AnalogueIndex":
        if not self.vectors:
            self.backend = "empty"
            return self
        data = np.stack(self.vectors).astype(np.float32)
        if hasattr(cv2, "ANNIndex_create") and len(self.vectors) >= 2:
            index = cv2.ANNIndex_create(self.dim, cv2.ANNINDEX_DIST_EUCLIDEAN)
            index.setSeed(42)
            index.addItems(data)
            index.build(trees)
            self._index = index
            self.backend = "opencv-annoy"
        else:
            self._index = data
            self.backend = "numpy"
        return self

    def query(self, vector: np.ndarray, k: int = 5, exclude_track: str | None = None) -> list[Analogue]:
        if exclude_track is not None:
            wanted = k
            k = min(len(self.vectors), k + sum(1 for m in self.metas if m.get("track_id") == exclude_track))
            found = [a for a in self.query(vector, k) if a.meta.get("track_id") != exclude_track]
            return found[:wanted]
        if self._index is None or not self.vectors:
            return []
        k = min(k, len(self.vectors))
        q = np.asarray(vector, np.float32).reshape(1, -1)
        if self.backend == "opencv-annoy":
            idx, dist = self._index.knnSearch(q, k)
            pairs = zip(idx.ravel().tolist(), dist.ravel().tolist())
        else:
            d = np.linalg.norm(self._index - q, axis=1)
            order = np.argsort(d)[:k]
            pairs = zip(order.tolist(), d[order].tolist())
        return [Analogue(self.metas[i], float(dv)) for i, dv in pairs if 0 <= i < len(self.metas)]


def summarise(analogues: list[Analogue]) -> dict:
    if not analogues:
        return {"n": 0}
    grew = [a.meta.get("grew") for a in analogues if a.meta.get("grew") is not None]
    reached = [a.meta.get("reached_road") for a in analogues if a.meta.get("reached_road") is not None]
    return {
        "n": len(analogues),
        "tracks": len({a.meta.get("track_id") for a in analogues}),
        "grew_share": round(sum(grew) / len(grew), 2) if grew else None,
        "reached_road_share": round(sum(reached) / len(reached), 2) if reached else None,
        "nearest": [{"track_id": a.meta.get("track_id"), "date": a.meta.get("date"), "distance": round(a.distance, 3)} for a in analogues[:3]],
    }

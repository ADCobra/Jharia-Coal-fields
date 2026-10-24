from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from firewatch.geo.grid import GridSpec
from firewatch.models import RawScene, SceneRef


@dataclass
class HighResPatch:
    scene_id: str
    acquired: dt.datetime
    grid: GridSpec
    nir: np.ndarray
    swir16: np.ndarray
    swir22: np.ndarray
    valid: np.ndarray
    source: str


class SceneSource(Protocol):
    name: str

    def list_scenes(self, start: dt.datetime, end: dt.datetime) -> list[SceneRef]: ...

    def get_ref(self, scene_id: str) -> SceneRef: ...

    def load(self, ref: SceneRef, grid: GridSpec) -> RawScene: ...

    def history(self, before: dt.datetime, n: int) -> list[SceneRef]: ...

    def highres(self, bounds: tuple[float, float, float, float], crs: str, when: dt.datetime) -> HighResPatch | None: ...

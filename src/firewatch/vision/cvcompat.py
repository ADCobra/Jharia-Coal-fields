from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2


@dataclass(frozen=True)
class Capabilities:
    opencv_version: str
    major: int
    aliked: bool
    disk: bool
    lightglue: bool
    ann_index: bool
    dnn_new_engine: bool
    tokenizer: bool
    usac_magsac: bool
    dis_flow: bool
    threads: int
    cpu_dispatch: str

    def as_dict(self) -> dict:
        return asdict(self)


def _cpu_dispatch() -> str:
    for line in cv2.getBuildInformation().splitlines():
        if "Dispatched code generation" in line:
            return line.split(":", 1)[1].strip()
    return ""


@lru_cache(maxsize=1)
def capabilities() -> Capabilities:
    version = cv2.__version__
    return Capabilities(
        opencv_version=version,
        major=int(version.split(".")[0]),
        aliked=hasattr(cv2, "ALIKED_create"),
        disk=hasattr(cv2, "DISK_create"),
        lightglue=hasattr(cv2, "LightGlueMatcher_create"),
        ann_index=hasattr(cv2, "ANNIndex_create"),
        dnn_new_engine=hasattr(cv2.dnn, "ENGINE_NEW"),
        tokenizer=hasattr(cv2.dnn, "Tokenizer") or hasattr(cv2, "dnn_Tokenizer"),
        usac_magsac=hasattr(cv2, "USAC_MAGSAC"),
        dis_flow=hasattr(cv2, "DISOpticalFlow_create"),
        threads=cv2.getNumThreads(),
        cpu_dispatch=_cpu_dispatch(),
    )


def dnn_engine(preferred: str = "new") -> int:
    names = {"new": ("ENGINE_NEW", "ENGINE_OPENCV"), "auto": ("ENGINE_AUTO",), "classic": ("ENGINE_CLASSIC",)}
    for name in names.get(preferred, ()) + ("ENGINE_AUTO",):
        if hasattr(cv2.dnn, name):
            return int(getattr(cv2.dnn, name))
    return 0


def model_path(models_dir: str | Path, filename: str) -> Path | None:
    path = Path(models_dir) / filename
    return path if path.is_file() and path.stat().st_size > 1024 else None


def quiet_opencv() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass


def dis_preset_medium() -> int:
    for name in ("DISOPTICAL_FLOW_PRESET_MEDIUM", "DISOpticalFlow_PRESET_MEDIUM"):
        if hasattr(cv2, name):
            return int(getattr(cv2, name))
    return 2

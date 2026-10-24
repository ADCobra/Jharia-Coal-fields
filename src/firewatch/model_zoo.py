from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    filename: str
    url: str
    sha1: str
    purpose: str


MODELS = (
    ModelFile(
        "aliked-n16rot-top1k-640.onnx",
        "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/aliked-n16rot-top1k-640.onnx",
        "41faa7bf5d7eb68a2851471ba03aa20c9db30e4c",
        "ALIKED keypoints and descriptors (cv2.ALIKED)",
    ),
    ModelFile(
        "aliked_lightglue.onnx",
        "https://raw.githubusercontent.com/YangGuanyuhan/lightglue_opencv_project/main/model/aliked_lightglue.onnx",
        "02723aa521990e57fe33d90b67977590c460e351",
        "LightGlue matcher for ALIKED (cv2.LightGlueMatcher)",
    ),
    ModelFile(
        "disk.onnx",
        "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0/disk.onnx",
        "5f6a9069aed0af7302b67dcfb6d24b0d46707aec",
        "DISK keypoints and descriptors (cv2.DISK), matched with a ratio test",
    ),
)


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models(dest: str | Path, force: bool = False) -> list[dict]:
    import httpx

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    for model in MODELS:
        path = dest / model.filename
        if path.exists() and not force and sha1_of(path) == model.sha1:
            report.append({"file": model.filename, "status": "present"})
            continue
        with httpx.stream("GET", model.url, follow_redirects=True, timeout=300) as response:
            response.raise_for_status()
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
        digest = sha1_of(tmp)
        if digest != model.sha1:
            tmp.unlink(missing_ok=True)
            report.append({"file": model.filename, "status": "checksum mismatch", "sha1": digest})
            continue
        tmp.replace(path)
        report.append({"file": model.filename, "status": "downloaded"})
    return report

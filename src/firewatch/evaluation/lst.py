from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from firewatch.config import Settings
from firewatch.data.synthetic import SyntheticWorld
from firewatch.evaluation.common import md_table, write_report
from firewatch.geo.grid import GridSpec
from firewatch.physics.thermal import derive_thermal


def error_stats(ours: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> dict:
    ok = valid & np.isfinite(ours) & np.isfinite(reference)
    diff = (ours - reference)[ok]
    if diff.size == 0:
        return {"pixels": 0}
    return {
        "pixels": int(diff.size),
        "bias_k": float(diff.mean()),
        "mae_k": float(np.abs(diff).mean()),
        "rmse_k": float(np.sqrt((diff ** 2).mean())),
        "p95_abs_k": float(np.percentile(np.abs(diff), 95)),
    }


def synthetic(settings: Settings, scale: float | None = None, indices=range(-3, 4)) -> list[dict]:
    world = SyntheticWorld.from_settings(settings, scale=scale, misregister=False)
    rows = []
    for index in indices:
        raw = world.raw_scene(index)
        thermal = derive_thermal(raw, settings.thermal)
        truth = cv2.GaussianBlur(raw.truth["lst_true"], (0, 0), 1.3)
        rows.append({"scene_id": raw.ref.scene_id, "against": "known surface temperature", **error_stats(thermal.lst, truth, thermal.valid)})
        rows.append({"scene_id": raw.ref.scene_id, "against": "synthetic Level-2 product", **error_stats(thermal.lst, raw.l2_surface_temp, thermal.valid)})
    return rows


def real(settings: Settings, scene_ids: list[str]) -> list[dict]:
    from firewatch.data.landsat import LandsatSource

    source = LandsatSource(settings, with_l2=True)
    grid = GridSpec.from_bbox(settings.aoi.bbox, settings.aoi.crs, settings.aoi.resolution_m)
    rows = []
    for scene_id in scene_ids:
        raw = source.load(source.get_ref(scene_id), grid)
        thermal = derive_thermal(raw, settings.thermal)
        if raw.l2_surface_temp is None:
            rows.append({"scene_id": scene_id, "against": "USGS Level-2 surface temperature", "pixels": 0, "note": "no Level-2 match found"})
            continue
        rows.append({"scene_id": scene_id, "against": "USGS Level-2 surface temperature", **error_stats(thermal.lst, raw.l2_surface_temp, thermal.valid)})
    return rows


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> Path:
    data = {"synthetic": synthetic(settings, scale)}
    scenes = [s for s in os.environ.get("FIREWATCH_EVAL_SCENES", "").split(",") if s.strip()]
    if scenes:
        data["real"] = real(settings, scenes)
    head = ["Scene", "Compared with", "Pixels", "Bias (K)", "MAE (K)", "RMSE (K)", "95th pct abs. error (K)"]
    to_row = lambda r: [r["scene_id"], r["against"], r.get("pixels"), r.get("bias_k"), r.get("mae_k"), r.get("rmse_k"), r.get("p95_abs_k")]
    md = [
        "# Land surface temperature from Landsat band 10",
        "",
        "Single-channel retrieval: digital number to radiance (RADIANCE_MULT/ADD), radiance to brightness temperature (K1, K2), then emissivity correction from NDVI (Sobrino et al. 2004 threshold method). No atmospheric correction is applied, so a bias against the USGS Level-2 product is expected on real scenes.",
        "",
        "## Synthetic check",
        "",
        "The synthetic scenes are encoded with the same constants, so this mainly confirms the physics code inverts cleanly; the error left comes from noise and a lower emissivity assumed for coal.",
        "",
        md_table(head, [to_row(r) for r in data["synthetic"]]),
    ]
    if "real" in data:
        md += ["", "## Real scenes", "", md_table(head, [to_row(r) for r in data["real"]])]
    else:
        md += ["", "## Real scenes", "", "Not run here. Set FIREWATCH_EVAL_SCENES to a comma-separated list of Landsat Collection 2 Level-1 scene ids and run `firewatch -c config/landsat.yaml eval lst`."]
    return write_report(out_dir, "lst", data, "\n".join(md))

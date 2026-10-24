from __future__ import annotations

import importlib
import logging
from pathlib import Path

from firewatch.config import Settings

log = logging.getLogger(__name__)

SUITES = {
    "lst": ("firewatch.evaluation.lst", "Land surface temperature retrieval"),
    "detection": ("firewatch.evaluation.detection", "Hotspot detection against synthetic ground truth"),
    "alignment": ("firewatch.evaluation.alignment", "Image alignment: ORB, SIFT, DISK, ALIKED + LightGlue"),
    "forecast": ("firewatch.evaluation.forecast", "Spread projection and precedent against synthetic truth"),
    "agent": ("firewatch.evaluation.agent_eval", "Agent decisions on scripted scenarios"),
    "nrsc": ("firewatch.evaluation.nrsc", "Fire area against the NRSC mapping (real data)"),
    "firms": ("firewatch.evaluation.firms", "Hotspots against NASA FIRMS (real data)"),
    "bench": ("firewatch.evaluation.bench", "Run time per stage"),
}


def run_suite(name: str, settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> dict[str, Path]:
    names = list(SUITES) if name == "all" else [name]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    for item in names:
        module = importlib.import_module(SUITES[item][0])
        log.info("running %s", item)
        results[item] = module.report(settings, base_dir, out_dir, scale=scale)
    if name == "all":
        from firewatch.evaluation.common import environment

        env = environment()
        lines = ["# Evaluation reports", "", f"Generated {env['time'][:16].replace('T', ' ')} UTC with OpenCV {env['opencv']} on {env['machine']}.", ""]
        lines += [f"- [{SUITES[k][1]}]({p.name})" for k, p in results.items()]
        index = out_dir / "README.md"
        index.write_text("\n".join(lines) + "\n")
        results["index"] = index
    return results

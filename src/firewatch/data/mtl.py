from __future__ import annotations

import json
from typing import Any

OLI_BANDS = {"coastal": 1, "blue": 2, "green": 3, "red": 4, "nir08": 5, "swir16": 6, "swir22": 7, "lwir11": 10, "lwir12": 11}


def _flatten(node: Any, out: dict[str, str]) -> dict[str, str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                _flatten(value, out)
            else:
                out[str(key)] = str(value)
    elif isinstance(node, list):
        for item in node:
            _flatten(item, out)
    return out


def parse_mtl(content: str | bytes | dict) -> dict[str, str]:
    if isinstance(content, dict):
        return _flatten(content, {})
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    text = text.strip()
    if text.startswith("{"):
        return _flatten(json.loads(text), {})
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ("GROUP", "END_GROUP"):
            continue
        out[key] = value.strip().strip('"')
    return out


def _number(mtl: dict[str, str], key: str) -> float:
    try:
        return float(mtl[key])
    except KeyError as exc:
        raise KeyError(f"MTL is missing {key}") from exc


def thermal_constants(mtl: dict[str, str], band: int = 10) -> dict[str, float]:
    return {
        "mult": _number(mtl, f"RADIANCE_MULT_BAND_{band}"),
        "add": _number(mtl, f"RADIANCE_ADD_BAND_{band}"),
        "k1": _number(mtl, f"K1_CONSTANT_BAND_{band}"),
        "k2": _number(mtl, f"K2_CONSTANT_BAND_{band}"),
    }


def reflectance_rescaling(mtl: dict[str, str], band: int) -> tuple[float, float]:
    return _number(mtl, f"REFLECTANCE_MULT_BAND_{band}"), _number(mtl, f"REFLECTANCE_ADD_BAND_{band}")


def sun_elevation(mtl: dict[str, str]) -> float | None:
    value = mtl.get("SUN_ELEVATION")
    return float(value) if value not in (None, "") else None

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import secrets
from pathlib import Path
from typing import Any

import numpy as np


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def new_id(prefix: str = "", now: dt.datetime | None = None) -> str:
    now = now or utcnow()
    return f"{prefix}{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def parse_time(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        out = value
    elif isinstance(value, dt.date):
        out = dt.datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        out = dt.datetime.fromisoformat(text)
    else:
        raise TypeError(f"cannot parse time from {value!r}")
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out


def days_between(a: Any, b: Any) -> float:
    return (parse_time(b) - parse_time(a)).total_seconds() / 86400.0


def jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(jsonable(obj), **kwargs)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, value)))


def bearing_deg(dx: float, dy: float) -> float:
    return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)

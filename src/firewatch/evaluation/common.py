from __future__ import annotations

import json
import platform
from pathlib import Path

import cv2
from matplotlib.figure import Figure

from firewatch.util import jsonable, utcnow

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def environment() -> dict:
    return {"time": utcnow().isoformat(), "python": platform.python_version(), "machine": platform.machine(), "opencv": cv2.__version__, "threads": cv2.getNumThreads()}


def write_report(out_dir: Path, name: str, data: dict, markdown: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(jsonable({**data, "environment": environment()}), indent=2))
    path = out_dir / f"{name}.md"
    path.write_text(markdown.rstrip() + "\n")
    return path


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def styled_axes(width: float = 7.0, height: float = 3.6, rows: int = 1):
    fig = Figure(figsize=(width, height), dpi=150)
    axes = fig.subplots(rows, 1, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in (axes if rows > 1 else [axes]):
        style_axis(ax)
    return fig, axes


def save(fig, path: Path, legend_space: float = 0.0) -> Path:
    fig.tight_layout(rect=(0, legend_space, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    return path

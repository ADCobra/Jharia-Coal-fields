from __future__ import annotations

import os
from pathlib import Path

import pytest

from firewatch.config import load_settings

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("FIREWATCH_MODELS", ROOT / "models"))


@pytest.fixture(scope="session")
def settings():
    return load_settings(environ={})


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS


def workspace_settings(base: Path, scale: float = 0.3):
    settings = load_settings(environ={})
    return settings.with_overrides({
        "data": {"source": "synthetic", "synthetic": {"scale": scale}},
        "storage": {"root": str(base / "blobs")},
        "state": {"sqlite_path": str(base / "state.db")},
        "notify": {"outbox": str(base / "outbox.jsonl")},
        "alignment": {"models_dir": str(MODELS)},
    })


@pytest.fixture(scope="session")
def demo_ctx(tmp_path_factory):
    from firewatch.pipeline.context import build_context
    from firewatch.pipeline.runner import run_demo

    base = tmp_path_factory.mktemp("demo")
    ctx = build_context(workspace_settings(base), base)
    summaries = run_demo(ctx, passes=4, archive=2)
    ctx.summaries = summaries
    return ctx

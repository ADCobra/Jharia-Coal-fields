from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from firewatch.config import Settings
from firewatch.data.synthetic import SyntheticSource
from firewatch.models import SceneRef
from firewatch.pipeline.context import Context
from firewatch.pipeline.process import PassResult, run_pass

log = logging.getLogger(__name__)


def backfill(ctx: Context, refs: list[SceneRef], agent: bool = False, on_pass: Callable[[PassResult], None] | None = None) -> list[dict]:
    summaries = []
    for ref in sorted(refs, key=lambda r: r.acquired):
        result = run_pass(ctx, ref, agent=agent)
        summaries.append(result.summary)
        if on_pass:
            on_pass(result)
    return summaries


def run_next_synthetic(ctx: Context, agent: bool = True) -> PassResult:
    if not isinstance(ctx.source, SyntheticSource):
        raise RuntimeError("run_next_synthetic needs the synthetic source")
    repo = ctx.services.repo
    index = int(repo.get_meta("synthetic_next_index", 0))
    if index > ctx.source.last_index:
        raise RuntimeError(f"the synthetic archive ends at pass {ctx.source.last_index}")
    result = run_pass(ctx, ctx.source.ref_for_index(index), agent=agent)
    repo.put_meta("synthetic_next_index", index + 1)
    return result


def run_next_scene(ctx: Context, agent: bool = True) -> PassResult:
    if isinstance(ctx.source, SyntheticSource):
        return run_next_synthetic(ctx, agent=agent)
    done = {r.get("scene_id") for r in ctx.services.repo.list_runs(5000)}
    for ref in ctx.source.refs():
        if ref.scene_id not in done:
            return run_pass(ctx, ref, agent=agent)
    raise RuntimeError("every downloaded scene has been processed; run scripts/fetch_real_data.py to get newer ones")


def run_real(ctx: Context, archive: int = 3, passes: int | None = None, on_pass: Callable[[PassResult], None] | None = None) -> list[dict]:
    refs = ctx.source.refs()
    done = {r.get("scene_id") for r in ctx.services.repo.list_runs(5000)}
    summaries, count = [], 0
    for index, ref in enumerate(refs):
        if ref.scene_id in done:
            continue
        if passes is not None and count >= passes:
            break
        result = run_pass(ctx, ref, agent=index >= archive)
        summaries.append(result.summary)
        count += 1
        if on_pass:
            on_pass(result)
    return summaries


def run_demo(ctx: Context, passes: int = 9, archive: int = 6, on_pass: Callable[[PassResult], None] | None = None) -> list[dict]:
    if not isinstance(ctx.source, SyntheticSource):
        raise RuntimeError("the demo runs on the synthetic source")
    source = ctx.source
    repo = ctx.services.repo
    summaries = []
    start = max(source.first_index, -archive)
    if repo.get_meta("synthetic_next_index") is None:
        history = [source.ref_for_index(i) for i in range(start, 0)]
        summaries += backfill(ctx, history, agent=False, on_pass=on_pass)
        repo.put_meta("synthetic_next_index", 0)
    for _ in range(passes):
        index = int(repo.get_meta("synthetic_next_index", 0))
        if index > source.last_index:
            break
        result = run_next_synthetic(ctx)
        summaries.append(result.summary)
        if on_pass:
            on_pass(result)
    return summaries


def reset_local_state(settings: Settings, base_dir: str | Path) -> list[str]:
    base = Path(base_dir)
    removed = []
    if settings.storage.backend == "local":
        root = Path(settings.storage.root)
        root = root if root.is_absolute() else base / root
        if root.exists():
            shutil.rmtree(root)
            removed.append(str(root))
    if settings.state.backend == "sqlite":
        db = Path(settings.state.sqlite_path)
        db = db if db.is_absolute() else base / db
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db) + suffix)
            if p.exists():
                p.unlink()
                removed.append(str(p))
    if settings.notify.backend == "console":
        outbox = Path(settings.notify.outbox)
        outbox = outbox if outbox.is_absolute() else base / outbox
        if outbox.exists():
            outbox.unlink()
            removed.append(str(outbox))
    return removed

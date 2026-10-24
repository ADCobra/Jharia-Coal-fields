from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from firewatch.config import Settings, load_settings
from firewatch.util import dumps, parse_time

log = logging.getLogger("firewatch")


def _settings(args) -> Settings:
    return load_settings(*args.config)


def _context(args, settings: Settings | None = None):
    from firewatch.pipeline.context import build_context

    return build_context(settings or _settings(args), Path(args.base_dir))


def _pass_line(result) -> str:
    s = result.summary
    al = s["alignment"]
    residual = f"{al['residual_px']:.2f} px" if al.get("residual_px") is not None else "n/a"
    parts = [
        f"{s['scene_id']}",
        f"hotspots {s['hotspots']}",
        f"levels {s['by_level']}",
        f"decisions {s['decisions']}",
        f"align {al['method']} ({'ok' if al['ok'] else 'failed'}, residual {residual})",
        f"{s['seconds']} s",
    ]
    lines = ["  ".join(parts)]
    for action in s["actions"]:
        if action["kind"] in ("close_road", "propose_reopen", "escalate", "blocked"):
            lines.append(f"    {action['actor']} {action['kind']} {', '.join(action['target']) or '-'} (track {action.get('track_id')})")
    return "\n".join(lines)


def cmd_demo(args) -> int:
    from firewatch.pipeline.runner import reset_local_state, run_demo

    settings = _settings(args)
    if args.scale:
        settings = settings.with_overrides({"data": {"synthetic": {"scale": args.scale}}})
    if settings.data.source != "synthetic":
        settings = settings.with_overrides({"data": {"source": "synthetic"}})
    if args.fresh:
        for path in reset_local_state(settings, args.base_dir):
            print(f"removed {path}")
    ctx = _context(args, settings)
    print(f"grid {ctx.grid.width}x{ctx.grid.height} at {ctx.grid.res:.0f} m, {len(ctx.roads)} road segments, alignment methods {ctx.registrar.available_methods()}")
    run_demo(ctx, passes=args.passes, archive=args.archive, on_pass=lambda r: print(_pass_line(r), flush=True))
    pending = ctx.services.repo.list_approvals("pending")
    print(f"\n{len(pending)} approval(s) waiting for a person:")
    for a in pending:
        print(f"  {a['approval_id']}  {a['kind']:<15} track {a.get('track_id')}  {', '.join(a.get('segments') or []) or '-'}  {a.get('reason', '')[:90]}")
    print("\nOpen the dashboard with: firewatch serve   (then http://localhost:8000, user/password from the api section of the config)")
    return 0


def _real_settings(args) -> Settings:
    import yaml

    settings = _settings(args)
    if settings.data.source != "real":
        overrides = Path(args.base_dir) / "config" / "real.yaml"
        if not overrides.exists():
            raise SystemExit("config/real.yaml is missing; pass -c with a config that sets data.source: real")
        settings = settings.with_overrides(yaml.safe_load(overrides.read_text()))
    return settings


def cmd_real(args) -> int:
    from firewatch.pipeline.runner import reset_local_state, run_real

    settings = _real_settings(args)
    if args.fresh:
        for path in reset_local_state(settings, args.base_dir):
            print(f"removed {path}")
    ctx = _context(args, settings)
    refs = ctx.source.refs()
    if not refs:
        print("No downloaded scenes found. Run: python scripts/fetch_real_data.py", file=sys.stderr)
        return 1
    print(f"{len(refs)} real Landsat scene(s) from {refs[0].acquired:%d %b %Y} to {refs[-1].acquired:%d %b %Y}; {len(ctx.roads)} road sections, {len(ctx.routes)} truck route(s)")
    run_real(ctx, archive=args.archive, passes=args.passes, on_pass=lambda r: print(_pass_line(r), flush=True))
    pending = ctx.services.repo.list_approvals("pending")
    print(f"\n{len(pending)} decision(s) waiting for a person.")
    print("Open the dashboard with: firewatch -c config/real.yaml serve   (then http://localhost:8000, user demo, password demo)")
    return 0


def cmd_run(args) -> int:
    from firewatch.pipeline.process import run_pass
    from firewatch.pipeline.runner import run_next_synthetic

    ctx = _context(args)
    result = run_next_synthetic(ctx, agent=not args.no_agent) if args.next else run_pass(ctx, args.scene_id, agent=not args.no_agent)
    print(_pass_line(result))
    return 0


def cmd_backfill(args) -> int:
    from firewatch.pipeline.runner import backfill

    ctx = _context(args)
    refs = ctx.source.list_scenes(parse_time(args.start), parse_time(args.end))
    print(f"{len(refs)} scene(s) to process")
    backfill(ctx, refs, agent=args.agent, on_pass=lambda r: print(_pass_line(r), flush=True))
    return 0


def cmd_search(args) -> int:
    settings = _settings(args)
    if settings.data.source == "synthetic":
        ctx = _context(args, settings)
        refs = ctx.source.list_scenes(parse_time(args.start), parse_time(args.end))
    else:
        from firewatch.data.stac import search_landsat

        refs = search_landsat(settings, parse_time(args.start), parse_time(args.end))
    for ref in refs:
        print(f"{ref.scene_id}  {ref.acquired:%Y-%m-%d}  {ref.platform}  cloud {ref.cloud_cover}  path/row {ref.path_row}")
    print(f"{len(refs)} scene(s)")
    return 0


def cmd_approvals(args) -> int:
    from firewatch.agent.approvals import ApprovalError, decide_approval
    from firewatch.agent.guard import Guard

    ctx = _context(args)
    repo = ctx.services.repo
    if args.action == "list":
        for a in repo.list_approvals(None if args.all else "pending"):
            print(f"{a['approval_id']}  {a['status']:<9} {a['kind']:<15} track {a.get('track_id')}  {', '.join(a.get('segments') or []) or '-'}")
        return 0
    if not args.approval_id:
        print("an approval id is required", file=sys.stderr)
        return 2
    try:
        decided = decide_approval(args.approval_id, args.action == "approve", args.user, args.note or "", repo=repo, roads=ctx.roads, routes=ctx.routes, guard=Guard(ctx.settings), notifier=ctx.services.notifier)
    except ApprovalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(dumps(decided, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from firewatch.api.app import create_app

    app = create_app(_settings(args), Path(args.base_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_worker(args) -> int:
    from firewatch.pipeline.worker import run_worker

    settings = _settings(args)
    queue = args.queue_url or settings.worker.queue_url
    if not queue:
        print("worker.queue_url is not set", file=sys.stderr)
        return 2
    run_worker(_context(args, settings), queue, settings.worker.wait_seconds, once=args.once)
    return 0


def cmd_capabilities(args) -> int:
    from firewatch.vision.cvcompat import capabilities
    from firewatch.vision.registration import FeatureRegistrar

    settings = _settings(args)
    models_dir = Path(settings.alignment.models_dir)
    models_dir = models_dir if models_dir.is_absolute() else Path(args.base_dir) / models_dir
    report = {"opencv": capabilities().as_dict(), "alignment_methods": FeatureRegistrar(settings.alignment, models_dir).available_methods()}
    print(json.dumps(report, indent=2))
    return 0


def cmd_download_models(args) -> int:
    from firewatch.model_zoo import download_models

    for item in download_models(Path(args.base_dir) / args.dest, force=args.force):
        print(f"{item['file']}: {item['status']}")
    return 0


def cmd_eval(args) -> int:
    from firewatch.evaluation.suite import run_suite

    results = run_suite(args.suite, _settings(args), Path(args.base_dir), Path(args.out), scale=args.scale)
    for name, path in results.items():
        print(f"{name}: {path}")
    return 0


def cmd_timelapse(args) -> int:
    from firewatch.pipeline.timelapse import build_timelapse

    ctx = _context(args)
    data, info = build_timelapse(ctx.services, frame_ms=args.frame_ms, ext=Path(args.out).suffix or ".webp")
    Path(args.out).write_bytes(data)
    print(f"{args.out}: {info['frames']} frames, {info['first']['scene_id']} to {info['last']['scene_id']}, {len(data) / 1e6:.1f} MB")
    return 0


def cmd_reset(args) -> int:
    from firewatch.pipeline.runner import reset_local_state

    for path in reset_local_state(_settings(args), args.base_dir):
        print(f"removed {path}")
    return 0


def cmd_export_world(args) -> int:
    from firewatch.data.synthetic import SyntheticWorld

    settings = _settings(args)
    world = SyntheticWorld.from_settings(settings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "roads.geojson").write_text(json.dumps(world.roads_geojson()))
    (out / "industrial.geojson").write_text(json.dumps(world.industrial_geojson()))
    (out / "routes.json").write_text(json.dumps(world.routes(), indent=2))
    print(f"wrote roads, industrial sites and routes to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from firewatch.evaluation.suite import SUITES

    parser = argparse.ArgumentParser(prog="firewatch", description="Jharia Fire Watch: coal-fire thermal watch and road-safety agent (OpenCV 5 + AWS)")
    parser.add_argument("-c", "--config", action="append", default=[], help="extra YAML config file, applied in order (repeatable)")
    parser.add_argument("--base-dir", default=".", help="directory that relative paths (var/, models/) resolve against")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the synthetic Jharia demo end to end")
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--archive", type=int, default=6, help="older passes processed first, without the agent, to build history")
    p.add_argument("--fresh", action="store_true", help="delete local state before running")
    p.add_argument("--scale", type=float, default=None, help="shrink the synthetic area (0-1) for a quicker run")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("real", help="process the real Landsat scenes downloaded by scripts/fetch_real_data.py")
    p.add_argument("--fresh", action="store_true", help="delete local state for real data before running")
    p.add_argument("--archive", type=int, default=3, help="first scenes processed without the agent, to build history")
    p.add_argument("--passes", type=int, default=None, help="process at most this many new scenes")
    p.set_defaults(func=cmd_real)

    p = sub.add_parser("run", help="process one pass")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-id")
    g.add_argument("--next", action="store_true", help="next synthetic pass")
    p.add_argument("--no-agent", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("backfill", help="process every scene in a date range")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--agent", action="store_true", help="let the agent act on historical passes too")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("search", help="list scenes available for the area of interest")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("approvals", help="list or decide pending approvals (the human step)")
    p.add_argument("action", choices=["list", "approve", "reject"])
    p.add_argument("approval_id", nargs="?")
    p.add_argument("--user", default="cli-user")
    p.add_argument("--note", default="")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_approvals)

    p = sub.add_parser("serve", help="start the API and dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("worker", help="process new Landsat scenes from an SQS queue")
    p.add_argument("--queue-url")
    p.add_argument("--once", action="store_true")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("capabilities", help="show which OpenCV 5 features this install has")
    p.set_defaults(func=cmd_capabilities)

    p = sub.add_parser("download-models", help="fetch the ALIKED, LightGlue and DISK ONNX models")
    p.add_argument("--dest", default="models")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download_models)

    p = sub.add_parser("eval", help="run an evaluation suite")
    p.add_argument("suite", choices=[*SUITES, "all"])
    p.add_argument("--out", default="reports")
    p.add_argument("--scale", type=float, default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("timelapse", help="write the season as an animated WebP or GIF (cv2.imencodeanimation)")
    p.add_argument("--out", default="season.webp")
    p.add_argument("--frame-ms", type=int, default=650)
    p.set_defaults(func=cmd_timelapse)

    p = sub.add_parser("reset", help="delete local state (blobs, SQLite, outbox)")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("export-world", help="write the synthetic world's roads and sites as GeoJSON")
    p.add_argument("--out", default="data/synthetic")
    p.set_defaults(func=cmd_export_world)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from firewatch.vision.cvcompat import quiet_opencv

    quiet_opencv()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

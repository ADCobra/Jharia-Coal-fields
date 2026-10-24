from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from firewatch.config import Settings
from firewatch.data.synthetic import TRUTH_THRESHOLD_K, SyntheticWorld
from firewatch.evaluation.common import INK, INK_2, SERIES, md_table, save, styled_axes, write_report
from firewatch.models import Track
from firewatch.pipeline.analogues import analogue_rows
from firewatch.vision.similarity import AnalogueIndex, summarise

WORLDS = (("tuned", {}), ("held-out", {"profile": "holdout", "seed": 23}))
STEP_DAYS = 4.0
BINS = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01))


def _settings(settings: Settings, base_dir: Path, tmp: str) -> Settings:
    models = Path(settings.alignment.models_dir)
    return settings.with_overrides({
        "data": {"source": "synthetic"},
        "storage": {"backend": "local", "root": f"{tmp}/blobs"},
        "state": {"backend": "sqlite", "sqlite_path": f"{tmp}/state.db"},
        "notify": {"backend": "console", "outbox": f"{tmp}/outbox.jsonl"},
        "alignment": {"models_dir": str(models if models.is_absolute() else (base_dir / models).resolve())},
        "observability": {"log_group": None, "metrics_namespace": None},
        "agent": {"planner": "policy"},
    })


def _polygon_mask(grid, polygon, shape) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if len(polygon) >= 3:
        pix = grid.map_polygon_to_pixel(polygon)
        cv2.fillPoly(mask, [np.round(pix).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def _same_area_disk(now: np.ndarray, area: int) -> np.ndarray:
    grown = now
    for radius in range(1, 80):
        if int(grown.sum()) >= area:
            break
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        grown = cv2.dilate(now.astype(np.uint8), kernel).astype(bool)
    return grown


def _share(part: np.ndarray, whole: np.ndarray) -> float | None:
    n = int(whole.sum())
    return float((part & whole).sum()) / n if n else None


def run_season(settings: Settings, base_dir: Path, world: SyntheticWorld, first: int = -6, last: int = 12) -> dict:
    from firewatch.pipeline.context import build_context
    from firewatch.pipeline.process import run_pass

    horizon = settings.projection.horizon_days
    fires = {f.name: f for f in world.fires}
    with tempfile.TemporaryDirectory() as tmp:
        ctx = build_context(_settings(settings, base_dir, tmp), tmp, world=world)
        road_mask = ctx.roads.rasterize(world.grid, 1.5 * world.grid.res)
        rows = []
        for index in range(first, last + 1):
            result = run_pass(ctx, ctx.source.ref_for_index(index), agent=False)
            day = world.day_of(index)
            _, now_fields = world.fire_fields(day)
            future = [world.fire_fields(day + STEP_DAYS * k)[1] for k in range(1, int(horizon / STEP_DAYS) + 1)]
            ids = [h.hotspot_id for h in result.detection.hotspots]
            for case in result.cases.values():
                p = case.projection
                if p is None or len(p.cone_polygon) < 4:
                    continue
                det = result.detection.labels == ids.index(case.hotspot.hotspot_id) + 1
                overlaps = {name: int((field > TRUTH_THRESHOLD_K)[det].sum()) for name, field in now_fields.items()}
                name = max(overlaps, key=overlaps.get) if overlaps else None
                if not name or overlaps[name] == 0:
                    continue
                swept = np.zeros(world.shape, bool)
                for fields in future:
                    swept |= fields[name] > TRUTH_THRESHOLD_K
                cone = _polygon_mask(world.grid, p.cone_polygon, world.shape)
                same_area = _same_area_disk(det, int(cone.sum()))
                truth_now = now_fields[name] > TRUTH_THRESHOLD_K
                new = swept & ~truth_now
                spec = fires[name]
                rows.append({
                    "scene_id": result.ref.scene_id,
                    "fire": name,
                    "track_id": case.hotspot.track_id,
                    "moving": bool(abs(spec.vcol) + abs(spec.vrow) > 0),
                    "goes_out": not swept.any(),
                    "cone_px": int(cone.sum()),
                    "swept_px": int(swept.sum()),
                    "now_px": int(det.sum()),
                    "cone_cover": _share(cone, swept),
                    "disk_cover": _share(same_area, swept),
                    "now_cover": _share(det, swept),
                    "new_px": int(new.sum()),
                    "cone_new": _share(cone, new) if new.sum() >= 3 else None,
                    "disk_new": _share(same_area, new) if new.sum() >= 3 else None,
                    "cone_precision": float((cone & swept).sum()) / max(int(cone.sum()), 1),
                    "reach_probability": p.reach_probability,
                    "touching_now": bool((det & road_mask).any() or (truth_now & road_mask).any()),
                    "reached": bool((swept & road_mask).any()),
                })
        tracks = [Track.from_dict(t) for t in ctx.services.repo.list_tracks()]
        return {"rows": rows, "tracks": tracks, "roads": ctx.roads, "fire_of": {t.track_id: _fire_of(t, world) for t in tracks}}


def _fire_of(track: Track, world: SyntheticWorld, reach_px: float = 15.0) -> str:
    votes: dict[str, int] = {}
    for point in track.points:
        col, row = world.grid.colrow(*point.centroid)
        day = (point.acquired - world.start).total_seconds() / 86400.0
        best, name = reach_px, None
        for fire in world.fires:
            state = fire.state(day)
            if state is not None and float(np.hypot(state[0] - col, state[1] - row)) < best:
                best, name = float(np.hypot(state[0] - col, state[1] - row)), fire.name
        if name:
            votes[name] = votes.get(name, 0) + 1
    return max(votes, key=votes.get) if votes else f"other:{track.track_id}"


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarise_cones(rows: list[dict]) -> dict:
    burning = [r for r in rows if not r["goes_out"]]
    reach = [r for r in rows if not r["touching_now"]]
    out = {
        "projections": len(rows),
        "fire_went_out": len(rows) - len(burning),
        "cone_cover": _mean(r["cone_cover"] for r in burning),
        "disk_cover": _mean(r["disk_cover"] for r in burning),
        "now_cover": _mean(r["now_cover"] for r in burning),
        "cone_new": _mean(r["cone_new"] for r in burning),
        "disk_new": _mean(r["disk_new"] for r in burning),
        "new_ground_cases": sum(1 for r in burning if r["cone_new"] is not None),
        "area_ratio": _mean(r["cone_px"] / r["swept_px"] for r in burning if r["swept_px"]),
        "cone_precision": _mean(r["cone_precision"] for r in burning),
        "reach_cases": len(reach),
        "reached": sum(r["reached"] for r in reach),
    }
    if reach:
        p = np.array([r["reach_probability"] for r in reach], float)
        o = np.array([r["reached"] for r in reach], float)
        out["brier"] = float(np.mean((p - o) ** 2))
        out["brier_base_rate"] = float(o.mean() * (1 - o.mean()))
        out["calibration"] = [{"bin": f"{lo:.1f} to {min(hi, 1.0):.1f}", "cases": int(((p >= lo) & (p < hi)).sum()), "mean_forecast": float(p[(p >= lo) & (p < hi)].mean()) if ((p >= lo) & (p < hi)).any() else None, "observed": float(o[(p >= lo) & (p < hi)].mean()) if ((p >= lo) & (p < hi)).any() else None} for lo, hi in BINS]
    for group in ("moving", "static"):
        sub = [r for r in burning if r["moving"] == (group == "moving")]
        out[group] = {"cases": len(sub), "cone_cover": _mean(r["cone_cover"] for r in sub), "disk_cover": _mean(r["disk_cover"] for r in sub), "cone_new": _mean(r["cone_new"] for r in sub), "disk_new": _mean(r["disk_new"] for r in sub)}
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    wins = sum(float((p > neg).sum()) + 0.5 * float((p == neg).sum()) for p in pos)
    return wins / (len(pos) * len(neg))


def precedent_test(seasons: dict, settings: Settings) -> dict:
    rows = []
    for world, season in seasons.items():
        for vector, meta in analogue_rows(season["tracks"], season["roads"], settings.risk.d0_m):
            rows.append((vector, {**meta, "track_id": f"{world}:{meta['track_id']}", "fire": f"{world}:{season['fire_of'].get(meta['track_id'], meta['track_id'])}"}))
    far = [(v, m) for v, m in rows if m["distance_m"] is not None and m["distance_m"] > settings.risk.d0_m]
    preds, labels, used = [], [], 0
    for vector, meta in far:
        index = AnalogueIndex()
        for v, m in rows:
            if m["fire"] != meta["fire"]:
                index.add(v, m)
        found = summarise(index.build().query(vector, k=5))
        share = found.get("reached_road_share")
        gated = found.get("n", 0) >= settings.risk.precedent_min_analogues and found.get("tracks", 0) >= settings.risk.precedent_min_tracks and share is not None
        used += gated
        preds.append(share if gated else None)
        labels.append(bool(meta["reached_road"]))
    o = np.array(labels, bool)
    out = {"rows": len(rows), "far_rows": len(far), "tracks": len({m["track_id"] for _, m in rows}), "fires": len({m["fire"] for _, m in rows}), "positives": int(o.sum()), "gated": used}
    scored = [(p, l) for p, l in zip(preds, labels) if p is not None]
    if scored:
        p = np.array([s for s, _ in scored], float)
        y = np.array([l for _, l in scored], bool)
        base = float(o.mean())
        out["brier"] = float(np.mean((p - y) ** 2))
        out["brier_base_rate"] = float(np.mean((base - y) ** 2))
        out["skill"] = 1.0 - out["brier"] / out["brier_base_rate"] if out["brier_base_rate"] > 0 else None
        out["auc"] = _auc(p, y)
        out["mean_share_when_reached"] = float(p[y].mean()) if y.any() else None
        out["mean_share_when_not"] = float(p[~y].mean()) if (~y).any() else None
    return out


def evaluate(settings: Settings, base_dir: Path, scale: float | None = None) -> dict:
    seasons = {name: run_season(settings, base_dir, SyntheticWorld.from_settings(settings, scale=scale, **kw)) for name, kw in WORLDS}
    return {
        "horizon_days": settings.projection.horizon_days,
        "worlds": {name: summarise_cones(season["rows"]) for name, season in seasons.items()},
        "cases": {name: season["rows"] for name, season in seasons.items()},
        "precedent": precedent_test(seasons, settings),
    }


def chart(result: dict, path: Path) -> Path:
    fig, ax = styled_axes(7.2, 3.6)
    groups = []
    for world, s in result["worlds"].items():
        groups.append((f"{world.capitalize()}: whole footprint", [s["cone_cover"], s["disk_cover"], s["now_cover"]]))
        groups.append((f"{world.capitalize()}: new ground", [s["cone_new"], s["disk_new"], 0.0]))
    names = ["Projected cone", "Circle of the same area", "Today's outline"]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        xs = [k + (i - 1) * width for k in range(len(groups))]
        values = [(g[1][i] or 0.0) * 100 for g in groups]
        bars = ax.bar(xs, values, width=width * 0.9, color=SERIES[i], label=name, linewidth=0)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], color=INK, fontsize=7.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Share covered (%)", color=INK_2, fontsize=8.5)
    fig.suptitle(f"Where the fire actually went in the next {result['horizon_days']:g} days", color=INK, fontsize=10.5, x=0.02, ha="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center")
    return save(fig, path, legend_space=0.08)


def report(settings: Settings, base_dir: Path, out_dir: Path, scale: float | None = None) -> Path:
    result = evaluate(settings, base_dir, scale)
    chart_path = chart(result, out_dir / "forecast.png")
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    rows = []
    for world, s in result["worlds"].items():
        rows.append([world, s["projections"], s["fire_went_out"], pct(s["cone_cover"]), pct(s["disk_cover"]), pct(s["now_cover"]), f"{pct(s['cone_new'])} vs {pct(s['disk_new'])}", s["area_ratio"], pct(s["cone_precision"])])
    group_rows = []
    for world, s in result["worlds"].items():
        for group in ("moving", "static"):
            g = s[group]
            group_rows.append([world, group, g["cases"], pct(g["cone_cover"]), pct(g["disk_cover"]), f"{pct(g['cone_new'])} vs {pct(g['disk_new'])}"])
    reach_rows = []
    for world, s in result["worlds"].items():
        if "brier" in s:
            reach_rows.append([world, s["reach_cases"], s["reached"], s["brier"], s["brier_base_rate"]])
    pr = result["precedent"]
    md = [
        "# Spread projection and precedent against synthetic truth",
        "",
        f"Every pass of a full season (19 passes) in both synthetic worlds goes through the pipeline. For each hotspot that gets a projection, the cone is compared with where the matching true fire actually burned over the next {result['horizon_days']:g} days, sampled every {STEP_DAYS:g} days. "
        "Two baselines get the same test: today's outline (the fire stays put) and a circle grown around today's outline to the same area as the cone (the fire spreads, but with no direction). "
        "\"New ground\" is the part of the future footprint that is not burning today, which is the part a projection has to earn.",
        "",
        md_table(["World", "Projections", "Fire went out", "Cone covers", "Same-area circle covers", "Today's outline covers", "New ground: cone vs circle", "Cone area / true area", "Cone precision"], rows),
        "",
        f"![Projection coverage]({chart_path.name})",
        "",
        "## Moving and static fires",
        "",
        md_table(["World", "Fires", "Projections", "Cone covers", "Circle covers", "New ground: cone vs circle"], group_rows),
        "",
        "## Will it reach a road?",
        "",
        "Cases where neither today's outline nor the true fire already touches a road. Brier score of the ensemble's reach probability, next to the score of always forecasting the observed rate.",
        "",
        md_table(["World", "Cases", "Actually reached", "Brier", "Brier of the base rate"], reach_rows) if reach_rows else "No cases.",
        "",
        "## Precedent from similar past fires",
        "",
        f"The analogue index (cv2.ANNIndex over size, heat, growth, speed, history and road distance) is tested on {pr['rows']} track states from {pr['tracks']} tracks in both worlds, leaving out one fire at a time ({pr['fires']} fires and plants), so that a fire whose track broke in two cannot vouch for itself. "
        f"Only states more than {settings.risk.d0_m:g} m from a road count, since only those can go on to reach one: {pr['far_rows']} states, {pr['positives']} of which did come within {settings.risk.d0_m:g} m within 64 days. "
        f"The precedent passed its gate (at least {settings.risk.precedent_min_analogues} analogues from at least {settings.risk.precedent_min_tracks} other tracks) for {pr['gated']} of them.",
        "",
    ]
    if "brier" in pr:
        md += [md_table(["Measure", "Value"], [
            ["Brier score of the precedent share", pr["brier"]],
            ["Brier score of the base rate", pr["brier_base_rate"]],
            ["Skill against the base rate (1 is perfect, 0 is no better)", pr["skill"]],
            ["Area under the ROC curve", pr["auc"]],
            ["Mean share when the fire did reach a road", pr["mean_share_when_reached"]],
            ["Mean share when it did not", pr["mean_share_when_not"]],
        ]), ""]
    md.append("The worlds are small: a handful of fires, a few of them moving. Treat these as a check that the projection and the precedent point the right way, not as accuracy figures for Jharia.")
    return write_report(out_dir, "forecast", result, "\n".join(md))

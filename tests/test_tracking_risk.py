import datetime as dt

import pytest
from shapely.geometry import Point

from firewatch.evaluation.agent_eval import ROUTES, circle, eval_roads
from firewatch.models import Hotspot, Kinematics, Track
from firewatch.risk.scoring import ConfidenceInputs, assess, confidence
from firewatch.util import parse_time
from firewatch.vision.tracking import associate, kinematics, point_from_hotspot, update_tracks


def _hotspot(hid, when, x, y, r=90.0, mean=8.0):
    return Hotspot(hid, f"S-{hid}", when, 3.1416 * r * r, 30, mean, mean * 1.8, 315.0, (x, y), (86.3, 23.75), circle(x, y, r), (0, 0, 10, 10), 0.8, 0.6, 1.2, 0.0, 0.0)


def test_track_follows_a_moving_hotspot(settings):
    t0 = parse_time("2026-01-01")
    counter = iter(range(1, 100))
    new_id = lambda: f"T{next(counter):04d}"
    always = lambda x, y: True
    tracks = []
    for k in range(4):
        h = _hotspot(f"h{k}", t0 + dt.timedelta(days=16 * k), 0.0, 1000.0 - 40.0 * k)
        assoc = associate(tracks, [h], settings.tracking)
        touched = update_tracks(tracks, [h], assoc, settings.tracking, new_id, always)
        tracks = list({t.track_id: t for t in tracks + touched}.values())
    assert len(tracks) == 1
    kin = kinematics(tracks[0])
    assert kin.n_points == 4
    assert kin.vy_m_per_day == pytest.approx(-2.5, abs=0.01)


def test_unseen_track_cools_only_when_observable(settings):
    t0 = parse_time("2026-01-01")
    track = Track("T0001", points=[point_from_hotspot(_hotspot("a", t0, 0.0, 500.0))])
    assoc = associate([track], [], settings.tracking)
    update_tracks([track], [], assoc, settings.tracking, lambda: "X", lambda x, y: False)
    assert track.missed == 0
    update_tracks([track], [], assoc, settings.tracking, lambda: "X", lambda x, y: True)
    assert track.missed == 1 and track.status == "cooling"


def test_road_network_queries_and_reroute():
    roads = eval_roads()
    sid, dist = roads.nearest(Point(250.0, 300.0))
    assert sid == "M-04" and dist == pytest.approx(300.0)
    assert roads.within(Point(250.0, 100.0), 250.0) == ["M-04"]
    assert roads.within(Point(250.0, 100.0), 300.0) == ["M-03", "M-04", "M-05"]
    assert roads.approach_direction("M-04", (250.0, 300.0)) == pytest.approx((0.0, -1.0))
    roads.close(["M-04"])
    report = {r["route"]: r for r in roads.reroute_report(ROUTES)}
    assert report["West yard to washery"]["extra_m"] == 3000
    roads.close(["B-04"])
    assert {r["route"]: r for r in roads.reroute_report(ROUTES)}["West yard to washery"]["blocked"]
    roads.reopen(["M-04", "B-04"])
    assert not roads.closed


def test_risk_rises_as_fire_nears_the_road(settings):
    roads = eval_roads()
    when = parse_time("2026-01-01")
    kin = Kinematics(3, 32.0, 0.0, 0.2, 0.0, -1.5, 1.5, 180.0, 0.0)
    inputs = ConfidenceInputs(0.0, True, 0.3, 3, 0.0, 0.05)
    conf, notes = confidence(inputs, settings.hotspots, settings.alignment)
    far = assess(_hotspot("far", when, 250.0, 1290.0), kin, None, roads, settings.risk, conf, notes)
    near = assess(_hotspot("near", when, 250.0, 190.0), kin, None, roads, settings.risk, conf, notes)
    assert near.score > far.score
    assert near.components["proximity"] > far.components["proximity"]
    assert near.approach_m_per_day == pytest.approx(1.5)
    assert near.nearest_segment == "M-04"


def test_confidence_penalties_and_evidence(settings):
    base = ConfidenceInputs(0.4, False, None, 1, 0.5, 0.6)
    low, notes = confidence(base, settings.hotspots, settings.alignment)
    assert low == pytest.approx(0.0)
    assert len(notes) == 5
    helped = ConfidenceInputs(0.0, True, 0.3, 1, 0.0, 0.0, evidence={"history": {"persistent": True}, "highres": {"confirmed": True}})
    score, _ = confidence(helped, settings.hotspots, settings.alignment)
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("precedent, y, raised", [
    ({"n": 5, "tracks": 3, "reached_road_share": 0.6}, 790.0, True),
    ({"n": 5, "tracks": 3, "reached_road_share": 0.6}, 290.0, False),
    ({"n": 2, "tracks": 2, "reached_road_share": 1.0}, 790.0, False),
    ({"n": 5, "tracks": 1, "reached_road_share": 1.0}, 790.0, False),
    (None, 790.0, False),
])
def test_precedent_raises_spread_only_far_from_the_road_with_enough_cases(settings, precedent, y, raised):
    roads = eval_roads()
    when = parse_time("2026-01-01")
    kin = Kinematics(3, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    plain = assess(_hotspot("p", when, 250.0, y), kin, None, roads, settings.risk, 0.9, [])
    with_cases = assess(_hotspot("p", when, 250.0, y), kin, None, roads, settings.risk, 0.9, [], precedent)
    assert (with_cases.score > plain.score) is raised
    assert ("precedent" in with_cases.components) is raised


def test_analogue_query_skips_the_track_itself():
    import numpy as np

    from firewatch.vision.similarity import AnalogueIndex

    index = AnalogueIndex()
    for k in range(6):
        index.add(np.full(7, 0.1 * k, np.float32), {"track_id": "T1" if k < 3 else f"T{k}", "reached_road": k >= 3})
    index.build()
    found = index.query(np.zeros(7, np.float32), k=3, exclude_track="T1")
    assert len(found) == 3 and all(a.meta["track_id"] != "T1" for a in found)


def _jitter_track(offsets, step_days=8):
    from firewatch.models import TrackPoint

    t0 = parse_time("2026-01-01")
    pts = [TrackPoint("s", t0 + dt.timedelta(days=step_days * i), f"h{i}", 50000.0, 9.0, 14.0, (1000.0 + dx, 2000.0 + dy), circle(1000.0 + dx, 2000.0 + dy, 120)) for i, (dx, dy) in enumerate(offsets)]
    return Track("T1", pts)


def test_position_noise_shrinks_jitter_but_keeps_steady_motion():
    jitter = _jitter_track([(0, 0), (70, -40)])
    raw = kinematics(jitter, 8)
    assert raw.speed_m_per_day > 9
    assert kinematics(jitter, 8, noise_m=90).speed_m_per_day == 0.0
    steady = _jitter_track([(5 * 8 * i, 0) for i in range(8)])
    kept = kinematics(steady, 8, noise_m=90).speed_m_per_day
    assert 4.0 < kept < 5.0

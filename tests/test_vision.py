import cv2
import numpy as np
import pytest

from firewatch.data.synthetic import SyntheticWorld
from firewatch.evaluation.alignment import CONDITIONS, make_pair
from firewatch.geo.grid import GridSpec
from firewatch.models import SceneRef
from firewatch.physics.thermal import derive_thermal
from firewatch.util import parse_time
from firewatch.vision.anomaly import compute_anomaly, masked_box_mean, robust_sigma
from firewatch.vision.hotspots import detect_hotspots
from firewatch.vision.registration import FeatureRegistrar, invert_affine, transform_error_px


def _field(shape=(160, 200), seed=0):
    rng = np.random.default_rng(seed)
    lst = 305.0 + 0.4 * rng.standard_normal(shape).astype(np.float32)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    lst += 9.0 * np.exp(-(((rr - 80) / 4.0) ** 2 + ((cc - 120) / 5.0) ** 2) / 2)
    return lst.astype(np.float32)


def _grid(shape):
    return GridSpec("EPSG:32645", (30.0, 0.0, 400000.0, 0.0, -30.0, 2620000.0), shape[1], shape[0])


def test_masked_mean_ignores_masked_pixels():
    values = np.ones((50, 50), np.float32)
    values[20:30, 20:30] = 100.0
    mask = np.ones_like(values, bool)
    mask[20:30, 20:30] = False
    out = masked_box_mean(values, mask, 21)
    assert np.nanmax(out) == pytest.approx(1.0)


def test_robust_sigma_resists_outliers():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(10000)
    v[:200] = 50.0
    assert robust_sigma(v) == pytest.approx(1.0, abs=0.1)


def test_single_fire_is_found(settings):
    lst = _field()
    valid = np.ones(lst.shape, bool)
    flat = np.full(lst.shape, 0.15, np.float32)
    cfg = settings.thermal.model_copy(update={"detrend": False})
    anomaly = compute_anomaly(lst, valid, flat, flat, settings.hotspots, cfg)
    ref = SceneRef("TEST", parse_time("2026-01-01"), "LANDSAT_8", "test")
    industrial = np.zeros(lst.shape, bool)
    industrial[70:90, 110:115] = True
    found = detect_hotspots(anomaly, lst, valid, _grid(lst.shape), ref, settings.hotspots, industrial)
    assert len(found.hotspots) == 1
    h = found.hotspots[0]
    r0, c0, r1, c1 = h.bbox_px
    assert r0 <= 80 <= r1 and c0 <= 120 <= c1
    assert h.mean_anomaly_k > settings.hotspots.min_contrast_k
    assert 0.0 < h.industrial_overlap < 0.5
    assert h.polygon[0] == h.polygon[-1]


def test_noise_alone_gives_no_hotspots(settings):
    rng = np.random.default_rng(3)
    lst = (305.0 + 0.4 * rng.standard_normal((160, 200))).astype(np.float32)
    valid = np.ones(lst.shape, bool)
    flat = np.full(lst.shape, 0.15, np.float32)
    anomaly = compute_anomaly(lst, valid, flat, flat, settings.hotspots, settings.thermal.model_copy(update={"detrend": False}))
    found = detect_hotspots(anomaly, lst, valid, _grid(lst.shape), SceneRef("N", parse_time("2026-01-01"), "LANDSAT_8", "test"), settings.hotspots)
    assert found.hotspots == []


def _weak_blob(peak_k, seed=5, shape=(160, 200)):
    rng = np.random.default_rng(seed)
    lst = (305.0 + 0.3 * rng.standard_normal(shape)).astype(np.float32)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    lst += (peak_k * np.exp(-((rr - 80) ** 2 + (cc - 120) ** 2) / (2 * 2.0 ** 2))).astype(np.float32)
    return lst


def _detect_with_prior(settings, lst, prior, persistence=True):
    valid = np.ones(lst.shape, bool)
    flat = np.full(lst.shape, 0.15, np.float32)
    anomaly = compute_anomaly(lst, valid, flat, flat, settings.hotspots, settings.thermal.model_copy(update={"detrend": False}))
    cfg = settings.hotspots.model_copy(update={"persistence": persistence})
    return detect_hotspots(anomaly, lst, valid, _grid(lst.shape), SceneRef("P", parse_time("2026-01-01"), "LANDSAT_8", "test"), cfg, None, prior)


def _disk(center, radius, shape=(160, 200)):
    out = np.zeros(shape, np.uint8)
    cv2.circle(out, (center[1], center[0]), radius, 1, -1)
    return out.astype(bool)


def test_weak_blob_alone_is_only_a_candidate(settings):
    found = _detect_with_prior(settings, _weak_blob(3.2), None)
    assert found.hotspots == []
    assert found.candidates[80, 120]


def test_persistence_keeps_a_weak_blob_seen_last_pass(settings):
    found = _detect_with_prior(settings, _weak_blob(3.2), _disk((81, 121), 2))
    assert len(found.hotspots) == 1
    assert found.hotspots[0].persistent
    assert found.hotspots[0].max_anomaly_k < settings.hotspots.min_contrast_k


@pytest.mark.parametrize("prior_at, persistence", [((20, 20), True), ((81, 121), False)])
def test_persistence_needs_an_overlapping_prior(settings, prior_at, persistence):
    found = _detect_with_prior(settings, _weak_blob(3.2), _disk(prior_at, 2), persistence)
    assert found.hotspots == []


def test_persistence_does_not_promote_noise(settings):
    found = _detect_with_prior(settings, _weak_blob(0.0, seed=9), _disk((80, 100), 40))
    assert found.hotspots == []


def test_pond_shore_by_a_pit_is_not_a_hotspot(settings):
    world = SyntheticWorld.from_settings(settings, misregister=False, profile="holdout", seed=23)
    crop = world.grid.crop(640, 230, 140, 140)
    raw = world.raw_scene(0, crop=crop)
    thermal = derive_thermal(raw, settings.thermal)
    assert thermal.water.any() and not any(m.any() for m in raw.truth["fire_masks"].values())

    def count(thermal_cfg, water):
        anomaly = compute_anomaly(thermal.lst, thermal.valid, thermal.albedo, thermal.ndvi, settings.hotspots, thermal_cfg, water)
        return len(detect_hotspots(anomaly, thermal.lst, thermal.valid, crop, raw.ref, settings.hotspots).hotspots)

    assert count(settings.thermal.model_copy(update={"psf_sigma_px": 0.0}), None) == 1
    assert count(settings.thermal, thermal.water) == 0


def test_synthetic_scene_hotspots_match_truth(settings):
    world = SyntheticWorld.from_settings(settings, scale=0.4, misregister=False)
    raw = world.raw_scene(2)
    thermal = derive_thermal(raw, settings.thermal)
    anomaly = compute_anomaly(thermal.lst, thermal.valid, thermal.albedo, thermal.ndvi, settings.hotspots, settings.thermal)
    found = detect_hotspots(anomaly, thermal.lst, thermal.valid, world.grid, raw.ref, settings.hotspots, world.industrial_mask)
    truth = np.zeros(world.shape, bool)
    for mask in raw.truth["fire_masks"].values():
        truth |= mask
    for h in found.hotspots:
        hit = (found.labels == found.hotspots.index(h) + 1)
        assert (hit & (truth | world.industrial_mask)).any(), f"{h.hotspot_id} is not on a fire or the plant"


@pytest.fixture(scope="module")
def pair(settings):
    world = SyntheticWorld.from_settings(settings, scale=0.4, misregister=False)
    h, w = world.shape
    size = min(320, h, w)
    crop = (slice((h - size) // 2, (h + size) // 2), slice((w - size) // 2, (w + size) // 2))
    return make_pair(world, crop, CONDITIONS["Geolocation only"], np.random.default_rng(42))


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_classic_features_recover_shift(settings, models_dir, pair, method):
    ref, mov, truth = pair
    result = FeatureRegistrar(settings.alignment, models_dir).register(ref, mov, method=method)
    assert result.ok, result.note
    assert transform_error_px(result.matrix, truth, ref.shape) < 0.5


def test_learned_features_recover_shift(settings, models_dir, pair):
    registrar = FeatureRegistrar(settings.alignment, models_dir)
    if "aliked_lightglue" not in registrar.available_methods():
        pytest.skip("ALIKED/LightGlue models not downloaded (firewatch download-models)")
    ref, mov, truth = pair
    result = registrar.register(ref, mov, method="aliked_lightglue")
    assert result.ok, result.note
    assert transform_error_px(result.matrix, truth, ref.shape) < 0.5
    assert result.shift_after_px < result.shift_before_px


def test_transform_error_of_pure_shift():
    shift = [[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]]
    assert transform_error_px(shift, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (100, 100)) == pytest.approx(5.0)
    back = invert_affine(shift)
    assert back[0][2] == pytest.approx(-3.0) and back[1][2] == pytest.approx(-4.0)


def test_opencv_is_version_5():
    assert int(cv2.__version__.split(".")[0]) >= 5


def test_second_matcher_checks_every_pass(settings, models_dir, pair):
    ref, mov, truth = pair
    result = FeatureRegistrar(settings.alignment, models_dir).register_checked(ref, mov)
    assert result.ok and result.method == "sift"
    assert result.check_method in ("aliked_lightglue", "orb") and result.check_ok
    assert result.agreement_px is not None and result.agreement_px < 0.3


def test_second_matcher_takes_over_when_the_first_fails(settings, models_dir):
    world = SyntheticWorld.from_settings(settings, scale=0.4, misregister=False)
    h, w = world.shape
    size = min(320, h, w)
    crop = (slice((h - size) // 2, (h + size) // 2), slice((w - size) // 2, (w + size) // 2))
    ref, mov, truth = make_pair(world, crop, CONDITIONS["Dry season, new dumps"], np.random.default_rng(3))
    cfg = settings.alignment.model_copy(update={"method": "orb", "check_preference": ["sift"]})
    result = FeatureRegistrar(cfg, models_dir).register_checked(ref, mov)
    assert result.ok and result.method == "sift" and "orb failed" in result.note
    assert transform_error_px(result.matrix, truth, ref.shape) < 0.5


@pytest.mark.parametrize("offset, agrees", [(0.0, True), (2.0, False)])
def test_third_matcher_settles_a_disagreement(settings, models_dir, pair, offset, agrees):
    from firewatch.agent.tools import ToolContext, Toolbox
    from firewatch.evaluation.agent_eval import build_case, eval_roads

    registrar = FeatureRegistrar(settings.alignment, models_dir)
    if "disk" not in registrar.available_methods():
        pytest.skip("DISK model not downloaded")
    ref, mov, truth = pair
    used = registrar.register_checked(ref, mov)
    used.matrix = (np.asarray(used.matrix) + np.array([[0, 0, offset], [0, 0, 0]])).tolist()
    s = settings.with_overrides({"alignment": {"tiebreak_window_px": 256}})
    roads = eval_roads()
    case = build_case({"id": "TB", "hotspot": {"gap_m": 80, "mean_k": 10.0, "max_k": 18.0}, "track": {"points": 3}}, s, roads)
    case.hotspot.bbox_px = (150, 150, 170, 170)
    ctx = ToolContext(settings=s, source=None, grid=None, registrar=registrar, alignment=used, roads=roads, vlm=None, repo=None, run_id="T", acquired=None, reference_image=ref, scene_image=mov, scene_mask=None)
    out = Toolbox(ctx).refine_alignment(case)
    assert out["method"] == "disk" and out["window_px"] == 256
    assert out["ok"] is agrees
    assert out["agreement_px"] == pytest.approx(offset, abs=0.3)


def test_mask_opening_is_exact_and_never_adds_pixels():
    from firewatch.vision.hotspots import _open2

    rng = np.random.default_rng(3)
    mask = rng.random((50, 70)) > 0.4
    opened = _open2(mask)
    expected = np.zeros_like(mask)
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            if mask[r:r + 2, c:c + 2].all():
                expected[r:r + 2, c:c + 2] = True
    assert not (opened & ~mask).any()
    assert (opened == expected).all()

import pytest

from firewatch.agent.guard import Guard
from firewatch.agent.policy import decide
from firewatch.evaluation.agent_eval import build_case, circle, eval_roads, load_scenarios, run_scenario, sequence_checks
from firewatch.models import Hotspot
from firewatch.util import parse_time
from firewatch.vision.false_heat import industrial_assessment

KNOWN_LIMITATIONS = {"C04"}
SCENARIOS = load_scenarios()


@pytest.mark.parametrize("spec", [s for s in SCENARIOS if s["id"] not in KNOWN_LIMITATIONS], ids=lambda s: s["id"])
def test_scenario(settings, spec):
    out = run_scenario(spec, settings)
    assert out["decision"] == spec["expect"]["decision"], out["reason"]
    failed = [c for c in out["checks"] if not c["passed"]]
    assert not failed, failed


def test_known_limitation_errs_toward_a_person(settings):
    for spec in SCENARIOS:
        if spec["id"] in KNOWN_LIMITATIONS:
            out = run_scenario(spec, settings)
            assert out["outcome"] == "conservative"
            assert out["closed"] == []


def test_safety_and_memory_checks(settings):
    rows = sequence_checks(settings)
    assert len(rows) >= 10
    failed = [r for r in rows if not r["passed"]]
    assert not failed, failed


def test_guard_keeps_reopening_human_only(settings):
    guard = Guard(settings)
    assert guard.check("reopen_road", "agent") == (False, "reopening a road always needs a human decision")
    assert guard.check("reopen_road", "human")[0]
    assert not guard.check("delete_everything", "agent")[0]
    assert not guard.check("close_road", "agent", None)[0]


def test_projection_alone_does_not_close_a_road(settings):
    roads = eval_roads()
    case = build_case({"id": "P", "hotspot": {"gap_m": 500, "radius_m": 100, "mean_k": 7.0, "max_k": 12.0}, "track": {"points": 3, "growth_30d": 0.2, "approach": 8.0}}, settings, roads)
    assert case.risk.level == "high"
    decision = decide(case, settings, roads)
    assert decision.kind == "escalate"
    assert "M-04" in decision.segments
    roads.close(decision.segments)
    assert decide(case, settings, roads).kind == "close"


def test_industrial_overlap_alone_is_not_enough():
    def hs(overlap, rect, sol, peak):
        return Hotspot("h", "s", parse_time("2026-01-01"), 150000.0, 160, 12.0, peak, 330.0, (0.0, 0.0), (86.3, 23.7), circle(0, 0, 200), (0, 0, 1, 1), sol, rect, 1.2, 0.0, overlap)

    assert industrial_assessment(hs(0.42, 0.88, 0.97, 31.8))["verdict"] == "industrial"
    assert industrial_assessment(hs(0.95, 0.5, 0.7, 12.0))["verdict"] == "industrial"
    assert industrial_assessment(hs(0.35, 0.55, 0.78, 15.0))["verdict"] == "uncertain"
    assert industrial_assessment(hs(0.0, 0.5, 0.7, 12.0))["verdict"] == "fire"

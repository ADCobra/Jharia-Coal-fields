from __future__ import annotations

import json
import logging
import re

from firewatch.agent.case import Case
from firewatch.config import Settings
from firewatch.services.llm import LLMClient

log = logging.getLogger(__name__)

TOOL_HELP = {
    "check_industrial": "compare the hotspot with mapped industrial sites and its shape; ask the vision-language model if unsure",
    "fetch_history": "load older Landsat passes of this spot and test whether the heat was already there",
    "refine_alignment": "align a window around the hotspot with a third matcher (DISK) and check which alignment it agrees with",
    "fetch_highres": "load the closest clear Sentinel-2 image (20 m) and look for hot pixels in shortwave infrared",
    "request_drone": "file a drone survey request for this spot (simulated in this build)",
}

PLANNER_SYSTEM = (
    "You choose the next evidence-gathering step for a coal-fire watch agent in the Jharia coalfield, India. "
    "Pick exactly one tool from allowed_tools. You cannot close or reopen roads; other code decides that. "
    "Prefer the tool that most reduces uncertainty about whether this hotspot threatens a road. "
    'Reply with one JSON object only: {"tool": "<one of allowed_tools>", "rationale": "<one sentence citing the numbers>"}'
)

ALERT_SYSTEM = (
    "You write short operational alerts for mine safety officers. Use only the facts given. "
    "Plain English, under 90 words, no greeting, no speculation beyond the facts, keep every number as given."
)


def policy_rationale(tool: str, case: Case, settings: Settings) -> str:
    r = case.risk
    seg = r.nearest_segment or "the nearest road"
    dist = f"{r.distance_m:.0f} m" if r.distance_m is not None else "an unknown distance"
    if tool == "check_industrial":
        return f"{case.hotspot.industrial_overlap:.0%} overlap with a mapped industrial site and a {case.hotspot.rectangularity:.2f} rectangularity; rule out a plant before treating it as a coal fire"
    if tool == "fetch_history":
        return f"first sighting of a {case.hotspot.mean_anomaly_k:.1f} K anomaly {dist} from {seg}; check older passes to see if it is new"
    if tool == "refine_alignment":
        ci = case.conf_inputs
        if not ci.alignment_ok:
            return "alignment failed on this pass; check the hotspot's window with a third matcher"
        if ci.alignment_agreement_px is not None and ci.alignment_agreement_px > settings.alignment.max_disagreement_px:
            return f"SIFT and ALIKED + LightGlue disagree by {ci.alignment_agreement_px:.2f} px; let a third matcher settle it around the hotspot"
        return f"alignment residual {ci.alignment_residual_px:.2f} px makes spread estimates unreliable; check with a third matcher"
    if tool == "fetch_highres":
        return f"risk {r.score:.2f} ({r.level}) {dist} from {seg}; look for hot pixels in sharper Sentinel-2 shortwave infrared"
    if tool == "request_drone":
        return f"moving toward {seg} at {r.approach_m_per_day:.1f} m/day with no satellite confirmation; ask for a drone survey"
    return tool


class PolicyPlanner:
    name = "policy"

    def __init__(self, settings: Settings):
        self.settings = settings

    def choose(self, case: Case, options: list[str]) -> tuple[str, str, str]:
        tool = options[0]
        return tool, policy_rationale(tool, case, self.settings), self.name

    def write_alert(self, facts: dict) -> str | None:
        return None


def _first_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMPlanner:
    def __init__(self, settings: Settings, llm: LLMClient):
        self.settings = settings
        self.llm = llm
        self.fallback = PolicyPlanner(settings)
        self.name = f"bedrock:{llm.model_id}"

    def choose(self, case: Case, options: list[str]) -> tuple[str, str, str]:
        prompt = json.dumps({"hotspot": case.summary(self.settings), "allowed_tools": options, "tool_descriptions": {t: TOOL_HELP[t] for t in options}}, default=str)
        text = self.llm.complete(PLANNER_SYSTEM, prompt, max_tokens=200)
        parsed = _first_json(text or "")
        if parsed and parsed.get("tool") in options:
            return parsed["tool"], str(parsed.get("rationale", "")).strip()[:400], self.name
        tool, rationale, _ = self.fallback.choose(case, options)
        reason = "no reply" if text is None else "reply was not a valid choice"
        return tool, f"{rationale} (LLM {reason}; policy fallback)", "policy-fallback"

    def write_alert(self, facts: dict) -> str | None:
        return self.llm.complete(ALERT_SYSTEM, json.dumps(facts, default=str), max_tokens=220)


def build_planner(settings: Settings, llm: LLMClient):
    if settings.agent.planner == "bedrock" and llm.available:
        return LLMPlanner(settings, llm)
    return PolicyPlanner(settings)

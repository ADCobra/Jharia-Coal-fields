from __future__ import annotations

from firewatch.agent.case import Case
from firewatch.config import Settings

HUMAN_ONLY = {"reopen_road"}
AGENT_ALLOWED = {"close_road", "keep_closed", "notify", "escalate", "request_drone", "propose_reopen", "log", "exclude"}


class Guard:
    def __init__(self, settings: Settings):
        self.settings = settings

    def check(self, kind: str, actor: str, case: Case | None = None) -> tuple[bool, str]:
        if kind in HUMAN_ONLY:
            if actor != "human":
                return False, "reopening a road always needs a human decision"
            return True, ""
        if actor == "agent" and kind not in AGENT_ALLOWED:
            return False, f"the agent is not allowed to {kind}"
        if kind == "close_road" and actor == "agent":
            if not self.settings.agent.allow_auto_close:
                return False, "automatic closure is switched off"
            if case is None or case.decision != "close" or case.risk.level != "high":
                return False, "the agent may only close roads for a high-risk hotspot it decided to close"
            if case.risk.confidence < self.settings.risk.min_confidence_auto:
                return False, "confidence too low for an automatic closure"
        return True, ""

    @staticmethod
    def check_tool(tool: str | None, options: list[str]) -> bool:
        return tool is not None and tool in options

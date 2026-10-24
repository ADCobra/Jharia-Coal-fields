from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    available: bool
    model_id: str

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str | None: ...


class NullLLM:
    available = False
    model_id = "none"

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str | None:
        return None


class BedrockLLM:
    available = True

    def __init__(self, model_id: str, region: str, client=None):
        import boto3

        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime", region_name=region)

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> str | None:
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
        except Exception as exc:
            log.warning("Bedrock call failed: %s", exc)
            return None
        parts = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(p.get("text", "") for p in parts) or None

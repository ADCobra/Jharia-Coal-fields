from __future__ import annotations

import json
import logging

from firewatch.models import TraceEvent
from firewatch.services.blobs import BlobStore
from firewatch.services.observability import TraceSink
from firewatch.util import jsonable, utcnow

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, run_id: str, sinks: list[TraceSink] | None = None):
        self.run_id = run_id
        self.sinks = sinks or []
        self.events: list[TraceEvent] = []

    def emit(self, node: str, hotspot_id: str | None = None, *, observation: dict | None = None, decision: str | None = None, tool: str | None = None, result: dict | None = None, rationale: str | None = None) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            seq=len(self.events) + 1,
            ts=utcnow(),
            node=node,
            hotspot_id=hotspot_id,
            observation=jsonable(observation or {}),
            decision=decision,
            tool=tool,
            result=jsonable(result or {}),
            rationale=rationale,
        )
        self.events.append(event)
        log.debug("trace %s %s %s %s", node, hotspot_id, tool or "", decision or "")
        return event

    def as_dicts(self) -> list[dict]:
        return [jsonable(e) for e in self.events]

    def flush(self, blobs: BlobStore, key: str) -> None:
        lines = "\n".join(json.dumps(e) for e in self.as_dicts())
        blobs.put_text(key, lines + ("\n" if lines else ""), "application/x-ndjson")
        for sink in self.sinks:
            try:
                sink.write(self.run_id, self.as_dicts())
            except Exception as exc:
                log.warning("trace sink %s failed: %s", type(sink).__name__, exc)

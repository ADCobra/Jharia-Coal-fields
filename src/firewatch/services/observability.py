from __future__ import annotations

import json
import logging
import time
from typing import Protocol

from firewatch.util import jsonable

log = logging.getLogger(__name__)


class TraceSink(Protocol):
    def write(self, stream: str, events: list[dict]) -> None: ...


class CloudWatchLogsSink:
    def __init__(self, log_group: str, region: str, client=None):
        import boto3

        self.log_group = log_group
        self.client = client or boto3.client("logs", region_name=region)

    def _ensure_stream(self, stream: str) -> None:
        try:
            self.client.create_log_stream(logGroupName=self.log_group, logStreamName=stream)
        except self.client.exceptions.ResourceAlreadyExistsException:
            pass

    def write(self, stream: str, events: list[dict]) -> None:
        if not events:
            return
        self._ensure_stream(stream)
        now = int(time.time() * 1000)
        batch: list[dict] = []
        size = 0
        for i, event in enumerate(events):
            message = json.dumps(jsonable(event))
            entry = {"timestamp": now + i, "message": message}
            if batch and (len(batch) >= 5000 or size + len(message) + 26 > 900_000):
                self.client.put_log_events(logGroupName=self.log_group, logStreamName=stream, logEvents=batch)
                batch, size = [], 0
            batch.append(entry)
            size += len(message) + 26
        if batch:
            self.client.put_log_events(logGroupName=self.log_group, logStreamName=stream, logEvents=batch)


class Metrics:
    def __init__(self, namespace: str | None = None, region: str | None = None, client=None):
        self.namespace = namespace
        self.recorded: list[dict] = []
        self.client = None
        if namespace:
            import boto3

            self.client = client or boto3.client("cloudwatch", region_name=region)

    def put(self, name: str, value: float, unit: str = "Count", dimensions: dict[str, str] | None = None) -> None:
        datum = {"MetricName": name, "Value": float(value), "Unit": unit}
        if dimensions:
            datum["Dimensions"] = [{"Name": k, "Value": str(v)} for k, v in dimensions.items()]
        self.recorded.append(datum)
        if self.client is None:
            return
        try:
            self.client.put_metric_data(Namespace=self.namespace, MetricData=[datum])
        except Exception as exc:
            log.warning("could not publish metric %s: %s", name, exc)

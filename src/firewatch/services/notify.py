from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from firewatch.util import jsonable, utcnow

log = logging.getLogger(__name__)


class Notifier(Protocol):
    def send(self, subject: str, message: str, severity: str = "info", data: dict | None = None) -> dict: ...


class ConsoleNotifier:
    def __init__(self, outbox: str | Path | None = None):
        self.outbox = Path(outbox) if outbox else None
        self.sent: list[dict] = []

    def send(self, subject: str, message: str, severity: str = "info", data: dict | None = None) -> dict:
        record = {"ts": utcnow().isoformat(), "channel": "console", "severity": severity, "subject": subject, "message": message, "data": jsonable(data or {})}
        self.sent.append(record)
        log.info("[%s] %s", severity.upper(), subject)
        if self.outbox:
            self.outbox.parent.mkdir(parents=True, exist_ok=True)
            with self.outbox.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        return record


class SNSNotifier:
    def __init__(self, topic_arn: str, region: str, client=None):
        import boto3

        self.topic_arn = topic_arn
        self.client = client or boto3.client("sns", region_name=region)
        self.sent: list[dict] = []

    def send(self, subject: str, message: str, severity: str = "info", data: dict | None = None) -> dict:
        response = self.client.publish(
            TopicArn=self.topic_arn,
            Subject=subject[:99],
            Message=message,
            MessageAttributes={"severity": {"DataType": "String", "StringValue": severity}},
        )
        record = {"ts": utcnow().isoformat(), "channel": "sns", "severity": severity, "subject": subject, "message_id": response.get("MessageId")}
        self.sent.append(record)
        return record

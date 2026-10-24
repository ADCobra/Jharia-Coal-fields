from __future__ import annotations

import json
import logging
import re

from firewatch.data.stac import intersects_aoi, keep_landsat
from firewatch.pipeline.context import Context
from firewatch.pipeline.process import run_pass

log = logging.getLogger(__name__)
PRODUCT_RE = re.compile(r"L[COTE]0[89]_L1(?:TP|GT|GS)_\d{6}_\d{8}_\d{8}_\d{2}_(?:T1|T2|RT)")


def extract_scene_ids(body: str) -> list[str]:
    text = body
    try:
        envelope = json.loads(body)
        if isinstance(envelope, dict) and "Message" in envelope:
            text = envelope["Message"] if isinstance(envelope["Message"], str) else json.dumps(envelope["Message"])
    except json.JSONDecodeError:
        pass
    seen: list[str] = []
    for match in PRODUCT_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def path_row_of(scene_id: str) -> str:
    return scene_id.split("_")[2]


def handle_message(ctx: Context, body: str) -> list[dict]:
    allowed = set(ctx.settings.worker.path_rows)
    results = []
    for scene_id in extract_scene_ids(body):
        if allowed and path_row_of(scene_id) not in allowed:
            results.append({"scene_id": scene_id, "status": "skipped", "reason": "path/row not watched"})
            continue
        ref = ctx.source.get_ref(scene_id)
        if not intersects_aoi(ref, ctx.settings.aoi.bbox):
            results.append({"scene_id": scene_id, "status": "skipped", "reason": "outside the area of interest"})
            continue
        ok, why = keep_landsat(ref, ctx.settings)
        if not ok:
            results.append({"scene_id": scene_id, "status": "skipped", "reason": why})
            continue
        result = run_pass(ctx, ref)
        results.append({"scene_id": scene_id, "status": "processed", "run_id": result.run_id})
    return results


def run_worker(ctx: Context, queue_url: str, wait_seconds: int = 20, once: bool = False) -> None:
    import boto3

    sqs = boto3.client("sqs", region_name=ctx.settings.aws.region)
    log.info("watching %s", queue_url)
    while True:
        response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=5, WaitTimeSeconds=wait_seconds, VisibilityTimeout=900)
        for message in response.get("Messages", []):
            try:
                for item in handle_message(ctx, message["Body"]):
                    log.info("scene %s: %s %s", item["scene_id"], item["status"], item.get("reason", ""))
            except Exception:
                log.exception("message %s failed; it will become visible again for a retry", message.get("MessageId"))
                continue
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
        if once:
            return

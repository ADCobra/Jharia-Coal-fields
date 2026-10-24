import json

import boto3
import numpy as np
import pytest
from moto import mock_aws

from firewatch.pipeline.worker import extract_scene_ids, path_row_of
from firewatch.services.blobs import S3BlobStore
from firewatch.services.kv import DynamoKV
from firewatch.services.llm import BedrockLLM
from firewatch.services.notify import SNSNotifier
from firewatch.services.observability import CloudWatchLogsSink, Metrics
from firewatch.services.repository import Repository

REGION = "us-west-2"


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    for key, value in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": REGION}.items():
        monkeypatch.setenv(key, value)


@mock_aws
def test_s3_blob_store_round_trip():
    boto3.client("s3", region_name=REGION).create_bucket(Bucket="firewatch-test", CreateBucketConfiguration={"LocationConstraint": REGION})
    store = S3BlobStore("firewatch-test", "firewatch", REGION)
    store.put_json("runs/R1/summary.json", {"ok": True})
    store.put_arrays("runs/R1/maps.npz", lst=np.arange(6, dtype=np.float32).reshape(2, 3))
    assert store.get_json("runs/R1/summary.json") == {"ok": True}
    assert store.get_arrays("runs/R1/maps.npz")["lst"].shape == (2, 3)
    assert store.get_bytes("runs/missing.json") is None
    assert store.list("runs/R1/") == ["runs/R1/maps.npz", "runs/R1/summary.json"]


@mock_aws
def test_dynamodb_store_behaves_like_sqlite():
    boto3.client("dynamodb", region_name=REGION).create_table(
        TableName="fw",
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    repo = Repository(DynamoKV("fw", REGION))
    for i in range(3):
        repo.put_run({"run_id": f"R{i}", "acquired": f"2026-01-0{i + 1}", "status": "done"})
    repo.put_hotspot("R1", {"hotspot_id": "h1", "risk": {"score": 0.7}})
    repo.put_hotspot("R10", {"hotspot_id": "h9"})
    assert [r["run_id"] for r in repo.list_runs()] == ["R2", "R1", "R0"]
    assert [h["hotspot_id"] for h in repo.list_hotspots("R1")] == ["h1"]
    assert repo.next_track_id() == "T0001" and repo.next_track_id() == "T0002"
    repo.kv.delete("run", "R0")
    assert repo.get_run("R0") is None


@mock_aws
def test_sns_notifier_publishes():
    sns = boto3.client("sns", region_name=REGION)
    arn = sns.create_topic(Name="fw-alerts")["TopicArn"]
    sent = SNSNotifier(arn, REGION).send("Road closure near T0005: H2-04", "body text", "critical", {"x": 1})
    assert sent["channel"] == "sns" and sent["message_id"]


@mock_aws
def test_trace_sink_and_metrics():
    logs = boto3.client("logs", region_name=REGION)
    logs.create_log_group(logGroupName="/firewatch/agent")
    sink = CloudWatchLogsSink("/firewatch/agent", REGION)
    sink.write("R1", [{"node": "plan", "seq": 1}, {"node": "act", "seq": 2}])
    sink.write("R1", [{"node": "review_closures", "seq": 3}])
    events = logs.get_log_events(logGroupName="/firewatch/agent", logStreamName="R1")["events"]
    assert [json.loads(e["message"])["seq"] for e in events] == [1, 2, 3]
    metrics = Metrics("FireWatch", REGION)
    metrics.put("Hotspots", 4, "Count", {"aoi": "jharia"})
    assert metrics.recorded[0]["Value"] == 4.0


def test_bedrock_failure_returns_none():
    class Broken:
        def converse(self, **kwargs):
            raise RuntimeError("no access to this model")

    assert BedrockLLM("us.amazon.nova-lite-v1:0", REGION, client=Broken()).complete("system", "prompt") is None


def test_bedrock_reply_is_joined():
    class Fake:
        def converse(self, **kwargs):
            assert kwargs["inferenceConfig"]["temperature"] == 0.0
            return {"output": {"message": {"content": [{"text": '{"tool": '}, {"text": '"fetch_history"}'}]}}}

    assert BedrockLLM("m", REGION, client=Fake()).complete("s", "p") == '{"tool": "fetch_history"}'


def test_worker_reads_usgs_notifications():
    scene = "LC09_L1TP_140044_20251121_20251121_02_T1"
    body = json.dumps({"Type": "Notification", "Message": json.dumps({"landsat_product_id": scene, "s3_location": f"s3://usgs-landsat/collection02/level-1/{scene}/"})})
    assert extract_scene_ids(body) == [scene]
    assert path_row_of(scene) == "140044"
    assert extract_scene_ids("nothing here") == []

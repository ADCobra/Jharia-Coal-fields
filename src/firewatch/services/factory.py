from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from firewatch.config import Settings
from firewatch.services.blobs import BlobStore, LocalBlobStore, S3BlobStore
from firewatch.services.kv import DynamoKV, SQLiteKV
from firewatch.services.llm import BedrockLLM, LLMClient, NullLLM
from firewatch.services.notify import ConsoleNotifier, Notifier, SNSNotifier
from firewatch.services.observability import CloudWatchLogsSink, Metrics, TraceSink
from firewatch.services.repository import Repository


@dataclass
class Services:
    blobs: BlobStore
    repo: Repository
    notifier: Notifier
    llm: LLMClient
    metrics: Metrics
    trace_sinks: list[TraceSink] = field(default_factory=list)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def build_services(settings: Settings, base_dir: str | Path | None = None) -> Services:
    base = Path(base_dir or Path.cwd())
    region = settings.aws.region
    if settings.storage.backend == "s3":
        if not settings.storage.bucket:
            raise ValueError("storage.bucket is required for the s3 backend")
        blobs: BlobStore = S3BlobStore(settings.storage.bucket, settings.storage.prefix, region)
    else:
        blobs = LocalBlobStore(_resolve(base, settings.storage.root))
    if settings.state.backend == "dynamodb":
        if not settings.state.table:
            raise ValueError("state.table is required for the dynamodb backend")
        kv = DynamoKV(settings.state.table, region)
    else:
        kv = SQLiteKV(_resolve(base, settings.state.sqlite_path))
    if settings.notify.backend == "sns":
        if not settings.notify.topic_arn:
            raise ValueError("notify.topic_arn is required for the sns backend")
        notifier: Notifier = SNSNotifier(settings.notify.topic_arn, region)
    else:
        notifier = ConsoleNotifier(_resolve(base, settings.notify.outbox))
    llm: LLMClient = BedrockLLM(settings.agent.bedrock_model_id, region) if settings.agent.planner == "bedrock" else NullLLM()
    metrics = Metrics(settings.observability.metrics_namespace, region)
    sinks: list[TraceSink] = []
    if settings.observability.log_group:
        sinks.append(CloudWatchLogsSink(settings.observability.log_group, region))
    return Services(blobs=blobs, repo=Repository(kv), notifier=notifier, llm=llm, metrics=metrics, trace_sinks=sinks)

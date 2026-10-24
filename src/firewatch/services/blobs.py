from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from firewatch.util import jsonable


class BlobStore:
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        raise NotImplementedError

    def get_bytes(self, key: str) -> bytes | None:
        raise NotImplementedError

    def list(self, prefix: str = "") -> list[str]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        return self.get_bytes(key) is not None

    def put_json(self, key: str, value) -> None:
        self.put_bytes(key, json.dumps(jsonable(value)).encode(), "application/json")

    def get_json(self, key: str, default=None):
        raw = self.get_bytes(key)
        return default if raw is None else json.loads(raw)

    def put_text(self, key: str, text: str, content_type: str = "text/plain") -> None:
        self.put_bytes(key, text.encode(), content_type)

    def put_arrays(self, key: str, **arrays: np.ndarray) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        self.put_bytes(key, buf.getvalue(), "application/octet-stream")

    def get_arrays(self, key: str) -> dict[str, np.ndarray] | None:
        raw = self.get_bytes(key)
        if raw is None:
            return None
        with np.load(io.BytesIO(raw)) as data:
            return {k: data[k] for k in data.files}


class LocalBlobStore(BlobStore):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError(f"key escapes the blob root: {key}")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def list(self, prefix: str = "") -> list[str]:
        base = self.root.resolve()
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file() and str(p.relative_to(base)).startswith(prefix) and not p.name.endswith(".tmp"))


class S3BlobStore(BlobStore):
    def __init__(self, bucket: str, prefix: str = "", region: str | None = None, client=None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or boto3.client("s3", region_name=region)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, **extra)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def list(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys.extend(obj["Key"][strip:] for obj in page.get("Contents", []))
        return sorted(keys)

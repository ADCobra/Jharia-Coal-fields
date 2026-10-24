from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Protocol

from firewatch.util import jsonable, utcnow


class KVStore(Protocol):
    def put(self, kind: str, key: str, data: dict) -> None: ...

    def get(self, kind: str, key: str) -> dict | None: ...

    def list(self, kind: str, prefix: str = "", limit: int | None = None, newest_first: bool = False) -> list[dict]: ...

    def delete(self, kind: str, key: str) -> None: ...


class SQLiteKV:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS items (kind TEXT NOT NULL, key TEXT NOT NULL, data TEXT NOT NULL, updated TEXT NOT NULL, PRIMARY KEY (kind, key))")
        self._conn.commit()

    def put(self, kind: str, key: str, data: dict) -> None:
        payload = json.dumps(jsonable(data))
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO items (kind, key, data, updated) VALUES (?, ?, ?, ?)", (kind, key, payload, utcnow().isoformat()))
            self._conn.commit()

    def get(self, kind: str, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM items WHERE kind = ? AND key = ?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind: str, prefix: str = "", limit: int | None = None, newest_first: bool = False) -> list[dict]:
        order = "DESC" if newest_first else "ASC"
        sql = f"SELECT data FROM items WHERE kind = ? AND key >= ? AND key < ? ORDER BY key {order}"
        params: list = [kind, prefix, prefix + "\U0010ffff"]
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r[0]) for r in rows]

    def delete(self, kind: str, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM items WHERE kind = ? AND key = ?", (kind, key))
            self._conn.commit()


class DynamoKV:
    def __init__(self, table: str, region: str, resource=None):
        import boto3

        self._resource = resource or boto3.resource("dynamodb", region_name=region)
        self.table = self._resource.Table(table)

    def put(self, kind: str, key: str, data: dict) -> None:
        self.table.put_item(Item={"pk": kind, "sk": key, "data": json.dumps(jsonable(data)), "updated": utcnow().isoformat()})

    def get(self, kind: str, key: str) -> dict | None:
        item = self.table.get_item(Key={"pk": kind, "sk": key}).get("Item")
        return json.loads(item["data"]) if item else None

    def list(self, kind: str, prefix: str = "", limit: int | None = None, newest_first: bool = False) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        condition = Key("pk").eq(kind)
        if prefix:
            condition = condition & Key("sk").begins_with(prefix)
        kwargs = {"KeyConditionExpression": condition, "ScanIndexForward": not newest_first}
        out: list[dict] = []
        while True:
            if limit:
                kwargs["Limit"] = max(1, limit - len(out))
            page = self.table.query(**kwargs)
            out.extend(json.loads(item["data"]) for item in page.get("Items", []))
            if (limit and len(out) >= limit) or "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        return out[:limit] if limit else out

    def delete(self, kind: str, key: str) -> None:
        self.table.delete_item(Key={"pk": kind, "sk": key})

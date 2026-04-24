from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": tool_call.get("id"),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": function.get("name") or "",
            "arguments": arguments,
        },
    }
    return normalized


def tool_call_signature(tool_call: dict[str, Any]) -> str:
    normalized = normalize_tool_call(tool_call)
    normalized.pop("id", None)
    canonical = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.append(str(tool_call["id"]))
    return ids


def message_signature(message: dict[str, Any]) -> str:
    tool_calls = [
        normalize_tool_call(tool_call)
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    ]
    payload = {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_scope_message(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {"role": message.get("role")}
    for key in ("content", "name", "tool_call_id", "prefix"):
        if key in message:
            canonical[key] = message[key]
    if message.get("tool_calls"):
        canonical["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in message.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
    return canonical


def conversation_scope(messages: list[dict[str, Any]]) -> str:
    payload = [canonical_scope_message(message) for message in messages]
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ReasoningStore:
    def __init__(self, reasoning_content_path: str | Path) -> None:
        if str(reasoning_content_path) == ":memory:":
            self.reasoning_content_path: str | Path = ":memory:"
        else:
            self.reasoning_content_path = Path(reasoning_content_path).expanduser()
            self.reasoning_content_path.parent.mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.reasoning_content_path, check_same_thread=False
        )
        if isinstance(self.reasoning_content_path, Path):
            self.reasoning_content_path.chmod(0o600)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                key TEXT PRIMARY KEY,
                reasoning TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def put(self, key: str, reasoning: str, message: dict[str, Any]) -> None:
        if not reasoning:
            return
        message_json = json.dumps(message, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reasoning_cache(key, reasoning, message_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    reasoning = excluded.reasoning,
                    message_json = excluded.message_json,
                    created_at = excluded.created_at
                """,
                (key, reasoning, message_json, time.time()),
            )
            self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT reasoning FROM reasoning_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def store_assistant_message(self, message: dict[str, Any], scope: str) -> int:
        if message.get("role") != "assistant":
            return 0
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            return 0

        keys = [f"scope:{scope}:signature:{message_signature(message)}"]
        keys.extend(
            f"scope:{scope}:tool_call:{tool_call_id}"
            for tool_call_id in tool_call_ids(message)
        )
        keys.extend(
            f"scope:{scope}:tool_call_signature:{tool_call_signature(tool_call)}"
            for tool_call in (message.get("tool_calls") or [])
            if isinstance(tool_call, dict)
        )
        for key in keys:
            self.put(key, reasoning, message)
        return len(keys)

    def lookup_for_message(self, message: dict[str, Any], scope: str) -> str | None:
        reasoning = self.get(f"scope:{scope}:signature:{message_signature(message)}")
        if reasoning:
            return reasoning
        for tool_call_id in tool_call_ids(message):
            reasoning = self.get(f"scope:{scope}:tool_call:{tool_call_id}")
            if reasoning:
                return reasoning
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            reasoning = self.get(
                f"scope:{scope}:tool_call_signature:{tool_call_signature(tool_call)}"
            )
            if reasoning:
                return reasoning
        return None

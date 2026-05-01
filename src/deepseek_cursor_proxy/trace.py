from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


TRACE_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def authorization_summary(authorization: str | None) -> dict[str, Any]:
    if not authorization:
        return {"present": False}
    return {"present": True, "sha256": sha256_text(authorization)}


def sanitized_headers(headers: dict[str, str] | None) -> dict[str, Any]:
    if not headers:
        return {}
    sanitized: dict[str, Any] = {}
    for name, value in headers.items():
        if name.lower() == "authorization":
            sanitized[name] = authorization_summary(value)
        else:
            sanitized[name] = value
    return sanitized


def jsonable_body(body: bytes) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}
    return {"json": payload}


def tool_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            names.append("")
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            names.append(str(function.get("name") or ""))
        else:
            names.append("")
    return names


def content_stats(content: Any) -> dict[str, Any]:
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return {"length": len(text), "sha256": sha256_text(text)}


def message_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    summaries: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            summaries.append({"index": index, "type": type(message).__name__})
            continue
        tool_calls = message.get("tool_calls")
        tool_call_ids = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get("id"):
                    tool_call_ids.append(str(tool_call["id"]))
        reasoning = message.get("reasoning_content")
        content = str(message.get("content") or "")
        summary: dict[str, Any] = {
            "index": index,
            "role": message.get("role"),
            "content": content_stats(message.get("content")),
            "has_tool_calls": bool(tool_call_ids or tool_calls),
            "tool_call_ids": tool_call_ids,
            "tool_call_id": message.get("tool_call_id"),
            "has_reasoning_content": isinstance(reasoning, str),
            "reasoning_content_length": (
                len(reasoning) if isinstance(reasoning, str) else 0
            ),
            "has_recovery_notice": content.startswith(
                (
                    "[deepseek-cursor-proxy] Refreshed reasoning_content history.",
                    "[deepseek-cursor-proxy] Recovered",
                )
            ),
        }
        summaries.append(summary)
    return summaries


def payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    system_hashes = [
        content_stats(message.get("content"))["sha256"]
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    return {
        "model": payload.get("model"),
        "stream": bool(payload.get("stream")),
        "message_count": len(messages),
        "tool_count": (
            len(payload.get("tools") or [])
            if isinstance(payload.get("tools"), list)
            else 0
        ),
        "tool_names": tool_names(payload),
        "system_prompt_hashes": system_hashes,
        "messages": message_summaries(payload),
    }


def write_json_private(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.chmod(0o600)
    tmp_path.replace(path)
    path.chmod(0o600)


class TraceWriter:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        session_name = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"-pid{os.getpid()}"
        )
        self.session_dir = self.base_dir / session_name
        self.session_dir.mkdir(mode=0o700)
        self._lock = threading.Lock()
        self._next_sequence = 1
        self._write_manifest()

    def start_request(
        self,
        *,
        method: str,
        path: str,
        client_address: str,
        headers: dict[str, str],
    ) -> "TraceRequest":
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        trace_path = self.session_dir / f"request-{sequence:06d}.json"
        return TraceRequest(
            writer=self,
            sequence=sequence,
            path=trace_path,
            data={
                "schema_version": TRACE_SCHEMA_VERSION,
                "sequence": sequence,
                "created_at": utc_now_iso(),
                "request": {
                    "method": method,
                    "path": path,
                    "client_address": client_address,
                    "headers": sanitized_headers(headers),
                },
                "transform": {},
                "upstream": {},
                "cursor_response": {},
                "completion": {},
            },
        )

    def _write_manifest(self) -> None:
        write_json_private(
            self.session_dir / "manifest.json",
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "created_at": utc_now_iso(),
                "pid": os.getpid(),
                "base_dir": str(self.base_dir),
                "session_dir": str(self.session_dir),
                "format": "one JSON file per traced HTTP request",
            },
        )


@dataclass
class TraceRequest:
    writer: TraceWriter
    sequence: int
    path: Path
    data: dict[str, Any]
    _started: float = field(default_factory=time.monotonic)
    _finished: bool = False

    def record_cursor_body(self, payload: dict[str, Any]) -> None:
        self.data["request"]["body"] = payload
        self.data["request"]["summary"] = payload_summary(payload)

    def record_cursor_body_bytes(self, body: bytes) -> None:
        self.data["request"]["body_bytes"] = len(body)
        text = body.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.data["request"]["body"] = {"text": text}
            return
        self.data["request"]["body"] = payload
        if isinstance(payload, dict):
            self.data["request"]["summary"] = payload_summary(payload)

    def record_cursor_body_omitted(
        self, *, reason: str, body_bytes: int | None = None
    ) -> None:
        omitted: dict[str, Any] = {"reason": reason}
        if body_bytes is not None:
            omitted["body_bytes"] = body_bytes
        self.data["request"]["body_omitted"] = omitted

    def record_transform(self, prepared: Any) -> None:
        self.data["transform"] = {
            "original_model": prepared.original_model,
            "upstream_model": prepared.upstream_model,
            "cache_namespace": prepared.cache_namespace,
            "patched_reasoning_messages": prepared.patched_reasoning_messages,
            "missing_reasoning_messages": prepared.missing_reasoning_messages,
            "recovered_reasoning_messages": prepared.recovered_reasoning_messages,
            "recovery_dropped_messages": prepared.recovery_dropped_messages,
            "recovery_notice": prepared.recovery_notice,
            "record_response_scope": prepared.record_response_scope,
            "record_response_scopes": [
                scope for scope, _messages in prepared.record_response_contexts
            ],
            "continued_recovery_boundary": prepared.continued_recovery_boundary,
            "retired_prefix_messages": prepared.retired_prefix_messages,
            "reasoning_diagnostics": prepared.reasoning_diagnostics,
            "recovery_steps": prepared.recovery_steps,
            "upstream_request_summary": payload_summary(prepared.payload),
            "upstream_request_body": prepared.payload,
        }

    def record_upstream_request(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body_bytes: bytes,
    ) -> None:
        self.data["upstream"]["request"] = {
            "url": url,
            "headers": sanitized_headers(headers),
            "body_bytes": len(body_bytes),
        }

    def record_upstream_response(
        self,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        stream: bool | None = None,
    ) -> None:
        response: dict[str, Any] = {"status": status}
        if headers is not None:
            response["headers"] = sanitized_headers(headers)
        if stream is not None:
            response["stream"] = stream
        if body is not None:
            response["body"] = jsonable_body(body)
        self.data["upstream"]["response"] = response

    def record_cursor_response(
        self,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> None:
        response: dict[str, Any] = {"status": status}
        if headers is not None:
            response["headers"] = sanitized_headers(headers)
        if body is not None:
            response["body"] = jsonable_body(body)
        self.data["cursor_response"].update(response)

    def record_stream_chunk(self, upstream_line: bytes, cursor_line: bytes) -> None:
        upstream_stream = self.data["upstream"].setdefault("stream", {"chunks": []})
        cursor_stream = self.data["cursor_response"].setdefault(
            "stream", {"chunks": []}
        )
        index = len(upstream_stream["chunks"])
        upstream_stream["chunks"].append(
            {
                "index": index,
                "line": upstream_line.decode("utf-8", errors="replace"),
            }
        )
        cursor_stream["chunks"].append(
            {
                "index": index,
                "line": cursor_line.decode("utf-8", errors="replace"),
            }
        )

    def record_usage(self, usage: Any) -> None:
        if isinstance(usage, dict):
            self.data["upstream"]["usage"] = usage

    def finish(self, status: str, **extra: Any) -> None:
        if self._finished:
            return
        completion = {
            "status": status,
            "finished_at": utc_now_iso(),
            "elapsed_ms": round((time.monotonic() - self._started) * 1000),
        }
        completion.update(extra)
        self.data["completion"] = completion
        write_json_private(self.path, self.data)
        self._finished = True

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gzip
from http.client import HTTPException
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zlib

from .config import (
    ProxyConfig,
    default_config_path,
    default_reasoning_content_path,
)
from .reasoning_store import ReasoningStore, conversation_scope
from .streaming import CursorReasoningDisplayAdapter, StreamAccumulator
from .trace import TraceRequest, TraceWriter
from .tunnel import NgrokTunnel, local_tunnel_target
from .transform import (
    PreparedRequest,
    RECOVERY_NOTICE_CONTENT,
    prepare_upstream_request,
    rewrite_response_body,
)


LOG = logging.getLogger("deepseek_cursor_proxy")


class RequestBodyTooLarge(ValueError):
    pass


@dataclass
class ProxyResponseResult:
    sent: bool
    usage: dict[str, Any] | None = None


class DeepSeekProxyServer(ThreadingHTTPServer):
    config: ProxyConfig
    reasoning_store: ReasoningStore
    trace_writer: TraceWriter | None
    _request_log_lock: threading.Lock
    _next_request_log_id: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_log_lock = threading.Lock()
        self._next_request_log_id = 1

    def next_request_log_id(self) -> int:
        with self._request_log_lock:
            request_id = self._next_request_log_id
            self._next_request_log_id += 1
        return request_id


class DeepSeekProxyHandler(BaseHTTPRequestHandler):
    server_version = "DeepSeekPythonProxy/0.1"

    @property
    def config(self) -> ProxyConfig:
        return self.server.config  # type: ignore[return-value]

    @property
    def reasoning_store(self) -> ReasoningStore:
        return self.server.reasoning_store  # type: ignore[return-value]

    @property
    def trace_writer(self) -> TraceWriter | None:
        return getattr(self.server, "trace_writer", None)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        request_path = urlparse(self.path).path
        if self.config.verbose:
            LOG.info(
                "incoming OPTIONS %s from %s",
                request_path,
                self.client_address[0],
            )
        self._send_response_headers(204, [], "sending CORS preflight response")

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if self.config.verbose:
            LOG.info("incoming GET %s from %s", request_path, self.client_address[0])
        if request_path in {"/healthz", "/v1/healthz"}:
            self._send_json(200, {"ok": True})
            return
        if request_path in {"/models", "/v1/models"}:
            self._send_models()
            return
        self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:
        started = time.monotonic()
        request_id = self.server.next_request_log_id()
        request_path = urlparse(self.path).path
        trace = self._start_trace(request_path)
        if self.config.verbose:
            LOG.info(
                "incoming POST %s from %s content_length=%s user_agent=%s",
                request_path,
                self.client_address[0],
                self.headers.get("Content-Length", "0"),
                self.headers.get("User-Agent", ""),
            )
        if request_path not in {"/chat/completions", "/v1/chat/completions"}:
            LOG.warning("rejected unsupported POST path=%s status=404", request_path)
            self._send_json(
                404,
                {"error": {"message": "Only /v1/chat/completions is supported"}},
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=404)
            return
        cursor_authorization = self._cursor_authorization()
        if cursor_authorization is None:
            LOG.warning(
                "rejected request path=%s status=401 reason=missing_bearer_token",
                request_path,
            )
            self._send_json(
                401,
                {"error": {"message": "Missing Authorization bearer token"}},
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=401)
            return

        try:
            payload = self._read_json_body()
        except RequestBodyTooLarge as exc:
            LOG.warning(
                "rejected request path=%s status=413 reason=%s", request_path, exc
            )
            self._send_json(413, {"error": {"message": str(exc)}}, trace=trace)
            self._finish_trace(trace, "rejected", http_status=413, reason=str(exc))
            return
        except ValueError as exc:
            LOG.warning(
                "rejected request path=%s status=400 reason=%s", request_path, exc
            )
            self._send_json(400, {"error": {"message": str(exc)}}, trace=trace)
            self._finish_trace(trace, "rejected", http_status=400, reason=str(exc))
            return

        if trace is not None:
            trace.record_cursor_body(payload)

        if self.config.verbose:
            log_json("cursor request body", payload)

        prepared = prepare_upstream_request(
            payload,
            self.config,
            self.reasoning_store,
            authorization=cursor_authorization,
        )
        if trace is not None:
            trace.record_transform(prepared)
        if prepared.missing_reasoning_messages:
            LOG.warning(
                (
                    "strict missing-reasoning mode rejected request path=%s "
                    "status=409 reason=missing_reasoning_content count=%s"
                ),
                request_path,
                prepared.missing_reasoning_messages,
            )
            self._send_json(
                409,
                {
                    "error": {
                        "message": (
                            "deepseek-cursor-proxy is running in strict "
                            "missing-reasoning mode and cannot automatically "
                            "recover this thinking-mode tool-call history because "
                            "cached DeepSeek reasoning_content is missing for "
                            f"{prepared.missing_reasoning_messages} assistant "
                            "message(s). Restart without "
                            "`--missing-reasoning-strategy reject`, or pass "
                            "`--missing-reasoning-strategy recover`, so the proxy "
                            "can recover from partial chat history automatically."
                        ),
                        "type": "missing_reasoning_content",
                        "code": "missing_reasoning_content",
                        "missing_reasoning_messages": prepared.missing_reasoning_messages,
                    }
                },
                trace=trace,
            )
            self._finish_trace(trace, "rejected", http_status=409)
            return

        if self.config.verbose:
            LOG.info(
                (
                    "upstream request metadata: original_model=%s upstream_model=%s "
                    "patched_reasoning=%s missing_reasoning=%s %s"
                ),
                prepared.original_model,
                prepared.upstream_model,
                prepared.patched_reasoning_messages,
                prepared.missing_reasoning_messages,
                summarize_chat_payload(prepared.payload),
            )

        if self.config.verbose:
            log_json("upstream request body", prepared.payload)

        upstream_body = json.dumps(
            prepared.payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        upstream_url = f"{self.config.upstream_base_url}/chat/completions"
        upstream_headers = self._upstream_headers(
            stream=bool(prepared.payload.get("stream")),
            authorization=cursor_authorization,
        )
        if trace is not None:
            trace.record_upstream_request(
                url=upstream_url,
                headers=upstream_headers,
                body_bytes=upstream_body,
            )
        request = Request(
            upstream_url,
            data=upstream_body,
            method="POST",
            headers=upstream_headers,
        )

        try:
            if self.config.verbose:
                LOG.info("forwarding to %s", upstream_url)
            response = urlopen(request, timeout=self.config.request_timeout)
        except HTTPError as exc:
            LOG.warning(
                "request failed upstream_status=%s stream=%s elapsed_ms=%s",
                exc.code,
                bool(prepared.payload.get("stream")),
                elapsed_ms(started),
            )
            self._send_upstream_error(exc, trace=trace)
            self._finish_trace(
                trace,
                "upstream_error",
                http_status=exc.code,
                stream=bool(prepared.payload.get("stream")),
            )
            return
        except URLError as exc:
            LOG.warning(
                "upstream request failed elapsed_ms=%s reason=%s",
                elapsed_ms(started),
                exc.reason,
            )
            self._send_json(
                502,
                {"error": {"message": f"Upstream request failed: {exc.reason}"}},
                trace=trace,
            )
            self._finish_trace(trace, "upstream_error", http_status=502)
            return

        with response:
            upstream_status = getattr(response, "status", 200)
            if self.config.verbose:
                LOG.info(
                    "upstream response status=%s stream=%s elapsed_ms=%s",
                    upstream_status,
                    bool(prepared.payload.get("stream")),
                    elapsed_ms(started),
                )
            if prepared.payload.get("stream"):
                sent_response = self._proxy_streaming_response(
                    response,
                    prepared.original_model,
                    prepared.payload["messages"],
                    prepared.cache_namespace,
                    prepared.recovery_notice,
                    trace=trace,
                    record_response_scope=prepared.record_response_scope,
                    record_response_messages=prepared.record_response_messages,
                    record_response_contexts=prepared.record_response_contexts,
                )
            else:
                sent_response = self._proxy_regular_response(
                    response,
                    prepared.original_model,
                    prepared.payload["messages"],
                    prepared.cache_namespace,
                    prepared.recovery_notice,
                    trace=trace,
                    record_response_scope=prepared.record_response_scope,
                    record_response_messages=prepared.record_response_messages,
                    record_response_contexts=prepared.record_response_contexts,
                )
            if not sent_response.sent:
                self._finish_trace(
                    trace,
                    "client_disconnected",
                    http_status=upstream_status,
                    stream=bool(prepared.payload.get("stream")),
                )
                return
            log_request_lifecycle(
                request_id=request_id,
                cursor_payload=payload,
                prepared=prepared,
                usage=sent_response.usage,
            )
            self._finish_trace(
                trace,
                "completed",
                http_status=upstream_status,
                stream=bool(prepared.payload.get("stream")),
            )

    def _start_trace(self, request_path: str) -> TraceRequest | None:
        writer = self.trace_writer
        if writer is None:
            return None
        try:
            return writer.start_request(
                method=self.command,
                path=request_path,
                client_address=self.client_address[0],
                headers={name: value for name, value in self.headers.items()},
            )
        except OSError as exc:
            LOG.warning("failed to start request trace: %s", exc)
            return None

    def _finish_trace(
        self,
        trace: TraceRequest | None,
        status: str,
        **extra: Any,
    ) -> None:
        if trace is None:
            return
        try:
            trace.finish(status, **extra)
        except OSError as exc:
            LOG.warning("failed to write request trace: %s", exc)

    def _cursor_authorization(self) -> str | None:
        auth_header = self.headers.get("Authorization", "")
        scheme, separator, token = auth_header.strip().partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token.strip():
            return None
        return f"Bearer {token.strip()}"

    def _send_cors_headers(self) -> None:
        if not self.config.cors:
            return
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Origin, Content-Type, Accept, Authorization",
        )
        self.send_header("Access-Control-Expose-Headers", "Content-Length")
        self.send_header("Access-Control-Allow-Credentials", "true")

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        trace: TraceRequest | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if trace is not None:
            trace.record_cursor_response(
                status=status,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
                body=body,
            )
        sent_headers = self._send_response_headers(
            status,
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ],
            "sending JSON response headers",
        )
        if sent_headers:
            self._write_to_client(body, "sending JSON response body")

    def _send_response_headers(
        self,
        status: int,
        headers: list[tuple[str, str]],
        disconnect_context: str,
    ) -> bool:
        try:
            self.send_response(status)
            self._send_cors_headers()
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
        except (BrokenPipeError, ConnectionError) as exc:
            LOG.warning("client disconnected while %s: %s", disconnect_context, exc)
            return False
        return True

    def _write_to_client(
        self,
        body: bytes,
        disconnect_context: str,
        *,
        flush: bool = False,
    ) -> bool:
        try:
            self.wfile.write(body)
            if flush:
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError) as exc:
            LOG.warning("client disconnected while %s: %s", disconnect_context, exc)
            return False
        return True

    def _send_models(self) -> None:
        created = int(time.time())
        model_ids = list(
            dict.fromkeys(
                [
                    self.config.upstream_model,
                    "deepseek-v4-pro",
                    "deepseek-v4-flash",
                ]
            )
        )
        models = [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
            }
            for model_id in model_ids
        ]
        self._send_json(200, {"object": "list", "data": models})

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0:
            raise ValueError("Invalid Content-Length")
        if length > self.config.max_request_body_bytes:
            raise RequestBodyTooLarge(
                f"Request body is too large; limit is {self.config.max_request_body_bytes} bytes"
            )
        raw_body = self.rfile.read(length)
        if not raw_body:
            raise ValueError("Request body is empty")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _upstream_headers(self, stream: bool, authorization: str) -> dict[str, str]:
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": self.server_version,
        }
        accept_language = self.headers.get("Accept-Language")
        if accept_language:
            headers["Accept-Language"] = accept_language
        return headers

    def _send_upstream_error(
        self,
        exc: HTTPError,
        *,
        trace: TraceRequest | None = None,
    ) -> None:
        body = read_response_body(exc)
        if self.config.verbose:
            log_bytes("upstream error body", body)
        headers = {
            "Content-Type": exc.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_upstream_response(
                status=exc.code,
                headers={name: value for name, value in exc.headers.items()},
                body=body,
            )
            trace.record_cursor_response(status=exc.code, headers=headers, body=body)
        sent_headers = self._send_response_headers(
            exc.code,
            [
                ("Content-Type", headers["Content-Type"]),
                ("Content-Length", headers["Content-Length"]),
            ],
            "sending upstream error headers",
        )
        if sent_headers:
            self._write_to_client(body, "sending upstream error body")

    def _proxy_regular_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
        record_response_scope: str | None = None,
        record_response_messages: list[dict[str, Any]] | None = None,
        record_response_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
    ) -> ProxyResponseResult:
        body = read_response_body(response)
        upstream_body = body
        usage = usage_from_body(upstream_body)
        try:
            body = rewrite_response_body(
                body,
                original_model,
                self.reasoning_store,
                request_messages,
                cache_namespace,
                content_prefix=recovery_notice,
                scope=record_response_scope,
                prior_messages=record_response_messages,
                recording_contexts=record_response_contexts,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOG.warning("failed to rewrite upstream JSON response: %s", exc)

        if self.config.verbose:
            log_bytes("cursor response body", body)

        headers = {
            "Content-Type": response.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        if trace is not None:
            trace.record_upstream_response(
                status=getattr(response, "status", 200),
                headers=response_headers(response),
                body=upstream_body,
                stream=False,
            )
            try:
                upstream_payload = json.loads(upstream_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                upstream_payload = None
            if isinstance(upstream_payload, dict):
                trace.record_usage(upstream_payload.get("usage"))
            trace.record_cursor_response(
                status=getattr(response, "status", 200),
                headers=headers,
                body=body,
            )

        sent_headers = self._send_response_headers(
            getattr(response, "status", 200),
            [
                ("Content-Type", headers["Content-Type"]),
                ("Content-Length", headers["Content-Length"]),
            ],
            "sending upstream response headers",
        )
        if not sent_headers:
            return ProxyResponseResult(False, usage)
        sent = self._write_to_client(body, "sending upstream response body")
        return ProxyResponseResult(sent, usage)

    def _proxy_streaming_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
        record_response_scope: str | None = None,
        record_response_messages: list[dict[str, Any]] | None = None,
        record_response_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
    ) -> ProxyResponseResult:
        if trace is not None:
            trace.record_upstream_response(
                status=getattr(response, "status", 200),
                headers=response_headers(response),
                stream=True,
            )
            trace.record_cursor_response(
                status=getattr(response, "status", 200),
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                },
            )
        sent_headers = self._send_response_headers(
            getattr(response, "status", 200),
            [
                ("Content-Type", "text/event-stream"),
                ("Cache-Control", "no-cache"),
                ("Connection", "close"),
            ],
            "sending streaming response headers",
        )
        if not sent_headers:
            return ProxyResponseResult(False)
        self.close_connection = True

        accumulator = StreamAccumulator()
        usage: dict[str, Any] | None = None
        display_adapter = (
            CursorReasoningDisplayAdapter()
            if self.config.cursor_display_reasoning
            else None
        )
        scope = (
            record_response_scope
            if record_response_scope is not None
            else conversation_scope(request_messages, cache_namespace)
        )
        response_prior_messages = (
            record_response_messages
            if record_response_messages is not None
            else request_messages
        )
        response_contexts = (
            record_response_contexts
            if record_response_contexts is not None
            else [(scope, response_prior_messages)]
        )
        finalized = False
        pending_recovery_notice = recovery_notice
        while True:
            try:
                line = response.readline()
            except (HTTPException, OSError) as exc:
                LOG.warning("upstream streaming response read failed: %s", exc)
                return ProxyResponseResult(False, usage)
            if not line:
                break
            (
                rewritten,
                finalized,
                pending_recovery_notice,
                chunk_usage,
            ) = self._rewrite_sse_line(
                line,
                original_model,
                accumulator,
                cache_namespace,
                response_contexts,
                display_adapter,
                pending_recovery_notice,
                trace,
            )
            if chunk_usage is not None:
                usage = chunk_usage
            if trace is not None:
                trace.record_stream_chunk(line, rewritten)
            if not self._write_to_client(
                rewritten, "sending streaming response chunk", flush=True
            ):
                return ProxyResponseResult(False, usage)
            if finalized:
                break

        if not finalized:
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = sum(
                accumulator.store_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
        return ProxyResponseResult(True, usage)

    def _rewrite_sse_line(
        self,
        line: bytes,
        original_model: str,
        accumulator: StreamAccumulator,
        cache_namespace: str,
        response_contexts: list[tuple[str, list[dict[str, Any]]]],
        display_adapter: CursorReasoningDisplayAdapter | None,
        recovery_notice: str | None = None,
        trace: TraceRequest | None = None,
    ) -> tuple[bytes, bool, str | None, dict[str, Any] | None]:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line, False, recovery_notice, None

        data = stripped[len(b"data:") :].strip()
        if data == b"[DONE]":
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = sum(
                accumulator.store_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            prefix = b""
            if display_adapter is None:
                if recovery_notice:
                    prefix += sse_data(
                        recovery_notice_chunk(original_model, recovery_notice)
                    )
                return prefix + b"data: [DONE]\n\n", True, None, None
            closing_chunk = display_adapter.flush_chunk(original_model)
            if closing_chunk is not None:
                prefix += sse_data(closing_chunk)
            if recovery_notice:
                prefix += sse_data(
                    recovery_notice_chunk(original_model, recovery_notice)
                )
            return prefix + b"data: [DONE]\n\n", True, None, None

        try:
            chunk = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line, False, recovery_notice, None

        if isinstance(chunk, dict):
            if recovery_notice and inject_recovery_notice(chunk, recovery_notice):
                recovery_notice = None
            accumulator.ingest_chunk(chunk)
            stored = sum(
                accumulator.store_ready_reasoning(
                    self.reasoning_store,
                    scope,
                    cache_namespace,
                    prior_messages,
                )
                for scope, prior_messages in response_contexts
            )
            if self.config.verbose and stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            chunk_usage = chunk.get("usage")
            if trace is not None:
                trace.record_usage(chunk_usage)
            if display_adapter is not None:
                display_adapter.rewrite_chunk(chunk)
            if "model" in chunk:
                chunk["model"] = original_model
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
            return (
                (
                    b"data: "
                    + json.dumps(
                        chunk, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    + ending
                ),
                False,
                recovery_notice,
                chunk_usage if isinstance(chunk_usage, dict) else None,
            )
        return line, False, recovery_notice, None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local DeepSeek Cursor proxy")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=f"YAML config file, default {default_config_path()}",
    )
    parser.add_argument("--host", help="Bind host, default from config or 127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port, default from config or 9000",
    )
    parser.add_argument(
        "--model",
        help=(
            "Fallback DeepSeek model when the request has no model, "
            "default from config or deepseek-v4-pro"
        ),
    )
    parser.add_argument(
        "--base-url",
        help=("DeepSeek base URL, default from config or https://api.deepseek.com"),
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled", "pass-through"],
        help="DeepSeek thinking mode, default from config or enabled",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max", "xhigh"],
        help="DeepSeek reasoning effort, default from config or high",
    )
    parser.add_argument(
        "--reasoning-content-path",
        type=Path,
        help=(
            "SQLite reasoning_content cache path, "
            f"default {default_reasoning_content_path()}"
        ),
    )
    parser.add_argument(
        "--ngrok",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start an ngrok tunnel and print the Cursor base URL",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log detailed request lifecycle metadata and full payloads",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Write full structured request traces to this directory",
    )
    parser.add_argument(
        "--display-reasoning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror reasoning_content into Cursor-visible <think> content",
    )
    parser.add_argument(
        "--cors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send permissive CORS headers",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="Upstream request timeout in seconds, default from config or 300",
    )
    parser.add_argument(
        "--max-request-body-bytes",
        type=int,
        help="Maximum accepted request body size, default from config",
    )
    parser.add_argument(
        "--reasoning-cache-max-age-seconds",
        type=int,
        help="Maximum reasoning cache row age in seconds, default from config",
    )
    parser.add_argument(
        "--reasoning-cache-max-rows",
        type=int,
        help="Maximum reasoning cache rows, default from config",
    )
    parser.add_argument(
        "--missing-reasoning-strategy",
        choices=["recover", "reject"],
        help=(
            "What to do when required reasoning_content is missing: "
            "recover (friendly default) or reject (strict debugging mode)"
        ),
    )
    parser.add_argument(
        "--clear-reasoning-cache",
        action="store_true",
        help="Clear the local reasoning_content SQLite cache and exit",
    )
    return parser


def elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def log_json(label: str, payload: Any) -> None:
    LOG.info(
        "%s:\n%s",
        label,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )


def log_bytes(label: str, body: bytes) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        LOG.info("%s:\n%s", label, body.decode("utf-8", errors="replace"))
        return
    log_json(label, payload)


def usage_from_body(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            return usage
    return None


def log_request_lifecycle(
    *,
    request_id: int,
    cursor_payload: dict[str, Any],
    prepared: PreparedRequest,
    usage: dict[str, Any] | None,
) -> None:
    block = request_lifecycle_block(
        request_id=request_id,
        cursor_payload=cursor_payload,
        prepared=prepared,
        usage=usage,
    )
    if prepared.recovered_reasoning_messages:
        LOG.warning("%s", block)
    else:
        LOG.info("%s", block)


def request_lifecycle_block(
    *,
    request_id: int,
    cursor_payload: dict[str, Any],
    prepared: PreparedRequest,
    usage: dict[str, Any] | None,
) -> str:
    cursor_messages = message_count(cursor_payload)
    cursor_tools = tool_count(cursor_payload)
    upstream_messages = message_count(prepared.payload)
    upstream_tools = tool_count(prepared.payload)
    status = "recovered" if prepared.recovered_reasoning_messages else "ok"

    return "\n".join(
        [
            (
                "┌ cursor   "
                f"id={request_id} model={prepared.original_model} "
                f"messages={format_count(cursor_messages)} "
                f"tools={format_count(cursor_tools)}"
            ),
            (
                "├ context  "
                f"filled={format_count(prepared.patched_reasoning_messages)} "
                f"missing={format_count(prepared.missing_reasoning_messages)} "
                f"recovered={format_count(prepared.recovered_reasoning_messages)} "
                f"dropped={format_count(prepared.recovery_dropped_messages)} "
                f"status={status}"
            ),
            (
                "├ send     "
                f"user_msgs={format_count(user_message_count(prepared.payload))} "
                f"messages={format_count(upstream_messages)} "
                f"tools={format_count(upstream_tools)} "
                f"reasoning_content={format_count(reasoning_content_count(prepared.payload))}"
            ),
            (
                "└ stats    "
                f"prompt={format_usage_count(usage, 'prompt_tokens')} "
                f"output={format_usage_count(usage, 'completion_tokens')} "
                f"reasoning={format_count(reasoning_token_count(usage))} "
                f"cache_hit={cache_hit_rate(usage)}"
            ),
        ]
    )


def message_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def tool_count(payload: dict[str, Any]) -> int:
    tools = payload.get("tools")
    return len(tools) if isinstance(tools, list) else 0


def user_message_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )


def reasoning_content_count(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("reasoning_content"), str)
    )


def format_usage_count(usage: dict[str, Any] | None, key: str) -> str:
    if not isinstance(usage, dict):
        return "?"
    return format_count(usage.get(key))


def reasoning_token_count(usage: dict[str, Any] | None) -> Any:
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    return details.get("reasoning_tokens")


def cache_hit_rate(usage: dict[str, Any] | None) -> str:
    if not isinstance(usage, dict):
        return "?"
    hit_tokens = usage.get("prompt_cache_hit_tokens")
    miss_tokens = usage.get("prompt_cache_miss_tokens")
    if hit_tokens is None and miss_tokens is None:
        return "?"
    hit = int_or_zero(hit_tokens)
    miss = int_or_zero(miss_tokens)
    total = hit + miss
    if not total:
        return "?"
    return f"{hit / total:.1%}"


def format_count(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def sse_data(payload: dict[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n\n"
    )


def inject_recovery_notice(chunk: dict[str, Any], notice: str) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if "content" not in delta and not delta.get("tool_calls"):
            continue
        existing_content = delta.get("content")
        delta["content"] = notice + (
            existing_content if isinstance(existing_content, str) else ""
        )
        return True
    return False


def recovery_notice_chunk(
    model: str,
    notice: str = RECOVERY_NOTICE_CONTENT,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-deepseek-cursor-proxy-recovery",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": notice},
                "finish_reason": None,
            }
        ],
    }


def summarize_chat_payload(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    tools = payload.get("tools")
    functions = payload.get("functions")
    return (
        f"model={payload.get('model')!r} "
        f"stream={bool(payload.get('stream'))} "
        f"messages={len(messages) if isinstance(messages, list) else 0} "
        f"tools={len(tools) if isinstance(tools, list) else 0} "
        f"functions={len(functions) if isinstance(functions, list) else 0} "
        f"tool_choice={payload.get('tool_choice')!r}"
    )


def read_response_body(response: Any) -> bytes:
    body = response.read()
    encoding = (response.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip":
        return gzip.decompress(body)
    if encoding == "deflate":
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)
    return body


def response_headers(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", {})
    if hasattr(headers, "items"):
        return {str(name): str(value) for name, value in headers.items()}
    return {}


def warn_if_insecure_upstream(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return
    host = parsed.hostname or ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        return
    LOG.warning("upstream base_url uses plain HTTP; bearer tokens may be exposed")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_arg_parser().parse_args(argv)
    try:
        config = ProxyConfig.from_file(config_path=args.config_path)
    except ValueError as exc:
        LOG.error("%s", exc)
        return 2
    updates: dict[str, Any] = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.model is not None:
        updates["upstream_model"] = args.model
    if args.base_url is not None:
        updates["upstream_base_url"] = args.base_url.rstrip("/")
    if args.thinking is not None:
        updates["thinking"] = args.thinking
    if args.reasoning_effort is not None:
        updates["reasoning_effort"] = args.reasoning_effort
    if args.reasoning_content_path is not None:
        updates["reasoning_content_path"] = args.reasoning_content_path
    if args.ngrok is not None:
        updates["ngrok"] = args.ngrok
    if args.verbose is not None:
        updates["verbose"] = args.verbose
    if args.trace_dir is not None:
        updates["trace_dir"] = args.trace_dir
    if args.display_reasoning is not None:
        updates["cursor_display_reasoning"] = args.display_reasoning
    if args.cors is not None:
        updates["cors"] = args.cors
    if args.request_timeout is not None:
        updates["request_timeout"] = args.request_timeout
    if args.max_request_body_bytes is not None:
        updates["max_request_body_bytes"] = args.max_request_body_bytes
    if args.reasoning_cache_max_age_seconds is not None:
        updates["reasoning_cache_max_age_seconds"] = (
            args.reasoning_cache_max_age_seconds
        )
    if args.reasoning_cache_max_rows is not None:
        updates["reasoning_cache_max_rows"] = args.reasoning_cache_max_rows
    if args.missing_reasoning_strategy is not None:
        updates["missing_reasoning_strategy"] = args.missing_reasoning_strategy
    if updates:
        config = replace(config, **updates)

    warn_if_insecure_upstream(config.upstream_base_url)
    store = ReasoningStore(
        config.reasoning_content_path,
        max_age_seconds=config.reasoning_cache_max_age_seconds,
        max_rows=config.reasoning_cache_max_rows,
    )
    if args.clear_reasoning_cache:
        deleted = store.clear()
        LOG.info("cleared %s reasoning cache row(s)", deleted)
        store.close()
        return 0
    trace_writer: TraceWriter | None = None
    if config.trace_dir is not None:
        try:
            trace_writer = TraceWriter(config.trace_dir)
        except OSError as exc:
            LOG.error("failed to initialize trace directory: %s", exc)
            store.close()
            return 2
    server = DeepSeekProxyServer((config.host, config.port), DeepSeekProxyHandler)
    server.config = config
    server.reasoning_store = store
    server.trace_writer = trace_writer

    LOG.info("listening on http://%s:%s/v1", config.host, config.port)
    LOG.info(
        "forwarding to %s/chat/completions default_model=%s",
        config.upstream_base_url,
        config.upstream_model,
    )
    LOG.info(
        (
            "thinking=%s reasoning_effort=%s cursor_display_reasoning=%s "
            "missing_reasoning_strategy=%s reasoning_content_path=%s"
        ),
        config.thinking,
        config.reasoning_effort,
        config.cursor_display_reasoning,
        config.missing_reasoning_strategy,
        config.reasoning_content_path,
    )
    if config.verbose:
        LOG.info("logging mode=verbose metadata=detailed bodies=true")
        LOG.warning(
            "verbose logging enabled; prompts and code may be written to stdout"
        )
    else:
        LOG.info("logging mode=normal metadata=safe_summaries bodies=false")
    if trace_writer is not None:
        LOG.info("trace session directory: %s", trace_writer.session_dir)
        LOG.warning("trace logging enabled; prompts and code will be written to disk")

    tunnel: NgrokTunnel | None = None
    if config.ngrok:
        target_url = local_tunnel_target(config.host, config.port)
        tunnel = NgrokTunnel(target_url)
        try:
            public_url = tunnel.start()
        except RuntimeError as exc:
            LOG.error("%s", exc)
            server.server_close()
            store.close()
            return 2
        LOG.info("ngrok tunnel forwarding %s -> %s", public_url, target_url)
        LOG.info("Cursor Base URL: %s/v1", public_url.rstrip("/"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutting down")
    finally:
        if tunnel is not None:
            tunnel.stop()
        server.server_close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
from http.client import HTTPException
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
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
from .tunnel import NgrokTunnel, local_tunnel_target
from .transform import (
    RECOVERY_NOTICE_CONTENT,
    prepare_upstream_request,
    rewrite_response_body,
)


LOG = logging.getLogger("deepseek_cursor_proxy")


class RequestBodyTooLarge(ValueError):
    pass


class DeepSeekProxyServer(ThreadingHTTPServer):
    config: ProxyConfig
    reasoning_store: ReasoningStore


class DeepSeekProxyHandler(BaseHTTPRequestHandler):
    server_version = "DeepSeekPythonProxy/0.1"

    @property
    def config(self) -> ProxyConfig:
        return self.server.config  # type: ignore[return-value]

    @property
    def reasoning_store(self) -> ReasoningStore:
        return self.server.reasoning_store  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

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
        request_path = urlparse(self.path).path
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
                404, {"error": {"message": "Only /v1/chat/completions is supported"}}
            )
            return
        cursor_authorization = self._cursor_authorization()
        if cursor_authorization is None:
            LOG.warning(
                "rejected request path=%s status=401 reason=missing_bearer_token",
                request_path,
            )
            self._send_json(
                401, {"error": {"message": "Missing Authorization bearer token"}}
            )
            return

        try:
            payload = self._read_json_body()
        except RequestBodyTooLarge as exc:
            LOG.warning(
                "rejected request path=%s status=413 reason=%s", request_path, exc
            )
            self._send_json(413, {"error": {"message": str(exc)}})
            return
        except ValueError as exc:
            LOG.warning(
                "rejected request path=%s status=400 reason=%s", request_path, exc
            )
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        if self.config.verbose:
            log_json("cursor request body", payload)

        LOG.info("cursor request: %s", summarize_chat_payload(payload))

        prepared = prepare_upstream_request(
            payload,
            self.config,
            self.reasoning_store,
            authorization=cursor_authorization,
        )
        if prepared.patched_reasoning_messages:
            LOG.info(
                "restored reasoning_content on %s assistant message(s)",
                prepared.patched_reasoning_messages,
            )
        if prepared.recovered_reasoning_messages:
            if prepared.recovery_notice:
                LOG.warning(
                    (
                        "recovered request because cached reasoning_content was "
                        "unavailable for %s assistant message(s); omitted %s "
                        "older message(s) from forwarded history and will show "
                        "a Cursor notice"
                    ),
                    prepared.recovered_reasoning_messages,
                    prepared.recovery_dropped_messages,
                )
            else:
                LOG.info(
                    (
                        "continued recovered request; omitted %s old message(s) "
                        "before the prior recovery boundary"
                    ),
                    prepared.recovery_dropped_messages,
                )
        if prepared.missing_reasoning_messages:
            LOG.warning(
                "rejected request path=%s status=409 reason=missing_reasoning_content count=%s",
                request_path,
                prepared.missing_reasoning_messages,
            )
            self._send_json(
                409,
                {
                    "error": {
                        "message": (
                            "Missing cached DeepSeek reasoning_content for a "
                            f"thinking-mode tool-call history on "
                            f"{prepared.missing_reasoning_messages} assistant "
                            "message(s). This usually means the chat has tool-call "
                            "turns that were not captured by this proxy/cache. Start "
                            "a new chat or retry from the original tool-call turn."
                        ),
                        "type": "missing_reasoning_content",
                        "code": "missing_reasoning_content",
                        "missing_reasoning_messages": prepared.missing_reasoning_messages,
                    }
                },
            )
            return
        LOG.info(
            "deepseek send: %s patched=%s recovered=%s",
            compact_request_stats(prepared.payload),
            prepared.patched_reasoning_messages,
            prepared.recovered_reasoning_messages,
        )

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
        request = Request(
            upstream_url,
            data=upstream_body,
            method="POST",
            headers=self._upstream_headers(
                stream=bool(prepared.payload.get("stream")),
                authorization=cursor_authorization,
            ),
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
            self._send_upstream_error(exc)
            return
        except URLError as exc:
            LOG.warning(
                "upstream request failed elapsed_ms=%s reason=%s",
                elapsed_ms(started),
                exc.reason,
            )
            self._send_json(
                502, {"error": {"message": f"Upstream request failed: {exc.reason}"}}
            )
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
                )
            else:
                sent_response = self._proxy_regular_response(
                    response,
                    prepared.original_model,
                    prepared.payload["messages"],
                    prepared.cache_namespace,
                    prepared.recovery_notice,
                )
            if not sent_response:
                return
            LOG.info(
                (
                    "request complete status=%s stream=%s elapsed_ms=%s "
                    "patched_reasoning=%s missing_reasoning=%s recovered_reasoning=%s"
                ),
                upstream_status,
                bool(prepared.payload.get("stream")),
                elapsed_ms(started),
                prepared.patched_reasoning_messages,
                prepared.missing_reasoning_messages,
                prepared.recovered_reasoning_messages,
            )

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

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
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

    def _send_upstream_error(self, exc: HTTPError) -> None:
        body = read_response_body(exc)
        if self.config.verbose:
            log_bytes("upstream error body", body)
        sent_headers = self._send_response_headers(
            exc.code,
            [
                ("Content-Type", exc.headers.get("Content-Type", "application/json")),
                ("Content-Length", str(len(body))),
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
    ) -> bool:
        body = read_response_body(response)
        try:
            body = rewrite_response_body(
                body,
                original_model,
                self.reasoning_store,
                request_messages,
                cache_namespace,
                content_prefix=recovery_notice,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOG.warning("failed to rewrite upstream JSON response: %s", exc)
        log_usage_from_body(body)

        if self.config.verbose:
            log_bytes("cursor response body", body)

        sent_headers = self._send_response_headers(
            getattr(response, "status", 200),
            [
                (
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                ),
                ("Content-Length", str(len(body))),
            ],
            "sending upstream response headers",
        )
        if not sent_headers:
            return False
        return self._write_to_client(body, "sending upstream response body")

    def _proxy_streaming_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
        recovery_notice: str | None = None,
    ) -> bool:
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
            return False
        self.close_connection = True

        accumulator = StreamAccumulator()
        display_adapter = (
            CursorReasoningDisplayAdapter()
            if self.config.cursor_display_reasoning
            else None
        )
        scope = conversation_scope(request_messages, cache_namespace)
        finalized = False
        pending_recovery_notice = recovery_notice
        while True:
            try:
                line = response.readline()
            except (HTTPException, OSError) as exc:
                LOG.warning("upstream streaming response read failed: %s", exc)
                return False
            if not line:
                break
            rewritten, finalized, pending_recovery_notice = self._rewrite_sse_line(
                line,
                original_model,
                accumulator,
                scope,
                display_adapter,
                pending_recovery_notice,
            )
            if not self._write_to_client(
                rewritten, "sending streaming response chunk", flush=True
            ):
                return False
            if finalized:
                break

        if not finalized:
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = accumulator.store_reasoning(self.reasoning_store, scope)
            if stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
        return True

    def _rewrite_sse_line(
        self,
        line: bytes,
        original_model: str,
        accumulator: StreamAccumulator,
        scope: str,
        display_adapter: CursorReasoningDisplayAdapter | None,
        recovery_notice: str | None = None,
    ) -> tuple[bytes, bool, str | None]:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line, False, recovery_notice

        data = stripped[len(b"data:") :].strip()
        if data == b"[DONE]":
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = accumulator.store_reasoning(self.reasoning_store, scope)
            if stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            prefix = b""
            if display_adapter is None:
                if recovery_notice:
                    prefix += sse_data(
                        recovery_notice_chunk(original_model, recovery_notice)
                    )
                return prefix + b"data: [DONE]\n\n", True, None
            closing_chunk = display_adapter.flush_chunk(original_model)
            if closing_chunk is not None:
                prefix += sse_data(closing_chunk)
            if recovery_notice:
                prefix += sse_data(
                    recovery_notice_chunk(original_model, recovery_notice)
                )
            return prefix + b"data: [DONE]\n\n", True, None

        try:
            chunk = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line, False, recovery_notice

        if isinstance(chunk, dict):
            if recovery_notice and inject_recovery_notice(chunk, recovery_notice):
                recovery_notice = None
            accumulator.ingest_chunk(chunk)
            stored = accumulator.store_ready_reasoning(self.reasoning_store, scope)
            if stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            log_usage(chunk.get("usage"))
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
            )
        return line, False, recovery_notice


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local DeepSeek Cursor proxy")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=f"YAML config file, default {default_config_path()}",
    )
    parser.add_argument(
        "--host", help="Bind host, default from config, PROXY_HOST, or 127.0.0.1"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port, default from config, PROXY_PORT, or 9000",
    )
    parser.add_argument(
        "--model",
        help="Fallback DeepSeek model when the request has no model, default from config, DEEPSEEK_MODEL, or deepseek-v4-pro",
    )
    parser.add_argument(
        "--base-url",
        help="DeepSeek base URL, default from config, DEEPSEEK_BASE_URL, or https://api.deepseek.com",
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
        action="store_true",
        help="Start an ngrok tunnel and print the Cursor base URL",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log detailed request lifecycle metadata and full payloads",
    )
    parser.add_argument(
        "--no-cursor-display-reasoning",
        action="store_true",
        help="Do not mirror reasoning_content into Cursor-visible <think> content",
    )
    parser.add_argument(
        "--missing-reasoning-strategy",
        choices=["recover", "reject"],
        help=(
            "What to do when required reasoning_content is missing: "
            "recover (friendly default) or reject (strict)"
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


def log_usage_from_body(body: bytes) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(payload, dict):
        log_usage(payload.get("usage"))


def log_usage(usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    summary = compact_usage_stats(usage)
    if summary is None:
        return
    LOG.info("deepseek usage: %s", summary)


def compact_request_stats(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    tools = payload.get("tools")
    reasoning_count = 0
    reasoning_chars = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            reasoning_count += 1
            reasoning_chars += len(reasoning)
    rounds = sum(
        1
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )
    return (
        f"model={payload.get('model')} stream={int(bool(payload.get('stream')))} "
        f"rounds={rounds} msgs={len(messages)} "
        f"tools={len(tools) if isinstance(tools, list) else 0} "
        f"reasoning={reasoning_count}/{reasoning_chars}ch"
    )


def compact_usage_stats(usage: dict[str, Any]) -> str | None:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    hit_tokens = usage.get("prompt_cache_hit_tokens")
    miss_tokens = usage.get("prompt_cache_miss_tokens")
    details = usage.get("completion_tokens_details")
    reasoning_tokens = None
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")

    if all(
        value is None
        for value in (
            prompt_tokens,
            completion_tokens,
            total_tokens,
            hit_tokens,
            miss_tokens,
            reasoning_tokens,
        )
    ):
        return None

    cache_summary = "cache=?"
    if hit_tokens is not None or miss_tokens is not None:
        hit = int_or_zero(hit_tokens)
        miss = int_or_zero(miss_tokens)
        cache_total = hit + miss
        if cache_total:
            cache_summary = f"cache={hit}/{miss} hit={hit / cache_total:.1%}"
        else:
            cache_summary = f"cache={hit}/{miss}"

    return (
        f"prompt={prompt_tokens if prompt_tokens is not None else '?'} "
        f"completion={completion_tokens if completion_tokens is not None else '?'} "
        f"total={total_tokens if total_tokens is not None else '?'} "
        f"{cache_summary} "
        f"reasoning={reasoning_tokens if reasoning_tokens is not None else '?'}"
    )


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
    if args.host:
        updates["host"] = args.host
    if args.port:
        updates["port"] = args.port
    if args.model:
        updates["upstream_model"] = args.model
    if args.base_url:
        updates["upstream_base_url"] = args.base_url.rstrip("/")
    if args.reasoning_content_path:
        updates["reasoning_content_path"] = args.reasoning_content_path
    if args.ngrok:
        updates["ngrok"] = True
    if args.verbose:
        updates["verbose"] = True
    if args.no_cursor_display_reasoning:
        updates["cursor_display_reasoning"] = False
    if args.missing_reasoning_strategy:
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
    server = DeepSeekProxyServer((config.host, config.port), DeepSeekProxyHandler)
    server.config = config
    server.reasoning_store = store

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

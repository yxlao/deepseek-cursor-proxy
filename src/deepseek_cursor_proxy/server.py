from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
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
from .transform import prepare_upstream_request, rewrite_response_body


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
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

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
                            "thinking-mode tool-call history. Retry the tool-call "
                            "turn so the proxy can capture the original reasoning."
                        ),
                        "type": "missing_reasoning_content",
                        "code": "missing_reasoning_content",
                    }
                },
            )
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
                self._proxy_streaming_response(
                    response,
                    prepared.original_model,
                    prepared.payload["messages"],
                    prepared.cache_namespace,
                )
            else:
                self._proxy_regular_response(
                    response,
                    prepared.original_model,
                    prepared.payload["messages"],
                    prepared.cache_namespace,
                )
            LOG.info(
                (
                    "request complete status=%s stream=%s elapsed_ms=%s "
                    "patched_reasoning=%s missing_reasoning=%s"
                ),
                upstream_status,
                bool(prepared.payload.get("stream")),
                elapsed_ms(started),
                prepared.patched_reasoning_messages,
                prepared.missing_reasoning_messages,
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
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        self.send_response(exc.code)
        self._send_cors_headers()
        self.send_header(
            "Content-Type", exc.headers.get("Content-Type", "application/json")
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_regular_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
    ) -> None:
        body = read_response_body(response)
        try:
            body = rewrite_response_body(
                body,
                original_model,
                self.reasoning_store,
                request_messages,
                cache_namespace,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOG.warning("failed to rewrite upstream JSON response: %s", exc)
        log_cache_usage_from_body(body)

        if self.config.verbose:
            log_bytes("cursor response body", body)

        self.send_response(getattr(response, "status", 200))
        self._send_cors_headers()
        self.send_header(
            "Content-Type", response.headers.get("Content-Type", "application/json")
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_streaming_response(
        self,
        response: Any,
        original_model: str,
        request_messages: list[dict[str, Any]],
        cache_namespace: str,
    ) -> None:
        self.send_response(getattr(response, "status", 200))
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        accumulator = StreamAccumulator()
        display_adapter = (
            CursorReasoningDisplayAdapter()
            if self.config.cursor_display_reasoning
            else None
        )
        scope = conversation_scope(request_messages, cache_namespace)
        finalized = False
        while True:
            line = response.readline()
            if not line:
                break
            rewritten, finalized = self._rewrite_sse_line(
                line, original_model, accumulator, scope, display_adapter
            )
            self.wfile.write(rewritten)
            self.wfile.flush()
            if finalized:
                break

        if not finalized:
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = accumulator.store_reasoning(self.reasoning_store, scope)
            if stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)

    def _rewrite_sse_line(
        self,
        line: bytes,
        original_model: str,
        accumulator: StreamAccumulator,
        scope: str,
        display_adapter: CursorReasoningDisplayAdapter | None,
    ) -> tuple[bytes, bool]:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line, False

        data = stripped[len(b"data:") :].strip()
        if data == b"[DONE]":
            if self.config.verbose:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = accumulator.store_reasoning(self.reasoning_store, scope)
            if stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            if display_adapter is None:
                return b"data: [DONE]\n\n", True
            closing_chunk = display_adapter.flush_chunk(original_model)
            if closing_chunk is None:
                return b"data: [DONE]\n\n", True
            return sse_data(closing_chunk) + b"data: [DONE]\n\n", True

        try:
            chunk = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line, False

        if isinstance(chunk, dict):
            accumulator.ingest_chunk(chunk)
            stored = accumulator.store_finished_reasoning(self.reasoning_store, scope)
            if stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            log_cache_usage(chunk.get("usage"))
            if display_adapter is not None:
                display_adapter.rewrite_chunk(chunk)
            if "model" in chunk:
                chunk["model"] = original_model
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
            return (
                b"data: "
                + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                + ending
            ), False
        return line, False


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


def log_cache_usage_from_body(body: bytes) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if isinstance(payload, dict):
        log_cache_usage(payload.get("usage"))


def log_cache_usage(usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        return
    LOG.info("deepseek prompt cache: hit_tokens=%s miss_tokens=%s", hit, miss)


def sse_data(payload: dict[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n\n"
    )


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
        "thinking=%s reasoning_effort=%s cursor_display_reasoning=%s reasoning_content_path=%s",
        config.thinking,
        config.reasoning_effort,
        config.cursor_display_reasoning,
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

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
from .streaming import StreamAccumulator
from .tunnel import NgrokTunnel, local_tunnel_target
from .transform import prepare_upstream_request, rewrite_response_body


LOG = logging.getLogger("deepseek_cursor_proxy")


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
            self._send_json(
                404, {"error": {"message": "Only /v1/chat/completions is supported"}}
            )
            return
        if not self._authorized():
            self._send_json(
                401, {"error": {"message": "Missing or invalid proxy API key"}}
            )
            return

        try:
            self.config.validate()
        except ValueError as exc:
            self._send_json(500, {"error": {"message": str(exc)}})
            return

        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        if self.config.log_bodies:
            log_json("cursor request body", payload)

        if self.config.verbose:
            LOG.info("cursor request: %s", summarize_chat_payload(payload))

        prepared = prepare_upstream_request(payload, self.config, self.reasoning_store)
        if prepared.patched_reasoning_messages:
            LOG.info(
                "restored reasoning_content on %s assistant message(s)",
                prepared.patched_reasoning_messages,
            )
        if prepared.fallback_reasoning_messages:
            LOG.warning(
                "added compatibility reasoning_content placeholder on %s uncached assistant message(s)",
                prepared.fallback_reasoning_messages,
            )

        if self.config.verbose:
            LOG.info(
                (
                    "upstream request metadata: original_model=%s upstream_model=%s "
                    "patched_reasoning=%s fallback_reasoning=%s %s"
                ),
                prepared.original_model,
                prepared.upstream_model,
                prepared.patched_reasoning_messages,
                prepared.fallback_reasoning_messages,
                summarize_chat_payload(prepared.payload),
            )

        if self.config.log_bodies:
            log_json("upstream request body", prepared.payload)

        upstream_body = json.dumps(
            prepared.payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        upstream_url = f"{self.config.upstream_base_url}/chat/completions"
        request = Request(
            upstream_url,
            data=upstream_body,
            method="POST",
            headers=self._upstream_headers(stream=bool(prepared.payload.get("stream"))),
        )

        try:
            if self.config.verbose:
                LOG.info("forwarding to %s", upstream_url)
            response = urlopen(request, timeout=self.config.request_timeout)
        except HTTPError as exc:
            if self.config.verbose:
                LOG.info(
                    "upstream error status=%s elapsed_ms=%s",
                    exc.code,
                    elapsed_ms(started),
                )
            self._send_upstream_error(exc)
            return
        except URLError as exc:
            if self.config.verbose:
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
            if self.config.verbose:
                LOG.info(
                    "upstream response status=%s stream=%s elapsed_ms=%s",
                    getattr(response, "status", 200),
                    bool(prepared.payload.get("stream")),
                    elapsed_ms(started),
                )
            if prepared.payload.get("stream"):
                self._proxy_streaming_response(
                    response, prepared.original_model, prepared.payload["messages"]
                )
            else:
                self._proxy_regular_response(
                    response, prepared.original_model, prepared.payload["messages"]
                )

    def _authorized(self) -> bool:
        expected = self.config.proxy_api_key
        if expected is None:
            return True
        auth_header = self.headers.get("Authorization", "")
        return auth_header == f"Bearer {expected}"

    def _send_cors_headers(self) -> None:
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
        seen: set[str] = set()
        models = []
        for model_id in (self.config.upstream_model, *self.config.model_list):
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "deepseek",
                }
            )
        self._send_json(200, {"object": "list", "data": models})

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
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

    def _upstream_headers(self, stream: bool) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.upstream_api_key}",
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
        if self.config.log_bodies:
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
    ) -> None:
        body = read_response_body(response)
        try:
            body = rewrite_response_body(
                body, original_model, self.reasoning_store, request_messages
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOG.warning("failed to rewrite upstream JSON response: %s", exc)

        if self.config.log_bodies:
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
    ) -> None:
        self.send_response(getattr(response, "status", 200))
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        accumulator = StreamAccumulator()
        scope = conversation_scope(request_messages)
        finalized = False
        while True:
            line = response.readline()
            if not line:
                break
            rewritten = self._rewrite_sse_line(line, original_model, accumulator, scope)
            if rewritten is None:
                finalized = True
                rewritten = b"data: [DONE]\n\n"
            self.wfile.write(rewritten)
            self.wfile.flush()
            if finalized:
                break

        if not finalized:
            if self.config.log_bodies:
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
    ) -> bytes | None:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return line

        data = stripped[len(b"data:") :].strip()
        if data == b"[DONE]":
            if self.config.log_bodies:
                log_json("model streaming assistant messages", accumulator.messages())
            stored = accumulator.store_reasoning(self.reasoning_store, scope)
            if stored:
                LOG.info("stored %s streaming reasoning cache key(s)", stored)
            return None

        try:
            chunk = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return line

        if isinstance(chunk, dict):
            accumulator.ingest_chunk(chunk)
            if "model" in chunk:
                chunk["model"] = original_model
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n"
            return (
                b"data: "
                + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                + ending
            )
        return line


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local DeepSeek Cursor proxy")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=f"Env config file, default {default_config_path()}",
    )
    parser.add_argument(
        "--host", help="Bind host, default from PROXY_HOST or 127.0.0.1"
    )
    parser.add_argument(
        "--port", type=int, help="Bind port, default from PROXY_PORT or 9000"
    )
    parser.add_argument(
        "--model", help="Upstream DeepSeek model, default from DEEPSEEK_MODEL"
    )
    parser.add_argument(
        "--base-url", help="DeepSeek base URL, default https://api.deepseek.com"
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
        help="Log request lifecycle metadata without bodies",
    )
    parser.add_argument(
        "--log-bodies",
        action="store_true",
        help="Log normalized upstream request bodies",
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_arg_parser().parse_args(argv)
    config = ProxyConfig.from_env(env_file_path=args.config_path)
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
    if args.log_bodies:
        updates["log_bodies"] = True
    if updates:
        config = replace(config, **updates)

    try:
        config.validate()
    except ValueError as exc:
        LOG.error("%s", exc)
        return 2

    store = ReasoningStore(config.reasoning_content_path)
    server = DeepSeekProxyServer((config.host, config.port), DeepSeekProxyHandler)
    server.config = config
    server.reasoning_store = store

    LOG.info("listening on http://%s:%s/v1", config.host, config.port)
    LOG.info(
        "forwarding to %s/chat/completions as %s",
        config.upstream_base_url,
        config.upstream_model,
    )
    LOG.info(
        "thinking=%s reasoning_effort=%s reasoning_content_path=%s",
        config.thinking,
        config.reasoning_effort,
        config.reasoning_content_path,
    )
    if config.verbose:
        LOG.info("verbose logging enabled")
    if config.log_bodies:
        LOG.warning(
            "request body logging enabled; prompts and code may be written to stdout"
        )

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
        if config.proxy_api_key:
            LOG.info("Cursor API key: value of PROXY_API_KEY")
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

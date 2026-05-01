"""Server boundary, CLI, and operational tests.

Pure helper tests (gzip, summarize) and stub-handler tests (client
disconnect) live near the top. The bottom of the file boots a real proxy +
tiny upstream to exercise things that need the HTTP layer: bearer token
forwarding, oversized body, missing-bearer rejection, logging modes, and
streaming connection close.
"""

from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import gzip
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
import zlib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import (
    DeepSeekProxyHandler,
    DeepSeekProxyServer,
    build_arg_parser,
    read_response_body,
    summarize_chat_payload,
)


# ---------------------------------------------------------------------------
# Stubs for fast in-process tests of internal handler methods
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes, encoding: str = "", status: int = 200) -> None:
        self._body = BytesIO(body)
        self.headers = {"Content-Encoding": encoding} if encoding else {}
        self.status = status

    def read(self) -> bytes:
        return self._body.read()


class _FakeStreamingResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self.readline_calls = 0

    def readline(self) -> bytes:
        self.readline_calls += 1
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FailingStreamingResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def readline(self) -> bytes:
        raise OSError("record layer failure")


class _BrokenPipeWfile:
    def write(self, body: bytes) -> None:
        raise BrokenPipeError("test disconnect")

    def flush(self) -> None:
        raise BrokenPipeError("test disconnect")


def _make_handler_stub(wfile: object, **config: object) -> DeepSeekProxyHandler:
    handler = object.__new__(DeepSeekProxyHandler)
    handler.server = SimpleNamespace(
        config=ProxyConfig(**config),
        reasoning_store=ReasoningStore(":memory:"),
    )
    handler.wfile = wfile
    handler.close_connection = False
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    return handler


# ---------------------------------------------------------------------------
# CLI / pure helpers
# ---------------------------------------------------------------------------


class CliAndHelperTests(unittest.TestCase):
    def test_cli_boolean_flags_have_on_and_off_forms(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--no-ngrok",
                "--no-verbose",
                "--no-display-reasoning",
                "--no-collasible-resoning",
                "--cors",
                "--trace-dir",
                "/tmp/dcp-traces",
            ]
        )
        self.assertFalse(args.ngrok)
        self.assertFalse(args.verbose)
        self.assertFalse(args.display_reasoning)
        self.assertFalse(args.collapsible_reasoning)
        self.assertTrue(args.cors)
        self.assertEqual(args.trace_dir, Path("/tmp/dcp-traces"))

    def test_read_response_body_decodes_gzip_and_deflate(self) -> None:
        self.assertEqual(
            read_response_body(_FakeResponse(gzip.compress(b'{"ok":1}'), "gzip")),
            b'{"ok":1}',
        )
        self.assertEqual(
            read_response_body(_FakeResponse(zlib.compress(b'{"ok":1}'), "deflate")),
            b'{"ok":1}',
        )

    def test_summarize_chat_payload_omits_message_content(self) -> None:
        summary = summarize_chat_payload(
            {
                "model": "deepseek-v4-pro",
                "stream": True,
                "messages": [{"role": "user", "content": "secret prompt"}],
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
            }
        )
        self.assertIn("model='deepseek-v4-pro'", summary)
        self.assertIn("messages=1", summary)
        self.assertNotIn("secret prompt", summary)


# ---------------------------------------------------------------------------
# Client-disconnect / upstream-failure stubs (no real HTTP needed)
# ---------------------------------------------------------------------------


class HandlerStubTests(unittest.TestCase):
    def test_regular_response_handles_client_disconnect(self) -> None:
        handler = _make_handler_stub(_BrokenPipeWfile())
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            }
        ).encode("utf-8")
        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                result = handler._proxy_regular_response(
                    _FakeResponse(body),
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "ns",
                )
        finally:
            handler.server.reasoning_store.close()
        self.assertFalse(result.sent)
        self.assertIn("sending upstream response body", "\n".join(captured.output))

    def test_streaming_response_stops_on_client_disconnect(self) -> None:
        handler = _make_handler_stub(_BrokenPipeWfile())
        chunk = {
            "id": "stream",
            "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
        }
        response = _FakeStreamingResponse(
            [
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )
        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                result = handler._proxy_streaming_response(
                    response,
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "ns",
                )
        finally:
            handler.server.reasoning_store.close()
        self.assertFalse(result.sent)
        self.assertEqual(response.readline_calls, 1)
        self.assertIn("sending streaming response chunk", "\n".join(captured.output))

    def test_streaming_response_handles_upstream_read_failure(self) -> None:
        handler = _make_handler_stub(BytesIO())
        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                result = handler._proxy_streaming_response(
                    _FailingStreamingResponse(),
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "ns",
                )
        finally:
            handler.server.reasoning_store.close()
        self.assertFalse(result.sent)
        self.assertIn(
            "upstream streaming response read failed", "\n".join(captured.output)
        )

    def test_collapsible_reasoning_no_effect_when_display_disabled(self) -> None:
        wfile = BytesIO()
        handler = _make_handler_stub(
            wfile, display_reasoning=False, collapsible_reasoning=True
        )
        chunk = {
            "id": "stream",
            "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "delta": {"reasoning_content": "Need context."}}],
        }
        response = _FakeStreamingResponse(
            [
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )
        try:
            handler._proxy_streaming_response(
                response,
                "deepseek-v4-pro",
                [{"role": "user", "content": "hi"}],
                "ns",
            )
        finally:
            handler.server.reasoning_store.close()
        body = wfile.getvalue().decode("utf-8")
        self.assertIn("reasoning_content", body)
        self.assertNotIn("<details>", body)


# ---------------------------------------------------------------------------
# HTTP-level boundary tests: real proxy + tiny upstream
# ---------------------------------------------------------------------------


class _PlainFakeUpstream(BaseHTTPRequestHandler):
    """Returns a fixed plain response and records every request."""

    requests: list[dict[str, object]] = []
    auth_headers: list[str] = []
    delay_after_done: float = 0.0
    response: dict[str, object] = {}

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)
        self.__class__.auth_headers.append(self.headers.get("Authorization", ""))

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            if self.__class__.delay_after_done:
                time.sleep(self.__class__.delay_after_done)
            return

        body = json.dumps(self.__class__.response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_BASE_RESPONSE: dict[str, object] = {
    "id": "x",
    "object": "chat.completion",
    "created": 1,
    "model": "deepseek-v4-pro",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_cache_hit_tokens": 12,
        "prompt_cache_miss_tokens": 8,
        "completion_tokens_details": {"reasoning_tokens": 3},
    },
}


class _Fixture:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _post(url: str, payload: dict, api_key: str = "sk-test") -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class HttpBoundaryTests(unittest.TestCase):
    """Real-HTTP tests that don't fit the protocol suite: things the proxy
    must do at the HTTP boundary regardless of what DeepSeek answers."""

    def setUp(self) -> None:
        _PlainFakeUpstream.requests = []
        _PlainFakeUpstream.auth_headers = []
        _PlainFakeUpstream.delay_after_done = 0.0
        _PlainFakeUpstream.response = dict(_BASE_RESPONSE)
        self.upstream = _Fixture(
            ThreadingHTTPServer(("127.0.0.1", 0), _PlainFakeUpstream)
        )
        self.store = ReasoningStore(":memory:")
        proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        proxy.config = ProxyConfig(
            upstream_base_url=self.upstream.url,
            upstream_model="deepseek-v4-pro",
            ngrok=False,
        )
        proxy.reasoning_store = self.store
        self.proxy = _Fixture(proxy)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def _request(self) -> dict:
        return {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_rejects_missing_bearer_token(self) -> None:
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(self._request()).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 401)
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_rejects_oversized_request_body(self) -> None:
        self.proxy.server.config = replace(
            self.proxy.server.config, max_request_body_bytes=10
        )
        status, payload = _post(
            f"{self.proxy.url}/v1/chat/completions", self._request()
        )
        self.assertEqual(status, 413)
        self.assertIn("too large", payload["error"]["message"])
        self.assertEqual(_PlainFakeUpstream.requests, [])

    def test_forwards_bearer_token_to_upstream(self) -> None:
        status, _ = _post(
            f"{self.proxy.url}/v1/chat/completions",
            self._request(),
            api_key="sk-from-cursor",
        )
        self.assertEqual(status, 200)
        self.assertEqual(_PlainFakeUpstream.auth_headers[0], "Bearer sk-from-cursor")

    def test_streaming_response_closes_after_done_when_upstream_lingers(
        self,
    ) -> None:
        """Cursor relies on the proxy ending the SSE stream at [DONE], even
        if the upstream socket stays open."""
        _PlainFakeUpstream.delay_after_done = 2.0
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream"}],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )
        started = time.monotonic()
        with urlopen(request, timeout=1) as response:
            body = response.read().decode("utf-8")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertIn("data: [DONE]", body)

    def test_normal_logging_summarizes_without_bodies_or_keys(self) -> None:
        with self.assertLogs("deepseek_cursor_proxy", level="INFO") as captured:
            status, _ = _post(
                f"{self.proxy.url}/v1/chat/completions",
                self._request(),
                api_key="sk-from-cursor",
            )
            # `└ stats` is emitted on the handler thread *after* the response
            # body hits the socket, so the client may return before it lands.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not any(
                "└ stats" in record for record in captured.output
            ):
                time.sleep(0.01)
        output = "\n".join(captured.output)
        self.assertEqual(status, 200)
        # Single-line stage records keep the log readable.
        for marker in ("┌ cursor", "├ context", "├ send", "└ stats"):
            self.assertIn(marker, output)
        self.assertNotIn("hi", output.split("┌ cursor")[1].split("\n")[0])
        self.assertNotIn("sk-from-cursor", output)

    def test_verbose_logging_includes_bodies_but_redacts_api_key(self) -> None:
        self.proxy.server.config = replace(self.proxy.server.config, verbose=True)
        with self.assertLogs("deepseek_cursor_proxy", level="INFO") as captured:
            _post(
                f"{self.proxy.url}/v1/chat/completions",
                self._request(),
                api_key="sk-from-cursor",
            )
        output = "\n".join(captured.output)
        self.assertIn("cursor request body", output)
        self.assertIn("upstream request body", output)
        self.assertNotIn("sk-from-cursor", output)

    def test_healthz_returns_ok(self) -> None:
        with urlopen(f"{self.proxy.url}/healthz", timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["ok"], True)


if __name__ == "__main__":
    unittest.main()

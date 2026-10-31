"""Unit tests for the pooled upstream transport.

The fake upstream speaks HTTP/1.1 and counts accepted connections, so
keep-alive reuse is asserted directly instead of inferred from timings. The
safety tests matter more than the reuse test: a connection handed back to the
pool too early would feed one response to the next request.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer
from deepseek_cursor_proxy.transport import UpstreamConnectionPool

OK_BODY = b'{"ok": true}'
ERROR_BODY = b'{"error": {"message": "invalid api key"}}'
STREAM_BODY = (
    b'data: {"choices": [{"delta": {"content": "one"}}]}\n'
    b"\n"
    b'data: {"choices": [{"delta": {"content": "two"}}]}\n'
    b"\n"
    b"data: [DONE]\n"
    b"\n"
)
COMPLETION_BODY = json.dumps(
    {
        "id": "chatcmpl-test",
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
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
).encode("utf-8")


class KeepAliveUpstream(BaseHTTPRequestHandler):
    """Minimal upstream that keeps connections alive like the real API."""

    protocol_version = "HTTP/1.1"
    connections = 0
    paths: list[str] = []

    def setup(self) -> None:
        type(self).connections += 1
        super().setup()

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/moved":
            self.send_response(307)
            self.send_header("Location", "/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._respond(200, COMPLETION_BODY, "application/json")

    def do_GET(self) -> None:
        type(self).paths.append(self.path)
        if self.path == "/error":
            self._respond(401, ERROR_BODY, "application/json")
        elif self.path == "/stream":
            self._respond(200, STREAM_BODY, "text/event-stream")
        elif self.path == "/close":
            self._respond(200, OK_BODY, "application/json", close=True)
        else:
            self._respond(200, OK_BODY, "application/json")

    def _respond(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        close: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        if close:
            self.close_connection = True


class _UpstreamFixture:
    def __init__(self) -> None:
        KeepAliveUpstream.connections = 0
        KeepAliveUpstream.paths = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), KeepAliveUpstream)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _idle_connections(pool: UpstreamConnectionPool) -> list[Any]:
    """Idle sockets, for tests that need to break one on purpose."""
    return [entry[0] for entries in pool._idle.values() for entry in entries]


class PooledTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = _UpstreamFixture()
        self.pool = UpstreamConnectionPool()

    def tearDown(self) -> None:
        self.pool.close()
        self.upstream.close()

    def test_sequential_requests_reuse_one_connection(self) -> None:
        for _ in range(3):
            with self.pool.request("GET", f"{self.upstream.url}/ok") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), OK_BODY)
        self.assertEqual(KeepAliveUpstream.connections, 1)
        self.assertEqual(self.pool.idle_count(), 1)

    def test_error_response_keeps_body_and_connection(self) -> None:
        with self.assertRaises(HTTPError) as raised:
            self.pool.request("GET", f"{self.upstream.url}/error")
        error = raised.exception
        self.assertEqual(error.code, 401)
        self.assertEqual(error.headers.get("Content-Type"), "application/json")
        self.assertEqual(error.read(), ERROR_BODY)
        error.close()
        # The error body was read to EOF, so the socket is still good.
        self.assertEqual(self.pool.idle_count(), 1)

        with self.pool.request("GET", f"{self.upstream.url}/ok") as response:
            self.assertEqual(response.read(), OK_BODY)
        self.assertEqual(KeepAliveUpstream.connections, 1)

    def test_half_read_response_is_not_reused(self) -> None:
        first_line = b'data: {"choices": [{"delta": {"content": "one"}}]}\n'
        with self.pool.request("GET", f"{self.upstream.url}/stream") as response:
            self.assertEqual(response.readline(), first_line)
            # Leaving the body unread must discard the socket: unread bytes
            # would otherwise be read as the answer to the next request.
        self.assertEqual(self.pool.idle_count(), 0)

        with self.pool.request("GET", f"{self.upstream.url}/ok") as response:
            self.assertEqual(response.read(), OK_BODY)
        self.assertEqual(KeepAliveUpstream.connections, 2)

    def test_streaming_lines_are_read_and_connection_is_reused(self) -> None:
        with self.pool.request("GET", f"{self.upstream.url}/stream") as response:
            lines = [line for line in response]
        self.assertEqual(len(lines), 6)
        self.assertTrue(lines[-2].startswith(b"data: [DONE]"))
        self.assertEqual(self.pool.idle_count(), 1)

        with self.pool.request("GET", f"{self.upstream.url}/ok") as response:
            self.assertEqual(response.read(), OK_BODY)
        self.assertEqual(KeepAliveUpstream.connections, 1)

    def test_server_requesting_close_is_not_pooled(self) -> None:
        with self.pool.request("GET", f"{self.upstream.url}/close") as response:
            self.assertEqual(response.read(), OK_BODY)
        self.assertEqual(self.pool.idle_count(), 0)

        with self.pool.request("GET", f"{self.upstream.url}/ok") as response:
            self.assertEqual(response.read(), OK_BODY)
        self.assertEqual(KeepAliveUpstream.connections, 2)

    def test_stale_pooled_socket_is_retried_once(self) -> None:
        with self.pool.request("GET", f"{self.upstream.url}/ok") as response:
            response.read()
        # The server dropped the idle socket without the client noticing,
        # which is what an expired keep-alive looks like.
        for connection in _idle_connections(self.pool):
            connection.sock.close()

        with self.pool.request("GET", f"{self.upstream.url}/ok") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), OK_BODY)
        self.assertEqual(KeepAliveUpstream.connections, 2)

    def test_failure_on_a_new_connection_is_not_retried(self) -> None:
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()

        with mock.patch.object(
            self.pool, "_connect", wraps=self.pool._connect
        ) as connect:
            with self.assertRaises(URLError):
                self.pool.request("POST", f"http://127.0.0.1:{port}/ok", body=b"{}")
        # Retrying a request that never established a pooled socket could
        # bill the same prompt twice.
        self.assertEqual(connect.call_count, 1)

    def test_pooling_can_be_disabled(self) -> None:
        pool = UpstreamConnectionPool(max_idle_per_host=0)
        try:
            for _ in range(2):
                with pool.request("GET", f"{self.upstream.url}/ok") as response:
                    response.read()
            self.assertEqual(pool.idle_count(), 0)
        finally:
            pool.close()
        self.assertEqual(KeepAliveUpstream.connections, 2)

    def test_idle_connections_expire(self) -> None:
        pool = UpstreamConnectionPool(max_idle_seconds=0.05)
        try:
            with pool.request("GET", f"{self.upstream.url}/ok") as response:
                response.read()
            self.assertEqual(pool.idle_count(), 1)
            time.sleep(0.08)
            with pool.request("GET", f"{self.upstream.url}/ok") as response:
                response.read()
            self.assertEqual(KeepAliveUpstream.connections, 2)
        finally:
            pool.close()

    def test_redirect_is_followed_with_the_same_method(self) -> None:
        # urlopen used to follow redirects; a config pointing at a moved
        # endpoint must keep working, and a POST must stay a POST.
        with self.pool.request(
            "POST", f"{self.upstream.url}/moved", body=b"{}"
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), COMPLETION_BODY)
        self.assertEqual(KeepAliveUpstream.connections, 1)


class ProxyConnectionReuseTests(unittest.TestCase):
    """Two turns through the real proxy share one upstream connection."""

    def setUp(self) -> None:
        self.upstream = _UpstreamFixture()
        self.server = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        self.server.config = ProxyConfig(
            upstream_base_url=self.upstream.url,
            upstream_model="deepseek-v4-pro",
            ngrok=False,
            verbose=False,
            cors=False,
        )
        self.store = ReasoningStore(":memory:")
        self.server.reasoning_store = self.store
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}/v1/chat/completions"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.store.close()
        self.upstream.close()

    def _post(self, question: str) -> dict[str, Any]:
        request = Request(
            self.url,
            data=json.dumps(
                {
                    "model": "deepseek-flash",
                    "messages": [{"role": "user", "content": question}],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_two_turns_share_one_upstream_connection(self) -> None:
        first = self._post("first")
        second = self._post("second")
        self.assertEqual(first["choices"][0]["message"]["content"], "ok")
        self.assertEqual(second["choices"][0]["message"]["content"], "ok")
        self.assertEqual(KeepAliveUpstream.connections, 1)
        self.assertEqual(self.server.upstream_pool.idle_count(), 1)

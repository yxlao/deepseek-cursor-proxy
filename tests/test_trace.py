"""Trace writer tests, both as a unit (writes/redacts files) and integrated
through the proxy (captures real request flow on disk)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import stat
import threading
from tempfile import TemporaryDirectory
import time
import unittest
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer
from deepseek_cursor_proxy.trace import TraceWriter


class TraceWriterUnitTests(unittest.TestCase):
    def test_writes_manifest_and_numbered_request_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            writer = TraceWriter(temp_dir)
            first = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={"User-Agent": "Cursor/1.0"},
            )
            second = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={"User-Agent": "Cursor/1.0"},
            )
            first.finish("completed", http_status=200)
            second.finish("completed", http_status=200)

            self.assertTrue((writer.session_dir / "manifest.json").exists())
            self.assertTrue((writer.session_dir / "request-000001.json").exists())
            self.assertTrue((writer.session_dir / "request-000002.json").exists())
            self.assertEqual(
                stat.S_IMODE(
                    (writer.session_dir / "request-000001.json").stat().st_mode
                ),
                0o600,
            )

    def test_authorization_header_is_redacted(self) -> None:
        with TemporaryDirectory() as temp_dir:
            writer = TraceWriter(temp_dir)
            trace = writer.start_request(
                method="POST",
                path="/v1/chat/completions",
                client_address="127.0.0.1",
                headers={"Authorization": "Bearer sk-secret"},
            )
            trace.finish("completed", http_status=200)
            serialized = trace.path.read_text(encoding="utf-8")
            self.assertNotIn("sk-secret", serialized)
            payload = json.loads(serialized)
            self.assertEqual(
                payload["request"]["headers"]["Authorization"]["present"], True
            )
            self.assertIn("sha256", payload["request"]["headers"]["Authorization"])


# ---------------------------------------------------------------------------
# Integration: trace writer attached to a running proxy.
# ---------------------------------------------------------------------------


class _CannedUpstream(BaseHTTPRequestHandler):
    """Returns a tool-call response for the first POST and a streamed
    reasoning response for the second."""

    requests: list[dict[str, object]] = []

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"id":"s","object":"chat.completion.chunk","choices":'
                b'[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"},'
                b'"finish_reason":null}]}\n\n'
            )
            self.wfile.write(
                b'data: {"id":"s","object":"chat.completion.chunk","choices":'
                b'[{"index":0,"delta":{"content":"answer"},"finish_reason":null}],'
                b'"usage":{"completion_tokens_details":{"reasoning_tokens":1}}}\n\n'
            )
            self.wfile.write(
                b'data: {"id":"s","object":"chat.completion.chunk",'
                b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        body = json.dumps(
            {
                "id": "tool",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "I need the date.",
                            "tool_calls": [
                                {
                                    "id": "call_date",
                                    "type": "function",
                                    "function": {
                                        "name": "get_date",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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


def _read_single_trace(session_dir: Path) -> dict:
    deadline = time.monotonic() + 2
    files = sorted(session_dir.glob("request-*.json"))
    while not files and time.monotonic() < deadline:
        time.sleep(0.01)
        files = sorted(session_dir.glob("request-*.json"))
    if len(files) != 1:
        raise AssertionError(f"expected one trace, found {files}")
    return json.loads(files[0].read_text(encoding="utf-8"))


class TraceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _CannedUpstream.requests = []
        self.upstream = _Fixture(ThreadingHTTPServer(("127.0.0.1", 0), _CannedUpstream))
        self.store = ReasoningStore(":memory:")
        self.temp_dir = TemporaryDirectory()
        self.writer = TraceWriter(self.temp_dir.name)
        proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        proxy.config = ProxyConfig(
            upstream_base_url=self.upstream.url,
            upstream_model="deepseek-v4-pro",
            ngrok=False,
        )
        proxy.reasoning_store = self.store
        proxy.trace_writer = self.writer
        self.proxy = _Fixture(proxy)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()
        self.temp_dir.cleanup()

    def _post(self, payload: dict) -> dict:
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-from-cursor",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def test_captures_non_streaming_replay_without_api_key(self) -> None:
        self._post(
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "What is tomorrow's date?"}],
            }
        )
        trace = _read_single_trace(self.writer.session_dir)
        serialized = json.dumps(trace)
        self.assertEqual(trace["completion"]["status"], "completed")
        self.assertEqual(
            trace["request"]["body"]["messages"][0]["content"],
            "What is tomorrow's date?",
        )
        self.assertEqual(
            trace["upstream"]["response"]["body"]["json"]["choices"][0]["message"][
                "reasoning_content"
            ],
            "I need the date.",
        )
        self.assertNotIn("sk-from-cursor", serialized)

    def test_captures_streaming_replay_chunks(self) -> None:
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
        with urlopen(request, timeout=2) as response:
            response.read()
        trace = _read_single_trace(self.writer.session_dir)
        self.assertEqual(trace["completion"]["status"], "completed")
        self.assertIn(
            "reasoning_content",
            trace["upstream"]["stream"]["chunks"][0]["line"],
        )
        self.assertIn(
            "<details>", trace["cursor_response"]["stream"]["chunks"][0]["line"]
        )

    def test_captures_recovery_diagnostics(self) -> None:
        """A request that triggers cold-cache recovery records the recovery
        steps + diagnostic counters in the trace."""
        self._post(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "old"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_x", "content": "result"},
                    {"role": "user", "content": "new"},
                ],
            }
        )
        trace = _read_single_trace(self.writer.session_dir)
        self.assertEqual(
            trace["transform"]["recovery_steps"][0]["strategy"], "latest_user"
        )
        self.assertGreaterEqual(
            len(
                [
                    item
                    for item in trace["transform"]["reasoning_diagnostics"]
                    if item["missing"]
                ]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()

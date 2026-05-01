from __future__ import annotations

from io import BytesIO
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import zlib

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import (
    DeepSeekProxyHandler,
    build_arg_parser,
    read_response_body,
    summarize_chat_payload,
)


class FakeResponse:
    def __init__(self, body: bytes, encoding: str = "", status: int = 200) -> None:
        self._body = BytesIO(body)
        self.headers = {"Content-Encoding": encoding} if encoding else {}
        self.status = status

    def read(self) -> bytes:
        return self._body.read()


class FakeStreamingResponse:
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


class FailingStreamingResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def readline(self) -> bytes:
        raise OSError("record layer failure")


class BrokenPipeWfile:
    def write(self, body: bytes) -> None:
        raise BrokenPipeError("test disconnect")

    def flush(self) -> None:
        raise BrokenPipeError("test disconnect")


def make_proxy_handler(wfile: object) -> DeepSeekProxyHandler:
    handler = object.__new__(DeepSeekProxyHandler)
    handler.server = SimpleNamespace(
        config=ProxyConfig(),
        reasoning_store=ReasoningStore(":memory:"),
    )
    handler.wfile = wfile
    handler.close_connection = False
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    return handler


class ServerTests(unittest.TestCase):
    def test_cli_boolean_overrides_have_on_and_off_forms(self) -> None:
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

    def test_read_response_body_handles_gzip(self) -> None:
        body = gzip.compress(b'{"ok":true}')

        self.assertEqual(read_response_body(FakeResponse(body, "gzip")), b'{"ok":true}')

    def test_read_response_body_handles_deflate(self) -> None:
        body = zlib.compress(b'{"ok":true}')

        self.assertEqual(
            read_response_body(FakeResponse(body, "deflate")), b'{"ok":true}'
        )

    def test_summarize_chat_payload_does_not_include_message_content(self) -> None:
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
        self.assertIn("stream=True", summary)
        self.assertIn("messages=1", summary)
        self.assertIn("tools=1", summary)
        self.assertNotIn("secret prompt", summary)

    def test_regular_response_handles_client_disconnect(self) -> None:
        handler = make_proxy_handler(BrokenPipeWfile())
        body = json.dumps(
            {
                "id": "chatcmpl-test",
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
                sent = handler._proxy_regular_response(
                    FakeResponse(body),
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "cache-namespace",
                )
        finally:
            handler.server.reasoning_store.close()

        self.assertFalse(sent.sent)
        self.assertIn("sending upstream response body", "\n".join(captured.output))

    def test_streaming_response_stops_on_client_disconnect(self) -> None:
        handler = make_proxy_handler(BrokenPipeWfile())
        chunk = {
            "id": "chatcmpl-stream",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                }
            ],
        }
        response = FakeStreamingResponse(
            [
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )

        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                sent = handler._proxy_streaming_response(
                    response,
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "cache-namespace",
                )
        finally:
            handler.server.reasoning_store.close()

        self.assertFalse(sent.sent)
        self.assertEqual(response.readline_calls, 1)
        self.assertIn("sending streaming response chunk", "\n".join(captured.output))

    def test_streaming_response_handles_upstream_read_failure(self) -> None:
        handler = make_proxy_handler(BytesIO())

        try:
            with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
                sent = handler._proxy_streaming_response(
                    FailingStreamingResponse(),
                    "deepseek-v4-pro",
                    [{"role": "user", "content": "hi"}],
                    "cache-namespace",
                )
        finally:
            handler.server.reasoning_store.close()

        self.assertFalse(sent.sent)
        self.assertIn(
            "upstream streaming response read failed",
            "\n".join(captured.output),
        )

    def test_collapsible_reasoning_has_no_effect_when_display_is_disabled(
        self,
    ) -> None:
        wfile = BytesIO()
        handler = make_proxy_handler(wfile)
        handler.server.config = ProxyConfig(
            display_reasoning=False,
            collapsible_reasoning=True,
        )
        chunk = {
            "id": "chatcmpl-stream",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Need context."},
                }
            ],
        }
        response = FakeStreamingResponse(
            [
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )

        try:
            sent = handler._proxy_streaming_response(
                response,
                "deepseek-v4-pro",
                [{"role": "user", "content": "hi"}],
                "cache-namespace",
            )
        finally:
            handler.server.reasoning_store.close()

        body = wfile.getvalue().decode("utf-8")
        self.assertTrue(sent.sent)
        self.assertIn("reasoning_content", body)
        self.assertNotIn("<details>", body)
        self.assertNotIn("<think>", body)


if __name__ == "__main__":
    unittest.main()

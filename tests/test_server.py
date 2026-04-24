from __future__ import annotations

from io import BytesIO
import gzip
import unittest
import zlib

from deepseek_cursor_proxy.server import read_response_body, summarize_chat_payload


class FakeResponse:
    def __init__(self, body: bytes, encoding: str = "") -> None:
        self._body = BytesIO(body)
        self.headers = {"Content-Encoding": encoding} if encoding else {}

    def read(self) -> bytes:
        return self._body.read()


class ServerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

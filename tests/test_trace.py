from __future__ import annotations

import json
import stat
from tempfile import TemporaryDirectory
import unittest

from deepseek_cursor_proxy.trace import TraceWriter


class TraceWriterTests(unittest.TestCase):
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

            payload = json.loads(trace.path.read_text(encoding="utf-8"))
            serialized = json.dumps(payload)

            self.assertNotIn("sk-secret", serialized)
            self.assertEqual(
                payload["request"]["headers"]["Authorization"]["present"],
                True,
            )
            self.assertIn("sha256", payload["request"]["headers"]["Authorization"])


if __name__ == "__main__":
    unittest.main()

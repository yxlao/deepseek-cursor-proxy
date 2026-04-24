from __future__ import annotations

from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest

from deepseek_cursor_proxy.reasoning_store import ReasoningStore


class ReasoningStoreTests(unittest.TestCase):
    def test_file_store_creates_private_database_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            reasoning_content_path = (
                Path(temp_dir) / "nested" / "reasoning_content.sqlite3"
            )

            store = ReasoningStore(reasoning_content_path)
            store.close()

            self.assertTrue(reasoning_content_path.exists())
            self.assertEqual(stat.S_IMODE(reasoning_content_path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()

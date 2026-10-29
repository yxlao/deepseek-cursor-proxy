from __future__ import annotations

from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest

from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope


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

    def test_store_prunes_to_max_rows_and_can_clear(self) -> None:
        store = ReasoningStore(":memory:", max_rows=2, prune_every_puts=1)
        try:
            store.put("a", "reasoning a", {"role": "assistant"})
            store.put("b", "reasoning b", {"role": "assistant"})
            store.put("c", "reasoning c", {"role": "assistant"})

            self.assertIsNone(store.get("a"))
            self.assertEqual(store.get("b"), "reasoning b")
            self.assertEqual(store.get("c"), "reasoning c")
            self.assertEqual(store.clear(), 2)
            self.assertIsNone(store.get("b"))
            self.assertIsNone(store.get("c"))
        finally:
            store.close()

    def test_puts_do_not_rescan_the_table_on_every_write(self) -> None:
        # A request stores hundreds of rows at once; scanning the whole table on
        # each of them is what made a single turn take a minute.
        store = ReasoningStore(
            ":memory:",
            max_rows=2,
            prune_interval_seconds=3600,
            prune_every_puts=1000,
        )
        try:
            for index in range(5):
                store.put(f"k{index}", f"reasoning {index}", {"role": "assistant"})

            # Five rows for a cap of two: nothing was dropped while writing.
            self.assertEqual(store.get("k0"), "reasoning 0")

            self.assertEqual(store.prune(), 3)
            self.assertIsNone(store.get("k0"))
            self.assertEqual(store.get("k4"), "reasoning 4")
        finally:
            store.close()

    def test_prune_runs_again_after_enough_puts(self) -> None:
        store = ReasoningStore(
            ":memory:",
            max_rows=1,
            prune_interval_seconds=3600,
            prune_every_puts=3,
        )
        try:
            for index in range(3):
                store.put(f"k{index}", f"reasoning {index}", {"role": "assistant"})

            # The third put crosses prune_every_puts, so the cap is applied.
            self.assertIsNone(store.get("k0"))
            self.assertEqual(store.get("k2"), "reasoning 2")
        finally:
            store.close()

    def test_empty_reasoning_content_is_stored_as_present_value(self) -> None:
        store = ReasoningStore(":memory:")
        try:
            scope = conversation_scope([{"role": "user", "content": "lookup"}])
            tool_call = {
                "id": "call_empty",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
            message = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [tool_call],
            }

            self.assertGreater(store.store_assistant_message(message, scope), 0)
            self.assertEqual(store.get(f"scope:{scope}:tool_call:call_empty"), "")
            self.assertEqual(
                store.lookup_for_message(
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    scope,
                ),
                "",
            )
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()

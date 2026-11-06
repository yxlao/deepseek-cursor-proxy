from __future__ import annotations

from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest

from deepseek_cursor_proxy.reasoning_store import (
    ReasoningStore,
    ScopeIndex,
    conversation_scope,
    turn_context_signature,
)


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
        store = ReasoningStore(":memory:", max_rows=2)
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


class ScopeIndexTests(unittest.TestCase):
    """ScopeIndex must reproduce the reference hashes for every prefix."""

    transcripts = (
        [],
        [{"role": "system", "content": "system"}],
        [{"role": "user", "content": "привет"}],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "user", "content": "d"},
        ],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "a"},
            {"role": "tool", "tool_call_id": "t", "content": "r"},
            {"role": "assistant", "content": 'кавычки " и \\ и\n\t и emoji 🎵'},
        ],
    )

    def test_hashes_match_reference_for_every_prefix(self) -> None:
        for namespace in ("", "namespace:pro"):
            for messages in self.transcripts:
                index = ScopeIndex(namespace)
                self.assertEqual(
                    index.conversation_scope(), conversation_scope([], namespace)
                )
                self.assertEqual(
                    index.turn_context_signature(), turn_context_signature([])
                )
                for position, message in enumerate(messages):
                    index.append(message)
                    prefix = messages[: position + 1]
                    self.assertEqual(
                        index.conversation_scope(),
                        conversation_scope(prefix, namespace),
                    )
                    self.assertEqual(
                        index.turn_context_signature(),
                        turn_context_signature(prefix),
                    )

    def test_index_keeps_the_messages_it_hashed(self) -> None:
        index = ScopeIndex("namespace:pro")
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": ""},
        ]

        for message in messages:
            index.append(message)

        self.assertEqual(index.messages, messages)


if __name__ == "__main__":
    unittest.main()

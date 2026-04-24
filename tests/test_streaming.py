from __future__ import annotations

import unittest

from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope
from deepseek_cursor_proxy.streaming import StreamAccumulator


class StreamAccumulatorTests(unittest.TestCase):
    def test_accumulates_reasoning_content_and_tool_call_deltas(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need ",
                        },
                    }
                ]
            }
        )
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "context.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path"',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ':"README.md"}'},
                                }
                            ],
                        },
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "read README"}])
        stored = accumulator.store_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(
            store.get(f"scope:{scope}:tool_call:call_stream"), "Need context."
        )
        store.close()

    def test_returns_accumulated_messages_for_logging(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Think.",
                            "content": "Answer.",
                        },
                    }
                ]
            }
        )

        self.assertEqual(
            accumulator.messages(),
            [
                {
                    "role": "assistant",
                    "content": "Answer.",
                    "reasoning_content": "Think.",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()

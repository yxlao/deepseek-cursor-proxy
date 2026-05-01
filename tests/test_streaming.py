from __future__ import annotations

import unittest

from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope
from deepseek_cursor_proxy.streaming import (
    CursorReasoningDisplayAdapter,
    StreamAccumulator,
)


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

    def test_stores_reasoning_when_choice_finishes_before_done(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need a tool.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "lookup"}])
        stored = accumulator.store_finished_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(
            store.get(f"scope:{scope}:tool_call:call_stream"), "Need a tool."
        )
        self.assertEqual(accumulator.store_reasoning(store, scope), 0)
        store.close()

    def test_stores_same_streaming_choice_under_multiple_scopes(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need a tool.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

        first_scope = conversation_scope([{"role": "user", "content": "full"}])
        second_scope = conversation_scope([{"role": "user", "content": "active"}])
        first_stored = accumulator.store_finished_reasoning(store, first_scope)
        second_stored = accumulator.store_finished_reasoning(store, second_scope)

        self.assertGreater(first_stored, 0)
        self.assertGreater(second_stored, 0)
        self.assertEqual(
            store.get(f"scope:{first_scope}:tool_call:call_stream"), "Need a tool."
        )
        self.assertEqual(
            store.get(f"scope:{second_scope}:tool_call:call_stream"), "Need a tool."
        )
        store.close()

    def test_stores_tool_call_reasoning_before_finish_reason(self) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need a tool.",
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
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stream",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query"',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "lookup"}])
        stored = accumulator.store_ready_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(
            store.get(f"scope:{scope}:tool_call:call_stream"), "Need a tool."
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
                                    "function": {"arguments": ':"README"}'},
                                }
                            ],
                        },
                    }
                ]
            }
        )

        self.assertGreater(accumulator.store_ready_reasoning(store, scope), 0)
        store.close()

    def test_stores_empty_reasoning_content_when_stream_field_is_present(
        self,
    ) -> None:
        store = ReasoningStore(":memory:")
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_empty",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )

        scope = conversation_scope([{"role": "user", "content": "lookup"}])
        stored = accumulator.store_finished_reasoning(store, scope)

        self.assertGreater(stored, 0)
        self.assertEqual(store.get(f"scope:{scope}:tool_call:call_empty"), "")
        self.assertEqual(accumulator.messages()[0]["reasoning_content"], "")
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


class CursorReasoningDisplayAdapterTests(unittest.TestCase):
    def test_mirrors_reasoning_content_into_details_content(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        reasoning_chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Need context."},
                    "finish_reason": None,
                }
            ],
        }
        answer_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Final answer."},
                    "finish_reason": None,
                }
            ],
        }

        adapter.rewrite_chunk(reasoning_chunk)
        adapter.rewrite_chunk(answer_chunk)

        reasoning_delta = reasoning_chunk["choices"][0]["delta"]
        answer_delta = answer_chunk["choices"][0]["delta"]
        self.assertEqual(reasoning_delta["reasoning_content"], "Need context.")
        self.assertEqual(
            reasoning_delta["content"],
            "<details>\n<summary>Thinking</summary>\n\nNeed context.",
        )
        self.assertEqual(answer_delta["content"], "\n</details>\n\nFinal answer.")

    def test_can_mirror_reasoning_content_into_legacy_think_content(self) -> None:
        adapter = CursorReasoningDisplayAdapter(collapsible=False)
        reasoning_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Need context."},
                    "finish_reason": None,
                }
            ],
        }
        answer_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Final answer."},
                    "finish_reason": None,
                }
            ],
        }

        adapter.rewrite_chunk(reasoning_chunk)
        adapter.rewrite_chunk(answer_chunk)

        self.assertEqual(
            reasoning_chunk["choices"][0]["delta"]["content"], "<think>\nNeed context."
        )
        self.assertEqual(
            answer_chunk["choices"][0]["delta"]["content"],
            "\n</think>\n\nFinal answer.",
        )

    def test_closes_thinking_block_before_tool_calls(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        adapter.rewrite_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Need a tool."},
                    }
                ]
            }
        )
        tool_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        }

        adapter.rewrite_chunk(tool_chunk)

        self.assertEqual(
            tool_chunk["choices"][0]["delta"]["content"], "\n</details>\n\n"
        )

    def test_flush_chunk_closes_unfinished_thinking_block_at_done(self) -> None:
        adapter = CursorReasoningDisplayAdapter()
        adapter.rewrite_chunk(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Still thinking."},
                    }
                ],
            }
        )

        closing_chunk = adapter.flush_chunk("deepseek-v4-pro")

        self.assertIsNotNone(closing_chunk)
        assert closing_chunk is not None
        self.assertEqual(closing_chunk["model"], "deepseek-v4-pro")
        self.assertEqual(
            closing_chunk["choices"][0]["delta"]["content"], "\n</details>\n\n"
        )
        self.assertIsNone(adapter.flush_chunk("deepseek-v4-pro"))


if __name__ == "__main__":
    unittest.main()

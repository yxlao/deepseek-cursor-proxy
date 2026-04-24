from __future__ import annotations

import json
import unittest

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope
from deepseek_cursor_proxy.transform import (
    extract_text_content,
    prepare_upstream_request,
    rewrite_response_body,
)


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ReasoningStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_extracts_text_from_cursor_style_content_parts(self) -> None:
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "input_text", "text": "world"},
        ]

        self.assertEqual(
            extract_text_content(content),
            "hello\n[image_url omitted by DeepSeek text proxy]\nworld",
        )

    def test_prepares_thinking_request_and_converts_legacy_functions(self) -> None:
        payload = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [{"name": "lookup", "parameters": {"type": "object"}}],
            "function_call": {"name": "lookup"},
            "max_completion_tokens": 123,
            "parallel_tool_calls": True,
        }
        config = ProxyConfig(upstream_api_key="key")

        prepared = prepare_upstream_request(payload, config, self.store)

        self.assertEqual(prepared.original_model, "deepseek-v4-flash")
        self.assertEqual(prepared.upstream_model, "deepseek-v4-pro")
        self.assertEqual(prepared.payload["model"], "deepseek-v4-pro")
        self.assertEqual(prepared.payload["thinking"], {"type": "enabled"})
        self.assertEqual(prepared.payload["reasoning_effort"], "high")
        self.assertEqual(prepared.payload["max_tokens"], 123)
        self.assertEqual(prepared.payload["tools"][0]["type"], "function")
        self.assertEqual(
            prepared.payload["tool_choice"],
            "auto",
        )
        self.assertNotIn("parallel_tool_calls", prepared.payload)

    def test_normalizes_unsupported_required_tool_choice_to_auto(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "call a tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "required",
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(prepared.payload["tool_choice"], "auto")

    def test_restores_reasoning_content_for_cached_tool_call(self) -> None:
        prior_messages = [{"role": "user", "content": "read README"}]
        assistant_message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Need the file contents before answering.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        }
        self.store.store_assistant_message(
            assistant_message, conversation_scope(prior_messages)
        )

        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "read README"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "file text"},
                {"role": "user", "content": "continue"},
            ],
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(prepared.patched_reasoning_messages, 1)
        self.assertEqual(
            prepared.payload["messages"][1]["reasoning_content"],
            "Need the file contents before answering.",
        )

    def test_restores_reasoning_content_for_cached_final_tool_turn_message(
        self,
    ) -> None:
        prior_messages = [
            {"role": "user", "content": "read README"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need the file contents before answering.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "file text"},
        ]
        assistant_message = {
            "role": "assistant",
            "content": "Final answer after using the tool.",
            "reasoning_content": "The tool result is enough to answer.",
        }
        self.store.store_assistant_message(
            assistant_message, conversation_scope(prior_messages)
        )

        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "read README"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "file text"},
                {"role": "assistant", "content": "Final answer after using the tool."},
                {"role": "user", "content": "another question"},
            ],
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(prepared.patched_reasoning_messages, 1)
        self.assertEqual(
            prepared.payload["messages"][3]["reasoning_content"],
            "The tool result is enough to answer.",
        )

    def test_reasoning_cache_is_scoped_by_conversation_prefix(self) -> None:
        tool_call = {
            "id": "call_reused",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        assistant_a = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Reasoning for thread A.",
            "tool_calls": [tool_call],
        }
        assistant_b = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Reasoning for thread B.",
            "tool_calls": [tool_call],
        }
        prior_a = [{"role": "user", "content": "thread A"}]
        prior_b = [{"role": "user", "content": "thread B"}]

        self.store.store_assistant_message(assistant_a, conversation_scope(prior_a))
        self.store.store_assistant_message(assistant_b, conversation_scope(prior_b))

        payload_a = {
            "model": "deepseek-v4-pro",
            "messages": [
                *prior_a,
                {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            ],
        }
        payload_b = {
            "model": "deepseek-v4-pro",
            "messages": [
                *prior_b,
                {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            ],
        }

        prepared_a = prepare_upstream_request(
            payload_a, ProxyConfig(upstream_api_key="key"), self.store
        )
        prepared_b = prepare_upstream_request(
            payload_b, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(
            prepared_a.payload["messages"][1]["reasoning_content"],
            "Reasoning for thread A.",
        )
        self.assertEqual(
            prepared_b.payload["messages"][1]["reasoning_content"],
            "Reasoning for thread B.",
        )

    def test_exact_message_signature_wins_over_tool_call_id_fallback(self) -> None:
        prior = [{"role": "user", "content": "same conversation prefix"}]
        scope = conversation_scope(prior)
        first_tool_call = {
            "id": "call_reused",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"value":"first"}'},
        }
        second_tool_call = {
            "id": "call_reused",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"value":"second"}'},
        }
        self.store.store_assistant_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "first reasoning",
                "tool_calls": [first_tool_call],
            },
            scope,
        )
        self.store.store_assistant_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "second reasoning",
                "tool_calls": [second_tool_call],
            },
            scope,
        )

        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    *prior,
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [first_tool_call],
                    },
                ],
            },
            ProxyConfig(upstream_api_key="key"),
            self.store,
        )

        self.assertEqual(
            prepared.payload["messages"][1]["reasoning_content"], "first reasoning"
        )

    def test_restores_reasoning_when_cursor_drops_tool_call_id_but_keeps_function_call(
        self,
    ) -> None:
        prior = [{"role": "user", "content": "inspect repo"}]
        assistant_message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Need to call the file tool.",
            "tool_calls": [
                {
                    "id": "call_original",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        }
        self.store.store_assistant_message(assistant_message, conversation_scope(prior))

        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                *prior,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
            ],
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(prepared.patched_reasoning_messages, 1)
        self.assertEqual(prepared.payload["messages"][1]["content"], "")
        self.assertEqual(
            prepared.payload["messages"][1]["reasoning_content"],
            "Need to call the file tool.",
        )

    def test_adds_fallback_reasoning_for_uncached_assistant_tool_call(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "read README"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_uncached",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_uncached",
                    "content": "file text",
                },
            ],
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(prepared.patched_reasoning_messages, 0)
        self.assertEqual(prepared.fallback_reasoning_messages, 1)
        self.assertIn("reasoning_content", prepared.payload["messages"][1])

    def test_adds_fallback_reasoning_for_uncached_assistant_after_tool_result(
        self,
    ) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "read README"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need file text.",
                    "tool_calls": [
                        {
                            "id": "call_uncached",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_uncached",
                    "content": "file text",
                },
                {"role": "assistant", "content": "Summary of file."},
                {"role": "user", "content": "continue"},
            ],
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(prepared.fallback_reasoning_messages, 1)
        self.assertIn("reasoning_content", prepared.payload["messages"][3])

    def test_does_not_add_fallback_reasoning_for_plain_chat_history(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(prepared.fallback_reasoning_messages, 0)
        self.assertNotIn("reasoning_content", prepared.payload["messages"][1])

    def test_converted_function_message_uses_tool_schema(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {
                    "role": "function",
                    "name": "lookup",
                    "tool_call_id": "call_1",
                    "content": {"ok": True},
                }
            ],
        }

        prepared = prepare_upstream_request(
            payload, ProxyConfig(upstream_api_key="key"), self.store
        )

        self.assertEqual(
            prepared.payload["messages"][0],
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
        )

    def test_rewrite_response_records_reasoning_and_restores_model_name(self) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "I need to inspect the repo.",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ).encode()

        request_messages = [{"role": "user", "content": "inspect repo"}]
        rewritten = rewrite_response_body(
            body, "deepseek-v4-flash", self.store, request_messages
        )
        payload = json.loads(rewritten)

        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(
            self.store.get(
                f"scope:{conversation_scope(request_messages)}:tool_call:call_abc"
            ),
            "I need to inspect the repo.",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import (
    ReasoningStore,
    conversation_scope,
    message_signature,
)
from deepseek_cursor_proxy.transform import (
    RECOVERY_NOTICE_CONTENT,
    extract_text_content,
    prepare_upstream_request,
    reasoning_cache_namespace,
    rewrite_response_body,
    strip_cursor_thinking_blocks,
)


DEFAULT_CONFIG = ProxyConfig()
DEFAULT_CACHE_NAMESPACE = reasoning_cache_namespace(
    DEFAULT_CONFIG,
    "deepseek-v4-pro",
    {"type": "enabled"},
    "high",
)


def cache_scope(messages: list[dict]) -> str:
    return conversation_scope(messages, DEFAULT_CACHE_NAMESPACE)


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

    def test_strips_cursor_display_thinking_blocks_from_assistant_content(
        self,
    ) -> None:
        self.assertEqual(
            strip_cursor_thinking_blocks(
                "<think>\nNeed context.\n</think>\n\nFinal answer."
            ),
            "Final answer.",
        )

    def test_prepares_assistant_content_without_mirrored_thinking_blocks(
        self,
    ) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "<think>\nHidden.\n</think>\n\nVisible answer.",
                },
                {"role": "user", "content": "continue"},
            ],
        }

        prepared = prepare_upstream_request(
            payload,
            ProxyConfig(missing_reasoning_strategy="reject"),
            self.store,
        )

        self.assertEqual(prepared.payload["messages"][1]["content"], "Visible answer.")

    def test_prepares_thinking_request_and_converts_legacy_functions(self) -> None:
        payload = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [{"name": "lookup", "parameters": {"type": "object"}}],
            "function_call": {"name": "lookup"},
            "max_completion_tokens": 123,
            "parallel_tool_calls": True,
        }
        config = ProxyConfig()

        prepared = prepare_upstream_request(payload, config, self.store)

        self.assertEqual(prepared.original_model, "deepseek-v4-flash")
        self.assertEqual(prepared.upstream_model, "deepseek-v4-flash")
        self.assertEqual(prepared.payload["model"], "deepseek-v4-flash")
        self.assertEqual(prepared.payload["thinking"], {"type": "enabled"})
        self.assertEqual(prepared.payload["reasoning_effort"], "high")
        self.assertEqual(prepared.payload["max_tokens"], 123)
        self.assertEqual(prepared.payload["tools"][0]["type"], "function")
        self.assertEqual(
            prepared.payload["tool_choice"],
            {"type": "function", "function": {"name": "lookup"}},
        )
        self.assertNotIn("parallel_tool_calls", prepared.payload)

    def test_uses_config_model_only_when_request_model_is_missing(self) -> None:
        prepared = prepare_upstream_request(
            {"messages": [{"role": "user", "content": "hi"}]},
            ProxyConfig(upstream_model="deepseek-v4-flash"),
            self.store,
        )

        self.assertEqual(prepared.original_model, "deepseek-v4-flash")
        self.assertEqual(prepared.upstream_model, "deepseek-v4-flash")
        self.assertEqual(prepared.payload["model"], "deepseek-v4-flash")

    def test_streaming_requests_include_usage_for_runtime_stats(self) -> None:
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "stream": True,
                "stream_options": {"include_usage": False},
                "messages": [{"role": "user", "content": "hi"}],
            },
            ProxyConfig(),
            self.store,
        )

        self.assertEqual(prepared.payload["stream_options"]["include_usage"], True)

    def test_preserves_required_tool_choice(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "call a tool"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "required",
        }

        prepared = prepare_upstream_request(
            payload,
            ProxyConfig(missing_reasoning_strategy="reject"),
            self.store,
        )

        self.assertEqual(prepared.payload["tool_choice"], "required")

    def test_preserves_named_tool_choice(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "call lookup"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup"},
            },
        }

        prepared = prepare_upstream_request(payload, ProxyConfig(), self.store)

        self.assertEqual(
            prepared.payload["tool_choice"],
            {"type": "function", "function": {"name": "lookup"}},
        )

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
            assistant_message, cache_scope(prior_messages)
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
            payload,
            ProxyConfig(missing_reasoning_strategy="reject"),
            self.store,
        )

        self.assertEqual(prepared.patched_reasoning_messages, 1)
        self.assertEqual(
            prepared.payload["messages"][1]["reasoning_content"],
            "Need the file contents before answering.",
        )

    def test_accepts_empty_reasoning_content_when_present_for_tool_call(
        self,
    ) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "read README"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [
                        {
                            "id": "call_empty",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_empty", "content": "file text"},
            ],
        }

        prepared = prepare_upstream_request(payload, ProxyConfig(), self.store)

        self.assertEqual(prepared.patched_reasoning_messages, 0)
        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertIn("reasoning_content", prepared.payload["messages"][1])
        self.assertEqual(prepared.payload["messages"][1]["reasoning_content"], "")

    def test_restores_empty_reasoning_content_from_cache(self) -> None:
        prior_messages = [{"role": "user", "content": "read README"}]
        tool_call = {
            "id": "call_empty",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        }
        self.store.store_assistant_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [tool_call],
            },
            cache_scope(prior_messages),
        )

        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    *prior_messages,
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    {
                        "role": "tool",
                        "tool_call_id": "call_empty",
                        "content": "file text",
                    },
                ],
            },
            ProxyConfig(),
            self.store,
        )

        self.assertEqual(prepared.patched_reasoning_messages, 1)
        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertIn("reasoning_content", prepared.payload["messages"][1])
        self.assertEqual(prepared.payload["messages"][1]["reasoning_content"], "")

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
            assistant_message, cache_scope(prior_messages)
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
            payload,
            ProxyConfig(missing_reasoning_strategy="reject"),
            self.store,
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

        self.store.store_assistant_message(assistant_a, cache_scope(prior_a))
        self.store.store_assistant_message(assistant_b, cache_scope(prior_b))

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

        prepared_a = prepare_upstream_request(payload_a, ProxyConfig(), self.store)
        prepared_b = prepare_upstream_request(payload_b, ProxyConfig(), self.store)

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
        scope = cache_scope(prior)
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
            ProxyConfig(),
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
        self.store.store_assistant_message(assistant_message, cache_scope(prior))

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

        prepared = prepare_upstream_request(payload, ProxyConfig(), self.store)

        self.assertEqual(prepared.patched_reasoning_messages, 1)
        self.assertEqual(prepared.payload["messages"][1]["content"], "")
        self.assertEqual(
            prepared.payload["messages"][1]["reasoning_content"],
            "Need to call the file tool.",
        )

    def test_restores_reasoning_when_cursor_history_contains_mirrored_think_block(
        self,
    ) -> None:
        prior = [{"role": "user", "content": "inspect repo"}]
        tool_call = {
            "id": "call_original",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        }
        self.store.store_assistant_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Need to call the file tool.",
                "tool_calls": [tool_call],
            },
            cache_scope(prior),
        )

        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    *prior,
                    {
                        "role": "assistant",
                        "content": "<think>\nNeed to call the file tool.\n</think>\n\n",
                        "tool_calls": [tool_call],
                    },
                ],
            },
            ProxyConfig(),
            self.store,
        )

        self.assertEqual(prepared.patched_reasoning_messages, 1)
        self.assertEqual(prepared.payload["messages"][1]["content"], "")
        self.assertEqual(
            prepared.payload["messages"][1]["reasoning_content"],
            "Need to call the file tool.",
        )

    def test_reports_missing_reasoning_for_uncached_assistant_tool_call(self) -> None:
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
            payload,
            ProxyConfig(missing_reasoning_strategy="reject"),
            self.store,
        )

        self.assertEqual(prepared.patched_reasoning_messages, 0)
        self.assertEqual(prepared.missing_reasoning_messages, 1)
        self.assertNotIn("reasoning_content", prepared.payload["messages"][1])

    def test_can_recover_uncached_tool_history_from_latest_user(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "Follow project rules."},
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
                {"role": "user", "content": "continue with the summary"},
            ],
        }

        prepared = prepare_upstream_request(
            payload,
            ProxyConfig(missing_reasoning_strategy="recover"),
            self.store,
        )

        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertEqual(prepared.recovered_reasoning_messages, 1)
        self.assertEqual(prepared.recovery_dropped_messages, 3)
        self.assertEqual(prepared.recovery_notice, RECOVERY_NOTICE_CONTENT)
        self.assertEqual(
            [message["role"] for message in prepared.payload["messages"]],
            ["system", "system", "user"],
        )
        self.assertIn(
            "recovered this request", prepared.payload["messages"][1]["content"]
        )
        self.assertEqual(
            prepared.payload["messages"][2],
            {"role": "user", "content": "continue with the summary"},
        )

    def test_recovery_boundary_preserves_later_deepseek_tool_context(self) -> None:
        old_tool_call = {
            "id": "call_old",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        }
        new_tool_call = {
            "id": "call_new",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"query":"new"}',
            },
        }
        first_recovered = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "old model turn"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [old_tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_old",
                        "content": "old result",
                    },
                    {"role": "user", "content": "continue with DeepSeek"},
                ],
            },
            ProxyConfig(missing_reasoning_strategy="recover"),
            self.store,
        )
        recovered_tool_message = {
            "role": "assistant",
            "content": RECOVERY_NOTICE_CONTENT,
            "reasoning_content": "Need the new lookup.",
            "tool_calls": [new_tool_call],
        }
        self.store.store_assistant_message(
            recovered_tool_message,
            conversation_scope(
                first_recovered.payload["messages"],
                first_recovered.cache_namespace,
            ),
        )

        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "old model turn"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [old_tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_old",
                        "content": "old result",
                    },
                    {"role": "user", "content": "continue with DeepSeek"},
                    {
                        "role": "assistant",
                        "content": RECOVERY_NOTICE_CONTENT,
                        "tool_calls": [new_tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_new",
                        "content": "new result",
                    },
                ],
            },
            ProxyConfig(missing_reasoning_strategy="recover"),
            self.store,
        )

        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertIsNone(prepared.recovery_notice)
        self.assertEqual(
            [message["role"] for message in prepared.payload["messages"]],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(
            prepared.payload["messages"][2]["reasoning_content"],
            "Need the new lookup.",
        )
        self.assertEqual(
            prepared.payload["messages"][3],
            {
                "role": "tool",
                "tool_call_id": "call_new",
                "content": "new result",
            },
        )

    def test_reports_missing_reasoning_for_uncached_assistant_after_tool_result(
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
            payload,
            ProxyConfig(missing_reasoning_strategy="reject"),
            self.store,
        )

        self.assertEqual(prepared.missing_reasoning_messages, 1)
        self.assertNotIn("reasoning_content", prepared.payload["messages"][3])

    def test_does_not_report_missing_reasoning_for_plain_chat_history(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "continue"},
            ],
        }

        prepared = prepare_upstream_request(payload, ProxyConfig(), self.store)

        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertNotIn("reasoning_content", prepared.payload["messages"][1])

    def test_does_not_repair_reasoning_when_thinking_is_disabled(self) -> None:
        payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "read README"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Should be removed in non-thinking mode.",
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
            payload, ProxyConfig(thinking="disabled"), self.store
        )

        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertNotIn("reasoning_content", prepared.payload["messages"][1])

    def test_reasoning_cache_is_namespaced_by_authorization(self) -> None:
        config = ProxyConfig(missing_reasoning_strategy="reject")
        prior = [{"role": "user", "content": "read README"}]
        namespace_a = reasoning_cache_namespace(
            config,
            config.upstream_model,
            {"type": "enabled"},
            "high",
            "Bearer key-a",
        )
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
        }
        self.store.store_assistant_message(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Reasoning for key A.",
                "tool_calls": [tool_call],
            },
            conversation_scope(prior, namespace_a),
        )

        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    *prior,
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ],
            },
            config,
            self.store,
            authorization="Bearer key-b",
        )

        self.assertEqual(prepared.missing_reasoning_messages, 1)
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

        prepared = prepare_upstream_request(payload, ProxyConfig(), self.store)

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

    def test_rewrite_response_can_prefix_recovery_notice_before_storing(
        self,
    ) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Summary.",
                            "reasoning_content": "Tool result is enough.",
                        },
                    }
                ],
            }
        ).encode()

        request_messages = [
            {"role": "user", "content": "read README"},
            {"role": "tool", "tool_call_id": "call_abc", "content": "file text"},
        ]
        rewritten = rewrite_response_body(
            body,
            "deepseek-v4-pro",
            self.store,
            request_messages,
            content_prefix=RECOVERY_NOTICE_CONTENT,
        )
        payload = json.loads(rewritten)
        stored_message = {
            "role": "assistant",
            "content": RECOVERY_NOTICE_CONTENT + "Summary.",
            "reasoning_content": "Tool result is enough.",
        }

        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            RECOVERY_NOTICE_CONTENT + "Summary.",
        )
        self.assertEqual(
            self.store.get(
                f"scope:{conversation_scope(request_messages)}:signature:"
                f"{message_signature(stored_message)}"
            ),
            "Tool result is enough.",
        )

    def test_rewrite_response_preserves_prompt_cache_usage_fields(self) -> None:
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
                "usage": {
                    "prompt_tokens": 10,
                    "prompt_cache_hit_tokens": 6,
                    "prompt_cache_miss_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            }
        ).encode()

        rewritten = rewrite_response_body(body, "deepseek-v4-flash", self.store, [])
        payload = json.loads(rewritten)

        self.assertEqual(payload["usage"]["prompt_cache_hit_tokens"], 6)
        self.assertEqual(payload["usage"]["prompt_cache_miss_tokens"], 4)


if __name__ == "__main__":
    unittest.main()

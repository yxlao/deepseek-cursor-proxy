"""Pure-function unit tests for transform.py.

Anything that requires a fake DeepSeek upstream lives in test_protocol.py.
This file only exercises helpers that take dicts/strings and return
dicts/strings — content extraction, request normalization, response rewrite,
recovery-notice stripping, and warning behaviour for dropped fields.
"""

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
    RECOVERY_NOTICE_TEXT,
    extract_text_content,
    normalize_reasoning_effort,
    prepare_upstream_request,
    reasoning_cache_namespace,
    rewrite_response_body,
    strip_cursor_thinking_blocks,
    strip_recovery_notice_for_upstream,
)


def _default_cache_namespace() -> str:
    return reasoning_cache_namespace(
        ProxyConfig(),
        "deepseek-v4-pro",
        {"type": "enabled"},
        "high",
    )


def _cache_scope(messages: list[dict]) -> str:
    return conversation_scope(messages, _default_cache_namespace())


class ContentHelpersTests(unittest.TestCase):
    def test_extract_text_content_flattens_multipart_array(self) -> None:
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "input_text", "text": "world"},
        ]
        self.assertEqual(
            extract_text_content(content),
            "hello\n[image_url omitted by DeepSeek text proxy]\nworld",
        )

    def test_extract_text_content_passes_through_string_and_none(self) -> None:
        self.assertEqual(extract_text_content("plain"), "plain")
        self.assertIsNone(extract_text_content(None))

    def test_strip_cursor_thinking_blocks_removes_details_and_think(self) -> None:
        self.assertEqual(
            strip_cursor_thinking_blocks(
                "<details>\n<summary>Thinking</summary>\n\nplan\n</details>\n\nanswer"
            ),
            "answer",
        )
        self.assertEqual(
            strip_cursor_thinking_blocks("<think>\nplan\n</think>\n\nanswer"),
            "answer",
        )

    def test_strip_cursor_thinking_blocks_preserves_unrelated_details(self) -> None:
        kept = "<details><summary>Diff</summary>\nrelevant\n</details>"
        self.assertEqual(strip_cursor_thinking_blocks(kept), kept)

    def test_normalize_reasoning_effort_aliases(self) -> None:
        self.assertEqual(normalize_reasoning_effort("low"), "high")
        self.assertEqual(normalize_reasoning_effort("medium"), "high")
        self.assertEqual(normalize_reasoning_effort("high"), "high")
        self.assertEqual(normalize_reasoning_effort("max"), "max")
        self.assertEqual(normalize_reasoning_effort("xhigh"), "max")
        self.assertEqual(normalize_reasoning_effort("nonsense"), "high")


class RequestPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ReasoningStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_legacy_functions_field_is_converted_to_tools(self) -> None:
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hi"}],
                "functions": [{"name": "lookup", "parameters": {"type": "object"}}],
                "function_call": "auto",
            },
            ProxyConfig(),
            self.store,
        )
        self.assertEqual(prepared.payload["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(prepared.payload["tool_choice"], "auto")
        self.assertNotIn("functions", prepared.payload)
        self.assertNotIn("function_call", prepared.payload)

    def test_named_function_call_becomes_named_tool_choice(self) -> None:
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hi"}],
                "function_call": {"name": "lookup"},
            },
            ProxyConfig(),
            self.store,
        )
        self.assertEqual(
            prepared.payload["tool_choice"],
            {"type": "function", "function": {"name": "lookup"}},
        )

    def test_max_completion_tokens_is_aliased_to_max_tokens(self) -> None:
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 256,
            },
            ProxyConfig(),
            self.store,
        )
        self.assertEqual(prepared.payload["max_tokens"], 256)

    def test_standard_openai_fields_are_forwarded_without_warning(self) -> None:
        # Cursor and the OpenAI SDK send these on every request; forward them
        # so DeepSeek can use what it understands and ignore the rest.
        with self.assertNoLogs("deepseek_cursor_proxy", level="WARNING"):
            prepared = prepare_upstream_request(
                {
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "hi"}],
                    "user": "user-abc",
                    "seed": 42,
                    "n": 1,
                    "logit_bias": {"50256": -100},
                },
                ProxyConfig(),
                self.store,
            )
        self.assertEqual(prepared.payload["user"], "user-abc")
        self.assertEqual(prepared.payload["seed"], 42)
        self.assertEqual(prepared.payload["n"], 1)
        self.assertEqual(prepared.payload["logit_bias"], {"50256": -100})

    def test_unknown_request_fields_are_dropped_with_warning(self) -> None:
        with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
            prepared = prepare_upstream_request(
                {
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "hi"}],
                    "parallel_tool_calls": True,
                    "service_tier": "fast",
                },
                ProxyConfig(),
                self.store,
            )
        self.assertNotIn("parallel_tool_calls", prepared.payload)
        self.assertNotIn("service_tier", prepared.payload)
        log = "\n".join(captured.output)
        self.assertIn("parallel_tool_calls", log)
        self.assertIn("service_tier", log)

    def test_non_deepseek_model_is_rewritten_with_warning(self) -> None:
        with self.assertLogs("deepseek_cursor_proxy", level="WARNING") as captured:
            prepared = prepare_upstream_request(
                {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                ProxyConfig(upstream_model="deepseek-v4-pro"),
                self.store,
            )
        self.assertEqual(prepared.payload["model"], "deepseek-v4-pro")
        self.assertIn("non-DeepSeek", "\n".join(captured.output))

    def test_thinking_disabled_strips_reasoning_from_assistant_history(self) -> None:
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "should be discarded",
                    },
                ],
            },
            ProxyConfig(thinking="disabled"),
            self.store,
        )
        self.assertEqual(prepared.payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_content", prepared.payload["messages"][1])

    def test_plain_chat_history_does_not_require_reasoning(self) -> None:
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"},
                ],
            },
            ProxyConfig(),
            self.store,
        )
        self.assertEqual(prepared.missing_reasoning_messages, 0)


class RecoveryNoticeStrippingTests(unittest.TestCase):
    def test_strips_only_the_recovery_notice_prefix(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": RECOVERY_NOTICE_CONTENT + "real answer",
            },
            {"role": "assistant", "content": "ordinary"},
        ]
        result = strip_recovery_notice_for_upstream(messages)
        self.assertEqual(result[1]["content"], "real answer")
        self.assertEqual(result[2]["content"], "ordinary")

    def test_returns_a_copy_so_caller_keeps_with_prefix_messages(self) -> None:
        original = [
            {
                "role": "assistant",
                "content": RECOVERY_NOTICE_CONTENT + "answer",
            }
        ]
        stripped = strip_recovery_notice_for_upstream(original)
        # The cache scope is computed on the with-prefix history, so the
        # caller's list must NOT be mutated in place.
        self.assertEqual(original[0]["content"], RECOVERY_NOTICE_CONTENT + "answer")
        self.assertEqual(stripped[0]["content"], "answer")
        self.assertIsNot(stripped[0], original[0])

    def test_text_constant_matches_content_prefix(self) -> None:
        # Sanity check that the user-visible text used as a boundary marker
        # is consistent with the wire-format prefix.
        self.assertTrue(RECOVERY_NOTICE_CONTENT.startswith(RECOVERY_NOTICE_TEXT))


class ResponseRewriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ReasoningStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_records_reasoning_and_restores_original_model_name(self) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Final.",
                            "reasoning_content": "Done thinking.",
                        },
                    }
                ],
            }
        ).encode()
        request_messages = [{"role": "user", "content": "hi"}]
        rewritten = rewrite_response_body(
            body, "deepseek-v4-pro", self.store, request_messages
        )
        payload = json.loads(rewritten)
        self.assertEqual(payload["model"], "deepseek-v4-pro")
        stored = self.store.get(
            f"scope:{conversation_scope(request_messages)}:signature:"
            f"{message_signature(payload['choices'][0]['message'])}"
        )
        self.assertEqual(stored, "Done thinking.")

    def test_recovery_notice_is_prefixed_into_response_content(self) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Final."},
                    }
                ],
            }
        ).encode()
        rewritten = rewrite_response_body(
            body,
            "deepseek-v4-pro",
            self.store,
            [{"role": "user", "content": "hi"}],
            content_prefix=RECOVERY_NOTICE_CONTENT,
        )
        self.assertIn(
            RECOVERY_NOTICE_CONTENT,
            json.loads(rewritten)["choices"][0]["message"]["content"],
        )

    def test_preserves_prompt_cache_usage_fields(self) -> None:
        body = json.dumps(
            {
                "id": "chatcmpl",
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
        usage = json.loads(rewritten)["usage"]
        self.assertEqual(usage["prompt_cache_hit_tokens"], 6)
        self.assertEqual(usage["prompt_cache_miss_tokens"], 4)


class CrossModeAndModelTests(unittest.TestCase):
    """Regression coverage for PR #28's cross-mode/model context preservation
    (Pro↔Flash family normalization, portable turn-scoped keys, recovery
    boundary continuation). Originally shipped with PR #28 in test_transform.py
    and dropped by PR #33's test refactor; restored from commit 5f14da3."""

    def setUp(self) -> None:
        self.store = ReasoningStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_deepseek_pro_and_flash_share_reasoning_namespace(self) -> None:
        config = ProxyConfig()
        namespace_pro = reasoning_cache_namespace(
            config,
            "deepseek-v4-pro",
            {"type": "enabled"},
            "high",
            "Bearer key-a",
        )
        namespace_flash = reasoning_cache_namespace(
            config,
            "deepseek-v4-flash",
            {"type": "enabled"},
            "high",
            "Bearer key-a",
        )
        self.assertEqual(namespace_pro, namespace_flash)

        prior = [{"role": "user", "content": "read README"}]
        tool_call = {
            "id": "call_shared",
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
                "reasoning_content": "Shared DeepSeek reasoning.",
                "tool_calls": [tool_call],
            },
            conversation_scope(prior, namespace_pro),
            namespace_pro,
            prior,
        )

        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-flash",
                "messages": [
                    *prior,
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ],
            },
            config,
            self.store,
            authorization="Bearer key-a",
        )

        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertEqual(
            prepared.payload["messages"][1]["reasoning_content"],
            "Shared DeepSeek reasoning.",
        )

    def test_strict_hit_backfills_portable_cache_for_mode_switch(self) -> None:
        agent_prior = [
            {"role": "system", "content": "Agent mode."},
            {"role": "user", "content": "set up the task"},
            {"role": "user", "content": "read README"},
        ]
        plan_prior = [
            {"role": "system", "content": "Plan mode."},
            {"role": "user", "content": "set up the task"},
            {"role": "user", "content": "read README"},
        ]
        tool_call = {
            "id": "call_mode_switch",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
        }
        assistant_message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Need README before answering.",
            "tool_calls": [tool_call],
        }
        # Store under Agent scope only — no portable aliases yet.
        self.store.store_assistant_message(
            assistant_message,
            _cache_scope(agent_prior),
        )

        # Agent re-request: strict scope hit, should backfill portable.
        strict_prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    *agent_prior,
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ],
            },
            ProxyConfig(),
            self.store,
        )
        # Plan re-request: scope changed (different system prompt) but the
        # turn signature still matches, so the portable alias hits.
        portable_prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    *plan_prior,
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ],
            },
            ProxyConfig(),
            self.store,
        )

        self.assertEqual(strict_prepared.patched_reasoning_messages, 1)
        self.assertEqual(portable_prepared.patched_reasoning_messages, 1)
        self.assertEqual(portable_prepared.missing_reasoning_messages, 0)
        self.assertEqual(
            portable_prepared.payload["messages"][3]["reasoning_content"],
            "Need README before answering.",
        )
        self.assertTrue(
            str(portable_prepared.reasoning_diagnostics[-1]["hit_kind"]).startswith(
                "portable_"
            )
        )

    def test_portable_turn_cache_restores_final_assistant_after_tool_result(
        self,
    ) -> None:
        agent_user = {"role": "user", "content": "look up project state"}
        plan_user = dict(agent_user)
        tool_call = {
            "id": "call_project_state",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"state"}'},
        }
        tool_result = {
            "role": "tool",
            "tool_call_id": "call_project_state",
            "content": '{"state":"ready"}',
        }
        tool_assistant = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Need the project state.",
            "tool_calls": [tool_call],
        }
        final_assistant = {
            "role": "assistant",
            "content": "The project is ready.",
            "reasoning_content": "The tool result is enough to answer.",
        }
        agent_initial_prior = [
            {"role": "system", "content": "Agent mode."},
            agent_user,
        ]
        agent_final_prior = [*agent_initial_prior, tool_assistant, tool_result]
        self.store.store_assistant_message(
            tool_assistant,
            _cache_scope(agent_initial_prior),
            _default_cache_namespace(),
            agent_initial_prior,
        )
        self.store.store_assistant_message(
            final_assistant,
            _cache_scope(agent_final_prior),
            _default_cache_namespace(),
            agent_final_prior,
        )

        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": "Plan mode."},
                    plan_user,
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    tool_result,
                    {"role": "assistant", "content": "The project is ready."},
                    {"role": "user", "content": "continue"},
                ],
            },
            ProxyConfig(missing_reasoning_strategy="reject"),
            self.store,
        )

        self.assertEqual(prepared.missing_reasoning_messages, 0)
        self.assertEqual(prepared.patched_reasoning_messages, 2)
        self.assertEqual(
            prepared.payload["messages"][4]["reasoning_content"],
            "The tool result is enough to answer.",
        )

    def test_portable_turn_cache_isolated_for_reused_tool_call_id(self) -> None:
        # Two different conversations both happen to reuse the same
        # tool_call.id. Cache must NOT cross-contaminate.
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
        prior_a = [
            {"role": "system", "content": "Agent mode."},
            {"role": "user", "content": "thread A"},
        ]
        prior_b = [
            {"role": "system", "content": "Agent mode."},
            {"role": "user", "content": "thread B"},
        ]
        self.store.store_assistant_message(
            assistant_a,
            _cache_scope(prior_a),
            _default_cache_namespace(),
            prior_a,
        )
        self.store.store_assistant_message(
            assistant_b,
            _cache_scope(prior_b),
            _default_cache_namespace(),
            prior_b,
        )

        # Plan-mode replay of thread A — should retrieve A's reasoning, not B's.
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": "Plan mode."},
                    {"role": "user", "content": "thread A"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                ],
            },
            ProxyConfig(),
            self.store,
        )

        self.assertEqual(
            prepared.payload["messages"][2]["reasoning_content"],
            "Reasoning for thread A.",
        )

    def test_recovered_response_is_recorded_under_pre_recovery_scope(self) -> None:
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
            "function": {"name": "lookup", "arguments": '{"query":"new"}'},
        }
        first_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "old model turn"},
                {"role": "assistant", "content": "", "tool_calls": [old_tool_call]},
                {"role": "tool", "tool_call_id": "call_old", "content": "old result"},
                {"role": "user", "content": "continue with DeepSeek"},
            ],
        }
        first_recovered = prepare_upstream_request(
            first_payload,
            ProxyConfig(missing_reasoning_strategy="recover"),
            self.store,
        )
        self.assertEqual(first_recovered.recovered_reasoning_messages, 1)

        # Simulate DeepSeek's response to the recovered request.
        response_body = json.dumps(
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
                            "reasoning_content": "Need the new lookup.",
                            "tool_calls": [new_tool_call],
                        },
                    }
                ],
            }
        ).encode()
        rewritten = rewrite_response_body(
            response_body,
            "deepseek-v4-pro",
            self.store,
            first_recovered.payload["messages"],
            first_recovered.cache_namespace,
            content_prefix=first_recovered.recovery_notice,
            recording_contexts=first_recovered.record_response_contexts,
        )
        recovered_assistant = json.loads(rewritten)["choices"][0]["message"]

        # Reasoning must be recorded under BOTH scopes — pre-recovery (so
        # subsequent Cursor requests echoing the with-prefix history hit) and
        # post-recovery (so an immediate continuation also hits).
        self.assertEqual(len(first_recovered.record_response_contexts), 2)
        for scope, _messages in first_recovered.record_response_contexts:
            self.assertEqual(
                self.store.get(
                    f"scope:{scope}:signature:{message_signature(recovered_assistant)}"
                ),
                "Need the new lookup.",
            )
        recovered_assistant.pop("reasoning_content", None)

        # Cursor's next request echoes the recovered assistant + tool result.
        # The proxy should detect the recovery boundary, retire the prefix,
        # and continue cleanly without recovering again.
        second_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                *first_payload["messages"],
                recovered_assistant,
                {"role": "tool", "tool_call_id": "call_new", "content": "new result"},
            ],
        }

        second_prepared = prepare_upstream_request(
            second_payload,
            ProxyConfig(missing_reasoning_strategy="recover"),
            self.store,
        )

        self.assertEqual(second_prepared.missing_reasoning_messages, 0)
        self.assertEqual(second_prepared.recovered_reasoning_messages, 0)
        self.assertEqual(second_prepared.recovery_dropped_messages, 0)
        self.assertTrue(second_prepared.continued_recovery_boundary)
        self.assertGreater(second_prepared.retired_prefix_messages, 0)
        self.assertEqual(
            second_prepared.payload["messages"][2]["reasoning_content"],
            "Need the new lookup.",
        )


class StopMidStreamingToolCallTests(unittest.TestCase):
    """Regression for the 'Stop pressed during streaming tool-call arguments'
    scenario. When the upstream stream is cut off before the tool_call.id
    chunk arrives, the cached message has tool_calls with no IDs. Cursor
    synthesises its own ID for its bookkeeping, so the next request looks
    nothing like the cached message at the id/signature/message-content
    levels. The tool_name fallback is the only thing that can rescue this."""

    def setUp(self) -> None:
        self.store = ReasoningStore(":memory:")

    def test_tool_name_fallback_restores_reasoning_when_id_missing(self) -> None:
        # Turn 1 prepares an upstream request and caches a partial assistant
        # message simulating a Stop before id arrived.
        first_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
            ],
        }
        first_prepared = prepare_upstream_request(
            first_payload,
            ProxyConfig(missing_reasoning_strategy="recover"),
            self.store,
        )

        partial_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Need to grep.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "grep_search",
                                    "arguments": '{"q":',
                                },
                            }
                        ],
                    },
                }
            ]
        }
        rewrite_response_body(
            json.dumps(partial_response).encode("utf-8"),
            original_model=first_prepared.original_model,
            store=self.store,
            request_messages=first_prepared.record_response_messages,
            cache_namespace=first_prepared.cache_namespace,
            scope=first_prepared.record_response_scope,
            prior_messages=first_prepared.record_response_messages,
            recording_contexts=first_prepared.record_response_contexts,
        )

        # Turn 2: Cursor saved the partial response with a synthesised id and
        # its own best guess for the arguments.
        second_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "cursor-synth-1",
                            "type": "function",
                            "function": {
                                "name": "grep_search",
                                "arguments": '{"q":"foo"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "cursor-synth-1",
                    "content": "match",
                },
                {"role": "user", "content": "u2"},
            ],
        }
        second_prepared = prepare_upstream_request(
            second_payload,
            ProxyConfig(missing_reasoning_strategy="recover"),
            self.store,
        )

        self.assertEqual(second_prepared.patched_reasoning_messages, 1)
        self.assertEqual(second_prepared.missing_reasoning_messages, 0)
        self.assertIsNone(second_prepared.recovery_notice)
        self.assertEqual(
            second_prepared.payload["messages"][2]["reasoning_content"],
            "Need to grep.",
        )

    def test_tool_name_keys_are_isolated_across_distinct_turns(self) -> None:
        # Two separate turns each interrupt with the same function name.
        # The strict scope already differs (each turn has more prior
        # messages) so the two cached entries should not collide and the
        # second turn's reasoning must not leak into the first turn's slot.
        config = ProxyConfig(missing_reasoning_strategy="recover")

        def cache_partial(payload: dict, reasoning: str, args_fragment: str) -> dict:
            prepared = prepare_upstream_request(payload, config, self.store)
            response = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": reasoning,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "grep_search",
                                        "arguments": args_fragment,
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
            rewrite_response_body(
                json.dumps(response).encode("utf-8"),
                original_model=prepared.original_model,
                store=self.store,
                request_messages=prepared.record_response_messages,
                cache_namespace=prepared.cache_namespace,
                scope=prepared.record_response_scope,
                prior_messages=prepared.record_response_messages,
                recording_contexts=prepared.record_response_contexts,
            )
            return prepared

        turn_a_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u-A"},
            ],
        }
        cache_partial(turn_a_payload, "Reasoning A.", '{"q":')

        turn_b_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u-A"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "synth-A",
                            "type": "function",
                            "function": {
                                "name": "grep_search",
                                "arguments": '{"q":"a"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "synth-A", "content": "ra"},
                {"role": "user", "content": "u-B"},
            ],
        }
        cache_partial(turn_b_payload, "Reasoning B.", '{"q":')

        # Now look up turn A's assistant under its own scope. It must still
        # return Reasoning A and never Reasoning B (no scope collision).
        recovery_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u-A"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "synth-A",
                            "type": "function",
                            "function": {
                                "name": "grep_search",
                                "arguments": '{"q":"a"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "synth-A", "content": "ra"},
                {"role": "user", "content": "u-A2"},
            ],
        }
        prepared = prepare_upstream_request(recovery_payload, config, self.store)
        self.assertEqual(
            prepared.payload["messages"][2]["reasoning_content"],
            "Reasoning A.",
        )


if __name__ == "__main__":
    unittest.main()

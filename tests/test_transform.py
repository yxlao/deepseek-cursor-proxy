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
    rewrite_response_body,
    strip_cursor_thinking_blocks,
    strip_recovery_notice_for_upstream,
)


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


if __name__ == "__main__":
    unittest.main()

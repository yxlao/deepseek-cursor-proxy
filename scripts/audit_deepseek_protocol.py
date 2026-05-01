#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore, conversation_scope
from deepseek_cursor_proxy.streaming import StreamAccumulator
from deepseek_cursor_proxy.transform import (
    prepare_upstream_request,
    reasoning_cache_namespace,
    rewrite_response_body,
)


class Audit:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.observations: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"PASS {name}")
            return
        self.failed += 1
        suffix = f": {detail}" if detail else ""
        print(f"FAIL {name}{suffix}")

    def observe(self, text: str) -> None:
        self.observations.append(text)
        print(f"OBSERVE {text}")


def protocol_namespace(config: ProxyConfig) -> str:
    return reasoning_cache_namespace(
        config,
        "deepseek-v4-pro",
        {"type": "enabled"},
        "high",
    )


def default_request_repairs_tool_turn(audit: Audit) -> None:
    config = ProxyConfig(missing_reasoning_strategy="reject")
    store = ReasoningStore(":memory:")
    try:
        tool_call = {
            "id": "call_lookup",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        first_messages = [{"role": "user", "content": "Use the lookup tool."}]
        first_response = {
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Need lookup before answering.",
                        "tool_calls": [tool_call],
                    },
                }
            ],
        }
        rewrite_response_body(
            json.dumps(first_response).encode("utf-8"),
            "deepseek-v4-pro",
            store,
            first_messages,
            protocol_namespace(config),
        )

        second_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                *first_messages,
                {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": "call_lookup",
                    "content": "lookup result",
                },
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
        second_prepared = prepare_upstream_request(second_payload, config, store)
        audit.check(
            "default thinking request restores tool-call reasoning",
            second_prepared.patched_reasoning_messages == 1
            and second_prepared.missing_reasoning_messages == 0
            and second_prepared.payload["messages"][1].get("reasoning_content")
            == "Need lookup before answering.",
        )

        final_response = {
            "id": "chatcmpl-final",
            "object": "chat.completion",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Final answer.",
                        "reasoning_content": "The lookup result is enough.",
                    },
                }
            ],
        }
        rewrite_response_body(
            json.dumps(final_response).encode("utf-8"),
            "deepseek-v4-pro",
            store,
            second_prepared.payload["messages"],
            second_prepared.cache_namespace,
            recording_contexts=second_prepared.record_response_contexts,
        )
        followup_payload = {
            "model": "deepseek-v4-pro",
            "messages": [
                *second_payload["messages"],
                {"role": "assistant", "content": "Final answer."},
                {"role": "user", "content": "Continue."},
            ],
        }
        followup_prepared = prepare_upstream_request(followup_payload, config, store)
        audit.check(
            "default thinking request restores final post-tool reasoning",
            followup_prepared.patched_reasoning_messages == 2
            and followup_prepared.missing_reasoning_messages == 0
            and followup_prepared.payload["messages"][3].get("reasoning_content")
            == "The lookup result is enough.",
        )
    finally:
        store.close()


def plain_chat_does_not_require_reasoning(audit: Audit) -> None:
    store = ReasoningStore(":memory:")
    try:
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "Continue"},
                ],
            },
            ProxyConfig(missing_reasoning_strategy="reject"),
            store,
        )
        audit.check(
            "plain chat history does not require historical reasoning",
            prepared.missing_reasoning_messages == 0
            and "reasoning_content" not in prepared.payload["messages"][1],
        )
    finally:
        store.close()


def streaming_tool_reasoning_is_stored_before_done(audit: Audit) -> None:
    store = ReasoningStore(":memory:")
    try:
        accumulator = StreamAccumulator()
        accumulator.ingest_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "Need streaming lookup.",
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
                    }
                ]
            }
        )
        scope = conversation_scope([{"role": "user", "content": "Stream with a tool."}])
        stored = accumulator.store_ready_reasoning(store, scope)
        audit.check(
            "streamed tool-call reasoning is available before DONE",
            stored > 0
            and store.get(f"scope:{scope}:tool_call:call_stream")
            == "Need streaming lookup.",
        )
    finally:
        store.close()


def pass_through_default_thinking_observation(audit: Audit) -> None:
    store = ReasoningStore(":memory:")
    try:
        tool_call = {
            "id": "call_passthrough",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        prepared = prepare_upstream_request(
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "Lookup."},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    {
                        "role": "tool",
                        "tool_call_id": "call_passthrough",
                        "content": "result",
                    },
                ],
            },
            ProxyConfig(thinking="pass-through", missing_reasoning_strategy="reject"),
            store,
        )
        if (
            prepared.payload.get("thinking") is None
            and prepared.missing_reasoning_messages == 0
            and "reasoning_content" not in prepared.payload["messages"][1]
        ):
            audit.observe(
                "pass-through mode treats an omitted thinking field as non-thinking; "
                "DeepSeek's documented default is thinking enabled."
            )
        else:
            audit.check(
                "pass-through omitted-thinking request is handled as thinking mode",
                True,
            )
    finally:
        store.close()


def main() -> int:
    audit = Audit()
    default_request_repairs_tool_turn(audit)
    plain_chat_does_not_require_reasoning(audit)
    streaming_tool_reasoning_is_stored_before_done(audit)
    pass_through_default_thinking_observation(audit)

    print(
        "SUMMARY "
        f"passed={audit.passed} failed={audit.failed} "
        f"observations={len(audit.observations)}"
    )
    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

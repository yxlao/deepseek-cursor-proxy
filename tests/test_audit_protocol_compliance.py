"""Integration tests adapted from scripts/audit_protocol_compliance.py.

A small in-process "fake DeepSeek" upstream strictly enforces the protocol
contract from docs/thinking-mode-tool-call-flow.md: every assistant message
that participated in a tool-calling turn must carry `reasoning_content`,
otherwise the upstream returns HTTP 400 — the same error real DeepSeek emits.
The proxy is booted in-process and the canonical four-turn tool-call loop is
walked end-to-end. If the proxy is protocol-compliant, every turn succeeds;
otherwise the upstream short-circuits with 400 and the test fails fast.
"""

from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer


THINKING_1_1 = "Thinking 1.1 - need to look up the date."
THINKING_1_2 = "Thinking 1.2 - I have the date, now I need the weather."
THINKING_1_3 = "Thinking 1.3 - tool results suffice for the answer."
THINKING_2_1 = "Thinking 2.1 - a brand new user turn."

ANSWER_1 = "Answer 1: Tomorrow is sunny on 2026-04-24."
ANSWER_2 = "Answer 2: Acknowledged follow-up."

CALL_ID_1 = "call_get_date"
CALL_ID_2 = "call_get_weather"


def _response_turn_1_1() -> dict[str, Any]:
    return {
        "id": "chatcmpl-turn-1-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": THINKING_1_1,
                    "tool_calls": [
                        {
                            "id": CALL_ID_1,
                            "type": "function",
                            "function": {"name": "get_date", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }


def _response_turn_1_2() -> dict[str, Any]:
    return {
        "id": "chatcmpl-turn-1-2",
        "object": "chat.completion",
        "created": 2,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": THINKING_1_2,
                    "tool_calls": [
                        {
                            "id": CALL_ID_2,
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"date":"2026-04-24"}',
                            },
                        }
                    ],
                },
            }
        ],
    }


def _response_turn_1_3() -> dict[str, Any]:
    return {
        "id": "chatcmpl-turn-1-3",
        "object": "chat.completion",
        "created": 3,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": ANSWER_1,
                    "reasoning_content": THINKING_1_3,
                },
            }
        ],
    }


def _response_turn_2_1() -> dict[str, Any]:
    return {
        "id": "chatcmpl-turn-2-1",
        "object": "chat.completion",
        "created": 4,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": ANSWER_2,
                    "reasoning_content": THINKING_2_1,
                },
            }
        ],
    }


class StrictFakeDeepSeek(BaseHTTPRequestHandler):
    """Strict reimplementation of DeepSeek's protocol contract.

    Rejects with HTTP 400 (same error real DeepSeek emits) whenever an
    assistant message that participated in a tool-call turn lacks
    `reasoning_content`. Plain assistant messages between user turns do not
    need reasoning_content.
    """

    requests: list[dict[str, Any]] = []

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)

        messages = payload.get("messages") or []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            if self._is_tool_turn_assistant(messages, index) and not isinstance(
                message.get("reasoning_content"), str
            ):
                return self._send(
                    400,
                    {
                        "error": {
                            "message": (
                                "The reasoning_content in the thinking mode "
                                "must be passed back to the API."
                            ),
                            "type": "invalid_request_error",
                            "code": "invalid_request_error",
                            "missing_index": index,
                        }
                    },
                )

        last_assistant_index = self._last_index(messages, "assistant")
        last_user_index = self._last_index(messages, "user")
        last_tool_index = self._last_index(messages, "tool")

        if (
            last_user_index == 0
            and last_assistant_index == -1
            and last_tool_index == -1
        ):
            return self._send(200, _response_turn_1_1())

        if last_user_index > 0 and last_user_index > last_tool_index:
            return self._send(200, _response_turn_2_1())

        if (
            last_tool_index != -1
            and messages[last_tool_index].get("tool_call_id") == CALL_ID_1
            and last_assistant_index < last_tool_index
        ):
            return self._send(200, _response_turn_1_2())

        if (
            last_tool_index != -1
            and messages[last_tool_index].get("tool_call_id") == CALL_ID_2
        ):
            return self._send(200, _response_turn_1_3())

        return self._send(
            400,
            {
                "error": {
                    "message": (
                        "audit harness: unexpected shape: "
                        f"roles={[m.get('role') for m in messages]}"
                    )
                }
            },
        )

    def _is_tool_turn_assistant(
        self, messages: list[dict[str, Any]], index: int
    ) -> bool:
        message = messages[index]
        if message.get("tool_calls"):
            return True
        for prior in reversed(messages[:index]):
            role = prior.get("role")
            if role == "tool":
                return True
            if role in {"user", "system"}:
                return False
        return False

    @staticmethod
    def _last_index(messages: list[dict[str, Any]], role: str) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == role:
                return index
        return -1

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def cursor_strip(message: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(message)
    cleaned.pop("reasoning_content", None)
    return cleaned


def post(
    url: str,
    payload: dict[str, Any],
    authorization: str = "Bearer sk-audit",
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class _Fixture:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start_strict_upstream() -> _Fixture:
    StrictFakeDeepSeek.requests = []
    return _Fixture(ThreadingHTTPServer(("127.0.0.1", 0), StrictFakeDeepSeek))


def start_proxy(
    upstream_url: str,
    store: ReasoningStore,
    **config_overrides: Any,
) -> _Fixture:
    proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
    proxy.config = ProxyConfig(
        upstream_base_url=upstream_url,
        upstream_model="deepseek-v4-pro",
        ngrok=False,
        verbose=False,
        cors=False,
        **config_overrides,
    )
    proxy.reasoning_store = store
    return _Fixture(proxy)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Returns the current date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns the weather.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        },
    },
]


class StrictUpstreamCase(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = start_strict_upstream()
        self.store = ReasoningStore(":memory:")
        self.proxy = start_proxy(self.upstream.url, self.store)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()


class CanonicalFourTurnLoopTests(StrictUpstreamCase):
    def test_canonical_four_turn_loop_succeeds(self) -> None:
        # Turn 1.1: user only
        status, response_1_1 = post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"}
                ],
                "tools": TOOLS,
            },
        )
        self.assertEqual(status, 200, response_1_1)
        first_assistant = response_1_1["choices"][0]["message"]
        self.assertEqual(first_assistant.get("reasoning_content"), THINKING_1_1)
        self.assertEqual(
            (first_assistant.get("tool_calls") or [{}])[0].get("id"), CALL_ID_1
        )

        # Turn 1.2: + tool result (Cursor strips reasoning_content from history)
        status, response_1_2 = post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"},
                    cursor_strip(first_assistant),
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_1,
                        "content": "2026-04-24",
                    },
                ],
                "tools": TOOLS,
            },
        )
        self.assertEqual(status, 200, response_1_2)
        upstream_seen_1_2 = StrictFakeDeepSeek.requests[1]["messages"]
        self.assertEqual(upstream_seen_1_2[1].get("reasoning_content"), THINKING_1_1)
        second_assistant = response_1_2["choices"][0]["message"]
        self.assertEqual(
            (second_assistant.get("tool_calls") or [{}])[0].get("id"), CALL_ID_2
        )

        # Turn 1.3: + tool result (proxy must patch BOTH prior reasonings)
        status, response_1_3 = post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"},
                    cursor_strip(first_assistant),
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_1,
                        "content": "2026-04-24",
                    },
                    cursor_strip(second_assistant),
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_2,
                        "content": "sunny",
                    },
                ],
                "tools": TOOLS,
            },
        )
        self.assertEqual(status, 200, response_1_3)
        upstream_seen_1_3 = StrictFakeDeepSeek.requests[2]["messages"]
        self.assertEqual(upstream_seen_1_3[1].get("reasoning_content"), THINKING_1_1)
        self.assertEqual(upstream_seen_1_3[3].get("reasoning_content"), THINKING_1_2)
        third_assistant = response_1_3["choices"][0]["message"]
        self.assertIn(ANSWER_1, third_assistant.get("content"))

        # Turn 2.1: brand new user turn (final assistant of prior turn also patched)
        status, response_2_1 = post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"},
                    cursor_strip(first_assistant),
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_1,
                        "content": "2026-04-24",
                    },
                    cursor_strip(second_assistant),
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_2,
                        "content": "sunny",
                    },
                    cursor_strip(third_assistant),
                    {"role": "user", "content": "Thanks. What about Saturday?"},
                ],
                "tools": TOOLS,
            },
        )
        self.assertEqual(status, 200, response_2_1)
        upstream_seen_2_1 = StrictFakeDeepSeek.requests[3]["messages"]
        self.assertEqual(upstream_seen_2_1[1].get("reasoning_content"), THINKING_1_1)
        self.assertEqual(upstream_seen_2_1[3].get("reasoning_content"), THINKING_1_2)
        self.assertEqual(upstream_seen_2_1[5].get("reasoning_content"), THINKING_1_3)
        self.assertEqual(
            upstream_seen_2_1[6],
            {"role": "user", "content": "Thanks. What about Saturday?"},
        )
        self.assertIn(ANSWER_2, response_2_1["choices"][0]["message"].get("content"))


class StrictRejectModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = start_strict_upstream()
        self.store = ReasoningStore(":memory:")
        self.proxy = start_proxy(
            self.upstream.url,
            self.store,
            missing_reasoning_strategy="reject",
        )

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_strict_mode_surfaces_missing_reasoning_without_calling_upstream(
        self,
    ) -> None:
        request = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": CALL_ID_1,
                            "type": "function",
                            "function": {"name": "get_date", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": CALL_ID_1,
                    "content": "2026-04-24",
                },
            ],
        }
        status, payload = post(f"{self.proxy.url}/v1/chat/completions", request)
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["error"]["missing_reasoning_messages"], 1)
        self.assertEqual(StrictFakeDeepSeek.requests, [])


class ThinkingDisabledTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = start_strict_upstream()
        self.store = ReasoningStore(":memory:")
        self.proxy = start_proxy(
            self.upstream.url,
            self.store,
            thinking="disabled",
            missing_reasoning_strategy="recover",
        )

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_thinking_disabled_never_injects_reasoning_content(self) -> None:
        request = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "ping"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Should be discarded by proxy.",
                    "tool_calls": [
                        {
                            "id": CALL_ID_1,
                            "type": "function",
                            "function": {"name": "get_date", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": CALL_ID_1,
                    "content": "2026-04-24",
                },
            ],
        }
        post(f"{self.proxy.url}/v1/chat/completions", request)
        sent = StrictFakeDeepSeek.requests[-1]
        self.assertEqual(sent.get("thinking"), {"type": "disabled"})
        self.assertNotIn("reasoning_content", sent["messages"][1])


class ColdCacheRecoveryTests(StrictUpstreamCase):
    def test_cold_cache_drops_unrecoverable_history_and_prefixes_notice(
        self,
    ) -> None:
        request = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "old work"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": CALL_ID_1,
                            "type": "function",
                            "function": {"name": "get_date", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": CALL_ID_1,
                    "content": "2026-04-24",
                },
                {"role": "user", "content": "Thanks. What about Saturday?"},
            ],
        }
        status, response = post(f"{self.proxy.url}/v1/chat/completions", request)
        self.assertEqual(status, 200, response)
        sent = StrictFakeDeepSeek.requests[-1]
        roles = [m.get("role") for m in sent["messages"]]
        self.assertEqual(roles, ["system", "system", "user"])
        self.assertEqual(
            sent["messages"][-1]["content"], "Thanks. What about Saturday?"
        )
        self.assertIn(
            "[deepseek-cursor-proxy] Refreshed reasoning",
            response["choices"][0]["message"]["content"],
        )


class StreamingThenNonStreamingTests(unittest.TestCase):
    """Cursor often streams Turn 1.1 then issues Turn 1.2 as a non-stream
    POST. The proxy must repair `reasoning_content` from the streamed cache
    before forwarding 1.2 upstream."""

    class _StreamingThenJsonHandler(BaseHTTPRequestHandler):
        requests: list[dict[str, Any]] = []

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.__class__.requests.append(payload)

            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {
                        "id": "stream-tool",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "deepseek-v4-pro",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning_content": THINKING_1_1,
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": CALL_ID_1,
                                            "type": "function",
                                            "function": {
                                                "name": "get_date",
                                                "arguments": "{}",
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "stream-tool",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "deepseek-v4-pro",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                ]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            messages = payload["messages"]
            assistant = messages[1]
            if assistant.get("reasoning_content") != THINKING_1_1:
                body = json.dumps(
                    {
                        "error": {
                            "message": (
                                "missing reasoning_content (got "
                                + repr(assistant.get("reasoning_content"))
                                + ")"
                            )
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = json.dumps(_response_turn_1_2()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def setUp(self) -> None:
        self._StreamingThenJsonHandler.requests = []
        self.upstream = _Fixture(
            ThreadingHTTPServer(("127.0.0.1", 0), self._StreamingThenJsonHandler)
        )
        self.store = ReasoningStore(":memory:")
        self.proxy = start_proxy(self.upstream.url, self.store)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_streaming_then_non_streaming_round_trip(self) -> None:
        stream_request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "stream": True,
                    "messages": [{"role": "user", "content": "go"}],
                    "tools": [TOOLS[0]],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-audit",
                "Content-Type": "application/json",
            },
        )
        with urlopen(stream_request, timeout=5) as response:
            stream_body = response.read().decode("utf-8")
        self.assertIn("data: [DONE]", stream_body)

        status, payload = post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": CALL_ID_1,
                                "type": "function",
                                "function": {
                                    "name": "get_date",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_1,
                        "content": "2026-04-24",
                    },
                ],
                "tools": [TOOLS[0]],
            },
        )
        self.assertEqual(status, 200, payload)
        sent = self._StreamingThenJsonHandler.requests[-1]
        self.assertEqual(sent["messages"][1].get("reasoning_content"), THINKING_1_1)


class AuthorizationNamespaceIsolationTests(StrictUpstreamCase):
    def test_key_swap_does_not_leak_cached_reasoning(self) -> None:
        post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "What is the date?"}],
                "tools": [TOOLS[0]],
            },
            authorization="Bearer sk-USER-A",
        )

        status, _ = post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What is the date?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": CALL_ID_1,
                                "type": "function",
                                "function": {
                                    "name": "get_date",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_1,
                        "content": "2026-04-24",
                    },
                ],
                "tools": [TOOLS[0]],
            },
            authorization="Bearer sk-USER-B",
        )
        self.assertEqual(status, 200)
        sent = StrictFakeDeepSeek.requests[-1]
        leaked = any(
            m.get("role") == "assistant" and m.get("reasoning_content") == THINKING_1_1
            for m in sent["messages"]
        )
        self.assertFalse(leaked)


if __name__ == "__main__":
    unittest.main()

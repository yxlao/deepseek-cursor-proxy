"""End-to-end protocol tests against an in-process fake DeepSeek.

Each test boots the proxy in-process against a small fake upstream and walks
a real HTTP request scenario. The strict variant rejects with HTTP 400 — the
same error real DeepSeek emits — whenever an assistant message that
participated in a tool-calling turn lacks `reasoning_content`. So if the
proxy is protocol-compliant, every turn succeeds; if not, the upstream
short-circuits and the test fails fast.

This file is the ground truth for "does the proxy speak DeepSeek correctly?"
"""

from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer


# Canonical fake-DeepSeek reasoning/answer text reused across tests.
THINKING_1_1 = "Thinking 1.1 - need to look up the date."
THINKING_1_2 = "Thinking 1.2 - I have the date, now I need the weather."
THINKING_1_3 = "Thinking 1.3 - tool results suffice for the answer."
THINKING_2_1 = "Thinking 2.1 - a brand new user turn."
ANSWER_1 = "Answer 1: Tomorrow is sunny on 2026-04-24."
ANSWER_2 = "Answer 2: Acknowledged follow-up."
CALL_ID_1 = "call_get_date"
CALL_ID_2 = "call_get_weather"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Fake upstreams
# ---------------------------------------------------------------------------


def _completion(
    *,
    chat_id: str,
    finish_reason: str,
    content: str = "",
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-v4-pro",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }


class StrictFakeDeepSeek(BaseHTTPRequestHandler):
    """DeepSeek protocol contract: tool-turn assistants must carry
    `reasoning_content` (string). Returns canned canonical responses keyed
    on the request shape; rejects with 400 otherwise."""

    requests: list[dict[str, Any]] = []
    auth_headers: list[str] = []

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)
        self.__class__.auth_headers.append(self.headers.get("Authorization", ""))

        messages = payload.get("messages") or []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if _is_tool_turn_assistant(messages, index) and not isinstance(
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

        last_user = _last_index(messages, "user")
        last_tool = _last_index(messages, "tool")
        last_assistant = _last_index(messages, "assistant")
        if last_user == 0 and last_assistant == -1 and last_tool == -1:
            return self._send(
                200,
                _completion(
                    chat_id="chatcmpl-1-1",
                    finish_reason="tool_calls",
                    reasoning=THINKING_1_1,
                    tool_calls=[
                        {
                            "id": CALL_ID_1,
                            "type": "function",
                            "function": {"name": "get_date", "arguments": "{}"},
                        }
                    ],
                ),
            )
        if last_user > 0 and last_user > last_tool:
            return self._send(
                200,
                _completion(
                    chat_id="chatcmpl-2-1",
                    finish_reason="stop",
                    content=ANSWER_2,
                    reasoning=THINKING_2_1,
                ),
            )
        if (
            last_tool != -1
            and messages[last_tool].get("tool_call_id") == CALL_ID_1
            and last_assistant < last_tool
        ):
            return self._send(
                200,
                _completion(
                    chat_id="chatcmpl-1-2",
                    finish_reason="tool_calls",
                    reasoning=THINKING_1_2,
                    tool_calls=[
                        {
                            "id": CALL_ID_2,
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"date":"2026-04-24"}',
                            },
                        }
                    ],
                ),
            )
        if last_tool != -1 and messages[last_tool].get("tool_call_id") == CALL_ID_2:
            return self._send(
                200,
                _completion(
                    chat_id="chatcmpl-1-3",
                    finish_reason="stop",
                    content=ANSWER_1,
                    reasoning=THINKING_1_3,
                ),
            )
        return self._send(
            400,
            {
                "error": {
                    "message": (
                        "test fake: unexpected shape: "
                        f"roles={[m.get('role') for m in messages]}"
                    )
                }
            },
        )

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _is_tool_turn_assistant(messages: list[dict[str, Any]], index: int) -> bool:
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


def _last_index(messages: list[dict[str, Any]], role: str) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == role:
            return i
    return -1


# ---------------------------------------------------------------------------
# HTTP fixtures
# ---------------------------------------------------------------------------


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


def _start_strict_upstream() -> _Fixture:
    StrictFakeDeepSeek.requests = []
    StrictFakeDeepSeek.auth_headers = []
    return _Fixture(ThreadingHTTPServer(("127.0.0.1", 0), StrictFakeDeepSeek))


def _start_proxy(
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


def _post(
    url: str,
    payload: dict[str, Any],
    authorization: str = "Bearer sk-test",
) -> tuple[int, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": authorization, "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _drop_reasoning(message: dict[str, Any]) -> dict[str, Any]:
    cleaned = deepcopy(message)
    cleaned.pop("reasoning_content", None)
    return cleaned


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------


class _StrictUpstreamCase(unittest.TestCase):
    """Common setup: strict fake DeepSeek + proxy."""

    config_overrides: dict[str, Any] = {}

    def setUp(self) -> None:
        self.upstream = _start_strict_upstream()
        self.store = ReasoningStore(":memory:")
        self.proxy = _start_proxy(
            self.upstream.url, self.store, **self.config_overrides
        )

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()


class CanonicalLoopTests(_StrictUpstreamCase):
    def test_canonical_four_turn_tool_call_loop(self) -> None:
        """Cursor strips reasoning_content from history; the proxy must
        patch every prior assistant message that participated in the tool
        chain so the strict upstream accepts each turn."""
        # Turn 1.1: bare user message.
        status, response_1_1 = _post(
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
        first = response_1_1["choices"][0]["message"]
        self.assertEqual(first["reasoning_content"], THINKING_1_1)
        self.assertEqual(first["tool_calls"][0]["id"], CALL_ID_1)

        # Turn 1.2: append tool result; Cursor drops reasoning.
        status, response_1_2 = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"},
                    _drop_reasoning(first),
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
        second = response_1_2["choices"][0]["message"]
        self.assertEqual(second["tool_calls"][0]["id"], CALL_ID_2)
        upstream_1_2 = StrictFakeDeepSeek.requests[1]["messages"]
        self.assertEqual(upstream_1_2[1]["reasoning_content"], THINKING_1_1)

        # Turn 1.3: both prior assistants need patching.
        status, response_1_3 = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"},
                    _drop_reasoning(first),
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_1,
                        "content": "2026-04-24",
                    },
                    _drop_reasoning(second),
                    {"role": "tool", "tool_call_id": CALL_ID_2, "content": "sunny"},
                ],
                "tools": TOOLS,
            },
        )
        self.assertEqual(status, 200, response_1_3)
        third = response_1_3["choices"][0]["message"]
        self.assertIn(ANSWER_1, third["content"])
        upstream_1_3 = StrictFakeDeepSeek.requests[2]["messages"]
        self.assertEqual(upstream_1_3[1]["reasoning_content"], THINKING_1_1)
        self.assertEqual(upstream_1_3[3]["reasoning_content"], THINKING_1_2)

        # Turn 2.1: brand new user turn; the prior final assistant also
        # needs patching since DeepSeek treats it as part of the tool turn.
        status, response_2_1 = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"},
                    _drop_reasoning(first),
                    {
                        "role": "tool",
                        "tool_call_id": CALL_ID_1,
                        "content": "2026-04-24",
                    },
                    _drop_reasoning(second),
                    {"role": "tool", "tool_call_id": CALL_ID_2, "content": "sunny"},
                    _drop_reasoning(third),
                    {"role": "user", "content": "Thanks. What about Saturday?"},
                ],
                "tools": TOOLS,
            },
        )
        self.assertEqual(status, 200, response_2_1)
        self.assertIn(ANSWER_2, response_2_1["choices"][0]["message"]["content"])
        upstream_2_1 = StrictFakeDeepSeek.requests[3]["messages"]
        self.assertEqual(upstream_2_1[1]["reasoning_content"], THINKING_1_1)
        self.assertEqual(upstream_2_1[3]["reasoning_content"], THINKING_1_2)
        self.assertEqual(upstream_2_1[5]["reasoning_content"], THINKING_1_3)

    def test_authorization_namespace_isolation(self) -> None:
        """A second user with the same conversation prefix must NOT see
        cached reasoning from the first user."""
        # Prime cache as user A.
        _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"}
                ],
                "tools": TOOLS,
            },
            authorization="Bearer sk-USER-A",
        )

        # User B replays a tool history with the exact same shape.
        status, _ = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"},
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
                "tools": TOOLS,
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


class StrictRejectModeTests(_StrictUpstreamCase):
    config_overrides = {"missing_reasoning_strategy": "reject"}

    def test_returns_409_without_calling_upstream(self) -> None:
        status, payload = _post(
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
            },
        )
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["error"]["missing_reasoning_messages"], 1)
        self.assertEqual(StrictFakeDeepSeek.requests, [])


class ThinkingDisabledTests(_StrictUpstreamCase):
    config_overrides = {"thinking": "disabled"}

    def test_disabled_does_not_inject_reasoning(self) -> None:
        _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
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
            },
        )
        sent = StrictFakeDeepSeek.requests[-1]
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_content", sent["messages"][1])


class RecoveryTests(_StrictUpstreamCase):
    def test_cold_cache_recovers_to_latest_user_with_notice(self) -> None:
        """Stale tool history with no cached reasoning: proxy keeps only
        the latest user message + recovery system message and prefixes a
        user-facing notice into the response."""
        status, response = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
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
            },
        )
        self.assertEqual(status, 200, response)
        sent = StrictFakeDeepSeek.requests[-1]
        self.assertEqual(
            [m["role"] for m in sent["messages"]], ["system", "system", "user"]
        )
        self.assertEqual(
            sent["messages"][-1]["content"], "Thanks. What about Saturday?"
        )
        self.assertIn(
            "[deepseek-cursor-proxy] Refreshed reasoning",
            response["choices"][0]["message"]["content"],
        )

    def test_recovery_notice_is_stripped_before_upstream_replay(self) -> None:
        """When Cursor echoes a previous response (which carried the proxy's
        recovery notice) back as assistant content, the notice serves as a
        boundary marker for the proxy but must not be replayed upstream as
        if DeepSeek had written it."""
        # Trigger initial recovery so the response carries the notice.
        status, first = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
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
            },
        )
        self.assertEqual(status, 200)

        # Cursor faithfully echoes the response (including the notice prefix).
        echoed = _drop_reasoning(first["choices"][0]["message"])
        status, _ = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
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
                    echoed,
                    {"role": "user", "content": "And Sunday?"},
                ],
            },
        )
        self.assertEqual(status, 200)
        sent = StrictFakeDeepSeek.requests[-1]
        for message in sent["messages"]:
            if message.get("role") != "assistant":
                continue
            self.assertNotIn("deepseek-cursor-proxy", message.get("content", ""))

    def test_recover_mode_does_not_short_circuit_with_409(self) -> None:
        """In `recover` mode, a payload with no user message leaves the
        recovery loop unable to drop anything (`dropped_messages == 0`),
        so `missing_indexes` stays populated. The proxy must NOT 409 in
        that case — it must forward to upstream and relay whatever
        DeepSeek decides. 409 is reserved for `reject` mode."""
        status, _ = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": "Be brief."},
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
            },
        )
        # Strict upstream rejects the missing-reasoning history with 400.
        # The point of this test is the proxy did NOT pre-empt with 409.
        self.assertNotEqual(status, 409)
        self.assertEqual(status, 400)
        self.assertEqual(len(StrictFakeDeepSeek.requests), 1)


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------


def _sse_chunks(*chunks: dict[str, Any]) -> bytes:
    out = b""
    for chunk in chunks:
        out += f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    out += b"data: [DONE]\n\n"
    return out


class _StreamingThenJsonHandler(BaseHTTPRequestHandler):
    """First request streams a tool call (with reasoning); subsequent
    non-streaming requests echo a final answer if the assistant message
    carries the previously-streamed reasoning_content."""

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
            self.wfile.write(
                _sse_chunks(
                    {
                        "id": "stream-1",
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
                        "id": "stream-1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "deepseek-v4-pro",
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                        ],
                    },
                )
            )
            self.wfile.flush()
            return

        # Non-streaming follow-up: enforce that proxy patched the prior
        # streamed reasoning_content into history.
        assistant = payload["messages"][1]
        if assistant.get("reasoning_content") != THINKING_1_1:
            self._send(400, {"error": {"message": "missing streamed reasoning"}})
            return
        self._send(
            200,
            _completion(
                chat_id="follow-up",
                finish_reason="stop",
                content="follow-up accepted",
                reasoning="post-tool reasoning",
            ),
        )

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class StreamingThenNonStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        _StreamingThenJsonHandler.requests = []
        self.upstream = _Fixture(
            ThreadingHTTPServer(("127.0.0.1", 0), _StreamingThenJsonHandler)
        )
        self.store = ReasoningStore(":memory:")
        self.proxy = _start_proxy(self.upstream.url, self.store)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_streamed_reasoning_is_replayed_in_next_non_streaming_request(
        self,
    ) -> None:
        """Cursor often streams Turn 1.1 then issues Turn 1.2 as a non-stream
        POST. The proxy must repair reasoning_content from the streaming
        cache before forwarding the follow-up."""
        # Stream a tool call.
        request = Request(
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
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=5) as response:
            self.assertIn("data: [DONE]", response.read().decode("utf-8"))

        # Non-streaming follow-up (Cursor strips reasoning_content).
        status, payload = _post(
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
        sent = _StreamingThenJsonHandler.requests[-1]
        self.assertEqual(sent["messages"][1]["reasoning_content"], THINKING_1_1)


class _ReasoningStreamHandler(BaseHTTPRequestHandler):
    """Streams reasoning_content tokens followed by a content answer."""

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(
            _sse_chunks(
                {
                    "id": "stream-r",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-v4-pro",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "reasoning_content": "Need ",
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "stream-r",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-v4-pro",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "context."},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "stream-r",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-v4-pro",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Final."},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "stream-r",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-v4-pro",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            )
        )
        self.wfile.flush()


class StreamingDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = _Fixture(
            ThreadingHTTPServer(("127.0.0.1", 0), _ReasoningStreamHandler)
        )
        self.store = ReasoningStore(":memory:")
        self.proxy = _start_proxy(self.upstream.url, self.store)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_streaming_response_mirrors_reasoning_into_details_block(self) -> None:
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream"}],
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=2) as response:
            body = response.read().decode("utf-8")

        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(
            chunks[0]["choices"][0]["delta"]["content"],
            "<details>\n<summary>Thinking</summary>\n\nNeed ",
        )
        self.assertEqual(
            chunks[2]["choices"][0]["delta"]["content"],
            "\n</details>\n\nFinal.",
        )


class NonStreamingDisplayTests(_StrictUpstreamCase):
    def test_non_streaming_response_mirrors_reasoning_into_details_block(
        self,
    ) -> None:
        """The README claims thinking tokens are displayed in Cursor; this
        must hold for non-streaming responses too, not only streaming ones."""
        status, response = _post(
            f"{self.proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": "What's the weather tomorrow?"}
                ],
                "tools": TOOLS,
            },
        )
        self.assertEqual(status, 200, response)
        # Turn 1.1 has empty content + reasoning + tool calls.
        content = response["choices"][0]["message"]["content"]
        self.assertEqual(
            content,
            f"<details>\n<summary>Thinking</summary>\n\n{THINKING_1_1}\n</details>\n\n",
        )


# ---------------------------------------------------------------------------
# Concurrent threads (independent fake to keep the canonical strict fake
# stateless across the suite).
# ---------------------------------------------------------------------------


class _PerThreadFakeDeepSeek(BaseHTTPRequestHandler):
    """Strict fake keyed on a thread-name embedded in the user message; each
    thread expects its own reasoning text. Catches cross-thread cache leaks."""

    requests: list[dict[str, Any]] = []

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)

        thread = self._thread_name(payload.get("messages") or [])
        expected_tool = f"tool reasoning {thread}"
        expected_final = f"final reasoning {thread}"
        final_content = f"answer {thread}"

        for index, message in enumerate(payload.get("messages") or []):
            if message.get("role") != "assistant":
                continue
            if (
                message.get("tool_calls")
                and message.get("reasoning_content") != expected_tool
            ):
                return self._send(400, {"error": {"missing_index": index}})
            if (
                message.get("content") == final_content
                and message.get("reasoning_content") != expected_final
            ):
                return self._send(400, {"error": {"missing_index": index}})

        messages = payload.get("messages") or []
        if len(messages) == 1:
            return self._send(
                200,
                _completion(
                    chat_id=f"tool-{thread}",
                    finish_reason="tool_calls",
                    reasoning=expected_tool,
                    tool_calls=[
                        {
                            "id": "call_reused",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                ),
            )
        if len(messages) == 3:
            return self._send(
                200,
                _completion(
                    chat_id=f"final-{thread}",
                    finish_reason="stop",
                    content=final_content,
                    reasoning=expected_final,
                ),
            )
        return self._send(
            200,
            _completion(
                chat_id=f"followup-{thread}",
                finish_reason="stop",
                content=f"follow-up {thread}",
                reasoning="follow-up reasoning",
            ),
        )

    @staticmethod
    def _thread_name(messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "?"
        text = str(messages[0].get("content") or "")
        if "thread A" in text:
            return "A"
        if "thread B" in text:
            return "B"
        return "?"

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ConcurrentThreadTests(unittest.TestCase):
    def setUp(self) -> None:
        _PerThreadFakeDeepSeek.requests = []
        self.upstream = _Fixture(
            ThreadingHTTPServer(("127.0.0.1", 0), _PerThreadFakeDeepSeek)
        )
        self.store = ReasoningStore(":memory:")
        self.proxy = _start_proxy(self.upstream.url, self.store)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_two_interleaved_threads_do_not_leak_reasoning(self) -> None:
        """Tool-call IDs are reused across both threads. The proxy must
        scope cache by conversation, not by tool_call_id, so each thread
        sees only its own reasoning."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        def first(thread: str) -> dict[str, Any]:
            return {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": f"Start thread {thread}."}],
                "tools": tools,
            }

        def second(thread: str, assistant: dict[str, Any]) -> dict[str, Any]:
            return {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "user", "content": f"Start thread {thread}."},
                    _drop_reasoning(assistant),
                    {
                        "role": "tool",
                        "tool_call_id": "call_reused",
                        "content": f"tool result {thread}",
                    },
                ],
                "tools": tools,
            }

        status, first_a = _post(f"{self.proxy.url}/v1/chat/completions", first("A"))
        self.assertEqual(status, 200)
        status, first_b = _post(f"{self.proxy.url}/v1/chat/completions", first("B"))
        self.assertEqual(status, 200)
        status, _ = _post(
            f"{self.proxy.url}/v1/chat/completions",
            second("B", first_b["choices"][0]["message"]),
        )
        self.assertEqual(status, 200)
        status, _ = _post(
            f"{self.proxy.url}/v1/chat/completions",
            second("A", first_a["choices"][0]["message"]),
        )
        self.assertEqual(status, 200)

        # If the cache leaked, the upstream's strict reasoning check would
        # have rejected one of these turns with 400.
        upstream_b = _PerThreadFakeDeepSeek.requests[2]["messages"]
        upstream_a = _PerThreadFakeDeepSeek.requests[3]["messages"]
        self.assertEqual(upstream_b[1]["reasoning_content"], "tool reasoning B")
        self.assertEqual(upstream_a[1]["reasoning_content"], "tool reasoning A")


# ---------------------------------------------------------------------------
# Streaming-cache timing: tool reasoning must be available before [DONE].
# ---------------------------------------------------------------------------


class _SlowToolStreamHandler(BaseHTTPRequestHandler):
    """Streams reasoning + tool_calls, then waits before sending [DONE].
    Lets us verify the proxy stores reasoning at finish_reason=tool_calls,
    not at [DONE]."""

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

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
                                "reasoning_content": "Streamed tool reasoning.",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_stream_tool",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
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
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                },
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
                if chunk["choices"][0]["finish_reason"] is None:
                    time.sleep(0.2)
            time.sleep(1)  # delay [DONE] so the follow-up beats it.
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        # Non-streaming follow-up.
        messages = payload.get("messages") or []
        if (
            len(messages) >= 2
            and messages[1].get("reasoning_content") == "Streamed tool reasoning."
        ):
            self._send(
                200,
                _completion(
                    chat_id="follow",
                    finish_reason="stop",
                    content="follow-up accepted",
                    reasoning="post-tool",
                ),
            )
            return
        self._send(400, {"error": {"message": "missing streamed reasoning"}})

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class StreamingCacheTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = _Fixture(
            ThreadingHTTPServer(("127.0.0.1", 0), _SlowToolStreamHandler)
        )
        self.store = ReasoningStore(":memory:")
        self.proxy = _start_proxy(self.upstream.url, self.store)

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_tool_reasoning_is_cached_before_done(self) -> None:
        """A follow-up POST issued before [DONE] must still find the
        streamed reasoning in cache."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        request = Request(
            f"{self.proxy.url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream tool"}],
                    "tools": tools,
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=3) as response:
            # Read until we see the tool_calls chunk; the upstream then
            # delays [DONE] for ~1s, giving us a window to send a follow-up.
            while True:
                line = response.readline().decode("utf-8")
                self.assertNotEqual(line, "")
                if '"finish_reason":"tool_calls"' in line:
                    break

            status, payload = _post(
                f"{self.proxy.url}/v1/chat/completions",
                {
                    "model": "deepseek-v4-pro",
                    "messages": [
                        {"role": "user", "content": "stream tool"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_stream_tool",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_stream_tool",
                            "content": "tool result",
                        },
                    ],
                    "tools": tools,
                },
            )
            response.read()
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            payload["choices"][0]["message"]["content"].split("</details>")[-1],
            "\n\nfollow-up accepted",
        )


if __name__ == "__main__":
    unittest.main()

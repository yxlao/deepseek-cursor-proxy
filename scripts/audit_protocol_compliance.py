"""Standalone audit harness for docs/thinking-mode-tool-call-flow.md.

This script does not import anything from tests/. It boots a real
DeepSeekProxyServer in-process, points it at a tiny "fake DeepSeek" upstream
that strictly enforces the protocol described in the doc, and walks the four
canonical turns:

    Turn 1.1: user only          -> assistant.tool_calls + reasoning_content
    Turn 1.2: + tool result      -> assistant.tool_calls + reasoning_content
    Turn 1.3: + tool result      -> final assistant.content
    Turn 2.1: + new user message -> follow-up

Cursor is simulated by stripping `reasoning_content` from every assistant
message before posting back to the proxy. The fake upstream rejects with HTTP
400 (the same DeepSeek error from the README) whenever an assistant message
that needs `reasoning_content` doesn't have it. So if the proxy is
protocol-compliant, all four turns succeed end-to-end. If it isn't, the
upstream returns 400 and we fail fast.

Run it with:

    uv run python scripts/audit_protocol_compliance.py
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer


# ----- canonical reasoning/content texts the doc describes -----
THINKING_1_1 = "Thinking 1.1 - need to look up the date."
THINKING_1_2 = "Thinking 1.2 - I have the date, now I need the weather."
THINKING_1_3 = "Thinking 1.3 - tool results suffice for the answer."
THINKING_2_1 = "Thinking 2.1 - a brand new user turn."

ANSWER_1 = "Answer 1: Tomorrow is sunny on 2026-04-24."
ANSWER_2 = "Answer 2: Acknowledged follow-up."

CALL_ID_1 = "call_get_date"
CALL_ID_2 = "call_get_weather"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# -------------------------------------------------------------------
# Fake DeepSeek upstream that strictly enforces the protocol contract.
# -------------------------------------------------------------------
class StrictFakeDeepSeek(BaseHTTPRequestHandler):
    """Strict reimplementation of DeepSeek's protocol contract.

    Rules enforced (as documented in
    docs/thinking-mode-tool-call-flow.md):

    1. If an assistant message in the request requested tool_calls, the same
       message must carry `reasoning_content`. Otherwise HTTP 400 with the
       canonical "reasoning_content in the thinking mode must be passed back"
       error.
    2. If an assistant `content`-only message follows a `tool` message,
       it also belongs to the tool-calling turn and must carry
       `reasoning_content`. Same 400 otherwise.
    3. Plain assistant messages between user turns (no tool involvement) do
       NOT need reasoning_content.
    """

    requests: list[dict[str, Any]] = []

    # silent
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)

        messages = payload.get("messages") or []

        # Validate: every assistant message that participated in a tool call
        # turn must carry reasoning_content (string, may be empty).
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            needs_reasoning = self._is_tool_turn_assistant(messages, index)
            if needs_reasoning and not isinstance(
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

        # Decide which canonical response to return based on the input shape.
        last_assistant_index = self._last_index(messages, "assistant")
        last_user_index = self._last_index(messages, "user")
        last_tool_index = self._last_index(messages, "tool")

        # Turn 1.1: only a single user message.
        if (
            last_user_index == 0
            and last_assistant_index == -1
            and last_tool_index == -1
        ):
            return self._send(200, _response_turn_1_1())

        # Turn 2.1: brand new user message after a complete prior turn.
        # Detect this BEFORE the tool-result branches because Turn 2.1
        # carries earlier tool messages too.
        if last_user_index > 0 and last_user_index > last_tool_index:
            return self._send(200, _response_turn_2_1())

        # Turn 1.2: tool result for call_1 has been appended.
        if (
            last_tool_index != -1
            and messages[last_tool_index].get("tool_call_id") == CALL_ID_1
            and last_assistant_index < last_tool_index
        ):
            return self._send(200, _response_turn_1_2())

        # Turn 1.3: tool result for call_2 has been appended.
        if (
            last_tool_index != -1
            and messages[last_tool_index].get("tool_call_id") == CALL_ID_2
        ):
            return self._send(200, _response_turn_1_3())

        # Anything else is an unexpected shape for this audit harness.
        return self._send(
            400,
            {
                "error": {
                    "message": f"audit harness: unexpected shape: roles={[m.get('role') for m in messages]}"
                }
            },
        )

    def _is_tool_turn_assistant(
        self, messages: list[dict[str, Any]], index: int
    ) -> bool:
        message = messages[index]
        if message.get("tool_calls"):
            return True
        # If a tool message appears between this assistant and the previous
        # user/system, this assistant is part of the tool-calling turn.
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


# -------------------------------------------------------------------
# Canonical DeepSeek responses for each turn
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Helpers to simulate Cursor stripping reasoning_content
# -------------------------------------------------------------------
def cursor_strip(message: dict[str, Any]) -> dict[str, Any]:
    """Cursor faithfully echoes assistant messages but drops
    reasoning_content from tool-call history."""
    cleaned = deepcopy(message)
    cleaned.pop("reasoning_content", None)
    return cleaned


def post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer sk-audit",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# -------------------------------------------------------------------
# Servers
# -------------------------------------------------------------------
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


def _start_upstream() -> _Fixture:
    StrictFakeDeepSeek.requests = []
    return _Fixture(ThreadingHTTPServer(("127.0.0.1", 0), StrictFakeDeepSeek))


def _start_proxy(upstream_url: str, store: ReasoningStore) -> _Fixture:
    proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
    proxy.config = ProxyConfig(
        upstream_base_url=upstream_url,
        upstream_model="deepseek-v4-pro",
        ngrok=False,
        verbose=False,
        cors=False,
    )
    proxy.reasoning_store = store
    return _Fixture(proxy)


# -------------------------------------------------------------------
# Assertions and mini test cases
# -------------------------------------------------------------------
def expect(name: str, condition: bool, detail: str = "") -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"  [{marker}] {name}{(': ' + detail) if detail else ''}")
    if not condition:
        raise AssertionError(name + (": " + detail if detail else ""))


def case_canonical_loop() -> None:
    """Walk through the exact Turn 1.1 -> 1.2 -> 1.3 -> 2.1 cadence."""
    print("\n[Case 1] Canonical four-turn tool-call loop")

    upstream = _start_upstream()
    store = ReasoningStore(":memory:")
    proxy = _start_proxy(upstream.url, store)
    try:
        tools = [
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

        # ----- Turn 1.1 -----
        turn_1_1_request = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "What's the weather tomorrow?"}],
            "tools": tools,
        }
        status, response_1_1 = post(
            f"{proxy.url}/v1/chat/completions", turn_1_1_request
        )
        expect("turn 1.1 status 200", status == 200, str(response_1_1))
        first_assistant = response_1_1["choices"][0]["message"]
        expect(
            "turn 1.1 response carries reasoning_content (proxy doesn't strip)",
            first_assistant.get("reasoning_content") == THINKING_1_1,
        )
        expect(
            "turn 1.1 returned tool_calls with id call_get_date",
            (first_assistant.get("tool_calls") or [{}])[0].get("id") == CALL_ID_1,
        )

        # ----- Turn 1.2 (Cursor strips reasoning_content) -----
        turn_1_2_request = {
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
            "tools": tools,
        }
        status, response_1_2 = post(
            f"{proxy.url}/v1/chat/completions", turn_1_2_request
        )
        expect(
            "turn 1.2 status 200 (proxy patched reasoning_content)",
            status == 200,
            str(response_1_2),
        )
        upstream_seen_1_2 = StrictFakeDeepSeek.requests[1]["messages"]
        expect(
            "turn 1.2 upstream saw THINKING_1_1 in the first assistant message",
            upstream_seen_1_2[1].get("reasoning_content") == THINKING_1_1,
        )
        second_assistant = response_1_2["choices"][0]["message"]
        expect(
            "turn 1.2 response: tool_calls call_get_weather",
            (second_assistant.get("tool_calls") or [{}])[0].get("id") == CALL_ID_2,
        )

        # ----- Turn 1.3 (Cursor strips reasoning from BOTH prior assistants) -----
        turn_1_3_request = {
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
            "tools": tools,
        }
        status, response_1_3 = post(
            f"{proxy.url}/v1/chat/completions", turn_1_3_request
        )
        expect(
            "turn 1.3 status 200 (proxy patched both prior reasonings)",
            status == 200,
            str(response_1_3),
        )
        upstream_seen_1_3 = StrictFakeDeepSeek.requests[2]["messages"]
        expect(
            "turn 1.3 upstream saw THINKING_1_1",
            upstream_seen_1_3[1].get("reasoning_content") == THINKING_1_1,
        )
        expect(
            "turn 1.3 upstream saw THINKING_1_2",
            upstream_seen_1_3[3].get("reasoning_content") == THINKING_1_2,
        )
        third_assistant = response_1_3["choices"][0]["message"]
        expect(
            "turn 1.3 final answer reaches the client",
            third_assistant.get("content") == ANSWER_1,
        )

        # ----- Turn 2.1 (new user turn) -----
        turn_2_1_request = {
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
            "tools": tools,
        }
        status, response_2_1 = post(
            f"{proxy.url}/v1/chat/completions", turn_2_1_request
        )
        expect(
            "turn 2.1 status 200 (final assistant of prior turn also patched)",
            status == 200,
            str(response_2_1),
        )
        upstream_seen_2_1 = StrictFakeDeepSeek.requests[3]["messages"]
        expect(
            "turn 2.1 upstream saw THINKING_1_1 on first assistant",
            upstream_seen_2_1[1].get("reasoning_content") == THINKING_1_1,
        )
        expect(
            "turn 2.1 upstream saw THINKING_1_2 on tool-calling assistant",
            upstream_seen_2_1[3].get("reasoning_content") == THINKING_1_2,
        )
        expect(
            "turn 2.1 upstream saw THINKING_1_3 on the final assistant",
            upstream_seen_2_1[5].get("reasoning_content") == THINKING_1_3,
        )
        expect(
            "turn 2.1 last message is the new user message",
            upstream_seen_2_1[6]
            == {
                "role": "user",
                "content": "Thanks. What about Saturday?",
            },
        )
        expect(
            "turn 2.1 final answer reaches the client",
            response_2_1["choices"][0]["message"].get("content") == ANSWER_2,
        )
    finally:
        proxy.close()
        upstream.close()
        store.close()


def case_strict_mode_rejects_uncached() -> None:
    """Strict mode (`--missing-reasoning-strategy reject`) must NOT silently
    forward broken history. Should respond 409 to the client. Doc covers this
    indirectly: the proxy "exists" because Cursor omits reasoning_content; in
    strict debugging mode the proxy must surface the gap rather than corrupt
    the upstream."""
    print("\n[Case 2] Strict mode surfaces missing reasoning_content")

    upstream = _start_upstream()
    store = ReasoningStore(":memory:")
    proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
    proxy.config = ProxyConfig(
        upstream_base_url=upstream.url,
        upstream_model="deepseek-v4-pro",
        ngrok=False,
        missing_reasoning_strategy="reject",
    )
    proxy.reasoning_store = store
    fixture = _Fixture(proxy)
    try:
        tool_call = {
            "id": CALL_ID_1,
            "type": "function",
            "function": {"name": "get_date", "arguments": "{}"},
        }
        request = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": CALL_ID_1,
                    "content": "2026-04-24",
                },
            ],
        }
        status, payload = post(f"{fixture.url}/v1/chat/completions", request)
        expect(
            "strict mode returns 409 instead of forwarding bad history",
            status == 409,
            str(payload),
        )
        expect(
            "strict mode includes missing_reasoning_messages count",
            payload["error"]["missing_reasoning_messages"] == 1,
        )
        expect(
            "strict mode does NOT call upstream",
            len(StrictFakeDeepSeek.requests) == 0,
        )
    finally:
        fixture.close()
        upstream.close()
        store.close()


def case_thinking_disabled_drops_reasoning() -> None:
    """Doc: `extra_body={"thinking": {"type": "disabled"}}` turns thinking off.
    With thinking disabled the proxy must NOT inject reasoning_content into
    outgoing requests, even if the cache has it."""
    print("\n[Case 3] thinking=disabled never injects reasoning_content")

    upstream = _start_upstream()
    store = ReasoningStore(":memory:")
    proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
    proxy.config = ProxyConfig(
        upstream_base_url=upstream.url,
        upstream_model="deepseek-v4-pro",
        ngrok=False,
        thinking="disabled",
        # In thinking=disabled the strict upstream wouldn't reject anyway,
        # but we want a clean signal that the proxy didn't add reasoning.
        missing_reasoning_strategy="recover",
    )
    proxy.reasoning_store = store
    fixture = _Fixture(proxy)
    try:
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
        # The strict upstream would reject because thinking is disabled-but
        # there's a tool turn. We don't care: we only inspect what the proxy
        # SENT to the upstream, regardless of whether DeepSeek 400s.
        post(f"{fixture.url}/v1/chat/completions", request)
        sent = StrictFakeDeepSeek.requests[-1]
        expect(
            "thinking={'type':'disabled'} forwarded",
            sent.get("thinking") == {"type": "disabled"},
        )
        expect(
            "no reasoning_content on assistant tool message when disabled",
            "reasoning_content" not in sent["messages"][1],
        )
    finally:
        fixture.close()
        upstream.close()
        store.close()


def case_pass_through_keeps_client_thinking() -> None:
    """Doc lets clients toggle thinking via `extra_body`. The proxy supports a
    `pass-through` mode where the client decides. Verify `thinking` is left
    untouched in pass-through mode."""
    print("\n[Case 4] thinking=pass-through honors client toggle")

    upstream = _start_upstream()
    store = ReasoningStore(":memory:")
    proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
    proxy.config = ProxyConfig(
        upstream_base_url=upstream.url,
        upstream_model="deepseek-v4-pro",
        ngrok=False,
        thinking="pass-through",
        missing_reasoning_strategy="recover",
    )
    proxy.reasoning_store = store
    fixture = _Fixture(proxy)
    try:
        request = {
            "model": "deepseek-v4-pro",
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "ping"}],
        }
        status, _ = post(f"{fixture.url}/v1/chat/completions", request)
        sent = StrictFakeDeepSeek.requests[-1]
        expect(
            "client-supplied thinking=disabled survives pass-through",
            sent.get("thinking") == {"type": "disabled"},
        )

        # When the client sends no thinking field, pass-through means the
        # proxy must also send none (so DeepSeek picks its own default).
        StrictFakeDeepSeek.requests.clear()
        post(
            f"{fixture.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        sent = StrictFakeDeepSeek.requests[-1]
        expect(
            "no thinking key when client omits it in pass-through",
            "thinking" not in sent,
        )
    finally:
        fixture.close()
        upstream.close()
        store.close()


def case_streaming_turn_1_2_round_trip() -> None:
    """Streaming variant of the protocol contract: stream the Turn 1.1 tool
    call, then post Turn 1.2 with reasoning stripped. Proxy must restore
    reasoning_content from the streamed cache before forwarding 1.2."""
    print("\n[Case 5] Streaming -> non-streaming tool-call round trip")

    class StreamingThenJson(BaseHTTPRequestHandler):
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

            # Non-streaming follow-up: validate prior assistant has reasoning.
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

    StreamingThenJson.requests = []
    upstream = _Fixture(ThreadingHTTPServer(("127.0.0.1", 0), StreamingThenJson))
    store = ReasoningStore(":memory:")
    proxy = _start_proxy(upstream.url, store)
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_date",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # ----- Turn 1.1 streaming -----
        request = Request(
            f"{proxy.url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "stream": True,
                    "messages": [{"role": "user", "content": "go"}],
                    "tools": tools,
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-audit",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=5) as response:
            stream_body = response.read().decode("utf-8")
        expect(
            "streaming response contains [DONE]",
            "data: [DONE]" in stream_body,
        )

        # Give the accumulator a moment to flush stores, then assert.
        # (StreamAccumulator stores on [DONE], synchronous, so no sleep needed.)
        # ----- Turn 1.2 non-streaming -----
        status, payload = post(
            f"{proxy.url}/v1/chat/completions",
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
                "tools": tools,
            },
        )
        expect(
            "streaming -> non-streaming round trip status 200",
            status == 200,
            str(payload),
        )
        sent = StreamingThenJson.requests[-1]
        expect(
            "proxy patched THINKING_1_1 from streaming cache",
            sent["messages"][1].get("reasoning_content") == THINKING_1_1,
        )
    finally:
        proxy.close()
        upstream.close()
        store.close()


def case_recovery_drops_unrecoverable_history() -> None:
    """README & doc: when reasoning is missing AND can't be recovered, the
    proxy continues from the latest user request and prefixes a recovery
    notice. Verify the proxy sends only the latest user message after a
    cold-cache strict gap, never a 400-baiting tool history."""
    print("\n[Case 6] Cold cache: drops unrecoverable history, prefixes notice")

    upstream = _start_upstream()
    store = ReasoningStore(":memory:")
    proxy = _start_proxy(upstream.url, store)
    try:
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
        status, response = post(f"{proxy.url}/v1/chat/completions", request)
        expect(
            "recover mode succeeds even with empty cache",
            status == 200,
            str(response),
        )
        sent = StrictFakeDeepSeek.requests[-1]
        roles = [m.get("role") for m in sent["messages"]]
        expect(
            "upstream only sees system/system/user after cold-cache recovery",
            roles == ["system", "system", "user"],
            str(roles),
        )
        expect(
            "kept the latest user message (the active query)",
            sent["messages"][-1]["content"] == "Thanks. What about Saturday?",
        )
        prefix = response["choices"][0]["message"]["content"]
        expect(
            "client receives recovery-notice prefix",
            prefix.startswith("[deepseek-cursor-proxy] Refreshed reasoning"),
            prefix,
        )
    finally:
        proxy.close()
        upstream.close()
        store.close()


def case_authorization_isolates_namespace() -> None:
    """The README claims the cache namespace includes a hash of the API key
    (so two users on the same machine never see each other's reasoning).
    Verify a key swap really does break cache hits."""
    print("\n[Case 7] Authorization-keyed cache namespace isolation")

    upstream = _start_upstream()
    store = ReasoningStore(":memory:")
    proxy = _start_proxy(upstream.url, store)
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_date",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        # Prime the cache via key A.
        post(
            f"{proxy.url}/v1/chat/completions",
            {
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "What is the date?"}],
                "tools": tools,
            },
        )

        # Now repeat the exact tool history but with key B.
        request = Request(
            f"{proxy.url}/v1/chat/completions",
            data=json.dumps(
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
                    "tools": tools,
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-OTHER-USER",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        # Note: under recover mode the proxy successfully sends a sanitized
        # request, so we look at what was actually sent upstream.
        sent = StrictFakeDeepSeek.requests[-1]
        # If the cache had leaked across keys, there would be a tool-call
        # assistant message with reasoning. Recovery mode means it should
        # have collapsed to a single user message instead.
        roles = [m.get("role") for m in sent["messages"]]
        expect(
            "key-B request after key-A priming did NOT see leaked reasoning",
            "assistant" not in roles
            or all(
                "reasoning_content" not in m
                or m.get("reasoning_content") in (None, "")
                or m.get("reasoning_content") != THINKING_1_1
                for m in sent["messages"]
                if m.get("role") == "assistant"
            ),
        )
        del payload  # unused
    finally:
        proxy.close()
        upstream.close()
        store.close()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> int:
    cases = [
        case_canonical_loop,
        case_strict_mode_rejects_uncached,
        case_thinking_disabled_drops_reasoning,
        case_pass_through_keeps_client_thinking,
        case_streaming_turn_1_2_round_trip,
        case_recovery_drops_unrecoverable_history,
        case_authorization_isolates_namespace,
    ]
    failed = 0
    for case in cases:
        try:
            case()
        except AssertionError as exc:
            failed += 1
            print(f"  CASE FAILED: {exc}")
        except Exception as exc:  # pragma: no cover - audit harness
            failed += 1
            print(f"  CASE ERRORED: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"FAIL: {failed} of {len(cases)} cases failed")
        return 1
    print(f"OK: {len(cases)} of {len(cases)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

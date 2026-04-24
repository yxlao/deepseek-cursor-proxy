from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer


TOOL_REASONING = "I need the current date before answering."
FINAL_REASONING = "The tool result gives the date, so I can answer."
FINAL_CONTENT = "Final answer after using the tool."


def post_json(
    url: str, payload: dict, api_key: str = "cursor-local-token"
) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        response = urlopen(request, timeout=5)
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class FakeDeepSeekHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)

        for index, message in enumerate(payload.get("messages", [])):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            requires_reasoning = (
                bool(message.get("tool_calls"))
                or message.get("content") == FINAL_CONTENT
            )
            if requires_reasoning and not message.get("reasoning_content"):
                self._send_json(
                    400,
                    {
                        "error": {
                            "message": "The reasoning_content in the thinking mode must be passed back to the API.",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "invalid_request_error",
                            "missing_index": index,
                        }
                    },
                )
                return

        call_number = len(self.__class__.requests)
        if call_number == 1:
            self._send_json(200, tool_call_response())
        elif call_number == 2:
            self._send_json(200, final_response())
        else:
            self._send_json(200, plain_response("Follow-up accepted."))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class InterleavedFakeDeepSeekHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)

        messages = payload.get("messages", [])
        thread_name = thread_name_from_messages(messages)
        if thread_name not in {"A", "B"}:
            self._send_json(400, {"error": {"message": "unknown test thread"}})
            return

        expected_tool_reasoning = f"tool reasoning for thread {thread_name}"
        expected_final_reasoning = f"final reasoning for thread {thread_name}"
        final_content = f"final answer for thread {thread_name}"

        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if (
                message.get("tool_calls")
                and message.get("reasoning_content") != expected_tool_reasoning
            ):
                self._send_missing_reasoning(index)
                return
            if (
                message.get("content") == final_content
                and message.get("reasoning_content") != expected_final_reasoning
            ):
                self._send_missing_reasoning(index)
                return

        if len(messages) == 1:
            self._send_json(200, interleaved_tool_call_response(thread_name))
        elif len(messages) == 3:
            self._send_json(200, interleaved_final_response(thread_name))
        else:
            self._send_json(
                200, plain_response(f"follow-up accepted for thread {thread_name}")
            )

    def _send_missing_reasoning(self, index: int) -> None:
        self._send_json(
            400,
            {
                "error": {
                    "message": "The reasoning_content in the thinking mode must be passed back to the API.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_request_error",
                    "missing_index": index,
                }
            },
        )

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SlowAfterDoneStreamingDeepSeekHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "streamed"},
                    "finish_reason": None,
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        time.sleep(2)


def tool_call_response() -> dict:
    return {
        "id": "chatcmpl-tool",
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
                    "reasoning_content": TOOL_REASONING,
                    "tool_calls": [
                        {
                            "id": "call_date",
                            "type": "function",
                            "function": {"name": "get_date", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }


def interleaved_tool_call_response(thread_name: str) -> dict:
    return {
        "id": f"chatcmpl-tool-{thread_name}",
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
                    "reasoning_content": f"tool reasoning for thread {thread_name}",
                    "tool_calls": [
                        {
                            "id": "call_reused",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }


def interleaved_final_response(thread_name: str) -> dict:
    return {
        "id": f"chatcmpl-final-{thread_name}",
        "object": "chat.completion",
        "created": 2,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": f"final answer for thread {thread_name}",
                    "reasoning_content": f"final reasoning for thread {thread_name}",
                },
            }
        ],
    }


def final_response() -> dict:
    return {
        "id": "chatcmpl-final",
        "object": "chat.completion",
        "created": 2,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": FINAL_CONTENT,
                    "reasoning_content": FINAL_REASONING,
                },
            }
        ],
    }


def plain_response(content: str) -> dict:
    return {
        "id": "chatcmpl-plain",
        "object": "chat.completion",
        "created": 3,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


class ServerFixture:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def start(self) -> "ServerFixture":
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class ProxyEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeDeepSeekHandler.requests = []
        self.upstream = ServerFixture(
            ThreadingHTTPServer(("127.0.0.1", 0), FakeDeepSeekHandler)
        ).start()
        self.store = ReasoningStore(":memory:")
        proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        proxy.config = ProxyConfig(
            upstream_api_key="upstream-key",
            proxy_api_key="cursor-local-token",
            upstream_base_url=self.upstream.url,
            upstream_model="deepseek-v4-pro",
        )
        proxy.reasoning_store = self.store
        self.proxy = ServerFixture(proxy).start()

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_missing_reasoning_reproduces_upstream_error_without_proxy_repair(
        self,
    ) -> None:
        status, payload = post_json(
            f"{self.upstream.url}/chat/completions",
            second_cursor_request(include_reasoning=False),
        )

        self.assertEqual(status, 400)
        self.assertIn("reasoning_content", payload["error"]["message"])

    def test_proxy_repairs_cursor_style_multi_round_tool_call_history(self) -> None:
        status, first = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            first_cursor_request(),
        )
        self.assertEqual(status, 200)
        tool_call_message = first["choices"][0]["message"]
        self.assertEqual(tool_call_message["reasoning_content"], TOOL_REASONING)

        status, second = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            second_cursor_request(include_reasoning=False),
        )
        self.assertEqual(status, 200)
        self.assertEqual(second["choices"][0]["message"]["content"], FINAL_CONTENT)

        status, third = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            third_cursor_request_missing_all_reasoning(),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            third["choices"][0]["message"]["content"], "Follow-up accepted."
        )

        second_upstream_messages = FakeDeepSeekHandler.requests[1]["messages"]
        self.assertEqual(
            second_upstream_messages[1]["reasoning_content"], TOOL_REASONING
        )

        third_upstream_messages = FakeDeepSeekHandler.requests[2]["messages"]
        self.assertEqual(
            third_upstream_messages[1]["reasoning_content"], TOOL_REASONING
        )
        self.assertEqual(
            third_upstream_messages[3]["reasoning_content"], FINAL_REASONING
        )

    def test_proxy_adds_fallback_reasoning_for_uncached_cursor_tool_history(
        self,
    ) -> None:
        status, _ = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            second_cursor_request(include_reasoning=False),
        )

        self.assertEqual(status, 200)
        upstream_messages = FakeDeepSeekHandler.requests[0]["messages"]
        self.assertIn("reasoning_content", upstream_messages[1])


class InterleavedConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        InterleavedFakeDeepSeekHandler.requests = []
        self.upstream = ServerFixture(
            ThreadingHTTPServer(("127.0.0.1", 0), InterleavedFakeDeepSeekHandler)
        ).start()
        self.store = ReasoningStore(":memory:")
        proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        proxy.config = ProxyConfig(
            upstream_api_key="upstream-key",
            proxy_api_key="cursor-local-token",
            upstream_base_url=self.upstream.url,
            upstream_model="deepseek-v4-pro",
        )
        proxy.reasoning_store = self.store
        self.proxy = ServerFixture(proxy).start()

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_one_proxy_repairs_two_interleaved_cursor_threads(self) -> None:
        status, first_a = post_json(
            f"{self.proxy.url}/v1/chat/completions", interleaved_first_request("A")
        )
        self.assertEqual(status, 200)

        status, first_b = post_json(
            f"{self.proxy.url}/v1/chat/completions", interleaved_first_request("B")
        )
        self.assertEqual(status, 200)

        status, final_b = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            interleaved_second_request("B", first_b["choices"][0]["message"]),
        )
        self.assertEqual(status, 200)

        status, final_a = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            interleaved_second_request("A", first_a["choices"][0]["message"]),
        )
        self.assertEqual(status, 200)

        status, followup_a = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            interleaved_third_request(
                "A", first_a["choices"][0]["message"], final_a["choices"][0]["message"]
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            followup_a["choices"][0]["message"]["content"],
            "follow-up accepted for thread A",
        )

        status, followup_b = post_json(
            f"{self.proxy.url}/v1/chat/completions",
            interleaved_third_request(
                "B", first_b["choices"][0]["message"], final_b["choices"][0]["message"]
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            followup_b["choices"][0]["message"]["content"],
            "follow-up accepted for thread B",
        )

        final_b_upstream_messages = InterleavedFakeDeepSeekHandler.requests[2][
            "messages"
        ]
        final_a_upstream_messages = InterleavedFakeDeepSeekHandler.requests[3][
            "messages"
        ]
        followup_a_upstream_messages = InterleavedFakeDeepSeekHandler.requests[4][
            "messages"
        ]
        followup_b_upstream_messages = InterleavedFakeDeepSeekHandler.requests[5][
            "messages"
        ]

        self.assertEqual(
            final_b_upstream_messages[1]["reasoning_content"],
            "tool reasoning for thread B",
        )
        self.assertEqual(
            final_a_upstream_messages[1]["reasoning_content"],
            "tool reasoning for thread A",
        )
        self.assertEqual(
            followup_a_upstream_messages[1]["reasoning_content"],
            "tool reasoning for thread A",
        )
        self.assertEqual(
            followup_a_upstream_messages[3]["reasoning_content"],
            "final reasoning for thread A",
        )
        self.assertEqual(
            followup_b_upstream_messages[1]["reasoning_content"],
            "tool reasoning for thread B",
        )
        self.assertEqual(
            followup_b_upstream_messages[3]["reasoning_content"],
            "final reasoning for thread B",
        )


class StreamingProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = ServerFixture(
            ThreadingHTTPServer(("127.0.0.1", 0), SlowAfterDoneStreamingDeepSeekHandler)
        ).start()
        self.store = ReasoningStore(":memory:")
        proxy = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        proxy.config = ProxyConfig(
            upstream_api_key="upstream-key",
            proxy_api_key="cursor-local-token",
            upstream_base_url=self.upstream.url,
            upstream_model="deepseek-v4-pro",
        )
        proxy.reasoning_store = self.store
        self.proxy = ServerFixture(proxy).start()

    def tearDown(self) -> None:
        self.proxy.close()
        self.upstream.close()
        self.store.close()

    def test_streaming_proxy_closes_after_done_even_if_upstream_stays_open(
        self,
    ) -> None:
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
                "Authorization": "Bearer cursor-local-token",
                "Content-Type": "application/json",
            },
        )

        started = time.monotonic()
        with urlopen(request, timeout=1) as response:
            body = response.read().decode("utf-8")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1)
        self.assertIn("data: [DONE]", body)


def first_cursor_request() -> dict:
    return {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "What is tomorrow's date?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_date",
                    "description": "Get the current date",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def second_cursor_request(include_reasoning: bool) -> dict:
    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_date",
                "type": "function",
                "function": {"name": "get_date", "arguments": "{}"},
            }
        ],
    }
    if include_reasoning:
        assistant_message["reasoning_content"] = TOOL_REASONING
    return {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "user", "content": "What is tomorrow's date?"},
            assistant_message,
            {"role": "tool", "tool_call_id": "call_date", "content": "2026-04-24"},
        ],
        "tools": first_cursor_request()["tools"],
    }


def third_cursor_request_missing_all_reasoning() -> dict:
    payload = second_cursor_request(include_reasoning=False)
    payload["messages"].append({"role": "assistant", "content": FINAL_CONTENT})
    payload["messages"].append({"role": "user", "content": "Thanks, now continue."})
    return payload


def thread_name_from_messages(messages: list[dict]) -> str | None:
    if not messages:
        return None
    content = str(messages[0].get("content") or "")
    if "thread A" in content:
        return "A"
    if "thread B" in content:
        return "B"
    return None


def interleaved_first_request(thread_name: str) -> dict:
    return {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": f"Start thread {thread_name}."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Return a value.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def interleaved_second_request(thread_name: str, assistant_message: dict) -> dict:
    cursor_assistant = dict(assistant_message)
    cursor_assistant.pop("reasoning_content", None)
    return {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "user", "content": f"Start thread {thread_name}."},
            cursor_assistant,
            {
                "role": "tool",
                "tool_call_id": "call_reused",
                "content": f"tool result for thread {thread_name}",
            },
        ],
        "tools": interleaved_first_request(thread_name)["tools"],
    }


def interleaved_third_request(
    thread_name: str, tool_assistant_message: dict, final_message: dict
) -> dict:
    payload = interleaved_second_request(thread_name, tool_assistant_message)
    cursor_final = dict(final_message)
    cursor_final.pop("reasoning_content", None)
    payload["messages"].append(cursor_final)
    payload["messages"].append(
        {"role": "user", "content": f"Follow up in thread {thread_name}."}
    )
    return payload


if __name__ == "__main__":
    unittest.main()

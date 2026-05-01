from __future__ import annotations

from copy import deepcopy
import json
import os
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler, DeepSeekProxyServer


LIVE_DEEPSEEK = os.getenv("RUN_LIVE_DEEPSEEK_TESTS") == "1" and bool(
    os.getenv("LIVE_DEEPSEEK_KEY")
)


def post_json(
    url: str, payload: dict, api_key: str, timeout: int = 180
) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        response = urlopen(request, timeout=timeout)
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class ProxyFixture:
    def __init__(self) -> None:
        self.store = ReasoningStore(":memory:")
        server = DeepSeekProxyServer(("127.0.0.1", 0), DeepSeekProxyHandler)
        server.config = ProxyConfig(
            upstream_base_url="https://api.deepseek.com",
            upstream_model="deepseek-v4-pro",
            request_timeout=180,
        )
        server.reasoning_store = self.store
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1/chat/completions"

    def start(self) -> "ProxyFixture":
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.store.close()


@unittest.skipUnless(
    LIVE_DEEPSEEK,
    "set RUN_LIVE_DEEPSEEK_TESTS=1 and LIVE_DEEPSEEK_KEY to run live tests",
)
class LiveDeepSeekProxyTests(unittest.TestCase):
    def test_proxy_repairs_real_deepseek_tool_call_history(self) -> None:
        api_key = os.environ["LIVE_DEEPSEEK_KEY"]
        proxy = ProxyFixture().start()
        try:
            first_status, first_response = post_json(
                proxy.url,
                first_request(),
                api_key=api_key,
            )
            self.assertEqual(first_status, 200, first_response.get("error"))
            assistant_with_reasoning = first_response["choices"][0]["message"]
            self.assertTrue(assistant_with_reasoning.get("reasoning_content"))
            self.assertTrue(assistant_with_reasoning.get("tool_calls"))

            cursor_assistant = deepcopy(assistant_with_reasoning)
            cursor_assistant.pop("reasoning_content", None)
            tool_messages = [
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "2026-04-24",
                }
                for tool_call in cursor_assistant["tool_calls"]
            ]
            missing_reasoning_payload = {
                "model": "deepseek-v4-pro",
                "messages": [
                    first_request()["messages"][0],
                    cursor_assistant,
                    *tool_messages,
                ],
                "tools": first_request()["tools"],
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            }

            direct_status, direct_response = post_json(
                "https://api.deepseek.com/chat/completions",
                missing_reasoning_payload,
                api_key=api_key,
            )
            self.assertEqual(direct_status, 400)
            self.assertIn("reasoning_content", direct_response["error"]["message"])

            proxy_status, second_response = post_json(
                proxy.url,
                missing_reasoning_payload,
                api_key=api_key,
            )
            self.assertEqual(proxy_status, 200, second_response.get("error"))
            final_assistant = second_response["choices"][0]["message"]
            self.assertTrue(
                final_assistant.get("content") or final_assistant.get("tool_calls")
            )

            if final_assistant.get("content"):
                cursor_final = deepcopy(final_assistant)
                cursor_final.pop("reasoning_content", None)
                followup_payload = {
                    "model": "deepseek-v4-pro",
                    "messages": [
                        first_request()["messages"][0],
                        cursor_assistant,
                        *tool_messages,
                        cursor_final,
                        {"role": "user", "content": "Reply with exactly: OK"},
                    ],
                    "tools": first_request()["tools"],
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
                }
                followup_status, followup_response = post_json(
                    proxy.url,
                    followup_payload,
                    api_key=api_key,
                )
                self.assertEqual(followup_status, 200, followup_response.get("error"))
        finally:
            proxy.close()


def first_request() -> dict:
    return {
        "model": "deepseek-v4-pro",
        "messages": [
            {
                "role": "user",
                "content": "Use the get_date tool exactly once, then tell me the date it returns.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_date",
                    "description": "Return the current date as YYYY-MM-DD.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "required",
    }


if __name__ == "__main__":
    unittest.main()

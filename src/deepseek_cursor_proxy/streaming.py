from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .reasoning_store import ReasoningStore


@dataclass
class StreamingChoice:
    role: str = "assistant"
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class StreamAccumulator:
    def __init__(self) -> None:
        self.choices: dict[int, StreamingChoice] = {}

    def ingest_chunk(self, chunk: dict[str, Any]) -> None:
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return

        for raw_choice in choices:
            if not isinstance(raw_choice, dict):
                continue
            index = int(raw_choice.get("index") or 0)
            choice = self.choices.setdefault(index, StreamingChoice())
            finish_reason = raw_choice.get("finish_reason")
            if isinstance(finish_reason, str):
                choice.finish_reason = finish_reason

            delta = raw_choice.get("delta")
            if not isinstance(delta, dict):
                continue

            role = delta.get("role")
            if isinstance(role, str) and role:
                choice.role = role

            content = delta.get("content")
            if isinstance(content, str):
                choice.content += content

            reasoning_content = delta.get("reasoning_content")
            if isinstance(reasoning_content, str):
                choice.reasoning_content += reasoning_content

            self._merge_tool_call_deltas(choice, delta.get("tool_calls"))

    def store_reasoning(self, store: ReasoningStore, scope: str) -> int:
        stored = 0
        for choice in self.choices.values():
            stored += store.store_assistant_message(choice.to_message(), scope)
        return stored

    def messages(self) -> list[dict[str, Any]]:
        return [choice.to_message() for _, choice in sorted(self.choices.items())]

    def _merge_tool_call_deltas(self, choice: StreamingChoice, deltas: Any) -> None:
        if not isinstance(deltas, list):
            return

        for raw_delta in deltas:
            if not isinstance(raw_delta, dict):
                continue
            index = raw_delta.get("index")
            if not isinstance(index, int):
                index = len(choice.tool_calls)
            while len(choice.tool_calls) <= index:
                choice.tool_calls.append(
                    {"type": "function", "function": {"name": "", "arguments": ""}}
                )

            tool_call = choice.tool_calls[index]
            if raw_delta.get("id"):
                tool_call["id"] = raw_delta["id"]
            if raw_delta.get("type"):
                tool_call["type"] = raw_delta["type"]

            function_delta = raw_delta.get("function")
            if not isinstance(function_delta, dict):
                continue
            function = tool_call.setdefault("function", {"name": "", "arguments": ""})
            if function_delta.get("name"):
                existing_name = function.get("name") or ""
                new_name = str(function_delta["name"])
                function["name"] = (
                    new_name if not existing_name else existing_name + new_name
                )
            if (
                "arguments" in function_delta
                and function_delta["arguments"] is not None
            ):
                function["arguments"] = (function.get("arguments") or "") + str(
                    function_delta["arguments"]
                )

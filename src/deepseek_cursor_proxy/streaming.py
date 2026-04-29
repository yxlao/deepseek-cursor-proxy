from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from .reasoning_store import ReasoningStore


THINKING_BLOCK_START = "<think>\n"
THINKING_BLOCK_END = "\n</think>\n\n"


@dataclass
class StreamingChoice:
    role: str = "assistant"
    content: str = ""
    reasoning_content: str = ""
    has_reasoning_content: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.has_reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class StreamAccumulator:
    def __init__(self) -> None:
        self.choices: dict[int, StreamingChoice] = {}
        self._stored_choices: dict[int, str] = {}

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
                choice.has_reasoning_content = True
                choice.reasoning_content += reasoning_content

            self._merge_tool_call_deltas(choice, delta.get("tool_calls"))

    def store_reasoning(
        self,
        store: ReasoningStore,
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        stored = 0
        for index, choice in self.choices.items():
            stored += self._store_choice(
                index, choice, store, scope, "final", cache_namespace, prior_messages
            )
        return stored

    def store_finished_reasoning(
        self,
        store: ReasoningStore,
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        stored = 0
        for index, choice in self.choices.items():
            if choice.finish_reason is not None:
                stored += self._store_choice(
                    index,
                    choice,
                    store,
                    scope,
                    "final",
                    cache_namespace,
                    prior_messages,
                )
        return stored

    def store_ready_reasoning(
        self,
        store: ReasoningStore,
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        stored = 0
        for index, choice in self.choices.items():
            if choice.finish_reason is not None:
                stored += self._store_choice(
                    index,
                    choice,
                    store,
                    scope,
                    "final",
                    cache_namespace,
                    prior_messages,
                )
            elif self._has_identified_tool_calls(choice):
                stored += self._store_choice(
                    index,
                    choice,
                    store,
                    scope,
                    "tool_call",
                    cache_namespace,
                    prior_messages,
                )
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

    def _store_choice(
        self,
        index: int,
        choice: StreamingChoice,
        store: ReasoningStore,
        scope: str,
        stage: str = "final",
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        stage_rank = {"tool_call": 1, "final": 2}
        previous_stage = self._stored_choices.get(index)
        if stage_rank.get(previous_stage or "", 0) >= stage_rank.get(stage, 0):
            return 0
        stored = store.store_assistant_message(
            choice.to_message(),
            scope,
            cache_namespace,
            prior_messages,
        )
        if stored:
            self._stored_choices[index] = stage
        return stored

    def _has_identified_tool_calls(self, choice: StreamingChoice) -> bool:
        if not choice.has_reasoning_content or not choice.tool_calls:
            return False
        return all(bool(tool_call.get("id")) for tool_call in choice.tool_calls)


class CursorReasoningDisplayAdapter:
    """Mirror reasoning_content into content for Cursor's visible thinking UI path."""

    def __init__(self) -> None:
        self._open_choices: set[int] = set()
        self._last_chunk_metadata: dict[str, Any] = {}

    def rewrite_chunk(self, chunk: dict[str, Any]) -> None:
        self._remember_chunk_metadata(chunk)
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return

        for raw_choice in choices:
            if not isinstance(raw_choice, dict):
                continue
            index = int(raw_choice.get("index") or 0)
            delta = raw_choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
                raw_choice["delta"] = delta

            mirrored_parts: list[str] = []
            reasoning_content = delta.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content:
                if index not in self._open_choices:
                    mirrored_parts.append(THINKING_BLOCK_START)
                    self._open_choices.add(index)
                mirrored_parts.append(reasoning_content)

            existing_content = delta.get("content")
            should_close = index in self._open_choices and (
                bool(existing_content)
                or bool(delta.get("tool_calls"))
                or raw_choice.get("finish_reason") is not None
            )
            if should_close:
                mirrored_parts.append(THINKING_BLOCK_END)
                self._open_choices.discard(index)

            if not mirrored_parts:
                continue
            if isinstance(existing_content, str):
                mirrored_parts.append(existing_content)
            delta["content"] = "".join(mirrored_parts)

    def flush_chunk(self, model: str) -> dict[str, Any] | None:
        if not self._open_choices:
            return None

        choices = [
            {
                "index": index,
                "delta": {"content": THINKING_BLOCK_END},
                "finish_reason": None,
            }
            for index in sorted(self._open_choices)
        ]
        self._open_choices.clear()

        chunk: dict[str, Any] = {
            "id": self._last_chunk_metadata.get("id", "chatcmpl-reasoning-close"),
            "object": self._last_chunk_metadata.get("object", "chat.completion.chunk"),
            "created": self._last_chunk_metadata.get("created", int(time.time())),
            "model": model,
            "choices": choices,
        }
        return chunk

    def _remember_chunk_metadata(self, chunk: dict[str, Any]) -> None:
        metadata = {
            key: chunk[key] for key in ("id", "object", "created") if key in chunk
        }
        if metadata:
            self._last_chunk_metadata.update(metadata)

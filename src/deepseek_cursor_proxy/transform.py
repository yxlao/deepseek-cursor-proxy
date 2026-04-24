from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .config import ProxyConfig
from .reasoning_store import ReasoningStore, conversation_scope


SUPPORTED_REQUEST_FIELDS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "response_format",
    "stop",
    "tools",
    "tool_choice",
    "thinking",
    "reasoning_effort",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
}

MESSAGE_FIELDS = {
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "reasoning_content",
    "prefix",
}

ROLE_MESSAGE_FIELDS = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {
        "role",
        "content",
        "name",
        "tool_calls",
        "reasoning_content",
        "prefix",
    },
    "tool": {"role", "content", "tool_call_id"},
}

EFFORT_ALIASES = {
    "low": "high",
    "medium": "high",
    "high": "high",
    "max": "max",
    "xhigh": "max",
}

CURSOR_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking)>[\s\S]*?(?:</(?:think|thinking)>|$)\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedRequest:
    payload: dict[str, Any]
    original_model: str
    upstream_model: str
    patched_reasoning_messages: int
    fallback_reasoning_messages: int


def normalize_reasoning_effort(value: Any) -> str:
    if not isinstance(value, str):
        return "high"
    return EFFORT_ALIASES.get(value.strip().lower(), "high")


def extract_text_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            text = item.get("text") or item.get("content")
            if item_type in {"text", "input_text"} and isinstance(text, str):
                parts.append(text)
            elif isinstance(text, str):
                parts.append(text)
            elif item_type:
                parts.append(f"[{item_type} omitted by DeepSeek text proxy]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, (dict, tuple)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def strip_cursor_thinking_blocks(content: str) -> str:
    return CURSOR_THINKING_BLOCK_RE.sub("", content).lstrip("\r\n")


def normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        tool_call = {}
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": str(tool_call.get("id") or ""),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": arguments,
        },
    }
    if not normalized["id"]:
        normalized.pop("id")
    return normalized


def normalize_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {
            "type": "function",
            "function": {"name": "", "description": "", "parameters": {}},
        }
    normalized = dict(tool)
    normalized["type"] = normalized.get("type") or "function"
    function = normalized.get("function")
    if isinstance(function, dict):
        normalized["function"] = function
    return normalized


def legacy_function_to_tool(function: Any) -> dict[str, Any]:
    if not isinstance(function, dict):
        function = {}
    return {"type": "function", "function": function}


def convert_function_call(function_call: Any) -> Any:
    if isinstance(function_call, str):
        if function_call in {"auto", "none"}:
            return function_call
        if function_call == "required":
            return "auto"
        return None
    if isinstance(function_call, dict) and function_call.get("name"):
        return "auto"
    return None


def normalize_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "required":
            return "auto"
        return None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return "auto"
        return tool_choice
    return tool_choice


def normalize_message(
    message: Any,
    store: ReasoningStore | None,
    prior_messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool, bool]:
    if not isinstance(message, dict):
        message = {"role": "user", "content": str(message)}
    normalized = {key: value for key, value in message.items() if key in MESSAGE_FIELDS}
    role = normalized.get("role") or "user"
    normalized["role"] = role

    if role == "function":
        normalized["role"] = "tool"

    if "content" in normalized:
        normalized["content"] = extract_text_content(normalized["content"]) or ""
    elif normalized["role"] in {"assistant", "tool", "system", "user"}:
        normalized["content"] = ""
    if normalized["role"] == "assistant" and isinstance(normalized.get("content"), str):
        normalized["content"] = strip_cursor_thinking_blocks(normalized["content"])

    if normalized.get("tool_calls"):
        normalized["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in normalized.get("tool_calls") or []
        ]

    patched = False
    fallback = False
    if normalized["role"] == "assistant":
        reasoning = normalized.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            normalized.pop("reasoning_content", None)
            if store is not None:
                restored = store.lookup_for_message(
                    normalized, conversation_scope(prior_messages)
                )
                if restored:
                    normalized["reasoning_content"] = restored
                    patched = True
            if not patched and assistant_needs_reasoning_for_tool_context(
                normalized, prior_messages
            ):
                normalized["reasoning_content"] = fallback_reasoning_content(normalized)
                fallback = True

    allowed_fields = ROLE_MESSAGE_FIELDS.get(str(normalized["role"]), MESSAGE_FIELDS)
    normalized = {
        key: value for key, value in normalized.items() if key in allowed_fields
    }
    return normalized, patched, fallback


def normalize_messages(
    messages: Any, store: ReasoningStore | None
) -> tuple[list[dict[str, Any]], int, int]:
    if not isinstance(messages, list):
        return [], 0, 0
    normalized_messages: list[dict[str, Any]] = []
    patched_count = 0
    fallback_count = 0
    for message in messages:
        normalized, patched, fallback = normalize_message(
            message, store, normalized_messages
        )
        normalized_messages.append(normalized)
        if patched:
            patched_count += 1
        if fallback:
            fallback_count += 1
    return normalized_messages, patched_count, fallback_count


def assistant_needs_reasoning_for_tool_context(
    message: dict[str, Any],
    prior_messages: list[dict[str, Any]],
) -> bool:
    if message.get("tool_calls"):
        return True
    for prior_message in reversed(prior_messages):
        role = prior_message.get("role")
        if role == "tool":
            return True
        if role in {"user", "system"}:
            return False
    return False


def fallback_reasoning_content(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        return "Compatibility placeholder: Cursor omitted DeepSeek reasoning_content for this tool-call turn."
    return "Compatibility placeholder: Cursor omitted DeepSeek reasoning_content for this tool-result turn."


def upstream_model_for(original_model: str, config: ProxyConfig) -> str:
    if config.allow_model_passthrough and original_model.startswith("deepseek-"):
        return original_model
    return config.upstream_model


def prepare_upstream_request(
    payload: dict[str, Any],
    config: ProxyConfig,
    store: ReasoningStore | None,
) -> PreparedRequest:
    original_model = str(payload.get("model") or config.upstream_model)
    upstream_model = upstream_model_for(original_model, config)

    prepared = {
        key: value for key, value in payload.items() if key in SUPPORTED_REQUEST_FIELDS
    }
    if "max_tokens" not in prepared and "max_completion_tokens" in payload:
        prepared["max_tokens"] = payload["max_completion_tokens"]

    prepared["model"] = upstream_model
    messages, patched_count, fallback_count = normalize_messages(
        payload.get("messages"), store
    )
    prepared["messages"] = messages

    if "tools" in prepared and isinstance(prepared["tools"], list):
        prepared["tools"] = [normalize_tool(tool) for tool in prepared["tools"]]
    elif isinstance(payload.get("functions"), list):
        prepared["tools"] = [
            legacy_function_to_tool(function) for function in payload["functions"]
        ]

    if "tool_choice" in prepared:
        tool_choice = normalize_tool_choice(prepared["tool_choice"])
        if tool_choice is None:
            prepared.pop("tool_choice", None)
        else:
            prepared["tool_choice"] = tool_choice
    elif "function_call" in payload:
        tool_choice = convert_function_call(payload.get("function_call"))
        if tool_choice is not None:
            prepared["tool_choice"] = tool_choice

    if config.thinking != "pass-through":
        prepared["thinking"] = {"type": config.thinking}

    thinking = prepared.get("thinking")
    thinking_enabled = isinstance(thinking, dict) and thinking.get("type") == "enabled"
    if thinking_enabled:
        prepared["reasoning_effort"] = normalize_reasoning_effort(
            prepared.get("reasoning_effort") or config.reasoning_effort
        )

    return PreparedRequest(
        payload=prepared,
        original_model=original_model,
        upstream_model=upstream_model,
        patched_reasoning_messages=patched_count,
        fallback_reasoning_messages=fallback_count,
    )


def record_response_reasoning(
    response_payload: dict[str, Any],
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
) -> int:
    if store is None:
        return 0
    stored = 0
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return stored
    scope = conversation_scope(request_messages)
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            stored += store.store_assistant_message(message, scope)
    return stored


def rewrite_response_body(
    body: bytes,
    original_model: str,
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
) -> bytes:
    response_payload = json.loads(body.decode("utf-8"))
    if isinstance(response_payload, dict):
        record_response_reasoning(response_payload, store, request_messages)
        if "model" in response_payload:
            response_payload["model"] = original_model
    return json.dumps(
        response_payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

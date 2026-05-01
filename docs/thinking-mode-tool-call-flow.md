# DeepSeek Thinking Mode Tool-Call Flow

Source: [DeepSeek API Docs - Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)

This note documents how DeepSeek thinking mode works with tool calls, and why this proxy needs to restore `reasoning_content` when Cursor omits it.

## Thinking mode request controls

Thinking mode is enabled by default in DeepSeek's current API. In OpenAI-compatible requests, the explicit toggle is passed through `extra_body`:

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
```

Use `extra_body={"thinking": {"type": "disabled"}}` to turn it off.

Thinking effort is controlled with `reasoning_effort` in OpenAI-compatible requests:

```json
{
  "reasoning_effort": "high"
}
```

DeepSeek supports `high` and `max` thinking effort values. Regular thinking-mode requests default to `high`; some complex agent requests may be upgraded to `max`. For compatibility, `low` and `medium` are treated as `high`, while `xhigh` is treated as `max`.

The Anthropic-compatible shape uses `output_config.effort` instead:

```json
{
  "output_config": {
    "effort": "high"
  }
}
```

## Output fields

In thinking mode, the assistant message can include both:

- `reasoning_content`: the model's thinking text.
- `content`: the user-visible answer text.

These fields are siblings on the assistant message. They are not interchangeable:

```json
{
  "role": "assistant",
  "reasoning_content": "Thinking text from DeepSeek...",
  "content": "Final answer shown to the user."
}
```

DeepSeek also notes that sampling parameters such as `temperature`, `top_p`, `presence_penalty`, and `frequency_penalty` do not affect thinking-mode requests. They may be accepted for compatibility, but they are ignored.

## Context rules

The important context rule depends on whether the assistant used tools.

If no tool call happened between two `user` messages, previous assistant `reasoning_content` is not needed for the next request. Sending it later is allowed, but DeepSeek ignores it.

If the assistant did call a tool during a user turn, the `reasoning_content` from that tool-calling turn must be included in later requests. This covers the assistant messages that request tools and the final assistant message that answers after the tool results. If required reasoning is missing, DeepSeek returns a 400 error.

That second rule is the reason this proxy exists: Cursor sends the tool-call conversation history, but does not preserve DeepSeek's `reasoning_content`. The proxy caches the field from prior DeepSeek responses and patches it back into outgoing requests.

## Tool-call loop

```text
Turn 1.1
Input
  [tools] Tools
  [user]  User message 1

Output
  [assistant.reasoning_content] Thinking 1.1
  [assistant.tool_calls]        Tool call 1.1

        |
        | Append the assistant message, run Tool call 1.1,
        | then append Tool result 1.1.
        v

Turn 1.2
Input
  [tools]                       Tools
  [user]                        User message 1
  [assistant.reasoning_content] Thinking 1.1
  [assistant.tool_calls]        Tool call 1.1
  [tool]                        Tool result 1.1

Output
  [assistant.reasoning_content] Thinking 1.2
  [assistant.tool_calls]        Tool call 1.2

        |
        | Append the assistant message, run Tool call 1.2,
        | then append Tool result 1.2.
        v

Turn 1.3
Input
  [tools]                       Tools
  [user]                        User message 1
  [assistant.reasoning_content] Thinking 1.1
  [assistant.tool_calls]        Tool call 1.1
  [tool]                        Tool result 1.1
  [assistant.reasoning_content] Thinking 1.2
  [assistant.tool_calls]        Tool call 1.2
  [tool]                        Tool result 1.2

Output
  [assistant.reasoning_content] Thinking 1.3
  [assistant.content]           Answer 1

        |
        | User sends another message. Keep the previous tool-call
        | reasoning chain in the conversation history.
        v

Turn 2.1
Input
  [tools]                       Tools
  [user]                        User message 1
  [assistant.reasoning_content] Thinking 1.1
  [assistant.tool_calls]        Tool call 1.1
  [tool]                        Tool result 1.1
  [assistant.reasoning_content] Thinking 1.2
  [assistant.tool_calls]        Tool call 1.2
  [tool]                        Tool result 1.2
  [assistant.reasoning_content] Thinking 1.3
  [assistant.content]           Answer 1
  [user]                        User message 2

Output
  [assistant.reasoning_content] Thinking 2.1
  [...]
```

In each model sub-turn:

1. Send the current `messages` plus the tool definitions.
2. Append the returned assistant message exactly enough to preserve `content`, `reasoning_content`, and `tool_calls`.
3. If `tool_calls` is absent or empty, stop. The assistant has produced the final `content`.
4. If `tool_calls` is present, execute each requested tool locally.
5. Append one `tool` message per result, using the matching `tool_call_id`.
6. Send the expanded message list back to DeepSeek and repeat.

The assistant message after a tool call should retain this shape:

```json
{
  "role": "assistant",
  "content": "",
  "reasoning_content": "Thinking text used before requesting the tool...",
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\":\"Hangzhou\",\"date\":\"2026-04-20\"}"
      }
    }
  ]
}
```

The following tool result must reference the same tool-call ID:

```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "Cloudy, 7-13 C"
}
```

After the tool result is appended, the next request includes the original user message, the assistant reasoning/tool-call message, and the tool result. DeepSeek can then continue from the previous thinking state instead of starting over.

## Non-streaming response handling

For non-streaming calls, preserve the returned assistant message directly when possible:

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    tools=tools,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

assistant_message = response.choices[0].message
messages.append(assistant_message)

if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        result = run_tool(tool_call)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
```

Appending the full assistant message is safest because it keeps `content`, `reasoning_content`, and `tool_calls` together.

## Streaming response handling

For streaming calls, `reasoning_content` and `content` arrive in deltas and need to be accumulated separately:

```python
reasoning_content = ""
content = ""

for chunk in response:
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        reasoning_content += delta.reasoning_content
    elif getattr(delta, "content", None):
        content += delta.content

messages.append({
    "role": "assistant",
    "reasoning_content": reasoning_content,
    "content": content,
    "tool_calls": collected_tool_calls,
})
```

If the streamed assistant message includes tool calls, the assembled `reasoning_content` must be stored with that message before the next request.

## Field legend

- `reasoning_content` is the "Thinking" block in the diagram.
- `tool_calls` is the assistant's requested tool invocation.
- `tool` messages are the results returned by local tool execution.
- `content` is the user-visible final answer.
- If a user turn contains `tool_calls`, preserve the turn's assistant `reasoning_content` in every later request, including the final assistant answer for that turn.

## Proxy behavior

`deepseek-cursor-proxy` follows the same API contract on behalf of clients that do not preserve `reasoning_content`:

1. It forwards the user's OpenAI-compatible request to DeepSeek.
2. It records each DeepSeek assistant message's `reasoning_content`, keyed to the corresponding message and tool-call context.
3. When a later request arrives without required `reasoning_content`, it restores the cached value before sending the request upstream.
4. It forwards DeepSeek's response back to the client, optionally mirroring thinking text into Cursor-visible Markdown.

This keeps Cursor-compatible traffic valid for DeepSeek thinking-mode tool calls without changing the visible chat flow.

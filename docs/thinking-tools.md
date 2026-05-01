# DeepSeek Thinking Tool Flow

Source: [DeepSeek Thinking Mode - Tool Calls](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)

DeepSeek thinking mode can use tools across multiple model sub-turns before producing the final answer. When a user turn uses tools, later requests must pass back the complete `reasoning_content` chain for that turn, not just the latest thinking block.

![DeepSeek thinking mode with tools](https://api-docs.deepseek.com/img/thinking_with_tools_en.jpg)

## The Rule

- `reasoning_content` is the model's thinking text.
- `content` is the user-visible answer text.
- `tool_calls` are assistant requests to run local tools.
- `tool` messages are local tool results.
- If no tool call happened between two `user` messages, old `reasoning_content` is optional and DeepSeek ignores it.
- If tool calls happened, every later request must include all assistant `reasoning_content` blocks from that tool-use chain: `Thinking 1.1`, `Thinking 1.2`, `Thinking 1.3`, and so on.

Missing required `reasoning_content` causes DeepSeek to return:

```text
The reasoning_content in the thinking mode must be passed back to the API.
```

## Tool Loop

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

The important part is the input to `Turn 2.1`: it contains the full previous tool-use reasoning chain, not only `Thinking 1.3`.

## Correct Message Loop

1. Send the current `messages` plus the tool definitions.
2. Append the returned assistant message with `content`, `reasoning_content`, and `tool_calls`.
3. If there are no `tool_calls`, stop. The assistant has produced the final `content`.
4. If `tool_calls` is present, execute each requested tool locally.
5. Append one `tool` result per call using the matching `tool_call_id`.
6. Send the expanded message list back to DeepSeek and repeat.

A tool-calling assistant message must keep this shape:

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

The tool result references the same ID:

```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "Cloudy, 7-13 C"
}
```

For streaming responses, accumulate `delta.reasoning_content`, `delta.content`, and `delta.tool_calls` separately, then store the assembled assistant message before the next request.

## Proxy behavior

Cursor sends the visible tool-call transcript back to the model, but it omits DeepSeek's `reasoning_content`. `deepseek-cursor-proxy` repairs that gap:

1. Forward the request to DeepSeek with thinking mode enabled.
2. Record each returned assistant `reasoning_content` in a local SQLite cache.
3. On the next Cursor request, detect assistant messages that need reasoning.
4. Restore the missing `reasoning_content` before sending the request upstream.
5. Mirror thinking text into Cursor-visible Markdown when display is enabled.

The proxy does not invent the reasoning. It only stores DeepSeek's original `reasoning_content` and puts the complete required chain back into later requests.

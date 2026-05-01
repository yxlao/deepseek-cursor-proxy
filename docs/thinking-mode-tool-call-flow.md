# DeepSeek Thinking Mode Tool-Call Flow

Source: [DeepSeek API Docs - Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)

This diagram shows how a single user turn can expand into multiple model sub-turns when thinking mode and tool calls are enabled. The key rule is that every assistant message that includes a tool call must be sent back later with its `reasoning_content`.

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

In short:

- `reasoning_content` is the "Thinking" block in the diagram.
- `tool_calls` is the assistant's requested tool invocation.
- `tool` messages are the results returned by local tool execution.
- `content` is the user-visible final answer.
- If an assistant message contains `tool_calls`, preserve its `reasoning_content` in every later request.

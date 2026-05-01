# DeepSeek Thinking Tool-Call Protocol Audit

Audit date: 2026-05-01

Auditor: Codex

Scope:

- Protocol reference: [`docs/thinking-tools.md`](thinking-tools.md)
- Implementation reviewed: [`src/deepseek_cursor_proxy/transform.py`](../src/deepseek_cursor_proxy/transform.py), [`src/deepseek_cursor_proxy/reasoning_store.py`](../src/deepseek_cursor_proxy/reasoning_store.py), [`src/deepseek_cursor_proxy/streaming.py`](../src/deepseek_cursor_proxy/streaming.py), [`src/deepseek_cursor_proxy/server.py`](../src/deepseek_cursor_proxy/server.py), and related tests.
- Verification added: [`scripts/audit_deepseek_protocol.py`](../scripts/audit_deepseek_protocol.py)

## Executive Summary

For the default Cursor-facing configuration, the proxy is largely consistent with the documented DeepSeek thinking-mode tool-call protocol. It forces `thinking: {"type": "enabled"}`, normalizes supported reasoning effort aliases, stores returned `reasoning_content`, restores missing `reasoning_content` for assistant tool-call messages, restores the final assistant message after tool results, and handles both non-streaming and streaming responses.

The most important protocol gap is in `thinking: pass-through` mode. DeepSeek's documented current behavior is that thinking mode is enabled by default when the request does not explicitly disable it. The proxy, however, only repairs tool-call reasoning when the forwarded request contains `thinking: {"type": "enabled"}`. If pass-through mode is used and the client omits `thinking`, the proxy treats the request as non-thinking for repair purposes and can forward invalid tool-call history without `reasoning_content`.

I did not change production code in this audit. I added a local audit script and wrote this report.

## Findings

### F1. Pass-through mode misses DeepSeek's default-thinking behavior

Severity: Medium

Protocol requirement: thinking mode is enabled by default, and tool-call histories in thinking mode must include the prior assistant `reasoning_content`.

Implementation: `prepare_upstream_request()` sets `thinking_enabled` only when `prepared["thinking"]` is a dict with `type == "enabled"` ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L722)). Repair and recovery are gated on that boolean ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L760), [`transform.py`](../src/deepseek_cursor_proxy/transform.py#L767)).

Impact: With `ProxyConfig(thinking="pass-through")` and a request that omits `thinking`, a thinking-mode DeepSeek request can still be sent upstream, but the proxy will not detect, restore, reject, or recover missing `reasoning_content`.

Evidence:

```text
uv run python scripts/audit_deepseek_protocol.py
OBSERVE pass-through mode treats an omitted thinking field as non-thinking; DeepSeek's documented default is thinking enabled.
```

Recommendation: In pass-through mode, treat omitted `thinking` as enabled for DeepSeek thinking models unless the request explicitly sends `{"thinking": {"type": "disabled"}}`. Alternatively, document that pass-through callers must send the `thinking` field explicitly or they lose reasoning repair.

### F2. Default config overrides request-level thinking controls

Severity: Low to Medium, depending on intended API semantics

Protocol reference: the protocol document describes explicit request controls: `thinking: {"type": "enabled"}` and `thinking: {"type": "disabled"}`.

Implementation: unless config is `thinking="pass-through"`, the proxy overwrites any client-provided `thinking` field with the configured value ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L722)).

Impact: The default config is good for Cursor because it consistently enables thinking repair. But for a general OpenAI-compatible proxy, a caller who explicitly disables thinking in a request will still get thinking enabled unless the proxy config is changed.

Recommendation: Either document this as an intentional proxy policy, or change the precedence to honor explicit request-level `thinking` while keeping the config as a default.

### F3. Anthropic-compatible `output_config.effort` is not implemented

Severity: Low

Protocol reference: the protocol note mentions Anthropic-compatible `output_config.effort`.

Implementation: the proxy only accepts OpenAI-compatible `/v1/chat/completions` and the allow-list does not include `output_config` ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L20)).

Impact: This is not a problem for Cursor's OpenAI-compatible flow. It is a gap only if this proxy is expected to support Anthropic-compatible DeepSeek requests.

Recommendation: Leave as out of scope unless Anthropic-compatible traffic is a goal. If it is, add a separate route/transform instead of mixing schemas silently.

### F4. Recovery mode is a controlled context-loss fallback, not full protocol replay

Severity: Informational

Protocol requirement: required `reasoning_content` should be present in later thinking-mode tool-call requests.

Implementation: when cached reasoning is unavailable and `missing_reasoning_strategy="recover"`, the proxy drops unrecoverable older history, injects a recovery system message, and prefixes the next response with a notice ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L523), [`transform.py`](../src/deepseek_cursor_proxy/transform.py#L776)).

Impact: The repaired upstream request avoids sending an invalid tool-call chain, but it is no longer an exact replay of the original conversation. This is a pragmatic fallback for cold cache, model/config switches, or lost state.

Recommendation: Keep this behavior, but treat `missing_reasoning_strategy="reject"` as the strict compliance/debugging mode.

### F5. Non-streaming responses are not mirrored into Cursor-visible thinking blocks

Severity: Low

Protocol requirement: `reasoning_content` and `content` are sibling fields; mirroring is optional proxy behavior.

Implementation: streaming responses use `CursorReasoningDisplayAdapter` to mirror `reasoning_content` into `<details><summary>Thinking</summary>...` content ([`streaming.py`](../src/deepseek_cursor_proxy/streaming.py#L214)). Non-streaming `rewrite_response_body()` stores reasoning and restores the model name, but does not mirror `reasoning_content` into `content` ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L867)).

Impact: This does not violate the DeepSeek protocol, but it may not fully match the README's broad claim that thinking tokens are displayed in Cursor.

Recommendation: If non-streaming Cursor display matters, add equivalent optional mirroring for non-streaming responses. Otherwise, document that the display adapter is streaming-only.

## Protocol Matrix

| Protocol requirement | Implementation status | Evidence |
| --- | --- | --- |
| OpenAI-compatible `thinking` control exists | Mostly aligned | `SUPPORTED_REQUEST_FIELDS` includes `thinking`; default config injects enabled thinking ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L20), [`transform.py`](../src/deepseek_cursor_proxy/transform.py#L722)). See F1 and F2 for edge cases. |
| `reasoning_effort` supports `high` and `max`; aliases map to supported values | Aligned in enabled-thinking mode | `low`/`medium` map to `high`, `xhigh` maps to `max` ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L64), [`transform.py`](../src/deepseek_cursor_proxy/transform.py#L730)). |
| Assistant messages keep `content`, `reasoning_content`, and `tool_calls` as siblings | Aligned | Assistant role allow-list preserves all three fields ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L50)). |
| Plain chat history between user turns does not require old reasoning | Aligned | `assistant_needs_reasoning_for_tool_context()` returns false when no tool context exists before the prior user/system boundary ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L613)). |
| Assistant tool-call messages need `reasoning_content` in later requests | Aligned by default | Any assistant with `tool_calls` is marked as needing reasoning and is looked up in the cache ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L617), [`transform.py`](../src/deepseek_cursor_proxy/transform.py#L271)). |
| Final assistant answer after tool results needs `reasoning_content` | Aligned by default | An assistant after a prior `tool` message is marked as needing reasoning until the next user/system boundary ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L619)). |
| Tool results use `role: "tool"` and `tool_call_id` | Mostly pass-through | Tool messages are normalized to role/content/tool_call_id ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L61)). The proxy does not validate that every result ID matches a prior tool call. |
| Non-streaming responses store returned reasoning | Aligned | `rewrite_response_body()` calls `record_response_reasoning()` before forwarding the response ([`transform.py`](../src/deepseek_cursor_proxy/transform.py#L867)). |
| Streaming responses accumulate reasoning/content separately | Aligned | `StreamAccumulator.ingest_chunk()` accumulates `reasoning_content`, `content`, and tool-call deltas separately ([`streaming.py`](../src/deepseek_cursor_proxy/streaming.py#L42)). |
| Streamed tool-call reasoning is available before the next request | Aligned | `store_ready_reasoning()` stores once reasoning and tool-call IDs are known, before `[DONE]` if necessary ([`streaming.py`](../src/deepseek_cursor_proxy/streaming.py#L110), [`server.py`](../src/deepseek_cursor_proxy/server.py#L750)). |
| Cache keys isolate conversations | Aligned | Cache keys include conversation scope, message signature, tool-call IDs/signatures, model family/config namespace, and optional authorization hash ([`reasoning_store.py`](../src/deepseek_cursor_proxy/reasoning_store.py#L86), [`transform.py`](../src/deepseek_cursor_proxy/transform.py#L640)). |

## Detailed Review

### Request preparation

The proxy accepts a conservative allow-list of upstream fields, including `messages`, `tools`, `tool_choice`, `thinking`, `reasoning_effort`, and sampling parameters that DeepSeek ignores in thinking mode. It converts legacy `functions`/`function_call` to `tools`/`tool_choice`, preserves named and `required` tool choice, and translates `max_completion_tokens` to `max_tokens`.

This is consistent with the OpenAI-compatible protocol. It intentionally strips unsupported fields such as `parallel_tool_calls`, which reduces upstream rejection risk but should be considered a compatibility policy rather than a pure transparent proxy.

### Message normalization and repair

The key protocol logic is in `normalize_message()` and `assistant_needs_reasoning_for_tool_context()`.

Positive observations:

- Assistant messages with `tool_calls` always require reasoning.
- Assistant messages after a `tool` result require reasoning until a user/system boundary.
- Plain assistant messages outside tool context do not require reasoning.
- Existing `reasoning_content` is preserved as long as thinking is not disabled.
- Missing reasoning is restored from cache when possible.
- Mirrored Cursor display blocks are stripped from assistant `content` before cache lookup, so streaming display markup does not poison later signatures.

This matches the protocol's rule that the reasoning chain must be preserved only for user turns that involved tool calls.

### Cache behavior

The cache design is stronger than a single tool-call-ID map. It stores reasoning under:

- scoped message signature,
- scoped tool-call ID,
- scoped tool-call function signature,
- portable turn-level aliases for mode/system-prompt changes.

The scope excludes `reasoning_content`, which is correct because Cursor omits that field later. The namespace includes upstream base URL, DeepSeek model family, thinking config, reasoning effort, and an authorization hash. This prevents obvious collisions across users, configurations, and unrelated conversations.

Residual risk: the function-signature fallback can still mis-associate reasoning if a client drops tool-call IDs and repeats identical function calls with identical arguments in the same scoped context. This is a low-probability recovery fallback, and exact message signature plus tool-call ID are tried first.

### Response handling

Non-streaming responses are decoded, optionally prefixed with a recovery notice, recorded into the reasoning cache, and returned with the original model name restored. This preserves `reasoning_content` in the response body.

Streaming responses are handled more carefully:

- Deltas are accumulated into an assistant message.
- `reasoning_content` and `content` are accumulated separately.
- Tool-call deltas are merged by index.
- Reasoning can be stored before `[DONE]` once tool-call IDs exist, which is important because Cursor may issue the next tool-result request as soon as it sees `finish_reason: "tool_calls"`.
- Display mirroring happens after cache accumulation, so cached signatures are based on DeepSeek's original content, not the Cursor-visible details block.

### Recovery behavior

Recovery mode is intentionally not a perfect protocol replay. If a required historical `reasoning_content` cannot be found, the proxy avoids sending invalid history by trimming to a recent user context and adding a system recovery message. This is reasonable for user experience, and strict rejection mode is available for debugging or full compliance.

## Verification

Commands run:

```bash
uv run python scripts/audit_deepseek_protocol.py
uv run python -m unittest discover -s tests
```

Audit helper result:

```text
PASS default thinking request restores tool-call reasoning
PASS default thinking request restores final post-tool reasoning
PASS plain chat history does not require historical reasoning
PASS streamed tool-call reasoning is available before DONE
OBSERVE pass-through mode treats an omitted thinking field as non-thinking; DeepSeek's documented default is thinking enabled.
SUMMARY passed=4 failed=0 observations=1
```

Unit test result:

```text
Ran 93 tests in 15.407s
OK (skipped=1)
```

The skipped test is the live DeepSeek integration test guarded by `RUN_LIVE_DEEPSEEK_TESTS=1` and `LIVE_DEEPSEEK_KEY`.

## Recommended Next Steps

1. Fix F1 by making omitted `thinking` in pass-through mode behave as enabled for DeepSeek thinking models, unless the request explicitly disables thinking.
2. Decide and document precedence for request-level `thinking` versus proxy config. If the proxy is meant to be a general compatibility layer, explicit request values should probably win.
3. Add a regression test for pass-through omitted-thinking behavior after fixing F1.
4. Add optional validation or diagnostics for mismatched `tool_call_id` in tool results.
5. Clarify streaming-only versus non-streaming reasoning display behavior in README, or add non-streaming display mirroring.

## Bottom Line

Default Cursor usage is protocol-compatible for the core DeepSeek thinking-mode tool-call requirement. The proxy correctly records and restores `reasoning_content` across regular and streamed tool-call loops. The main alignment issue is limited to pass-through mode when callers rely on DeepSeek's default thinking behavior but omit the explicit `thinking` field.

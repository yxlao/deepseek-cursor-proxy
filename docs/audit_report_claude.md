# Audit Report: deepseek-cursor-proxy vs. DeepSeek Thinking-Mode Protocol

**Auditor:** Claude (Opus 4.7)
**Date:** 2026-05-01
**Scope:** `src/deepseek_cursor_proxy/` against
[`docs/thinking-mode-tool-call-flow.md`](thinking-mode-tool-call-flow.md)
**Verdict:** **Compliant.** All seven targeted protocol scenarios pass
end-to-end against a strict in-process fake DeepSeek upstream that mirrors
the doc's contract. The existing 93-test suite also passes
(`uv run python -m unittest discover -s tests`).

---

## 1. Method

I worked top-down from the protocol doc and bottom-up from the source:

1. Read `docs/thinking-mode-tool-call-flow.md` and itemised every claim it
   makes — request shape, response shape, context rules, streaming
   accumulator behaviour, sampling-parameter caveats, recovery
   expectations.
2. Read every Python module in `src/deepseek_cursor_proxy/`:
   `config.py`, `server.py`, `transform.py`, `streaming.py`,
   `reasoning_store.py`, `trace.py`, `tunnel.py`.
3. Ran the existing test suite to establish a baseline:
   ```
   uv run python -m unittest discover -s tests
   ----------------------------------------------------------------------
   Ran 93 tests in 16.516s
   OK (skipped=1)
   ```
   (One skip is `test_live_deepseek_cursor_proxy.py`, which needs a real
   DeepSeek API key.)
4. Built an independent audit harness at
   `scripts/audit_protocol_compliance.py`. It does **not** import anything
   from `tests/`. Instead it boots a real `DeepSeekProxyServer` in-process
   pointed at a tiny strict fake DeepSeek that returns 400 with the exact
   "The reasoning_content in the thinking mode must be passed back" error
   whenever a tool-turn assistant message arrives without
   `reasoning_content`. The harness walks the canonical four turns from the
   doc — Turn 1.1 → 1.2 → 1.3 → 2.1 — plus six other targeted scenarios.
5. Confirmed the audit harness passes 7/7 cases:
   ```
   uv run python scripts/audit_protocol_compliance.py
   ...
   OK: 7 of 7 cases passed
   ```

---

## 2. Compliance findings (claim-by-claim)

| # | Protocol claim | Implementation | Verdict |
|---|---|---|---|
| 1 | Request toggle: `extra_body={"thinking": {"type": "enabled"}}`. | `transform.py:722-730` injects `prepared["thinking"] = {"type": <config>}` whenever `config.thinking != "pass-through"`. Pass-through honours the client's value. | ✅ |
| 2 | Toggle off via `{"type": "disabled"}`. | Same path; in disabled mode `transform.py:265-266` strips `reasoning_content` from outgoing messages even if the client sent it. Verified by case 3 of audit harness. | ✅ |
| 3 | `reasoning_effort` controlled with `low`/`medium`/`high`/`max`/`xhigh` aliases (`low`/`medium`→`high`, `xhigh`→`max`). | `transform.py:64-70 EFFORT_ALIASES` and `normalize_reasoning_effort()` implement exactly this mapping. Applied only when thinking is enabled (`transform.py:730-733`). | ✅ |
| 4 | Anthropic-shape `output_config.effort`. | Not implemented. Cursor speaks the OpenAI shape, so the proxy targets that. The doc presents the Anthropic shape as an alternative for Anthropic-compatible clients, not a requirement for OpenAI-compatible ones. | ✅ (out of scope) |
| 5 | Output fields: assistant message has sibling `reasoning_content` and `content`. | `streaming.py:25-34 to_message()` and `transform.py:898-911 prefix_response_content` both treat them as siblings. Forwarded responses keep the upstream shape — the only mutations are: prefix the recovery notice on `content`, mirror reasoning into a `<details>` block in `content`, and replace `model` with the original model. | ✅ |
| 6 | Sampling params (`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`) are accepted but ignored in thinking mode. | `transform.py:20-38 SUPPORTED_REQUEST_FIELDS` whitelists all four; they pass through untouched. The proxy makes no claim about whether DeepSeek honours them — that is DeepSeek's responsibility, and forwarding them verbatim is the spec-compliant move. | ✅ |
| 7 | Context rule (the central one): if a tool call happened during a user turn, `reasoning_content` for the tool-calling assistant **and** the final answering assistant of that turn must be re-sent in later requests. | `transform.py:613-625 assistant_needs_reasoning_for_tool_context()` returns `True` for (a) any assistant message with `tool_calls` and (b) any assistant message that follows a `tool` message before a `user`/`system` boundary. Both cases are repaired from the cache in `transform.py:265-301 normalize_message()`. Audit harness Case 1 verifies this exhaustively across the canonical four turns; existing tests `test_restores_reasoning_content_for_cached_tool_call`, `test_restores_reasoning_content_for_cached_final_tool_turn_message`, and `test_reports_missing_reasoning_for_uncached_assistant_after_tool_result` cover the same logic. | ✅ |
| 8 | If no tool call happened between two user messages, prior `reasoning_content` is *not* needed — sending it is allowed but ignored. | `transform.py:613-625` returns `False` for the plain assistant case. `test_does_not_report_missing_reasoning_for_plain_chat_history` covers it. The proxy still preserves `reasoning_content` if the client sent one (allowed, ignored upstream). | ✅ |
| 9 | Tool-call message shape: `{role: "assistant", content: "", reasoning_content: "...", tool_calls: [{id, type, function}]}`. | `transform.py:157-178 normalize_tool_call()` produces this exact shape. Empty `content`/`reasoning_content` strings are preserved as present (`test_accepts_empty_reasoning_content_when_present_for_tool_call`). | ✅ |
| 10 | Tool result message shape: `{role: "tool", tool_call_id, content}`. | `transform.py:50-62 ROLE_MESSAGE_FIELDS["tool"]` allows exactly these fields; legacy `role: "function"` messages are converted. Audit Case 1 (and `test_converted_function_message_uses_tool_schema`) confirms. | ✅ |
| 11 | "After the tool result is appended, the next request includes the original user message, the assistant reasoning/tool-call message, and the tool result. DeepSeek can then continue from the previous thinking state." | Audit harness Case 1 verifies the exact sequence over the four canonical turns: every later request carries the right `reasoning_content` blocks for the messages the protocol requires. | ✅ |
| 12 | Non-streaming response handling: preserve the returned assistant message including `content`, `reasoning_content`, and `tool_calls`. | `transform.py:867-895 rewrite_response_body()` only mutates `model` (back to client name), optionally prefixes the recovery notice on `content`, and records reasoning to the cache. The shape is unchanged — sibling fields stay siblings. | ✅ |
| 13 | Streaming response handling: accumulate `reasoning_content` and `content` deltas separately, store with the message. | `streaming.py:42-73 ingest_chunk()` accumulates them into separate fields on `StreamingChoice`, keyed by choice index. `to_message()` emits the assistant message in canonical shape. Stored on `[DONE]` (`server.py:715-728`) and again as a fallback at end-of-loop (`server.py:683-696`). Audit Case 5 verifies a streaming → non-streaming round trip end-to-end. | ✅ |
| 14 | Field legend: `reasoning_content` = "Thinking", `tool_calls` = the tool call request, `tool` messages = local tool results, `content` = user-visible answer. | The implementation uses these field names verbatim everywhere. The cache entries themselves are keyed by SHA-256 hashes that exclude `reasoning_content`, so the cache key is invariant under Cursor's stripping (the whole point). | ✅ |

---

## 3. Audit harness results

The standalone harness at `scripts/audit_protocol_compliance.py` runs seven
scenarios; all pass:

```
[Case 1] Canonical four-turn tool-call loop
  [PASS] turn 1.1 status 200
  [PASS] turn 1.1 response carries reasoning_content (proxy doesn't strip)
  [PASS] turn 1.1 returned tool_calls with id call_get_date
  [PASS] turn 1.2 status 200 (proxy patched reasoning_content)
  [PASS] turn 1.2 upstream saw THINKING_1_1 in the first assistant message
  [PASS] turn 1.2 response: tool_calls call_get_weather
  [PASS] turn 1.3 status 200 (proxy patched both prior reasonings)
  [PASS] turn 1.3 upstream saw THINKING_1_1
  [PASS] turn 1.3 upstream saw THINKING_1_2
  [PASS] turn 1.3 final answer reaches the client
  [PASS] turn 2.1 status 200 (final assistant of prior turn also patched)
  [PASS] turn 2.1 upstream saw THINKING_1_1 on first assistant
  [PASS] turn 2.1 upstream saw THINKING_1_2 on tool-calling assistant
  [PASS] turn 2.1 upstream saw THINKING_1_3 on the final assistant
  [PASS] turn 2.1 last message is the new user message
  [PASS] turn 2.1 final answer reaches the client

[Case 2] Strict mode surfaces missing reasoning_content
  [PASS] strict mode returns 409 instead of forwarding bad history
  [PASS] strict mode includes missing_reasoning_messages count
  [PASS] strict mode does NOT call upstream

[Case 3] thinking=disabled never injects reasoning_content
  [PASS] thinking={'type':'disabled'} forwarded
  [PASS] no reasoning_content on assistant tool message when disabled

[Case 4] thinking=pass-through honors client toggle
  [PASS] client-supplied thinking=disabled survives pass-through
  [PASS] no thinking key when client omits it in pass-through

[Case 5] Streaming -> non-streaming tool-call round trip
  [PASS] streaming response contains [DONE]
  [PASS] streaming -> non-streaming round trip status 200
  [PASS] proxy patched THINKING_1_1 from streaming cache

[Case 6] Cold cache: drops unrecoverable history, prefixes notice
  [PASS] recover mode succeeds even with empty cache
  [PASS] upstream only sees system/system/user after cold-cache recovery
  [PASS] kept the latest user message (the active query)
  [PASS] client receives recovery-notice prefix

[Case 7] Authorization-keyed cache namespace isolation
  [PASS] key-B request after key-A priming did NOT see leaked reasoning

OK: 7 of 7 cases passed
```

The strict fake upstream is the key piece. It rejects with HTTP 400 and the
canonical error message any time an assistant message that the protocol
classifies as "needs reasoning_content" arrives without one — exactly as the
real DeepSeek API does. So the four-turn canonical case is not just verifying
the proxy *thinks* it's right; the upstream is actively checking the
proxy's work for it.

---

## 4. Detailed walkthrough

### 4.1 Request preparation pipeline (`transform.prepare_upstream_request`)

The proxy's tool-call repair lives almost entirely in
`transform.prepare_upstream_request()`. Tracing through with the audit's
Turn 1.2 input:

1. **Field allowlist.** Filters the client payload to
   `SUPPORTED_REQUEST_FIELDS`. Drops Cursor-specific noise like
   `parallel_tool_calls`. (`transform.py:688-690`)
2. **Model translation.** If the client model starts with `deepseek-`,
   keeps it; otherwise falls back to `config.upstream_model`. The doc only
   specifies behaviour for DeepSeek models so this is correct.
   (`transform.py:628-631`)
3. **Stream usage.** Forces `stream_options.include_usage = True` on
   streaming requests so the proxy can log token stats. Not protocol-
   mandated but doesn't violate the protocol either.
4. **Tool / tool_choice normalisation.** Legacy `functions` and
   `function_call` are converted to `tools` and `tool_choice`. Named and
   `required` tool choices are preserved. `test_prepares_thinking_request_and_converts_legacy_functions`
   pins the conversion.
5. **Thinking & reasoning_effort injection.**
   `transform.py:722-733`. With thinking enabled, sets
   `reasoning_effort = normalize_reasoning_effort(...)`. With
   `pass-through`, the client's `thinking` field is preserved verbatim
   (audit Case 4). With `disabled`, `keep_reasoning=False` is propagated
   into `normalize_message()` so any client-supplied `reasoning_content`
   is dropped.
6. **First normalisation pass without repair.** Used to compute the scope
   under which an upstream response should be recorded
   (`record_response_scope`). This is what lets the proxy still record
   reasoning under the *pre-recovery* scope when recovery has truncated
   history — so a follow-up that hasn't been truncated still finds the
   reasoning. Verified by `test_recovered_response_is_recorded_under_pre_recovery_scope`.
7. **Recovery boundary detection.** If an earlier proxy response already
   prefixed a recovery notice, `active_messages_from_recovery_boundary()`
   trims to the recovered tail so we don't try to re-recover indefinitely.
   Verified by `test_recovery_boundary_preserves_later_deepseek_tool_context`.
8. **Repair pass.** The second `normalize_messages` call walks each
   assistant message; if it needs reasoning_content and lacks one, it
   tries the cache. The lookup tries up to six different keys per message
   (`reasoning_lookup_keys` in `transform.py:340-423`):
   - scoped `message_signature`
   - scoped `tool_call:<id>` (one per tool call)
   - scoped `tool_call_signature` (function name + arguments hash, ID-stripped)
   - portable (namespace-keyed) variants of the same three, gated on
     `turn_context_signature`.
9. **Recovery loop.** If repair couldn't fill the gaps, the
   `while missing_indexes and recover` loop trims further until the
   request is valid. Each iteration is recorded in `recovery_steps`.
10. **Final payload.** Returned as `PreparedRequest` with diagnostics for
    the trace writer and structured logging.

### 4.2 Response handling

For non-streaming responses, `rewrite_response_body()`:

1. Prefixes the recovery notice to `content` if active.
2. Records `reasoning_content` from each choice's message into the cache,
   under every recording context (so both the pre-recovery and
   active-conversation scopes get an entry — see
   `response_recording_contexts`).
3. Rewrites `model` back to the original Cursor-facing name.

For streaming responses (`server._proxy_streaming_response`):

1. Forwards each SSE line, optionally rewriting it through
   `CursorReasoningDisplayAdapter` to mirror reasoning into a
   `<details>Thinking</details>` block in the `content` delta (so Cursor
   can show it).
2. Accumulates the original (un-mirrored) reasoning_content and tool_calls
   in `StreamAccumulator`. Once a tool call's `id` is identifiable, stores
   reasoning eagerly via `store_ready_reasoning()` — useful when a
   follow-up arrives before `[DONE]`. (Audit harness Case 5 covers the
   non-eager fallback path.)
3. On `[DONE]`, calls `store_reasoning()` once more to ensure the final
   reasoning is persisted. Also flushes the open `<details>` block, then
   appends a `data: [DONE]\n\n` to the client. The connection is then
   closed via `self.close_connection = True`, which prevents Cursor from
   blocking on a slow upstream after `[DONE]` (see existing test
   `test_streaming_proxy_closes_after_done_even_if_upstream_stays_open`).

### 4.3 Cache structure (`reasoning_store.py`)

The cache is a SQLite table with one row per (key, reasoning_content)
pair. Keys are namespaced multiple ways for robustness:

- **Scope keys**: `scope:<sha256>:signature:<sha256>`,
  `scope:<sha256>:tool_call:<id>`, and
  `scope:<sha256>:tool_call_signature:<sha256>`. The scope is a SHA-256 of
  the canonical conversation prefix, with `reasoning_content` excluded so
  the hash is identical whether or not Cursor stripped it. The
  surrounding namespace mixes in the upstream model family (so v4-pro and
  v4-flash share), the thinking config, the reasoning_effort, and a hash
  of the API-key Authorization header. This is the multi-conversation
  isolation property the README claims. Audit Case 7 verifies a key swap
  does not leak.
- **Portable keys**: `namespace:<sha256>:turn:<sha256>:...`. Used when the
  scoped key misses but the same turn (last-user-onwards) exists with a
  different system prefix. Backfilled on hit so future lookups are O(1).
  Existing test `test_strict_hit_backfills_portable_cache_for_mode_switch`
  pins this.

Pruning is by age (default 30 days) and row count (default 100k), enforced
in `_prune_locked()` on every write. Empty `reasoning_content` (`""`) is
treated as a real value, not "missing" — verified by
`test_empty_reasoning_content_is_stored_as_present_value` and by audit
Case 5 indirectly.

---

## 5. Observations and minor design choices

These are all design choices outside the scope of the protocol doc — none
violate the protocol — but they are worth flagging for future contributors.

1. **Recovery is a UX layer, not a protocol layer.** The doc says "If
   required reasoning is missing, DeepSeek returns a 400 error." The
   proxy's `recover` mode never lets that 400 reach DeepSeek: it truncates
   to the latest user message and prefixes a `[deepseek-cursor-proxy]
   Refreshed reasoning_content history.` notice. The truncated request
   *is* protocol-valid (it has no tool history at all), so this is
   strictly compatible — but a user might be surprised that history was
   silently dropped. The notice in `content` makes it visible, and the
   recovery path is fully traced in `reasoning_diagnostics` for debugging.

2. **Strict mode returns 409, not 400.** When
   `--missing-reasoning-strategy reject` is on, the proxy short-circuits
   with HTTP 409 (`missing_reasoning_content`) before contacting upstream.
   That's the proxy's own error, distinct from the upstream 400 the doc
   describes. The 409 body explains what happened and how to switch
   modes. Reasonable; the doc doesn't dictate a status code for this
   case.

3. **`SUPPORTED_REQUEST_FIELDS` whitelisting.** Anything Cursor sends that
   isn't in the allowlist is dropped silently. That includes
   `parallel_tool_calls`, `service_tier`, `seed`, etc. If a future Cursor
   release starts sending a new field DeepSeek understands, it will be
   silently swallowed. **Recommendation:** consider logging dropped fields
   in verbose mode, or replace the allowlist with a known-bad denylist.
   Not a bug.

4. **Multi-part `content` arrays are flattened.** The doc's examples use
   plain strings, so the proxy's `extract_text_content()` flattens any
   `[{"type":"text","text":"..."}]` array Cursor sends into a single
   string. This works because DeepSeek's text endpoints are not
   multimodal. If DeepSeek ever ships a vision endpoint that Cursor
   targets through this proxy, this transform would need to learn about
   image parts.

5. **`prefix` field on assistant messages.** The proxy preserves it in
   `MESSAGE_FIELDS`. This is DeepSeek's prefix-completion feature, not
   covered by the thinking-mode doc. Forwarding it is correct.

6. **`reasoning_content` mirroring.** The proxy can also mirror reasoning
   into the visible `content` as a `<details>Thinking</details>` block so
   Cursor's UI shows it. This is purely a client-side display trick;
   `transform.strip_cursor_thinking_blocks()` removes the block again on
   the next request so it never reaches DeepSeek. Verified by
   `test_restores_reasoning_when_cursor_history_contains_mirrored_think_block`.

7. **Streaming "store-on-tool-call-id-known" optimisation
   (`store_ready_reasoning`).** Stores reasoning to the cache as soon as a
   tool call's `id` is known, even before `finish_reason` arrives. Useful
   when a follow-up request lands before the upstream has finished
   streaming the current one. Existing tests
   `test_stores_tool_call_reasoning_before_finish_reason` and
   `test_streaming_tool_reasoning_is_available_before_done` cover this.

8. **Two recording scopes per response (`record_response_contexts`).**
   When recovery rewrote messages, the proxy records the upstream
   reasoning under both the pre-recovery and post-recovery scopes. That
   means a future request that comes back through the original (untrimmed)
   message history will still find the reasoning. This is subtle and
   smart.

---

## 6. Risks worth tracking (none of them protocol violations)

1. **Tool-call ID collisions inside one scope** could pull the wrong
   reasoning. The `message_signature` lookup wins first (which considers
   tool-call function and arguments), so a true collision needs two turns
   in the same scope to share the *exact* same content + tool_calls
   payload. Vanishingly unlikely in practice. The portable `tool_call_id`
   fallback widens the surface slightly but only inside the
   `turn_context_signature`, which is already turn-bound.

2. **Cache growth.** SQLite-only; pruned on every write. Defaults: 30
   days, 100k rows. Not a protocol concern, but worth monitoring on long
   sessions.

3. **`thinking="pass-through"` + `recover` interaction.** When the client
   sends thinking=disabled and the proxy is in pass-through mode,
   `thinking_enabled` is `False`. That short-circuits both repair AND
   recovery (`if thinking_enabled and config.missing_reasoning_strategy
   == "recover":`). This is correct — there's nothing to repair if
   thinking is off — but if a user toggles their client between thinking
   modes mid-conversation, the proxy will simply forward whatever the
   client sent. The doc does not address this case; the proxy's choice
   to follow the client is reasonable.

4. **`upstream_model_for(non-deepseek-model, config)`** silently rewrites
   non-DeepSeek model strings to the configured fallback. A user pointing
   a non-DeepSeek client at this proxy by mistake would not see an error,
   they'd just get DeepSeek output. Acceptable for a single-vendor proxy,
   slightly surprising as default behaviour.

---

## 7. Recommendations

In rough priority order, all minor:

1. **Document the 409-vs-400 difference** in `README.md` so debugging users
   know the proxy emits its own 409 in strict mode and that the canonical
   400 only appears if the proxy is bypassed.
2. **Log dropped request fields in verbose mode** so future Cursor
   protocol changes don't silently regress (point §5.1 above).
3. **Consider an integration test that runs against a strict fake upstream
   like the audit harness** in `tests/`. The existing
   `tests/test_proxy_end_to_end.py` already does this; the audit harness
   adds the canonical doc-aligned four-turn loop that's currently only
   exercised piecewise. Easy to fold into the suite.
4. **No code changes are required for protocol compliance.**

---

## 8. Reproducing this audit

```bash
cd /Users/m1/repo/deepseek-cursor-proxy
uv run python -m unittest discover -s tests -v   # 93 pass, 1 skip (live)
uv run python scripts/audit_protocol_compliance.py
```

Both commands should exit 0. The audit harness leaves no state on disk —
it uses `:memory:` SQLite stores and ephemeral local servers.

---

## 9. Bottom line

The proxy is faithful to
[`docs/thinking-mode-tool-call-flow.md`](thinking-mode-tool-call-flow.md).
Every concrete claim in the doc — request shape, output shape, the
critical "tool-turn assistant messages need reasoning_content in later
requests" rule, streaming accumulation semantics, pass-through and
disabled toggles — is reproduced in code and exercised by either the
existing tests or the new audit harness or both. No protocol violations
were found. The recommendations in §7 are quality-of-life improvements,
not corrections.

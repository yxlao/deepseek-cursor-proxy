# DeepSeek Thinking-Mode Protocol Audit

**Auditors:** Claude (Opus 4.7) and Codex
**Date:** 2026-05-01
**Reference:** [`docs/thinking-mode-tool-call-flow.md`](thinking-mode-tool-call-flow.md)

For default Cursor usage the proxy is protocol-compliant: 93 unit tests pass,
the canonical four-turn tool-call loop succeeds against a strict fake
DeepSeek, and a real 21-request trace contains zero missing-reasoning or
tool-result-ID failures. The issues below are gaps in non-default modes,
quality concerns, or operational observations.

## Summary

1. Recovery-notice text mismatch — older Cursor transcripts re-recover instead of continuing the active tool-call chain.
2. `thinking="pass-through"` with an omitted client `thinking` field disables reasoning repair even though DeepSeek defaults to thinking enabled.
3. Recovery notices that the proxy added to assistant `content` are later echoed back upstream as if DeepSeek had written them.
4. Outside pass-through mode the proxy overwrites any client-supplied `thinking` field with the configured value.
5. Anthropic-shape `output_config.effort` is not implemented; only the OpenAI-compatible shape works.
6. Non-streaming responses do not mirror `reasoning_content` into the Cursor-visible `<details>` block; only streamed responses do.
7. The "needs reasoning" predicate stops at `system` messages, but the protocol describes the boundary in terms of `user` messages only.
8. Strict mode returns HTTP 409, not the documented upstream HTTP 400, and this difference is undocumented.
9. Tool-result `tool_call_id`s are not validated against pending tool calls.
10. Unknown request fields are silently dropped by the `SUPPORTED_REQUEST_FIELDS` allow-list.
11. Multi-part `content` arrays are flattened to plain text, so the proxy would need work to support a future vision endpoint.
12. Non-DeepSeek model names are silently rewritten to the configured fallback.

---

## Issue 1 — Recovery-notice text mismatch

**Severity: High.** The current `transform.py` recognises only two
recovery-notice prefixes (`"[deepseek-cursor-proxy] Refreshed reasoning_content history."`
and the legacy `"Note: recovered this DeepSeek chat ..."`). Older proxy
versions emitted a third variant: `"[deepseek-cursor-proxy] Recovered this
DeepSeek chat because older tool-call reasoning was unavailable; ..."`. When
a Cursor transcript carrying that older notice arrives today, the boundary
detector misses it and the proxy re-recovers from the latest user message on
every subsequent request, repeatedly losing the active tool-call reasoning
chain. **Fix:** widen the recognized prefixes; add a regression test that
asserts `continued_recovery_boundary == True` on the older notice.

## Issue 2 — `thinking="pass-through"` ignores DeepSeek's default-enabled behaviour

**Severity: High in pass-through deployments; not exercised in default Cursor mode.**
DeepSeek's API defaults to thinking mode enabled. The proxy's repair and
recovery logic, however, runs only when the *outgoing* request explicitly
contains `thinking: {"type": "enabled"}`. In pass-through mode with a client
that omits the `thinking` field, DeepSeek still treats the request as
thinking-mode and demands `reasoning_content`, but the proxy doesn't try to
restore it — so the documented 400 reaches the client. **Fix:** treat
omitted `thinking` as enabled for repair purposes in pass-through mode (still
don't inject the field upstream), or document that pass-through callers must
send `thinking` explicitly to keep reasoning repair active.

## Issue 3 — Recovery notices are replayed upstream as assistant `content`

**Severity: Medium.** The proxy prefixes its recovery notice into the
assistant `content` it returns to Cursor. Cursor faithfully echoes that
content on the next request, and the proxy then forwards the proxy-generated
prose upstream as if DeepSeek had written it. DeepSeek tolerates this in
practice, but it is not exact protocol replay and weakens
message-signature cache lookups (the cache compensates with `tool_call_id`
and `tool_call_signature` keys). **Fix:** strip recovery-notice prefixes
from assistant `content` before forwarding upstream, or express recovery
state as a synthetic `system` message in the trimmed recovery request only.

## Issue 4 — Config overrides request-level `thinking`

**Severity: Low/Medium.** Outside pass-through mode the proxy overwrites
any client-supplied `thinking` field with the configured value. This is
ideal for Cursor (consistent thinking-mode repair) but surprising for a
general-purpose OpenAI-compatible proxy: a caller who explicitly sends
`{"type": "disabled"}` will still get thinking enabled. **Fix:** either
document this as intentional policy, or change precedence so explicit
request-level `thinking` wins with config as the default.

## Issue 5 — Anthropic-shape `output_config.effort` is not implemented

**Severity: Low (out of scope for Cursor).** The protocol doc mentions an
Anthropic-compatible alternative (`output_config.effort`); the proxy
recognises only the OpenAI-compatible `reasoning_effort`. Cursor uses the
OpenAI shape, so this is a non-issue for the project's stated audience.
**Fix:** document as out-of-scope, or add a separate route/transform if
Anthropic-compatible DeepSeek traffic ever becomes a goal.

## Issue 6 — Non-streaming responses don't mirror reasoning into Cursor display

**Severity: Low.** Streamed responses are run through the
`CursorReasoningDisplayAdapter` to mirror `reasoning_content` into a
`<details><summary>Thinking</summary>…</details>` block in the visible
`content`. Non-streaming responses skip that step, so the README's broad
claim that thinking tokens are displayed in Cursor is in fact streaming-only.
Not a protocol violation. **Fix:** either add equivalent mirroring for
non-streaming responses or document the streaming-only scope in the README.

## Issue 7 — `system` messages treated as tool-reasoning boundaries

**Severity: Low.** The "needs reasoning" predicate stops scanning backward
at any `user` *or* `system` message, but the protocol doc phrases the rule
as "between two `user` messages." If a client inserts a `system` message
inside a user turn after a tool result, the trailing assistant could be
marked as not needing `reasoning_content` — too lenient. No such pattern
appears in the captured 21-request trace. **Fix:** stop only at `user`
messages, or document that `system` is intentionally a turn boundary and
add a regression test.

## Issue 8 — Strict mode returns 409, not 400

**Severity: Low (documentation gap).** With
`--missing-reasoning-strategy reject` the proxy short-circuits with HTTP
409 (`missing_reasoning_content`) before contacting upstream. The
documented DeepSeek behaviour for missing reasoning is HTTP 400. This is a
deliberate design choice — the proxy uses 409 to distinguish proxy-detected
gaps from genuine upstream errors — but it is undocumented and can confuse
users debugging against the documented status code. **Fix:** document the
409-vs-400 difference in the README.

## Issue 9 — Tool-result `tool_call_id`s are not validated

**Severity: Low.** When normalising tool messages the proxy checks the
field shape but does not verify that each tool-result `tool_call_id`
corresponds to a pending assistant `tool_calls[i].id`. A misshaped client
could send orphan tool results that the proxy quietly forwards. The
captured trace contained zero such mismatches. **Fix:** add optional
validation or diagnostics in verbose mode.

## Issue 10 — Unknown request fields are silently dropped

**Severity: Low (policy choice).** `SUPPORTED_REQUEST_FIELDS` is an
allow-list. Anything Cursor sends that isn't on the list (e.g.
`parallel_tool_calls`, `service_tier`, `seed`) is dropped without notice.
If a future Cursor release starts sending a new field DeepSeek understands,
it will be silently swallowed. **Fix:** log dropped fields in verbose
mode, or replace the allow-list with a known-bad denylist.

## Issue 11 — Multi-part `content` arrays are flattened

**Severity: Low (forward-compat risk).** `extract_text_content()`
flattens any `[{"type":"text","text":"..."}]` array into a single string.
This works because DeepSeek's text endpoints are not multimodal. If
DeepSeek ever ships a vision endpoint that Cursor targets through this
proxy, this transform would need to learn about image parts. **Fix:**
none required today; revisit when DeepSeek adds vision support.

## Issue 12 — Non-DeepSeek model names are silently rewritten

**Severity: Low.** `upstream_model_for(non-deepseek-model, config)`
silently rewrites any client model name that doesn't start with
`deepseek-` to the configured fallback. A user pointing a non-DeepSeek
client at this proxy by mistake gets DeepSeek output with no error.
Acceptable for a single-vendor proxy; surprising as default behaviour.
**Fix:** consider returning a 4xx for clearly non-DeepSeek model names, or
log a warning.

---

## Reproducing this audit

```bash
uv run python -m unittest discover -s tests             # 93 pass, 1 skip
uv run python scripts/audit_protocol_compliance.py      # 7/7 cases pass
uv run python scripts/audit_deepseek_protocol.py        # 4/4 + 1 OBSERVE
```

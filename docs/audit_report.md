# DeepSeek Thinking-Mode Protocol Compliance Audit

**Auditors:** Claude (Opus 4.7) and Codex
**Date:** 2026-05-01
**Scope:**

- Protocol reference: [`docs/thinking-tools.md`](thinking-tools.md) (mirroring <https://api-docs.deepseek.com/guides/thinking_mode>)
- Implementation reviewed: every Python module under [`src/deepseek_cursor_proxy/`](../src/deepseek_cursor_proxy/) — `transform.py`, `reasoning_store.py`, `streaming.py`, `server.py`, `config.py`, `trace.py`, `tunnel.py`.
- Test surface: the existing `tests/` suite (93 tests).
- Captured DeepSeek traffic: trace session `trace-dumps/20260429T134418.628275Z-pid80943` (21 real Cursor → proxy → DeepSeek round trips).
- Verification scripts added: [`scripts/audit_protocol_compliance.py`](../scripts/audit_protocol_compliance.py) and [`scripts/audit_deepseek_protocol.py`](../scripts/audit_deepseek_protocol.py).

---

## 1. Executive Summary

**Verdict: Substantially compliant in default Cursor configuration; several
gaps in non-default modes plus one concrete regression risk for older trace
shapes.**

For the default Cursor-facing configuration (`thinking="enabled"`,
`missing_reasoning_strategy="recover"`), the proxy faithfully implements the
DeepSeek thinking-mode tool-call protocol:

- It forces `thinking: {"type": "enabled"}`, normalizes `reasoning_effort`
  aliases, and forwards sampling parameters that DeepSeek ignores.
- It records `reasoning_content` from every DeepSeek response (regular and
  streamed) and restores it onto outgoing tool-call assistant messages and
  the final assistant message of a tool turn.
- It correctly distinguishes plain chat history (no reasoning needed) from
  tool-turn history (reasoning required).
- The canonical four-turn loop from the protocol doc (Turn 1.1 → 1.2 → 1.3 →
  2.1) succeeds end-to-end against a strict in-process fake DeepSeek that
  enforces the documented contract.
- A real captured trace of 21 round trips contains zero missing-reasoning or
  tool-result-ID mismatches.

However, the implementation is **not** fully protocol-exact in every mode.
The audit found:

1. **`thinking="pass-through"` + omitted client `thinking` field misses
   DeepSeek's default-enabled behaviour** — repair is gated on the proxy
   _seeing_ an explicit `thinking: enabled`, so a client that relies on
   DeepSeek's default could trigger upstream 400s. _(Severity: High in
   pass-through deployments; not exercised in default Cursor mode.)_
2. **Recovery-notice text mismatch with older trace shapes** — current
   `transform.py` recognises only two recovery-notice strings. The captured
   `20260429T134418.628275Z` trace uses a _third_, slightly older variant
   that today's code would no longer treat as a recovery boundary, causing
   cascading re-recovery on replay. _(Severity: High for users with stored
   transcripts from older proxy versions; the runtime that produced the
   trace had handled it correctly at the time.)_
3. **Recovery notices leak into upstream `content`** — when the proxy
   prefixes its recovery message into the _response_ `content`, that string
   is later echoed back by Cursor, and the proxy then forwards it upstream
   as part of the assistant message that DeepSeek "wrote". _(Severity:
   Medium — works in practice but is not exact replay.)_
4. **Default-config policy overrides request-level `thinking`** — outside
   pass-through mode, any client-supplied `thinking` field is overwritten
   with the configured value. Intentional for Cursor; surprising for a
   general-purpose OpenAI-compatible proxy. _(Severity: Low/Medium.)_
5. **Anthropic-shape `output_config.effort` is not implemented** — only the
   OpenAI-compatible shape is. _(Severity: Low; out of scope for Cursor.)_
6. **Non-streaming responses don't mirror reasoning into a `<details>`
   block** — only streamed responses do. _(Severity: Low; the streaming-only
   policy isn't documented.)_
7. **`assistant_needs_reasoning_for_tool_context` treats `system` as a
   boundary** even though the doc phrases the rule in terms of `user`
   boundaries. _(Severity: Low; no instances seen in real trace.)_

Items 1, 2, 3 are the only items with real protocol consequences in
production; 4–7 are policy/design observations or low-probability edge
cases. **No code changes were required to make the canonical default
Cursor flow work** — every existing test still passes, and both audit
harnesses confirm the default tool-call loop is correct.

---

## 2. Verification Methodology

This audit combined three independent approaches whose results agree on the
default path and complement each other on the edges:

### 2.1 Source-level review

Read each of the seven modules in `src/deepseek_cursor_proxy/` against every
concrete claim in `docs/thinking-tools.md`. Specific citations
are noted inline in §4.

### 2.2 In-process strict-fake harness — `scripts/audit_protocol_compliance.py`

Boots a real `DeepSeekProxyServer` against a tiny in-process upstream that
returns the canonical
`"The reasoning_content in the thinking mode must be passed back to the API."`
HTTP 400 whenever an assistant message that needs `reasoning_content`
arrives without one. Walks seven scenarios:

```
[Case 1] Canonical four-turn tool-call loop                       PASS
[Case 2] Strict mode surfaces missing reasoning_content           PASS
[Case 3] thinking=disabled never injects reasoning_content        PASS
[Case 4] thinking=pass-through honors client toggle               PASS
[Case 5] Streaming -> non-streaming tool-call round trip          PASS
[Case 6] Cold cache: drops unrecoverable history, prefixes notice PASS
[Case 7] Authorization-keyed cache namespace isolation            PASS
OK: 7 of 7 cases passed
```

Case 1 alone runs the protocol doc's exact canonical flow (Turn 1.1 → 1.2 →
1.3 → 2.1) and verifies on every turn that the upstream saw the right
`reasoning_content` on every prior assistant message. The strict upstream
would 400 on the first protocol slip, so the harness is also actively
checking the proxy's work — not just self-asserting.

### 2.3 Protocol-checker harness — `scripts/audit_deepseek_protocol.py`

Independently developed audit script that exercises four scenarios and
emits structured `PASS`/`OBSERVE` lines:

```
PASS    default thinking request restores tool-call reasoning
PASS    default thinking request restores final post-tool reasoning
PASS    plain chat history does not require historical reasoning
PASS    streamed tool-call reasoning is available before DONE
OBSERVE pass-through mode treats an omitted thinking field as non-thinking;
        DeepSeek's documented default is thinking enabled.
SUMMARY passed=4 failed=0 observations=1
```

The single `OBSERVE` line is Finding F1 below.

### 2.4 Existing test suite

```
uv run python -m unittest discover -s tests
----------------------------------------------------------------------
Ran 93 tests in 15.4s
OK (skipped=1)
```

The single skip is `test_live_deepseek_cursor_proxy.py`, gated on
`RUN_LIVE_DEEPSEEK_TESTS=1` and `LIVE_DEEPSEEK_KEY`.

### 2.5 Trace audit of real Cursor → DeepSeek traffic

Replayed the 21 `request-*.json` files from
`trace-dumps/20260429T134418.628275Z-pid80943` through a stricter checker
over `transform.upstream_request_body`:

| Check                                                       | Result |
| ----------------------------------------------------------- | -----: |
| Requests audited                                            |     21 |
| Completed HTTP 200 requests                                 |     21 |
| Cursor messages in trace                                    |    612 |
| Upstream messages sent to DeepSeek                          |    279 |
| Assistant messages requiring `reasoning_content`            |     96 |
| Required assistant messages **missing** `reasoning_content` |  **0** |
| Assistant messages with `tool_calls`                        |     80 |
| Tool result messages                                        |     87 |
| Tool result ID/order mismatches                             |      0 |
| Unsupported upstream request keys found by checker          |      0 |
| Requests with unexpected `thinking` or `reasoning_effort`   |      0 |
| Forwarded ignored sampling params found                     |      0 |

Captured trace behaviour:

- Requests 1–12: clean repair path, no recovery.
- Request 13: one recovery event; 40 Cursor messages reduced to 3 upstream
  messages, with 9 missing-reasoning diagnostics and 38 dropped messages.
- Requests 14–21: the runtime treated the prior recovery notice as a
  boundary continuation and preserved the recovered active slice. No
  cascading recovery occurred in the captured trace.

**The captured runtime sent only valid DeepSeek tool-call requests — no
400s — for the entire 21-request session.** However, replaying the same
Cursor inputs through current `transform.py` does _not_ preserve that
boundary continuation (see Finding 2 below).

---

## 3. Compliance Matrix

| #   | Protocol claim                                                                                                                                 | Implementation status                                                         | Evidence                                                                                                                                                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Request toggle: `extra_body={"thinking": {"type": "enabled"}}`                                                                                 | Aligned for default config; see Finding 1 for pass-through edge               | `transform.py:722-730` injects `prepared["thinking"] = {"type": <config>}` whenever `config.thinking != "pass-through"`.                                                                                                                         |
| 2   | Toggle off via `{"type": "disabled"}`                                                                                                          | Aligned (audit Case 3)                                                        | Disabled mode propagates `keep_reasoning=False` into `normalize_message()` so `reasoning_content` is stripped (`transform.py:265-266`).                                                                                                          |
| 3   | Thinking mode is **enabled by default** at DeepSeek                                                                                            | Aligned only when proxy injects `enabled` explicitly; **gap in pass-through** | Finding 1 below. `thinking_enabled` is true only when `prepared["thinking"] == {"type": "enabled"}` (`transform.py:726`).                                                                                                                        |
| 4   | `reasoning_effort` supports `high` and `max`; aliases `low`/`medium` → `high`, `xhigh` → `max`                                                 | Aligned                                                                       | `EFFORT_ALIASES` (`transform.py:64-70`) implements the mapping; applied only when thinking is enabled (`transform.py:730-733`).                                                                                                                  |
| 5   | Anthropic-shape `output_config.effort`                                                                                                         | Not implemented (Finding 5)                                                   | `output_config` is not in `SUPPORTED_REQUEST_FIELDS`. Out of scope for Cursor's OpenAI-compatible flow.                                                                                                                                          |
| 6   | Output: assistant message has sibling `reasoning_content` and `content`                                                                        | Aligned                                                                       | `streaming.py:25-34` and `transform.py:867-895` both treat them as siblings. The only mutations are: prefix the recovery notice on `content`, mirror reasoning into `<details>` in streaming `content`, replace `model` with the original model. |
| 7   | Sampling params (`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`) accepted but ignored in thinking mode                        | Aligned                                                                       | `SUPPORTED_REQUEST_FIELDS` (`transform.py:20-38`) whitelists all four; pass through verbatim. Trace audit confirmed zero forwarded sampling params (none were sent by Cursor; the proxy would have forwarded them if present).                   |
| 8   | If a tool call happened during a user turn, prior tool-call assistant `reasoning_content` AND final answer `reasoning_content` must be re-sent | Aligned (audit Case 1; trace)                                                 | `assistant_needs_reasoning_for_tool_context()` (`transform.py:613-625`) flags both cases; repair runs in `normalize_message()` (`transform.py:265-301`).                                                                                         |
| 9   | If no tool call happened between two user messages, prior reasoning is optional and ignored                                                    | Aligned                                                                       | Same predicate returns `False` for plain assistants; `test_does_not_report_missing_reasoning_for_plain_chat_history` covers it.                                                                                                                  |
| 10  | Tool-call assistant shape `{role, content, reasoning_content, tool_calls:[{id, type, function}]}`                                              | Aligned                                                                       | `normalize_tool_call()` (`transform.py:157-178`); empty strings preserved as present.                                                                                                                                                            |
| 11  | Tool result shape `{role: "tool", tool_call_id, content}`                                                                                      | Aligned                                                                       | `ROLE_MESSAGE_FIELDS["tool"]` (`transform.py:50-62`); legacy `role: "function"` is converted; trace shows zero ID mismatches. The proxy does not validate every result ID matches a pending tool call — minor.                                   |
| 12  | Non-streaming response: preserve assistant message including `content`, `reasoning_content`, `tool_calls`                                      | Aligned                                                                       | `rewrite_response_body()` (`transform.py:867-895`) only rewrites `model`, optionally prefixes recovery notice on `content`, and stores reasoning.                                                                                                |
| 13  | Streaming response: accumulate `reasoning_content` and `content` deltas separately                                                             | Aligned (audit Case 5)                                                        | `StreamAccumulator.ingest_chunk()` (`streaming.py:42-73`); stored on `[DONE]` (`server.py:715-728`) plus end-of-loop fallback (`server.py:683-696`).                                                                                             |
| 14  | Streamed tool-call reasoning available before next request                                                                                     | Aligned                                                                       | `store_ready_reasoning()` (`streaming.py:110-139`) stores once tool-call IDs exist, even before `[DONE]`. Existing `test_streaming_tool_reasoning_is_available_before_done` covers it.                                                           |
| 15  | Cache keys isolate conversations across users / configurations                                                                                 | Aligned (audit Case 7)                                                        | Scope = SHA-256 of canonical conversation prefix excluding `reasoning_content`; namespace mixes upstream URL, model family, thinking config, reasoning_effort, and Authorization hash (`reasoning_store.py:86-92`, `transform.py:640-660`).      |
| 16  | Field legend: `reasoning_content`/`content`/`tool_calls`/`tool` used verbatim                                                                  | Aligned                                                                       | All field names match the doc; empty `reasoning_content` (`""`) is treated as a present value, not missing (`test_empty_reasoning_content_is_stored_as_present_value`).                                                                          |

---

## 4. Findings

Severity reflects impact on protocol compliance in real deployments.
Default Cursor users are not affected by Findings 1, 4, 5, 6, 7. Finding 2
matters when replaying older traces. Finding 3 is a quality issue that the
real DeepSeek API tolerates today.

### Finding 1 — `thinking="pass-through"` misclassifies DeepSeek's default-enabled behaviour

**Severity: High in pass-through deployments. Not exercised in default Cursor mode.**

**Protocol requirement:** the doc states "Thinking mode is enabled by
default in DeepSeek's current API." Tool-call histories sent under thinking
mode require `reasoning_content` on prior tool-turn assistants.

**Implementation:** `prepare_upstream_request()` sets `thinking_enabled =
True` only when `prepared["thinking"]` is a dict with `type == "enabled"`
(`transform.py:725-727`). Repair (`transform.py:767`) and recovery
(`transform.py:760`) are gated on that flag.

**Impact:** With `ProxyConfig(thinking="pass-through")` and a client request
that _omits_ `thinking`, the proxy forwards no `thinking` field, DeepSeek
applies its default (enabled), but the proxy's repair logic is off — so a
Cursor-style tool-call history without `reasoning_content` flows upstream
unmodified and triggers the documented 400.

**Evidence:**

```
$ uv run python scripts/audit_deepseek_protocol.py
OBSERVE pass-through mode treats an omitted thinking field as non-thinking;
        DeepSeek's documented default is thinking enabled.
```

Audit Case 4 in the strict-fake harness directly verifies that the proxy
sends no `thinking` key when the client omits it under pass-through:

```
[PASS] no thinking key when client omits it in pass-through
```

The remaining gap is that _repair_ doesn't run in this state.

**Recommendation:** treat omitted `thinking` as enabled for repair purposes
in pass-through mode (still don't inject a `thinking` field upstream), or
document that pass-through callers must send `thinking` explicitly to keep
reasoning repair active. Add a regression test for pass-through + omitted
`thinking` + missing tool-call reasoning.

### Finding 2 — Current code does not recognise this trace's recovery boundary

**Severity: High for replays of older trace shapes; not currently exercised by fresh runs.**

**Background:** `transform.has_recovery_notice()` recognises only two
prefixes today:

- `"[deepseek-cursor-proxy] Refreshed reasoning_content history."` (current)
- `"Note: recovered this DeepSeek chat because older tool-call reasoning was
unavailable; continuing with recent context only."` (legacy)

**Trace evidence:** the captured session `20260429T134418.628275Z-pid80943`
contains a _third_ notice variant:

```
[deepseek-cursor-proxy] Recovered this DeepSeek chat because older tool-call
reasoning was unavailable; continuing with recent context only.
```

`trace.py` still recognises this older prefix in
`message_summaries.has_recovery_notice` (`trace.py:108-112`); `transform.py`
does not.

**Impact:** Replaying Cursor request bodies 14–21 from this trace through
current `prepare_upstream_request()` produces:

| Request | Captured upstream msgs | Replay upstream msgs | Replay recovered | Replay dropped | Boundary detected? |
| ------- | ---------------------: | -------------------: | ---------------: | -------------: | ------------------ |
| 13      |                      3 |                    3 |                9 |             38 | false              |
| 14      |                      5 |                    3 |                9 |             40 | false              |
| 15      |                      7 |                    3 |                9 |             42 | false              |
| 16      |                      9 |                    3 |                9 |             44 | false              |
| 17      |                     11 |                    3 |                9 |             46 | false              |
| 18      |                     13 |                    3 |                9 |             48 | false              |
| 19      |                     15 |                    3 |                9 |             50 | false              |
| 20      |                     17 |                    3 |                9 |             52 | false              |
| 21      |                     19 |                    3 |                9 |             54 | false              |

The captured runtime correctly _continued_ the recovered chain after
request 13 (preserving 5/7/9/… upstream messages); current code re-recovers
every request back to the latest user message. This avoids 400s but loses
the active tool-call reasoning chain — directly contrary to the protocol's
goal of preserving reasoning across the turn.

**Recommendation:** add the older `"[deepseek-cursor-proxy] Recovered ..."`
prefix to recovery-notice recognition in `transform.py`, and add a
regression test that feeds the request-14 transcript through
`prepare_upstream_request` and asserts `continued_recovery_boundary == True`.

### Finding 3 — Recovery notices are replayed upstream as assistant `content`

**Severity: Medium.**

**Behaviour:** in recovery mode the proxy prefixes a Cursor-visible notice
into assistant `content`. When Cursor echoes that assistant message back on
the next request, the proxy forwards it upstream verbatim. Real captured
example from request 14:

```json
{
  "role": "assistant",
  "content": "[deepseek-cursor-proxy] Recovered this DeepSeek chat because older tool-call reasoning was unavailable; continuing with recent context only.\n\n",
  "reasoning_content": "The user wants to extend a script ...",
  "tool_calls": [...]
}
```

**Impact:**

- Not exact protocol replay — DeepSeek's example flow appends
  `response.choices[0].message`, but this `content` was generated by the
  proxy.
- DeepSeek accepts the request (the protocol does not forbid
  proxy-generated text), so this is not a hard violation.
- Message-signature cache lookups become weaker because content no longer
  matches the upstream-original message; the proxy compensates by also
  keying on `tool_call_id` and `tool_call_signature`, which works in
  practice.

**Recommendation:** strip recovery-notice prefixes from assistant `content`
before forwarding upstream (keep the notice Cursor-visible only), or
express recovery state as a synthetic `system` message that lives only in
the trimmed recovery request. Add tests for both current and legacy notice
prefixes round-tripping through Cursor.

### Finding 4 — Default config overrides request-level `thinking`

**Severity: Low/Medium depending on intended audience.**

**Behaviour:** unless `config.thinking == "pass-through"`, the proxy
overwrites any client-supplied `thinking` field with the configured value
(`transform.py:722-723`).

**Impact:** Excellent for Cursor (consistent thinking-mode repair), but a
caller who explicitly disables thinking in a request will still get
thinking enabled unless they reconfigure the proxy. As a general-purpose
OpenAI-compatible compatibility layer this is surprising.

**Recommendation:** either document this as intentional proxy policy, or
change precedence so explicit request-level `thinking` wins, with config as
a default.

### Finding 5 — Anthropic-shape `output_config.effort` is not implemented

**Severity: Low (out of scope for Cursor).**

`output_config` is not in the `SUPPORTED_REQUEST_FIELDS` allow-list. Cursor
uses the OpenAI-compatible shape, so this is a non-issue for the project's
stated audience. Document as out-of-scope, or add a separate route/transform
if Anthropic-compatible DeepSeek traffic ever becomes a goal.

### Finding 6 — Non-streaming responses don't mirror reasoning into Cursor-visible content

**Severity: Low.**

**Behaviour:** streaming responses run through `CursorReasoningDisplayAdapter`
to mirror `reasoning_content` into a `<details><summary>Thinking</summary>…`
block in `content` (`streaming.py:214`). Non-streaming `rewrite_response_body()`
records reasoning and rewrites the model name but does not mirror.

**Impact:** the README's broad claim "Displays DeepSeek's thinking tokens in
Cursor by forwarding them into Cursor-visible collapsible Markdown" is
streaming-only. Non-streaming Cursor flows would not see the thinking text.
This is a UX issue, not a protocol violation.

**Recommendation:** add equivalent optional mirroring for non-streaming
responses, or document the streaming-only scope.

### Finding 7 — `system` messages are treated as tool-reasoning boundaries

**Severity: Low (no instances in real trace).**

**Behaviour:** `assistant_needs_reasoning_for_tool_context()` stops scanning
backward at any `user` _or_ `system` message (`transform.py:619-624`).

**Protocol text:** the doc phrases the boundary as "between two `user`
messages." It does not say a `system` message ends a tool-call turn.

**Impact:** if a client inserts a `system` message inside a user turn after
a tool result and before an assistant message, the predicate would mark the
trailing assistant as not needing `reasoning_content` — too lenient. No such
pattern was seen in the 21-request trace.

**Recommendation:** stop only at `user` messages, or document that
`system` is treated as a turn boundary by design and add a regression test.

### Observation — Strict mode returns 409 instead of upstream's 400

When `--missing-reasoning-strategy reject` is on, the proxy short-circuits
with HTTP 409 (`missing_reasoning_content`) before contacting upstream
(audit Case 2). This is the proxy's own error code, distinct from the
upstream 400 the doc describes. Reasonable; the doc doesn't dictate a
status code for proxy-detected gaps. Worth documenting.

### Observation — Recovery is a UX layer, not a strict protocol replay

When required `reasoning_content` is unavailable, recover mode trims to the
latest user request, prepends a recovery-system message, and prefixes the
Cursor response with a notice (`transform.py:523-610`). The truncated
request _is_ protocol-valid (it has no tool history), so this is strictly
compatible — but a user might be surprised that history was silently
dropped. The notice in `content` makes it visible, and
`reasoning_diagnostics`/`recovery_steps` give full debugging trails.

### Observation — Portable cache fallbacks restore reasoning across mode/system-prompt switches

The cache stores reasoning under both **scoped** keys (full conversation
prefix) and **portable** keys (turn-context only, system-prompt-stripped).
Portable hits worked in the trace:

- Requests 6–12 used portable fallback hits after Cursor mode/model surface
  changes.
- Request 19 switched from `deepseek-v4-pro` to `deepseek-v4-flash` and
  still had 6 portable hits.
- DeepSeek accepted those requests.

**Tradeoff:** a portable hit is correct when the assistant tool-call
message is genuinely the same message in a rewritten transcript, but less
exact when the surrounding system prompt or upstream model differs from
the prefix that originally produced the cached reasoning.
**Recommendation:** keep portable fallback (it solves real Cursor
mode-switch problems); mark portable hits as compatibility repairs in
logs/trace; consider making portable fallback configurable for strict
debugging. The code already prefers exact scoped hits when available.

---

## 5. Detailed Architecture Walkthrough

### 5.1 Request preparation pipeline (`transform.prepare_upstream_request`)

Tracing through with the audit's Turn 1.2 input:

1. **Field allowlist** — filters the client payload to
   `SUPPORTED_REQUEST_FIELDS` (`transform.py:688-690`). Drops Cursor noise
   like `parallel_tool_calls`. _Policy choice, not pure passthrough._
2. **Model translation** — keeps client model if it starts with `deepseek-`,
   else falls back to `config.upstream_model` (`transform.py:628-631`).
3. **Stream usage injection** — forces
   `stream_options.include_usage = True` on streaming requests for stats
   logging.
4. **Tool / tool_choice normalisation** — legacy `functions` and
   `function_call` are converted; named and `required` tool choices are
   preserved (`test_prepares_thinking_request_and_converts_legacy_functions`).
5. **Thinking & reasoning_effort injection** (`transform.py:722-733`). With
   thinking enabled, sets `reasoning_effort = normalize_reasoning_effort()`.
   Pass-through preserves the client's value verbatim (audit Case 4) — but
   see Finding 1. Disabled propagates `keep_reasoning=False` so any
   client-supplied `reasoning_content` is stripped.
6. **First normalisation pass without repair** — used to compute the scope
   under which the upstream response should be recorded
   (`record_response_scope`). This lets the proxy still record reasoning
   under the _pre-recovery_ scope after recovery has truncated history, so
   a follow-up that hasn't been truncated still finds it. Verified by
   `test_recovered_response_is_recorded_under_pre_recovery_scope`.
7. **Recovery-boundary detection** — if an earlier proxy response prefixed a
   recovery notice, `active_messages_from_recovery_boundary()` trims to the
   recovered tail to avoid re-recovering indefinitely. Verified by
   `test_recovery_boundary_preserves_later_deepseek_tool_context`. **Caveat:
   only recognises two notice strings — see Finding 2.**
8. **Repair pass** — the second `normalize_messages` call walks each
   assistant message; if it needs reasoning_content and lacks one, it tries
   the cache. The lookup tries up to six different keys per message
   (`reasoning_lookup_keys` in `transform.py:340-423`):
   - scoped `message_signature`
   - scoped `tool_call:<id>` (one per tool call)
   - scoped `tool_call_signature` (function name + arguments hash, ID-stripped)
   - portable variants of the same three, gated on `turn_context_signature`.
9. **Recovery loop** — if repair couldn't fill the gaps, a `while
missing_indexes and recover` loop trims further until the request is
   valid. Each iteration is recorded in `recovery_steps`.
10. **Final payload** — returned as `PreparedRequest` with diagnostics for
    the trace writer and structured logging.

### 5.2 Response handling

**Non-streaming** (`rewrite_response_body`):

1. Prefixes the recovery notice to `content` if active.
2. Records `reasoning_content` from each choice's message under every
   recording context (so both pre-recovery and active-conversation scopes
   get an entry — see `response_recording_contexts`).
3. Rewrites `model` back to the original Cursor-facing name.

**Streaming** (`server._proxy_streaming_response`):

1. Forwards each SSE line, optionally rewriting through
   `CursorReasoningDisplayAdapter` to mirror reasoning into a
   `<details>Thinking</details>` block.
2. Accumulates the original (un-mirrored) reasoning_content and tool_calls
   in `StreamAccumulator`.
3. Once a tool call's `id` is identifiable, stores reasoning eagerly via
   `store_ready_reasoning()` — useful when a follow-up arrives before
   `[DONE]` (audit Case 5; existing
   `test_streaming_tool_reasoning_is_available_before_done`).
4. On `[DONE]`, calls `store_reasoning()` once more, flushes the open
   `<details>` block, sends `data: [DONE]\n\n`, and sets
   `self.close_connection = True`. Existing
   `test_streaming_proxy_closes_after_done_even_if_upstream_stays_open`
   pins the close-on-DONE behaviour.

### 5.3 Cache structure (`reasoning_store.py`)

SQLite table with one row per (key, reasoning_content) pair. Three scoped
key shapes plus three portable variants:

- **Scoped** — `scope:<sha256>:signature:<sha256>`,
  `scope:<sha256>:tool_call:<id>`,
  `scope:<sha256>:tool_call_signature:<sha256>`.
  Scope = SHA-256 of the canonical conversation prefix with
  `reasoning_content` excluded → identical hash whether or not Cursor
  stripped the field. Namespace mixes upstream model family (so v4-pro and
  v4-flash share), thinking config, reasoning_effort, and a hash of the
  Authorization header. Audit Case 7 verifies a key swap doesn't leak.
- **Portable** — `namespace:<sha256>:turn:<sha256>:...`. Used when the
  scoped key misses but the same turn (last-user-onwards) exists with a
  different system prefix. Backfilled on hit so future lookups are O(1).
  `test_strict_hit_backfills_portable_cache_for_mode_switch` pins this.

Pruning: by age (default 30 days) and row count (default 100k), enforced
in `_prune_locked()` on every write. Empty `reasoning_content` (`""`) is
treated as a real value, not missing
(`test_empty_reasoning_content_is_stored_as_present_value`).

### 5.4 Message normalisation and repair — protocol-relevant details

- Assistant messages with `tool_calls` always require reasoning.
- Assistant messages after a `tool` result require reasoning until a
  user/system boundary (Finding 7 notes the system-boundary deviation).
- Plain assistant messages outside tool context do not require reasoning.
- Existing `reasoning_content` is preserved as long as thinking is not
  disabled.
- Missing reasoning is restored from cache when possible.
- Mirrored Cursor display blocks are stripped from assistant `content`
  before cache lookup (`strip_cursor_thinking_blocks`), so streaming
  display markup doesn't poison later signatures
  (`test_restores_reasoning_when_cursor_history_contains_mirrored_think_block`).

### 5.5 Recovery behaviour

Recovery is intentionally _not_ a perfect protocol replay. If a required
historical `reasoning_content` cannot be found, the proxy avoids sending
invalid history by trimming to a recent user context and adding a system
recovery message. Strict rejection mode is available for full compliance
debugging.

---

## 6. Areas That Comply In The Captured Trace

The following protocol areas were correct in every one of the 21 captured
upstream requests:

- All upstream requests explicitly sent `thinking: {"type": "enabled"}`.
- All upstream requests sent normalised `reasoning_effort: "high"`.
- No upstream request forwarded `temperature`, `top_p`, `presence_penalty`,
  or `frequency_penalty`.
- All assistant messages with `tool_calls` had string `reasoning_content`.
- All final assistant messages after tool results (and before the next user
  boundary) had string `reasoning_content`.
- Every `tool` message referenced a pending tool-call ID.
- Streaming response chunks preserved `reasoning_content` in the
  Cursor-facing stream while also mirroring it into visible `content`.
- Streamed tool-call reasoning was stored before `[DONE]` once tool-call
  IDs were known.

---

## 7. Risks worth tracking

These are not protocol violations — flagged for future contributors:

1. **Tool-call ID collisions inside one scope** could pull the wrong
   reasoning. The `message_signature` lookup wins first (which considers
   tool-call function and arguments), so a true collision requires two
   turns in the same scope to share the _exact_ same content + tool_calls
   payload. Vanishingly unlikely. The portable `tool_call_id` fallback
   widens the surface slightly but only inside the
   `turn_context_signature`, which is already turn-bound.
2. **Function-signature fallback** (`tool_call_signature`) can mis-associate
   reasoning if a client drops tool-call IDs and repeats identical function
   calls with identical arguments in the same scoped context. Low
   probability; exact message signature plus tool-call ID are tried first.
3. **Cache growth.** SQLite-only; pruned on every write. Defaults: 30
   days, 100k rows. Worth monitoring on long sessions.
4. **`thinking="pass-through"` + `recover` interaction.** When the client
   sends `thinking=disabled` and the proxy is in pass-through mode,
   `thinking_enabled` is False, which short-circuits both repair and
   recovery. Correct — there's nothing to repair if thinking is off — but
   subtle when a user toggles their client between thinking modes
   mid-conversation. (Distinct from Finding 1, which is _omitted_ thinking
   in pass-through.)
5. **`upstream_model_for(non-deepseek-model, config)`** silently rewrites
   non-DeepSeek model strings to the configured fallback. A user pointing a
   non-DeepSeek client at this proxy by mistake gets DeepSeek output with
   no error. Acceptable for a single-vendor proxy.
6. **`SUPPORTED_REQUEST_FIELDS` whitelisting.** Anything Cursor sends that
   isn't in the allowlist is dropped silently. If a future Cursor release
   starts sending a new field DeepSeek understands, it will be silently
   swallowed. Consider logging dropped fields in verbose mode, or replacing
   the allowlist with a known-bad denylist.
7. **Multi-part `content` arrays are flattened.** Works because DeepSeek
   text endpoints are not multimodal. If DeepSeek ever ships a vision
   endpoint Cursor targets through this proxy, this transform would need
   to learn about image parts.

---

## 8. Recommendations

In rough priority order:

1. **(Finding 1)** Treat omitted `thinking` as enabled for repair purposes
   in pass-through mode unless the request explicitly disables thinking.
   Keep pass-through from injecting a field, but don't let it disable
   reasoning repair. Add a regression test for `ProxyConfig(thinking=
"pass-through")` with an omitted `thinking` field and missing
   tool-call `reasoning_content`.
2. **(Finding 2)** Add the older
   `"[deepseek-cursor-proxy] Recovered ..."` prefix to recovery-notice
   recognition in `transform.py`. Add a regression test that feeds the
   request-14-style transcript from the captured trace into
   `prepare_upstream_request` and asserts
   `continued_recovery_boundary == True`.
3. **(Finding 3)** Strip proxy recovery notices from assistant `content`
   before forwarding upstream. Keep the notice Cursor-visible only, or
   express recovery state as a synthetic `system` message that lives only
   in the reduced recovery request. Add tests for both current and legacy
   notice prefixes round-tripping through Cursor.
4. **(Finding 4)** Decide and document precedence between request-level
   `thinking` and proxy config. If the proxy is intended as a general
   compatibility layer, explicit request values should probably win.
5. **(Finding 6)** Clarify in the README that the
   `<details><summary>Thinking</summary>` mirroring is streaming-only, or
   add equivalent non-streaming display mirroring.
6. **(Finding 7)** Either change `assistant_needs_reasoning_for_tool_context`
   to stop only at `user` messages, or document that `system` is
   intentionally a turn boundary and add a regression test.
7. **(Operational)** Add optional validation or diagnostics for mismatched
   `tool_call_id` in tool results.
8. **(Operational)** Document the 409-vs-400 difference in `README.md` so
   debugging users know the proxy emits its own 409 in strict mode and
   that the canonical 400 only appears if the proxy is bypassed.
9. **(Operational)** Log dropped request fields in verbose mode so future
   Cursor protocol changes don't silently regress.
10. **(Test surface)** Fold the four-turn canonical loop from
    `scripts/audit_protocol_compliance.py` into `tests/`.

---

## 9. Reproducing this audit

```bash
cd /Users/m1/repo/deepseek-cursor-proxy

# Existing unit and integration tests
uv run python -m unittest discover -s tests
# Expected: Ran 93 tests ... OK (skipped=1)

# Strict-fake protocol harness (Claude)
uv run python scripts/audit_protocol_compliance.py
# Expected: OK: 7 of 7 cases passed

# Protocol-checker harness (Codex)
uv run python scripts/audit_deepseek_protocol.py
# Expected: SUMMARY passed=4 failed=0 observations=1
```

Both audit harnesses leave no state on disk — they use `:memory:` SQLite
stores and ephemeral local servers. The trace replay analysis (Finding 2)
requires the captured trace at
`trace-dumps/20260429T134418.628275Z-pid80943` to be present.

---

## 10. Bottom Line

**For default Cursor usage, the proxy is protocol-compatible** — every
existing test passes, both audit harnesses pass their default scenarios,
the canonical four-turn loop works end-to-end against a strict fake
DeepSeek, and a real 21-request trace shows zero missing-reasoning or
tool-result-ID failures. The main fixes worth shipping are:

- **Finding 1:** make pass-through mode honour DeepSeek's documented
  default-enabled thinking, or document that pass-through callers must
  send the field explicitly to keep reasoning repair active.
- **Finding 2:** widen recovery-notice recognition so older transcripts
  don't repeatedly re-recover and lose their active tool-call chain.
- **Finding 3:** stop replaying proxy recovery-notice prose upstream as
  if DeepSeek had written it.

Findings 4–7 are policy / quality-of-life improvements rather than
protocol corrections. No code change is required to make the canonical
default Cursor flow correct against today's DeepSeek API.

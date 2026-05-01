# Fork analysis: `mipselqq/deepseek-cursor-proxy`

A user maintains a fork of this project at <https://github.com/mipselqq/deepseek-cursor-proxy> with a few patches on top of our `main`. This doc groups those patches into independent **changes** (ignoring how they were split into commits in the fork), describes the real-world problem each one tries to solve, evaluates the trade-offs, and gives a recommendation on whether we should bring it into our `main`.

The analysis assumes our `main` is at PR #33 (the audit refactor).

One change from the fork — collapsible `<details>` blocks for the thinking display — is already on our `main` as PR #32, so it does not need analysis. The remaining changes are below.

---

## Change 1 — Namespace-Independent (NI) cache keys

**Status:** not on our `main`. **However, the underlying problem is largely already solved by PR #28**, which is on our `main`. See below.

### What is the problem they are trying to solve?

The proxy stores DeepSeek's `reasoning_content` in a SQLite cache, keyed by something called a **cache namespace**. The namespace is a hash of:

- The upstream base URL
- The model family
- The thinking mode (`enabled` vs `disabled`)
- The reasoning effort
- A hash of the API key

This namespace is intentional — it isolates one user's reasoning content from another user's, and one model's reasoning from another model's.

But it has a side effect. Imagine the user has a long conversation going with `deepseek-v4-pro` in Cursor's **Agent** mode. They've sent many tool-call turns; the proxy has cached the `reasoning_content` for each assistant message. All of those cache entries are stored under namespace `A`.

Now the user switches Cursor to **Ask** mode, or switches the model to `deepseek-v4-flash`. If the namespace changes to `B`, none of the cache entries under namespace `A` are visible.

The proxy then sees assistant messages in the conversation history that have tool-calls but no `reasoning_content`, can't find any matching cache entry under the new namespace, marks them as "missing", and falls back to the `latest_user` recovery strategy — which strips out almost all the conversation history down to the latest user message and prefixes the response with `[deepseek-cursor-proxy] Refreshed reasoning_content history.`. To the user this looks like the proxy ate their conversation.

### How does the fork fix it?

It introduces a second set of cache keys that have **no namespace prefix**:

```
ni:signature:{message_signature}
ni:tool_call:{tool_call_id}
ni:tool_call_signature:{tool_call_signature}
```

These are stored *alongside* the existing scope/portable keys. Whenever the proxy stores reasoning, it also stores it under these "namespace-independent" keys. Whenever it looks up reasoning, after trying the scope-based and namespace-portable keys, it falls back to looking under these NI keys. Since the NI key is derived purely from the message content (not from the namespace), it survives any model/mode/thinking switch.

### What our `main` already does (PR #28)

I verified the following by reading the code, not just the PR description.

1. **Family normalization** — `reasoning_model_family()` in `transform.py:670` maps both `deepseek-v4-pro` and `deepseek-v4-flash` to the family `deepseek-v4`. The namespace (`reasoning_cache_namespace()` in `transform.py:676`) hashes the *family*, not the model, so switching between Pro and Flash is invisible to the cache. **Caveat:** the family list is a hardcoded set `{deepseek-v4-pro, deepseek-v4-flash}`. A future `deepseek-v5-pro` will diverge silently until we update this function — that is a maintenance hazard worth tracking.

2. **Portable turn-scoped keys** — `portable_reasoning_keys()` in `reasoning_store.py:131` builds keys of the form `namespace:{ns}:turn:{turn_signature}:signature:{...}`. The `turn_signature` (`turn_context_signature()` in `reasoning_store.py:94`) hashes only messages from the latest user turn onward, **explicitly skipping system messages**. So even when Cursor's Agent↔Plan mode swap changes the system prompt or the tool surface, past assistant messages keep the same turn signature and are still findable.

3. **Strict-hit backfill** — `transform.py:303` calls `store.backfill_portable_aliases()` whenever a scope-only key hits. Existing scope-keyed entries become mode-stable on the way through, without requiring a fresh write.

4. **Recovery-boundary handling** — `active_messages_from_recovery_boundary()` retires the prefix when a recovery notice is detected; the `continued_recovery_boundary` flag stops the recovery loop from cascading on every later request.

PR #28's validation trace: 21 requests across Agent↔Plan and Pro↔Flash, only **one** recovery triggered (when the user routed through `composer-2` in the middle and came back), and **no cascade** afterward. Screenshots in the PR show the exact traces.

So in practice, on our current `main`:

- Pro↔Flash switching: no namespace change, cache hits cleanly. No NI keys needed.
- Agent↔Plan switching (within same model/mode): portable keys catch it. No NI keys needed.
- Going through a non-DeepSeek model and back: triggers one recovery, then continues cleanly. PR #28 documents this as "expected".

**Action item — restore PR #28's regression tests.** PR #28 originally shipped five regression tests in `tests/test_transform.py`:

- `test_deepseek_pro_and_flash_share_reasoning_namespace`
- `test_strict_hit_backfills_portable_cache_for_mode_switch`
- `test_portable_turn_cache_restores_final_assistant_after_tool_result`
- `test_portable_turn_cache_isolated_for_reused_tool_call_id`
- `test_recovered_response_is_recorded_under_pre_recovery_scope`

All five were dropped by PR #33's test refactor (they were in `test_transform.py`, which was trimmed from 1489 → 321 lines). None were migrated to `test_protocol.py`. The fix code is intact, but the regression coverage is gone — anyone refactoring `reasoning_store.py` or `transform.py`'s recovery path could re-break this without a test failing.

**Plan:** restore the originals from git history (commit `5f14da3`, the merge of PR #28) and re-home them in `test_protocol.py` as a `CrossModeAndModelTests` class. The original tests already operate against the in-process store and `prepare_upstream_request` API, so they should drop in with minimal adjustment for the post-PR-#33 imports and helper layout. Concretely:

```bash
# Recover the five tests verbatim from PR #28's commit.
git show 5f14da3:tests/test_transform.py > /tmp/pr28_test_transform.py
# Then port the relevant test methods into a new class inside
# tests/test_protocol.py and adjust imports.
```

This should land before — or alongside — any further work on the cross-mode/model code path, so we don't rely on PR #28's mechanisms long-term without test coverage.

### What NI keys would still buy us

NI keys would close gaps that PR #28 does *not* cover:

- Switching `thinking` (`enabled`↔`disabled`)
- Switching `reasoning_effort` (e.g. `high`↔`max`)
- Switching `base_url`
- Eliminating that one "expected" recovery when bouncing through a non-DeepSeek model

These are all real, but they are also rare and arguably *intentional* user actions (the user is saying "I want a different mode now"). It is not obviously wrong for the proxy to treat those as a fresh boundary.

### Trade-off

The fork's NI key drops not just the model and mode but also the **API key hash**. That last omission is meaningful:

- On a single-user local proxy: harmless. There is only one user.
- On a shared proxy: the cache becomes visible across users. If user X and user Y happen to send an assistant message with identical content + tool calls, user Y's lookup finds user X's reasoning. The fork dismisses this as "harmless because same content means same reasoning" — that is true for the *content* but loses *tenant isolation*.

If we want NI keys, we should at minimum scope them by `auth_hash`:

```
ni:auth:{auth_hash}:signature:{message_signature}
ni:auth:{auth_hash}:tool_call:{tool_call_id}
ni:auth:{auth_hash}:tool_call_signature:{tool_call_signature}
```

### Recommendation

**Likely skip.** The motivating problem (cross-mode/model context loss) is already addressed by PR #28 for the common cases. NI keys would only help with rare config switches (`thinking`, `reasoning_effort`, `base_url`), and even there the user-visible damage is one recovery notice — not the cascading context loss the fork's CHANGELOG describes. The marginal benefit is small, the complexity is non-trivial (two more key namespaces, more lookups per request, more storage), and the fork's specific implementation has a tenant-isolation hole.

If we ever do adopt NI keys (e.g. user reports show `thinking`-mode switches eating context in real workflows), use the auth-scoped variant above so we don't lose isolation.

---

## Change 2 — 409 strategy guard

**Status:** not on our `main`.

### What is the problem?

In `server.py:164`, when the proxy detects assistant messages with missing reasoning, it does this:

```python
if prepared.missing_reasoning_messages:
    LOG.warning("strict missing-reasoning mode rejected request ...")
    self._send_json(409, {"error": {...}})
    return
```

Three things to notice:

1. The `if` gate has no strategy check.
2. The `LOG.warning` claims strict mode is the reason regardless of actual mode.
3. The 409 error body recommends switching to `--missing-reasoning-strategy recover` — i.e. the message itself assumes this should only fire in `reject` mode.

The *intent* is clearly "only fire when the user has opted into `reject`". But the gate fires whenever `missing_reasoning_messages > 0`, regardless of strategy.

### Is the bug actually reachable?

I traced through the recovery loop (`transform.py:817-839`) and `recover_messages_from_missing_reasoning()` (`transform.py:554-641`). The loop only runs when strategy is `"recover"`. It calls the recovery function, then breaks early if `not dropped_messages`. The recovery function has three return paths and **two of them can return `dropped_messages = 0`**:

- **No user message in the conversation:** `last_user_index = -1` (line 612) → returns `(messages, 0, None, ...)`. The loop hits `if not dropped_messages: break` and exits with `missing_indexes` still populated.
- **Recovery boundary at index 0** (a recovery notice was at the very start, with no real content before it): `omitted_messages = recovery_boundary_index - len(leading_messages) - kept_context_messages = 0` → same break.

When either fires, `missing_reasoning_messages` stays non-zero, the recovery-loop exits without calling `normalize_messages` again, and the unconditional 409 in `server.py:164` triggers — even though the user is in `recover` mode.

Concrete cases that can hit this:

- Cursor sends `[system, assistant_with_tool_calls, tool]` with no user message — possible for `/summarize` or auto-generated traffic.
- A continuation where the only user-visible content above is a recovery notice the proxy itself emitted earlier, with no preceding user turn before that notice.

Neither is common in ordinary chat flow, but both *can* occur — especially around Cursor's auto-summary endpoints.

### The fork's fix

Add the strategy check explicitly:

```python
if (
    prepared.missing_reasoning_messages
    and self.config.missing_reasoning_strategy == "reject"
):
    self._send_json(409, ...)
    return
```

### What changes after the fix?

In `recover` mode with leftover `missing_indexes`:

- The proxy stops 409ing.
- It forwards the request to DeepSeek as-is.
- DeepSeek will probably 400 with "the reasoning_content in the thinking mode must be passed back".
- The proxy relays that 400 to Cursor.

That is consistent with the contract we offer in `recover` mode: we try, and if DeepSeek refuses, we relay the refusal. We do not pre-empt with a synthetic 409.

### Recommendation

**Take it as-is.** One-line correctness fix. The current code is inconsistent with the documented `recover` contract, the bug is reachable in edge-case Cursor traffic (verified by tracing the recovery loop), and the fix has no downside.

We should also add a `test_protocol.py` test that constructs one of the no-user-message cases above in `recover` mode and asserts the proxy *does not* 409 (it should forward to upstream and propagate whatever DeepSeek returns). This locks the contract in.

---

## Change 3 — Passthrough non-DeepSeek models

**Status:** not on our `main`.

### The problem

Cursor lets you set a custom URL for the OpenAI-compatible endpoint. Once you do, **every** chat-completions request goes through that URL — not just calls to your DeepSeek model. That includes `/summarize` (Cursor's auto-summary), Composer, GPT-4o requests for other features, and so on.

Our proxy currently handles those by silently rewriting the model name to `deepseek-v4-pro` (with a `WARNING` log we just added in PR #33). The result: Cursor's `/summarize` call asks for `gpt-4o-mini` but receives a DeepSeek answer — wrong tokenizer, wrong style, sometimes nonsense.

### The fork's fix

When the requested model does not start with `deepseek-`:

1. Skip our entire pipeline (no `prepare_upstream_request`, no reasoning patching, no response rewriting).
2. Forward the request payload byte-for-byte to a new configurable upstream — `passthrough_url`, defaulting to `https://api.openai.com`.
3. Relay the response (regular or SSE) byte-for-byte back to Cursor.

This is implemented as ~180 lines added to `server.py` (`_proxy_passthrough`, `_relay_passthrough_regular`, `_relay_passthrough_streaming`) plus a new config field.

### Why this fix is broken

The proxy forwards **Cursor's `Authorization` header** to the passthrough URL. But Cursor's bearer token is the user's **DeepSeek** API key (the user typed it into Cursor's "API key" field). Sending a DeepSeek key to OpenAI's `/v1/chat/completions` produces a 401 instantly — OpenAI does not know what to do with it.

So the passthrough only "works" if the user happened to configure their OpenAI key in Cursor's API-key field. But then their DeepSeek calls would fail, because DeepSeek does not know what to do with an OpenAI key. There is no setup where both work simultaneously, and the fork does not add a separate `passthrough_api_key` config to disentangle them.

In other words: the fork ships ~180 lines of code that 401 in production for almost everyone.

A complete passthrough would also need:

- A separate `passthrough_api_key` config field (and a way to thread it through without leaking it to DeepSeek).
- Probably a way to map model names (`gpt-4o-mini` from Cursor → whatever the upstream actually calls it).
- Documentation around tenancy, cost, and which features Cursor will then route through us.

That is real work, real surface area, and real risk.

### Recommendation

**Don't take this as-is.** A simpler, equivalent UX improvement is to **stop silently rewriting non-DeepSeek models** and instead return a clean 400:

```json
{
  "error": {
    "message": "deepseek-cursor-proxy only serves models starting with `deepseek-`; received `gpt-4o-mini`. Configure that model directly in Cursor's settings instead of routing it through this proxy.",
    "type": "unsupported_model",
    "code": "unsupported_model"
  }
}
```

That is a few lines, no new pipeline, no auth confusion, and tells the user exactly what went wrong and how to fix it. We are honest about being a DeepSeek-only proxy. If passthrough ever becomes a real requirement, we can implement it correctly with an opt-in config and a separate auth path; we shouldn't ship a half-working version in the meantime.

---

## Change 4 — CHANGELOG.md

**Status:** not on our `main`.

The fork adds a `CHANGELOG.md` documenting the two fork-only fixes plus a tour of the project's architecture (request flow, cache key hierarchy, file-by-file responsibilities, why the tests are structured the way they are).

The architecture description is accurate as of the fork point but is now slightly out of date relative to our `main` (e.g. it references `test_proxy_end_to_end.py`, which we removed in PR #33; it doesn't mention the strict fake DeepSeek harness in `test_protocol.py`).

### Recommendation

If we want a `CHANGELOG.md` we should write our own from our own commit history. Theirs is fine as inspiration but not worth importing wholesale.

---

## Summary

| Change | Verdict | Why |
|---|---|---|
| **1. NI cache keys** | **Likely skip** | PR #28 already solves the common cases (Pro↔Flash family normalization, Agent↔Plan portable keys, recovery-boundary handling). NI keys would only help rare switches (`thinking`, `reasoning_effort`, `base_url`); marginal benefit, real complexity, and the fork's version leaks reasoning across tenants. |
| **2. 409 strategy guard** | **Take** | One-line correctness fix; current code 409s in `recover` mode in edge cases despite the user's explicit opt-in. |
| **3. Non-DeepSeek passthrough** | **Don't take** | The auth model is wrong (forwards DeepSeek key to OpenAI → 401). Better fix: return a clear 400 instead of silently rewriting. |
| **4. CHANGELOG** | Optional | Write our own if we want one; theirs is partly out of date relative to our current `main`. |

If you agree with these, the natural next step is one PR off `main` doing:

1. The 409 strategy guard (one-line `server.py` change + a `test_protocol.py` test for `recover` mode with a non-recoverable history).
2. A clean 400 response for non-`deepseek-*` models in `server.py`, replacing today's silent rewrite + warning log.

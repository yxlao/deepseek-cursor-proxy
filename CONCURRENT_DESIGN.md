# Concurrency Design & Performance Guide

This document explains how the proxy handles concurrent requests, what went wrong
under multi-agent Cursor load, what the `fix/subagent-resilience` branch changed,
and what the performance optimisations added on top of that branch do.

---

## 1. What the proxy does

Cursor treats this proxy as a standard OpenAI-compatible API endpoint.  The
proxy's unique job is to solve a **DeepSeek-specific API requirement**: when you
send a conversation history that contains an assistant turn with tool calls,
DeepSeek's API requires that turn to also include the original `reasoning_content`
(chain-of-thought tokens) from when it was first generated.  Cursor strips that
field before echoing the history back, so the proxy must:

1. **Cache** every `reasoning_content` it receives from DeepSeek into SQLite.
2. **Restore** cached reasoning into outgoing messages before forwarding to DeepSeek.

Everything else — CORS handling, model name rewriting, ngrok tunnelling, recovery
notices — is scaffolding around that core loop.

### Per-request flow

```
Cursor IDE
  │
  │  POST /v1/chat/completions  (OpenAI-compatible)
  ▼
DeepSeekProxyHandler.do_POST()
  │
  ├─ parse & validate request body
  │
  ├─ prepare_upstream_request()
  │    ├─ normalize messages (strip unknown fields, fix content types)
  │    └─ for each assistant message with tool_calls:
  │         └─ ReasoningStore.get() ← look up cached reasoning_content
  │
  ├─ forward to DeepSeek API (urlopen, streaming or non-streaming)
  │
  ├─ [streaming path]
  │    ├─ for each SSE chunk:
  │    │    ├─ rewrite model name, mirror reasoning into content display
  │    │    └─ StreamAccumulator.store_ready_reasoning()
  │    │         └─ ReasoningStore.put() ← cache reasoning when tool_calls complete
  │    └─ ReasoningStore.put() ← cache on [DONE] or client disconnect
  │
  └─ [non-streaming path]
       └─ rewrite_response_body() → ReasoningStore.put() ← cache the response
```

---

## 2. Concurrency model

The server is a Python `ThreadingHTTPServer`: each incoming connection spawns a
new OS thread.  When Cursor runs multiple agents or subagents simultaneously,
several threads are active at once — each reading from the SQLite cache before
forwarding its request upstream, and each writing back after receiving a response.

### The SQLite layer

SQLite in its default (journal) mode allows only one writer at a time and blocks
readers during writes.  This made the original code prone to lock contention under
parallel requests.

---

## 3. Bug history

### Bug 1 — Startup crash cascade (pre-branch)

**Symptom:** Multiple "database is locked" tracebacks in `autostart.log`, followed
by repeated "Address already in use" errors.  Agents froze because the proxy
crashed in a loop and never became available.

**Root cause:**

1. Every new Cursor terminal (including agent tool-call shells) sources `~/.bashrc`,
   which runs the proxy autostart script.
2. The script uses `flock` + PID check to be idempotent, but has a startup window.
3. When several agents opened simultaneously, multiple proxy processes started.
4. Each called `ReasoningStore.__init__()` → `self.prune()`.
5. Without `WAL` mode or `busy_timeout`, the second process immediately received
   `sqlite3.OperationalError: database is locked` and crashed.
6. The crash triggered another autostart.  Loop.
7. Meanwhile the first healthy process held the port, so restarted processes hit
   `OSError: [Errno 98] Address already in use` and crashed again.

**Fix (in this branch):**

- `PRAGMA journal_mode = WAL` — readers never block writers; writers never block
  readers at the SQLite file level.
- `PRAGMA busy_timeout = 5000` — wait up to 5 s for a lock instead of crashing.
- `_startup_prune()` — retries + gracefully skips if still locked after retries.
- `_proxy_already_running()` health check in `main()` — if the proxy is already
  responding on `/healthz`, a new startup attempt exits `0` immediately without
  touching the port or the database.

### Bug 2 — Multi-second freezes under concurrent agent load (pre- and post-branch)

**Symptom:** Log lines showing 10+ seconds between "┌ cursor request" and
"├ context filled", followed by Cursor dropping the connection with a broken-pipe
warning.

**Root cause:**

The `ReasoningStore` used a single `threading.RLock` that serialised **all**
database operations — reads and writes — across all handler threads.  The
`put()` method called `_prune_locked()` on **every single write**:

```python
# OLD code — runs a full-table sort on every write
def _prune_locked(self):
    self._conn.execute("""
        DELETE FROM reasoning_cache
        WHERE key NOT IN (
            SELECT key FROM reasoning_cache
            ORDER BY created_at DESC   -- ← full-table sort, no index
            LIMIT 100000
        )
    """)
```

With 55,925 rows and no index on `created_at`, this was an O(N log N) sort on
every write.  While one thread held the lock for this sort:

- Every other thread's `store.get()` call (which normalises 15–30 assistant
  messages per request) queued behind the lock.
- Each blocked `get()` added to the "context filled" latency visible in the logs.

**Fix (this PR):** — see Section 4.

---

## 4. Performance optimisations

### 4a — Thread-local read connections

**Before:** One shared `sqlite3.Connection`, one `threading.RLock`.  Reads blocked
on writes and vice versa.

**After:** Two connection layers:

| Layer | Object | Protected by | Used for |
|---|---|---|---|
| Write | `self._write_conn` | `self._write_lock` (plain `Lock`) | INSERT / UPDATE / DELETE / COMMIT |
| Read | `self._local.conn` (per-thread) | Nothing — WAL handles it | SELECT |

In WAL mode SQLite guarantees that concurrent readers never see a partial write and
writers never wait for readers.  Each request handler thread opens its own read
connection the first time it calls `get()`.  Reads across multiple concurrent
requests never contend with each other or with the writer.

The special case `":memory:"` (used in tests) cannot share data across multiple
connections, so it falls back to the write connection for reads.

### 4b — `created_at` index

```sql
CREATE INDEX IF NOT EXISTS idx_reasoning_cache_created_at
ON reasoning_cache(created_at)
```

Applied on first startup against existing databases (`CREATE INDEX IF NOT EXISTS`
is idempotent).  This index turns:

| Query | Before | After |
|---|---|---|
| Age prune (`WHERE created_at < ?`) | O(N) full scan | O(log N) range seek |
| Row-count prune (top-N keep) | O(N log N) sort | O(max\_rows) index walk |

### 4c — Throttled row-count prune

The age-based prune (`DELETE WHERE created_at < cutoff`) is now fast with the
index and still runs on every write.

The row-count prune (keeping at most `max_rows` entries) only runs when the
approximate row count exceeds `max_rows` **or** every `_ROWCOUNT_PRUNE_INTERVAL`
(default 50) writes.  The row count is seeded from the real DB size at startup so
the threshold is accurate from the first write.

When the DB is well under the limit (the common case — 55k rows vs 100k limit),
the row-count DELETE skips entirely, saving a subquery evaluation on every request.

### 4d — SQLite PRAGMA tuning

| PRAGMA | Value | Effect |
|---|---|---|
| `journal_mode` | `WAL` | Concurrent readers + writer |
| `busy_timeout` | `5000` ms | Wait instead of crash on contention |
| `synchronous` | `NORMAL` | Skip extra fsync; safe under WAL |
| `cache_size` (write conn) | `-32768` (32 MiB) | Hot rows stay in RAM |
| `cache_size` (read conns) | `-8192` (8 MiB) | Per-thread read cache |

---

## 5. Autostart design change

The old `autostart.bash` automatically launched the proxy inside every new bash
session (including Cursor agent tool-call shells) using `flock` to prevent
duplicates.  The flock race window was the original source of the cascade.

**New behaviour:** The script is a **status check only**.

- Interactive shell, proxy running → silent (no noise on every tab open).
- Interactive shell, proxy **not** running → one clear warning + start command.
- Non-interactive shell (agent tool calls) → does nothing at all.

The proxy is started manually once.  Agents just talk to it; they never try to
start it.

---

## 6. Reasoning cache lifetime

Entries are pruned by two independent policies (both configurable):

| Policy | Default | Purpose |
|---|---|---|
| `reasoning_cache_max_age_seconds` | 30 days | Remove old conversation history |
| `reasoning_cache_max_rows` | 100,000 | Cap total DB size |

**Active conversations are never pruned** as long as entries are younger than
`max_age_seconds`.  If an entry is pruned (old conversation, restarted machine,
cache cleared), the proxy falls back to "recover" mode: it truncates the
conversation to the latest user message, adds a recovery notice, and continues.

On Cursor restart, reasoning_content for recent tool calls is restored from the
SQLite cache transparently, so the conversation resumes correctly.

---

## 7. Summary of changes in `fix/subagent-resilience`

| File | Change | Fixes |
|---|---|---|
| `reasoning_store.py` | WAL mode + busy\_timeout | Startup crash cascade |
| `reasoning_store.py` | `_startup_prune()` with retry | Startup crash cascade |
| `reasoning_store.py` | `created_at` index | Multi-second read latency |
| `reasoning_store.py` | Thread-local read connections | Read/write contention |
| `reasoning_store.py` | Throttled row-count prune | Per-write lock contention |
| `reasoning_store.py` | 32 MiB write + 8 MiB read caches | I/O reduction |
| `server.py` | `_proxy_already_running()` health check | "Address already in use" cascade |
| `server.py` | `LOG.info` instead of `LOG.warning` for broken pipes | Log noise |
| `server.py` | `client_disconnected` flag | Unnecessary DB writes on subagent churn |
| `autostart.bash` | Status check only, no auto-start | Cascade crash root cause |

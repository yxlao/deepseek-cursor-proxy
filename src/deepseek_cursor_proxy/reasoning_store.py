from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": tool_call.get("id"),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": function.get("name") or "",
            "arguments": arguments,
        },
    }
    return normalized


def tool_call_signature(tool_call: dict[str, Any]) -> str:
    normalized = normalize_tool_call(tool_call)
    normalized.pop("id", None)
    canonical = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.append(str(tool_call["id"]))
    return ids


def tool_call_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names


def message_signature(message: dict[str, Any]) -> str:
    tool_calls = [
        normalize_tool_call(tool_call)
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    ]
    payload = {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_scope_message(message: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {"role": message.get("role")}
    for key in ("content", "name", "tool_call_id", "prefix"):
        if key in message:
            canonical[key] = message[key]
    if message.get("tool_calls"):
        canonical["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in message.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
    return canonical


def conversation_scope(messages: list[dict[str, Any]], namespace: str = "") -> str:
    scope_messages = [canonical_scope_message(message) for message in messages]
    payload: Any = scope_messages
    if namespace:
        payload = {"namespace": namespace, "messages": scope_messages}
    return _sha256_json(payload)


def turn_context_signature(prior_messages: list[dict[str, Any]]) -> str:
    last_user_index = next(
        (
            index
            for index in range(len(prior_messages) - 1, -1, -1)
            if prior_messages[index].get("role") == "user"
        ),
        -1,
    )
    start_index = 0
    if last_user_index != -1:
        start_index = last_user_index
        while start_index > 0 and prior_messages[start_index - 1].get("role") == "user":
            start_index -= 1

    context_messages = [
        canonical_scope_message(message)
        for message in prior_messages[start_index:]
        if message.get("role") != "system"
    ]
    return _sha256_json(context_messages)


def scoped_reasoning_keys(message: dict[str, Any], scope: str) -> list[str]:
    keys = [f"scope:{scope}:signature:{message_signature(message)}"]
    keys.extend(
        f"scope:{scope}:tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"scope:{scope}:tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    # Recovery-of-last-resort key. Catches the case where a streaming response
    # was interrupted (user pressed Stop) before the tool_call.id chunk arrived,
    # so neither tool_call_id nor tool_call_signature (which canonicalizes
    # arguments) survives the round-trip through Cursor's transcript.
    keys.extend(
        f"scope:{scope}:tool_name:{tool_name}" for tool_name in tool_call_names(message)
    )
    return keys


def portable_reasoning_keys(
    message: dict[str, Any],
    cache_namespace: str,
    prior_messages: list[dict[str, Any]],
) -> list[str]:
    if not cache_namespace:
        return []

    turn_signature = turn_context_signature(prior_messages)
    keys = [
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"signature:{message_signature(message)}"
    ]
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"tool_call:{tool_call_id}"
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:"
        f"tool_call_signature:{tool_call_signature(tool_call)}"
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
    keys.extend(
        f"namespace:{cache_namespace}:turn:{turn_signature}:" f"tool_name:{tool_name}"
        for tool_name in tool_call_names(message)
    )
    return keys


_STARTUP_PRUNE_BUSY_RETRIES = 3
_STARTUP_PRUNE_BUSY_WAIT = 0.5  # seconds between retries

# Prune the max-rows limit at most once every N writes.  Running the full-table
# DELETE on every write was the primary cause of multi-second lock contention
# under concurrent subagent load.  Age-based pruning is cheap (index range
# scan) and still happens on every write; row-count pruning is expensive
# (table scan) and only needs to run occasionally.
_ROWCOUNT_PRUNE_INTERVAL = 50


class ReasoningStore:
    """Thread-safe reasoning_content cache backed by SQLite.

    Concurrency model
    -----------------
    The proxy runs as a ThreadingHTTPServer: each request is handled on its
    own thread.  To allow concurrent reads without blocking on writes we use
    two connection layers:

    * **Write connection** (``self._write_conn``) — one shared connection,
      serialised by ``self._write_lock`` (a plain ``threading.Lock``).  All
      INSERT/UPDATE/DELETE/COMMIT operations go through this path.

    * **Thread-local read connections** (``self._local.conn``) — each handler
      thread opens its own SQLite connection the first time it calls ``get()``.
      In WAL mode SQLite guarantees that readers never block writers and
      writers never block readers at the *file* level; the Python-level
      per-thread connections ensure there is no intra-process contention
      either.

    The special case ``":memory:"`` (used in tests) cannot have multiple
    connections pointing at the same data, so it falls back to the write
    connection for reads (still serialised by ``_write_lock``).
    """

    def __init__(
        self,
        reasoning_content_path: str | Path,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.max_age_seconds = max_age_seconds
        self.max_rows = max_rows
        self._in_memory = str(reasoning_content_path) == ":memory:"
        if self._in_memory:
            self.reasoning_content_path: str | Path = ":memory:"
        else:
            self.reasoning_content_path = Path(reasoning_content_path).expanduser()
            self.reasoning_content_path.parent.mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
        self._write_lock = threading.Lock()
        self._write_conn = self._open_conn(writer=True)
        if isinstance(self.reasoning_content_path, Path):
            self.reasoning_content_path.chmod(0o600)
        self._write_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                key TEXT PRIMARY KEY,
                reasoning TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # Index on created_at powers both the age-based DELETE (range scan) and
        # the max-rows DELETE (top-N scan), turning O(N log N) full-table sorts
        # into O(log N) index seeks.  CREATE INDEX IF NOT EXISTS is a no-op on
        # databases that already have the index, so it is safe to run every
        # startup and will transparently add the index to existing DBs.
        self._write_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reasoning_cache_created_at
            ON reasoning_cache(created_at)
            """
        )
        self._write_conn.commit()
        if not self._in_memory:
            self._local: threading.local = threading.local()
        # Seed the write counter with the current row count so the rowcount
        # prune triggers correctly: once _write_count exceeds max_rows we know
        # the DB could be over capacity and prune on every write; below that
        # threshold we only prune every _ROWCOUNT_PRUNE_INTERVAL writes.
        row = self._write_conn.execute(
            "SELECT COUNT(*) FROM reasoning_cache"
        ).fetchone()
        self._write_count: int = int(row[0]) if row else 0
        self._startup_prune()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _open_conn(self, *, writer: bool = False) -> sqlite3.Connection:
        """Open a new SQLite connection with appropriate PRAGMAs."""
        conn = sqlite3.connect(
            str(self.reasoning_content_path), check_same_thread=False
        )
        # busy_timeout: wait up to 5 s for a concurrent writer instead of
        # immediately raising OperationalError("database is locked").
        conn.execute("PRAGMA busy_timeout = 5000")
        # WAL mode: readers never block writers; writers never block readers.
        conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL durability is safe under WAL (no data loss on OS crash) and
        # avoids the extra fsync that FULL mode adds on every commit.
        conn.execute("PRAGMA synchronous = NORMAL")
        if writer:
            # Large page cache for the write connection.  The negative value
            # is in kibibytes; 524288 = 512 MiB, enough to hold the entire
            # 300 MB database in Python heap and absorb large write bursts.
            conn.execute("PRAGMA cache_size = -524288")  # 512 MiB
        else:
            # Read connections each get their own cache, but mmap_size (below)
            # means most reads never touch this cache at all.
            conn.execute("PRAGMA cache_size = -65536")   # 64 MiB per thread
        # Memory-map the database file directly into the process address space.
        # With mmap enabled the OS page cache becomes the working set — all
        # connections share the same physical pages, reads are zero-copy, and
        # the entire 300 MB DB can stay warm in RAM indefinitely as long as
        # there is free memory.  2 GiB ceiling is far above the current DB
        # size; SQLite only maps what actually exists.
        conn.execute("PRAGMA mmap_size = 2147483648")    # 2 GiB ceiling
        return conn

    def _read_conn(self) -> sqlite3.Connection:
        """Return the thread-local read connection, creating it if needed."""
        if self._in_memory:
            # All connections to :memory: are independent DBs; share the write
            # connection (serialised by _write_lock in the caller).
            return self._write_conn
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open_conn(writer=False)
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _startup_prune(self) -> None:
        """Prune on startup with retry + graceful degradation.

        Another proxy instance starting simultaneously can hold a write lock
        for a short window.  busy_timeout handles most of this, but if the
        lock persists longer than busy_timeout we catch the error and skip
        the prune rather than crashing — stale rows left behind are harmless
        and will be cleaned up on the next successful prune.
        """
        for attempt in range(_STARTUP_PRUNE_BUSY_RETRIES):
            try:
                self.prune()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                if attempt < _STARTUP_PRUNE_BUSY_RETRIES - 1:
                    time.sleep(_STARTUP_PRUNE_BUSY_WAIT)
        import logging as _logging

        _logging.getLogger("deepseek_cursor_proxy").warning(
            "reasoning store: skipped startup prune (database is locked)"
        )

    def close(self) -> None:
        with self._write_lock:
            self._write_conn.close()
        # Thread-local read connections are closed when their threads exit or
        # when the process terminates; the OS reclaims file handles either way.

    # ------------------------------------------------------------------
    # Core read / write
    # ------------------------------------------------------------------

    def put(self, key: str, reasoning: str, message: dict[str, Any]) -> None:
        if not isinstance(reasoning, str):
            return
        message_json = json.dumps(message, ensure_ascii=False, sort_keys=True)
        with self._write_lock:
            self._write_conn.execute(
                """
                INSERT INTO reasoning_cache(key, reasoning, message_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    reasoning = excluded.reasoning,
                    message_json = excluded.message_json,
                    created_at = excluded.created_at
                """,
                (key, reasoning, message_json, time.time()),
            )
            self._write_count += 1
            # Skip the expensive rowcount prune when we're well under
            # capacity, running it at most every _ROWCOUNT_PRUNE_INTERVAL
            # writes.  Once _write_count exceeds max_rows the DB could be
            # over capacity, so prune on every write until we're safely
            # back under the limit.
            at_or_over_limit = (
                self.max_rows is not None
                and self._write_count > self.max_rows
            )
            skip_rowcount = (
                not at_or_over_limit
                and self._write_count % _ROWCOUNT_PRUNE_INTERVAL != 0
            )
            deleted = self._prune_write_locked(skip_rowcount=skip_rowcount)
            self._write_count -= deleted
            self._write_conn.commit()

    def get(self, key: str) -> str | None:
        # File-based DB: no lock needed — WAL allows concurrent reads even
        # while the write connection is mid-transaction.  Each thread has its
        # own connection so there is no intra-process sharing either.
        # :memory: DB: falls back to the write connection (see _read_conn).
        if self._in_memory:
            with self._write_lock:
                row = self._read_conn().execute(
                    "SELECT reasoning FROM reasoning_cache WHERE key = ?",
                    (key,),
                ).fetchone()
        else:
            row = self._read_conn().execute(
                "SELECT reasoning FROM reasoning_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def store_assistant_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> int:
        if message.get("role") != "assistant":
            return 0
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            return 0

        keys = scoped_reasoning_keys(message, scope)
        if prior_messages is not None:
            keys.extend(
                portable_reasoning_keys(message, cache_namespace, prior_messages)
            )
        keys = list(dict.fromkeys(keys))
        for key in keys:
            self.put(key, reasoning, message)
        return len(keys)

    def lookup_for_message(
        self,
        message: dict[str, Any],
        scope: str,
        cache_namespace: str = "",
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> str | None:
        keys = scoped_reasoning_keys(message, scope)
        if prior_messages is not None:
            keys.extend(
                portable_reasoning_keys(message, cache_namespace, prior_messages)
            )
        for key in keys:
            reasoning = self.get(key)
            if reasoning is not None:
                return reasoning
        return None

    def backfill_portable_aliases(
        self,
        message: dict[str, Any],
        reasoning: str,
        cache_namespace: str,
        prior_messages: list[dict[str, Any]],
    ) -> int:
        if not isinstance(reasoning, str):
            return 0
        keys = portable_reasoning_keys(message, cache_namespace, prior_messages)
        if not keys:
            return 0
        message_with_reasoning = dict(message)
        message_with_reasoning["reasoning_content"] = reasoning
        for key in dict.fromkeys(keys):
            self.put(key, reasoning, message_with_reasoning)
        return len(keys)

    def clear(self) -> int:
        with self._write_lock:
            row = self._write_conn.execute(
                "SELECT COUNT(*) FROM reasoning_cache"
            ).fetchone()
            count = int(row[0] if row else 0)
            self._write_conn.execute("DELETE FROM reasoning_cache")
            self._write_conn.commit()
        return count

    def prune(self) -> int:
        with self._write_lock:
            deleted = self._prune_write_locked()
            self._write_conn.commit()
        return deleted

    def _prune_write_locked(self, *, skip_rowcount: bool = False) -> int:
        """Run pruning queries.  Must be called with ``_write_lock`` held."""
        deleted = 0
        if self.max_age_seconds is not None and self.max_age_seconds > 0:
            cutoff = time.time() - self.max_age_seconds
            # idx_reasoning_cache_created_at makes this a fast index range scan.
            cursor = self._write_conn.execute(
                "DELETE FROM reasoning_cache WHERE created_at < ?",
                (cutoff,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        if not skip_rowcount and self.max_rows is not None and self.max_rows > 0:
            # Keep the top max_rows rows by created_at; delete everything else.
            # The subquery walks the created_at index and materialises exactly
            # max_rows rowids — O(max_rows) index seek.  The outer DELETE is
            # O(rows deleted).  Using rowid NOT IN handles equal-timestamp ties
            # correctly (rowid is unique; created_at is not).
            cursor = self._write_conn.execute(
                """
                DELETE FROM reasoning_cache
                WHERE rowid NOT IN (
                    SELECT rowid FROM reasoning_cache
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (self.max_rows,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0
        return deleted

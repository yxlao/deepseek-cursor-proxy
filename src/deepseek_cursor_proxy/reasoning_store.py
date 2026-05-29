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


def compute_conversation_scopes(
    messages: list[dict[str, Any]], namespace: str = ""
) -> list[str]:
    """Pre-compute conversation scopes for every prefix position in O(n) time.

    Returns a list where ``scopes[i]`` equals
    ``conversation_scope(messages[:i], namespace)``.

    Uses incremental SHA-256 hashing so the entire message list is serialized
    only once, avoiding the O(n²) cost of calling ``conversation_scope``
    repeatedly inside a message-normalization loop.
    """
    scopes: list[str] = []
    hasher = hashlib.sha256()

    if namespace:
        ns_json = json.dumps(
            namespace, ensure_ascii=False, separators=(",", ":")
        )
        # sort_keys=True → {"messages":[...],"namespace":"ns"}
        prefix = b'{"messages":['
        suffix = b'],"namespace":' + ns_json.encode("utf-8") + b"}"
    else:
        prefix = b"["
        suffix = b"]"

    hasher.update(prefix)

    # Scope for empty prefix (index 0, before any messages)
    clone = hasher.copy()
    clone.update(suffix)
    scopes.append(clone.hexdigest())

    first = True
    for message in messages:
        if not first:
            hasher.update(b",")
        first = False

        canonical = canonical_scope_message(message)
        msg_json = json.dumps(
            canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        hasher.update(msg_json.encode("utf-8"))

        clone = hasher.copy()
        clone.update(suffix)
        scopes.append(clone.hexdigest())

    return scopes


def compute_turn_signatures(
    messages: list[dict[str, Any]],
) -> list[str]:
    """Pre-compute turn_context_signature for every prefix position in O(n) time.

    Returns a list where ``result[i]`` equals
    ``turn_context_signature(messages[:i])``.

    Uses incremental SHA-256 hashing.  The turn context resets at user-message
    boundaries, so the total work is proportional to the number of messages
    rather than O(n²).
    """
    signatures: list[str] = []
    hasher = hashlib.sha256()
    hasher.update(b"[")
    first_in_turn = True

    for i in range(len(messages) + 1):
        # Save signature for prefix of size i
        clone = hasher.copy()
        clone.update(b"]")
        signatures.append(clone.hexdigest())

        if i >= len(messages):
            break

        msg = messages[i]
        if msg.get("role") == "user":
            # New turn: find the start of the consecutive-user group
            turn_start = i
            while (
                turn_start > 0
                and messages[turn_start - 1].get("role") == "user"
            ):
                turn_start -= 1
            # Rebuild the hash from turn_start through i (one-time cost
            # per turn, which is rare compared to the total message count).
            hasher = hashlib.sha256()
            hasher.update(b"[")
            first_in_turn = True
            for j in range(turn_start, i + 1):
                if messages[j].get("role") == "system":
                    continue
                if not first_in_turn:
                    hasher.update(b",")
                first_in_turn = False
                canonical = canonical_scope_message(messages[j])
                msg_json = json.dumps(
                    canonical,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                hasher.update(msg_json.encode("utf-8"))
        elif msg.get("role") != "system":
            if not first_in_turn:
                hasher.update(b",")
            first_in_turn = False
            canonical = canonical_scope_message(msg)
            msg_json = json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            hasher.update(msg_json.encode("utf-8"))

    return signatures


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
    turn_signatures: list[str] | None = None,
) -> list[str]:
    if not cache_namespace:
        return []

    turn_signature = (
        turn_signatures[len(prior_messages)]
        if turn_signatures is not None
        else turn_context_signature(prior_messages)
    )
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


class ReasoningStore:
    def __init__(
        self,
        reasoning_content_path: str | Path,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.max_age_seconds = max_age_seconds
        self.max_rows = max_rows
        if str(reasoning_content_path) == ":memory:":
            self.reasoning_content_path: str | Path = ":memory:"
        else:
            self.reasoning_content_path = Path(reasoning_content_path).expanduser()
            self.reasoning_content_path.parent.mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.reasoning_content_path, check_same_thread=False
        )
        if isinstance(self.reasoning_content_path, Path):
            self.reasoning_content_path.chmod(0o600)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                key TEXT PRIMARY KEY,
                reasoning TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reasoning_cache_created_at "
            "ON reasoning_cache(created_at)"
        )
        self._conn.commit()
        self.prune()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def put(self, key: str, reasoning: str, message: dict[str, Any]) -> None:
        if not isinstance(reasoning, str):
            return
        message_json = json.dumps(message, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._conn.execute(
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
            self._prune_locked()
            self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
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
        turn_signatures: list[str] | None = None,
    ) -> int:
        if not isinstance(reasoning, str):
            return 0
        keys = portable_reasoning_keys(
            message, cache_namespace, prior_messages,
            turn_signatures=turn_signatures,
        )
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            return 0
        message_with_reasoning = dict(message)
        message_with_reasoning["reasoning_content"] = reasoning
        message_json = json.dumps(
            message_with_reasoning, ensure_ascii=False, sort_keys=True
        )
        now = time.time()
        with self._lock:
            for key in unique_keys:
                self._conn.execute(
                    """
                    INSERT INTO reasoning_cache(key, reasoning, message_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        reasoning = excluded.reasoning,
                        message_json = excluded.message_json,
                        created_at = excluded.created_at
                    """,
                    (key, reasoning, message_json, now),
                )
            self._prune_locked()
            self._conn.commit()
        return len(unique_keys)

    def clear(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM reasoning_cache").fetchone()
            count = int(row[0] if row else 0)
            self._conn.execute("DELETE FROM reasoning_cache")
            self._conn.commit()
        return count

    def prune(self) -> int:
        with self._lock:
            deleted = self._prune_locked()
            self._conn.commit()
        return deleted

    def _prune_locked(self) -> int:
        deleted = 0
        if self.max_age_seconds is not None and self.max_age_seconds > 0:
            cutoff = time.time() - self.max_age_seconds
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reasoning_cache WHERE created_at < ?",
                (cutoff,),
            ).fetchone()
            if row and int(row[0]) > 0:
                cursor = self._conn.execute(
                    "DELETE FROM reasoning_cache WHERE created_at < ?",
                    (cutoff,),
                )
                deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        if self.max_rows is not None and self.max_rows > 0:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reasoning_cache"
            ).fetchone()
            count = int(row[0] if row else 0)
            if count > self.max_rows:
                cursor = self._conn.execute(
                    """
                    DELETE FROM reasoning_cache
                    WHERE key NOT IN (
                        SELECT key
                        FROM reasoning_cache
                        ORDER BY created_at DESC
                        LIMIT ?
                    )
                    """,
                    (self.max_rows,),
                )
                deleted += cursor.rowcount if cursor.rowcount != -1 else 0
        return deleted

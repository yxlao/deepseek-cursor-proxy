"""Keep-alive connections to the upstream chat completions API.

`urllib` opens a new connection for every request, so each completion paid a
TCP handshake plus a TLS handshake before the upstream saw a byte. From a
European host that is about 52 ms to the DeepSeek API, on every turn. The API
speaks HTTP/1.1 and keeps connections open, so the same socket can serve the
next request.

Two rules make reuse safe:

- A connection goes back to the pool only after its response body was read to
  EOF. Closing a half-read response leaves unread bytes in the socket, and the
  next request would read them as its own answer. The wrapper tracks EOF rather
  than guessing from the connection state.
- A pooled socket may have been closed by the server while it sat idle. That
  failure happens before the request reaches the server, so one retry on a
  fresh connection is safe. A fresh connection that fails is reported as is.
"""

from __future__ import annotations

import http.client
import io
import ssl
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, urljoin, urlsplit

from .logging import LOG

DEFAULT_MAX_IDLE_PER_HOST = 4
DEFAULT_MAX_IDLE_SECONDS = 60.0
MAX_REDIRECTS = 3
# 303 is missing on purpose: it turns a POST into a GET, which a chat
# completion must never do silently.
REDIRECT_STATUSES = {301, 302, 307, 308}


class PooledResponse:
    """Upstream response that returns its connection once the body is drained.

    Callers use it like the `urlopen` response it replaces: read it whole,
    iterate SSE lines, or use it as a context manager. The difference is that
    closing it hands the socket back instead of discarding it.
    """

    def __init__(
        self,
        response: http.client.HTTPResponse,
        release: Callable[[bool], None],
    ) -> None:
        self._response = response
        self._release = release
        self._complete = False
        self._released = False

    def __getattr__(self, name: str) -> Any:
        # Guard against recursing when the instance is not built yet.
        if name == "_response":
            raise AttributeError(name)
        return getattr(self._response, name)

    def __iter__(self) -> Any:
        while True:
            line = self.readline()
            if not line:
                return
            yield line

    def __enter__(self) -> "PooledResponse":
        self._response.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        try:
            return self._response.__exit__(exc_type, exc, traceback)
        finally:
            self._finish()

    def read(self, amt: int | None = None) -> bytes:
        data = self._response.read(amt)
        if not data or amt is None:
            # No length means read to EOF; empty means EOF was reached.
            self._complete = True
        return data

    def readline(self, limit: int = -1) -> bytes:
        line = self._response.readline(limit)
        if not line:
            self._complete = True
        return line

    def readlines(self, hint: int = -1) -> list[bytes]:
        return list(self)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._released:
            return
        self._released = True
        self._release(self._complete)

    def __del__(self) -> None:
        # The handler always closes the response; this only covers a caller
        # that raises before entering the context manager.
        try:
            if not self._released:
                self._response.close()
                self._finish()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


class UpstreamConnectionPool:
    """Thread-safe pool of idle keep-alive connections, keyed by origin."""

    def __init__(
        self,
        max_idle_per_host: int = DEFAULT_MAX_IDLE_PER_HOST,
        max_idle_seconds: float = DEFAULT_MAX_IDLE_SECONDS,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.max_idle_per_host = max(0, max_idle_per_host)
        self.max_idle_seconds = max(0.0, max_idle_seconds)
        self._ssl_context = ssl_context or ssl.create_default_context()
        self._lock = threading.Lock()
        self._idle: dict[tuple[str, str, int], list[tuple[Any, float]]] = {}

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> PooledResponse:
        """Send one request, raising `HTTPError`/`URLError` like `urlopen`."""
        request_headers = dict(headers or {})
        parts = urlsplit(url)
        redirects = 0
        while True:
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise URLError(f"unsupported upstream URL: {url!r}")
            key, connection, response = self._send(
                method, parts, body, request_headers, timeout
            )
            pooled = PooledResponse(
                response, self._release_callback(key, connection, response)
            )
            if response.status in REDIRECT_STATUSES:
                location = response.headers.get("Location")
                self._drain(pooled)
                if not location:
                    raise URLError("upstream redirected without a Location header")
                if redirects >= MAX_REDIRECTS:
                    raise URLError(f"too many upstream redirects for {url!r}")
                redirects += 1
                url = urljoin(url, location)
                parts = urlsplit(url)
                continue
            if response.status >= 400:
                raise self._http_error(url, pooled)
            return pooled

    def _send(
        self,
        method: str,
        parts: SplitResult,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float | None,
    ) -> tuple[tuple[str, str, int], http.client.HTTPConnection, Any]:
        """Send once, retrying a stale pooled socket on a fresh connection."""
        target = self._target(parts)
        for attempt in (0, 1):
            key, connection, reused = self._checkout(parts, timeout)
            try:
                connection.request(method, target, body=body, headers=headers)
                return key, connection, connection.getresponse()
            except (http.client.HTTPException, OSError) as exc:
                self._close(connection)
                if reused and attempt == 0:
                    LOG.warning(
                        "upstream connection was stale, retrying on a new one: %s",
                        exc,
                    )
                    continue
                raise URLError(exc) from exc

        # Only reachable if the retry loop is ever shortened.
        raise URLError("upstream connection could not be established")

    def _drain(self, pooled: PooledResponse) -> None:
        """Read a small body to EOF so the socket can serve the next request."""
        try:
            pooled.read()
        except (http.client.HTTPException, OSError):
            pass
        finally:
            pooled.close()

    def close(self) -> None:
        """Close every idle connection; in-flight ones are left alone."""
        with self._lock:
            idle = [entry for entries in self._idle.values() for entry in entries]
            self._idle.clear()
        for connection, _ in idle:
            self._close(connection)

    def idle_count(self) -> int:
        with self._lock:
            return sum(len(entries) for entries in self._idle.values())

    @staticmethod
    def _target(parts: SplitResult) -> str:
        target = parts.path or "/"
        if parts.query:
            return f"{target}?{parts.query}"
        return target

    @staticmethod
    def _origin(parts: SplitResult) -> tuple[str, str, int]:
        scheme = parts.scheme
        port = parts.port or (443 if scheme == "https" else 80)
        return (scheme, parts.hostname or "", port)

    def _connect(
        self, scheme: str, host: str, port: int, timeout: float | None
    ) -> http.client.HTTPConnection:
        if scheme == "https":
            return http.client.HTTPSConnection(
                host, port, timeout=timeout, context=self._ssl_context
            )
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def _checkout(
        self, parts: SplitResult, timeout: float | None
    ) -> tuple[tuple[str, str, int], http.client.HTTPConnection, bool]:
        key = self._origin(parts)
        oldest_usable = time.monotonic() - self.max_idle_seconds
        with self._lock:
            entries = self._idle.get(key)
            while entries:
                connection, idle_since = entries.pop()
                if idle_since >= oldest_usable and connection.sock is not None:
                    if timeout is not None:
                        connection.sock.settimeout(timeout)
                    connection.timeout = timeout
                    return key, connection, True
                self._close(connection)
        connection = self._connect(key[0], key[1], key[2], timeout)
        connection.timeout = timeout
        return key, connection, False

    def _release_callback(
        self,
        key: tuple[str, str, int],
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
    ) -> Callable[[bool], None]:
        def release(complete: bool) -> None:
            if (
                complete
                and self.max_idle_per_host > 0
                and not response.will_close
                and connection.sock is not None
            ):
                with self._lock:
                    entries = self._idle.setdefault(key, [])
                    if len(entries) < self.max_idle_per_host:
                        entries.append((connection, time.monotonic()))
                        return
            self._close(connection)

        return release

    @staticmethod
    def _http_error(url: str, pooled: PooledResponse) -> HTTPError:
        """Turn an error response into the `HTTPError` urllib would raise."""
        try:
            body = pooled.read()
        except (http.client.HTTPException, OSError):
            body = b""
        finally:
            pooled.close()
        error = HTTPError(
            url, pooled.status, pooled.reason, pooled.headers, io.BytesIO(body)
        )
        return error

    @staticmethod
    def _close(connection: http.client.HTTPConnection) -> None:
        try:
            connection.close()
        except OSError:  # pragma: no cover - close is best effort
            pass

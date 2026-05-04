from __future__ import annotations

import logging as stdlib_logging
import sys
import threading
from typing import Any


LOG = stdlib_logging.getLogger("deepseek_cursor_proxy")

DEFAULT_INFO_LOG_FORMAT = "%(message)s"
DEFAULT_WARNING_LOG_FORMAT = "%(levelname)s %(message)s"
VERBOSE_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


class ConsoleLogFormatter(stdlib_logging.Formatter):
    def __init__(self, *, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self._verbose_formatter = stdlib_logging.Formatter(VERBOSE_LOG_FORMAT)
        self._info_formatter = stdlib_logging.Formatter(DEFAULT_INFO_LOG_FORMAT)
        self._warning_formatter = stdlib_logging.Formatter(DEFAULT_WARNING_LOG_FORMAT)

    def format(self, record: stdlib_logging.LogRecord) -> str:
        if self.verbose:
            return self._verbose_formatter.format(record)
        if record.levelno <= stdlib_logging.INFO:
            return self._info_formatter.format(record)
        return self._warning_formatter.format(record)


def configure_logging(*, verbose: bool) -> None:
    handler = stdlib_logging.StreamHandler()
    handler.setFormatter(ConsoleLogFormatter(verbose=verbose))
    stdlib_logging.basicConfig(
        level=stdlib_logging.INFO,
        handlers=[handler],
        force=True,
    )


class TerminalSpinner:
    hide_cursor = "\x1b[?25l"
    show_cursor = "\x1b[?25h"
    frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(
        self,
        *,
        enabled: bool,
        text: str,
        stream: Any | None = None,
        interval: float = 0.12,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled and bool(getattr(self.stream, "isatty", lambda: False)())
        self.text = text
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._visible = False

    def start(self) -> "TerminalSpinner":
        if not self.enabled or self._thread is not None:
            return self
        self.stream.write(self.hide_cursor)
        self.stream.flush()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._visible:
            self.stream.write("\r" + (" " * self._clear_width()) + "\r")
            self.stream.flush()
            self._visible = False
        self.stream.write(self.show_cursor)
        self.stream.flush()

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            self.stream.write("\r" + self.text.format(frame=self.frames[index]))
            self.stream.flush()
            self._visible = True
            index = (index + 1) % len(self.frames)
            self._stop.wait(self.interval)

    def _clear_width(self) -> int:
        return max(len(self.text.format(frame=frame)) for frame in self.frames)

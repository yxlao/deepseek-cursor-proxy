#!/usr/bin/env python
"""Silent launcher for DeepSeek Cursor Proxy (single instance)."""
import datetime
import os
import socket
import subprocess
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "startup.log")
BAT_PATH = os.path.join(SCRIPT_DIR, "start-deepseek-proxy.bat")
PROXY_PORT = 9000

CREATE_NO_WINDOW = 0x08000000


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> None:
    log("=== Silent launcher started ===")
    log(f"Script dir: {SCRIPT_DIR}")
    log(f"Batch path: {BAT_PATH}")
    log(f"Python: {sys.executable}")

    if port_is_open(PROXY_PORT):
        log(f"SKIP: port {PROXY_PORT} already in use (proxy already running)")
        return

    if not os.path.isfile(BAT_PATH):
        log("ERROR: start-deepseek-proxy.bat not found!")
        return

    try:
        proc = subprocess.Popen(
            ["cmd", "/c", BAT_PATH],
            cwd=SCRIPT_DIR,
            creationflags=CREATE_NO_WINDOW,
        )
        log(f"Process spawned: PID={proc.pid}")
    except Exception:
        log(f"ERROR spawning process:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()

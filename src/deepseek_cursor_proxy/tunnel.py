from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import shutil
import subprocess
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


LOG = logging.getLogger("deepseek_cursor_proxy")


DEFAULT_NGROK_API_URL = "http://127.0.0.1:4040/api"


def local_tunnel_target(host: str, port: int) -> str:
    local_host = host.strip() or "127.0.0.1"
    if local_host in {"0.0.0.0", "::"}:
        local_host = "127.0.0.1"
    if ":" in local_host and not local_host.startswith("["):
        local_host = f"[{local_host}]"
    return f"http://{local_host}:{port}"


def parse_ngrok_public_url(payload: dict[str, Any]) -> str | None:
    records = payload.get("endpoints")
    if not isinstance(records, list):
        records = payload.get("tunnels")
    if not isinstance(records, list):
        return None

    public_urls = [
        public_url
        for record in records
        if isinstance(record, dict)
        for public_url in (record.get("url"), record.get("public_url"))
        if isinstance(public_url, str)
    ]
    for public_url in public_urls:
        if public_url.startswith("https://"):
            return public_url
    for public_url in public_urls:
        if public_url.startswith("http://"):
            return public_url
    return None


def ngrok_agent_urls(api_url: str) -> list[str]:
    normalized = api_url.rstrip("/")
    if normalized.endswith("/endpoints") or normalized.endswith("/tunnels"):
        return [normalized]
    return [f"{normalized}/endpoints", f"{normalized}/tunnels"]


@dataclass
class NgrokTunnel:
    target_url: str
    command: str = "ngrok"
    api_url: str = DEFAULT_NGROK_API_URL
    startup_timeout: float = 15.0

    process: subprocess.Popen[bytes] | None = None

    def start(self) -> str:
        if shutil.which(self.command) is None:
            raise RuntimeError(
                "ngrok is not installed or is not on PATH. Install it, then run "
                "`ngrok config add-authtoken <token>` once."
            )

        self.process = subprocess.Popen(
            [self.command, "http", self.target_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            return self.wait_for_public_url()
        except Exception:
            self.stop()
            raise

    def wait_for_public_url(self) -> str:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "ngrok did not report a public URL"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("ngrok exited before creating a tunnel")
            for api_url in ngrok_agent_urls(self.api_url):
                try:
                    with urlopen(api_url, timeout=1) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    public_url = parse_ngrok_public_url(payload)
                    if public_url:
                        return public_url
                except (OSError, URLError, json.JSONDecodeError) as exc:
                    last_error = str(exc)
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for ngrok tunnel: {last_error}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        LOG.info("stopping ngrok tunnel")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

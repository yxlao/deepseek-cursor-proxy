from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

import yaml

APP_DIR_NAME = ".deepseek-cursor-proxy"
CONFIG_FILE_NAME = "config.yaml"
REASONING_CONTENT_FILE_NAME = "reasoning_content.sqlite3"

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
MISSING = object()
DEFAULT_CONFIG_TEXT = """# This file was created automatically at ~/.deepseek-cursor-proxy/config.yaml.
# API keys are read from Cursor's Authorization header and forwarded upstream.

base_url: https://api.deepseek.com
model: deepseek-v4-pro
thinking: enabled
reasoning_effort: high
display_reasoning: true

host: 127.0.0.1
port: 9000
ngrok: true
verbose: false
log_bodies: false
request_timeout: 300

reasoning_content_path: reasoning_content.sqlite3
"""


def default_app_dir() -> Path:
    return Path.home() / APP_DIR_NAME


def default_config_path() -> Path:
    return default_app_dir() / CONFIG_FILE_NAME


def default_reasoning_content_path() -> Path:
    return default_app_dir() / REASONING_CONTENT_FILE_NAME


def populate_default_config_file(config_path: Path) -> None:
    config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_path.parent.chmod(0o700)
    config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    config_path.chmod(0o600)


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        return {}

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config at {config_path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return dict(loaded)


def resolve_config_path(
    env: Mapping[str, str] | None, config_path: str | Path | None
) -> Path:
    live_env = os.environ if env is None else env
    return Path(
        config_path
        or live_env.get("DEEPSEEK_CURSOR_PROXY_CONFIG_PATH")
        or default_config_path()
    ).expanduser()


def setting_value(
    settings: Mapping[str, Any],
    env: Mapping[str, str],
    key: str,
    env_name: str,
) -> Any:
    if env_name in env:
        return env[env_name]
    return settings.get(key, MISSING)


def as_str(value: Any, default: str) -> str:
    if value is MISSING or value is None:
        return default
    return str(value)


def as_bool(value: Any, default: bool) -> bool:
    if value is MISSING or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def as_int(value: Any, default: int) -> int:
    if value is MISSING or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    if value is MISSING or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_path(value: Any, default_path: Path, relative_base: Path) -> Path:
    if value is MISSING or value is None or value == "":
        return default_path
    candidate_path = Path(str(value)).expanduser()
    if candidate_path.is_absolute():
        return candidate_path
    return relative_base / candidate_path


def settings_and_env(
    env: Mapping[str, str] | None, config_path: str | Path | None
) -> tuple[dict[str, Any], dict[str, str], Path]:
    live_env = dict(os.environ if env is None else env)
    config_path = resolve_config_path(live_env, config_path)
    if (
        config_path == default_config_path()
        and "DEEPSEEK_CURSOR_PROXY_CONFIG_PATH" not in live_env
        and not config_path.exists()
    ):
        populate_default_config_file(config_path)
    return load_config_file(config_path), live_env, config_path


@dataclass(frozen=True)
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    upstream_base_url: str = "https://api.deepseek.com"
    upstream_model: str = "deepseek-v4-pro"
    allow_model_passthrough: bool = False
    thinking: str = "enabled"
    reasoning_effort: str = "high"
    request_timeout: float = 300.0
    reasoning_content_path: Path = field(default_factory=default_reasoning_content_path)
    cursor_display_reasoning: bool = True
    verbose: bool = False
    log_bodies: bool = False
    ngrok: bool = False

    @classmethod
    def from_file(
        cls: type[ProxyConfig],
        env: Mapping[str, str] | None = None,
        config_path: str | Path | None = None,
    ) -> "ProxyConfig":
        settings, live_env, resolved_config_path = settings_and_env(env, config_path)
        config_dir = resolved_config_path.parent

        thinking = (
            as_str(
                setting_value(
                    settings,
                    live_env,
                    "thinking",
                    "DEEPSEEK_THINKING",
                ),
                "enabled",
            )
            .strip()
            .lower()
        )
        if thinking in {"passthrough", "pass-through", "pass_through"}:
            thinking = "pass-through"
        if thinking not in {"enabled", "disabled", "pass-through"}:
            thinking = "enabled"

        return cls(
            host=as_str(
                setting_value(
                    settings,
                    live_env,
                    "host",
                    "PROXY_HOST",
                ),
                "127.0.0.1",
            ),
            port=as_int(
                setting_value(
                    settings,
                    live_env,
                    "port",
                    "PROXY_PORT",
                ),
                9000,
            ),
            upstream_base_url=as_str(
                setting_value(
                    settings,
                    live_env,
                    "base_url",
                    "DEEPSEEK_BASE_URL",
                ),
                "https://api.deepseek.com",
            ).rstrip("/"),
            upstream_model=as_str(
                setting_value(
                    settings,
                    live_env,
                    "model",
                    "DEEPSEEK_MODEL",
                ),
                "deepseek-v4-pro",
            ),
            allow_model_passthrough=as_bool(
                setting_value(
                    settings,
                    live_env,
                    "allow_model_passthrough",
                    "DEEPSEEK_ALLOW_MODEL_PASSTHROUGH",
                ),
                False,
            ),
            thinking=thinking,
            reasoning_effort=as_str(
                setting_value(
                    settings,
                    live_env,
                    "reasoning_effort",
                    "DEEPSEEK_REASONING_EFFORT",
                ),
                "high",
            ),
            request_timeout=as_float(
                setting_value(
                    settings,
                    live_env,
                    "request_timeout",
                    "PROXY_REQUEST_TIMEOUT",
                ),
                300.0,
            ),
            reasoning_content_path=as_path(
                setting_value(
                    settings,
                    live_env,
                    "reasoning_content_path",
                    "REASONING_CONTENT_PATH",
                ),
                default_reasoning_content_path(),
                config_dir,
            ),
            cursor_display_reasoning=as_bool(
                setting_value(
                    settings,
                    live_env,
                    "display_reasoning",
                    "CURSOR_DISPLAY_REASONING",
                ),
                True,
            ),
            verbose=as_bool(
                setting_value(
                    settings,
                    live_env,
                    "verbose",
                    "PROXY_VERBOSE",
                ),
                False,
            ),
            log_bodies=as_bool(
                setting_value(
                    settings,
                    live_env,
                    "log_bodies",
                    "PROXY_LOG_BODIES",
                ),
                False,
            ),
            ngrok=as_bool(
                setting_value(
                    settings,
                    live_env,
                    "ngrok",
                    "PROXY_NGROK",
                ),
                False,
            ),
        )

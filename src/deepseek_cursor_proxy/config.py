from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

APP_DIR_NAME = ".deepseek-cursor-proxy"
CONFIG_FILE_NAME = "config.yaml"
REASONING_CONTENT_FILE_NAME = "reasoning_content.sqlite3"

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
MISSING = object()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_UPSTREAM_BASE_URL = "https://api.deepseek.com"
DEFAULT_UPSTREAM_MODEL = "deepseek-v4-pro"
DEFAULT_THINKING = "enabled"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_CURSOR_DISPLAY_REASONING = True
DEFAULT_NGROK = True
DEFAULT_VERBOSE = False
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_MAX_REQUEST_BODY_BYTES = 20 * 1024 * 1024
DEFAULT_CORS = False
DEFAULT_MISSING_REASONING_STRATEGY = "recover"
DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
DEFAULT_REASONING_CACHE_MAX_ROWS = 100_000

DEFAULT_CONFIG_HEADER = (
    "# This file was created automatically at ~/.deepseek-cursor-proxy/config.yaml."
)
DEFAULT_CONFIG_TEXT = f"""{DEFAULT_CONFIG_HEADER}
# API keys are read from Cursor's Authorization header and forwarded upstream.

# `model` is the fallback when a request has no model; Cursor's requested
# DeepSeek model name is otherwise respected.
base_url: {DEFAULT_UPSTREAM_BASE_URL}
model: {DEFAULT_UPSTREAM_MODEL}
thinking: {DEFAULT_THINKING}
reasoning_effort: {DEFAULT_REASONING_EFFORT}
display_reasoning: {str(DEFAULT_CURSOR_DISPLAY_REASONING).lower()}

host: {DEFAULT_HOST}
port: {DEFAULT_PORT}
ngrok: {str(DEFAULT_NGROK).lower()}
verbose: {str(DEFAULT_VERBOSE).lower()}
request_timeout: {DEFAULT_REQUEST_TIMEOUT:g}
max_request_body_bytes: {DEFAULT_MAX_REQUEST_BODY_BYTES}
cors: {str(DEFAULT_CORS).lower()}

reasoning_content_path: {REASONING_CONTENT_FILE_NAME}
missing_reasoning_strategy: {DEFAULT_MISSING_REASONING_STRATEGY}
reasoning_cache_max_age_seconds: {DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS}
reasoning_cache_max_rows: {DEFAULT_REASONING_CACHE_MAX_ROWS}
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


def resolve_config_path(config_path: str | Path | None) -> Path:
    return Path(config_path or default_config_path()).expanduser()


def setting_value(settings: Mapping[str, Any], key: str) -> Any:
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


def settings_from_config(
    config_path: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    resolved_config_path = resolve_config_path(config_path)
    if config_path is None and not resolved_config_path.exists():
        populate_default_config_file(resolved_config_path)
    return load_config_file(resolved_config_path), resolved_config_path


def normalize_thinking(value: Any) -> str:
    thinking = as_str(value, DEFAULT_THINKING).strip().lower()
    if thinking in {"passthrough", "pass-through", "pass_through"}:
        return "pass-through"
    if thinking in {"enabled", "disabled"}:
        return thinking
    return DEFAULT_THINKING


def normalize_missing_reasoning_strategy(value: Any) -> str:
    strategy = as_str(value, DEFAULT_MISSING_REASONING_STRATEGY).strip().lower()
    if strategy in {"recover", "reject"}:
        return strategy
    return DEFAULT_MISSING_REASONING_STRATEGY


@dataclass(frozen=True)
class ProxyConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL
    upstream_model: str = DEFAULT_UPSTREAM_MODEL
    thinking: str = DEFAULT_THINKING
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES
    reasoning_content_path: Path = field(default_factory=default_reasoning_content_path)
    missing_reasoning_strategy: str = DEFAULT_MISSING_REASONING_STRATEGY
    reasoning_cache_max_age_seconds: int = DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS
    reasoning_cache_max_rows: int = DEFAULT_REASONING_CACHE_MAX_ROWS
    cursor_display_reasoning: bool = DEFAULT_CURSOR_DISPLAY_REASONING
    cors: bool = DEFAULT_CORS
    verbose: bool = DEFAULT_VERBOSE
    ngrok: bool = DEFAULT_NGROK
    trace_dir: Path | None = None

    @classmethod
    def from_file(
        cls: type[ProxyConfig],
        config_path: str | Path | None = None,
    ) -> "ProxyConfig":
        settings, resolved_config_path = settings_from_config(config_path)
        config_dir = resolved_config_path.parent

        return cls(
            host=as_str(
                setting_value(settings, "host"),
                DEFAULT_HOST,
            ),
            port=as_int(
                setting_value(settings, "port"),
                DEFAULT_PORT,
            ),
            upstream_base_url=as_str(
                setting_value(settings, "base_url"),
                DEFAULT_UPSTREAM_BASE_URL,
            ).rstrip("/"),
            upstream_model=as_str(
                setting_value(settings, "model"),
                DEFAULT_UPSTREAM_MODEL,
            ),
            thinking=normalize_thinking(setting_value(settings, "thinking")),
            reasoning_effort=as_str(
                setting_value(settings, "reasoning_effort"),
                DEFAULT_REASONING_EFFORT,
            ),
            request_timeout=as_float(
                setting_value(settings, "request_timeout"),
                DEFAULT_REQUEST_TIMEOUT,
            ),
            max_request_body_bytes=as_int(
                setting_value(settings, "max_request_body_bytes"),
                DEFAULT_MAX_REQUEST_BODY_BYTES,
            ),
            reasoning_content_path=as_path(
                setting_value(settings, "reasoning_content_path"),
                default_reasoning_content_path(),
                config_dir,
            ),
            missing_reasoning_strategy=normalize_missing_reasoning_strategy(
                setting_value(settings, "missing_reasoning_strategy")
            ),
            reasoning_cache_max_age_seconds=as_int(
                setting_value(settings, "reasoning_cache_max_age_seconds"),
                DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS,
            ),
            reasoning_cache_max_rows=as_int(
                setting_value(settings, "reasoning_cache_max_rows"),
                DEFAULT_REASONING_CACHE_MAX_ROWS,
            ),
            cursor_display_reasoning=as_bool(
                setting_value(settings, "display_reasoning"),
                DEFAULT_CURSOR_DISPLAY_REASONING,
            ),
            cors=as_bool(
                setting_value(settings, "cors"),
                DEFAULT_CORS,
            ),
            verbose=as_bool(
                setting_value(settings, "verbose"),
                DEFAULT_VERBOSE,
            ),
            ngrok=as_bool(
                setting_value(settings, "ngrok"),
                DEFAULT_NGROK,
            ),
        )

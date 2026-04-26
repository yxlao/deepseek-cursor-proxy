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
DEFAULT_CONFIG_HEADER = (
    "# This file was created automatically at ~/.deepseek-cursor-proxy/config.yaml."
)
DEFAULT_CONFIG_TEXT = f"""{DEFAULT_CONFIG_HEADER}
# API keys are read from Cursor's Authorization header and forwarded upstream.

# `model` is the fallback when a request has no model; Cursor's requested
# DeepSeek model name is otherwise respected.
base_url: https://api.deepseek.com
model: deepseek-v4-pro
thinking: enabled
reasoning_effort: high
display_reasoning: true

host: 127.0.0.1
port: 9000
ngrok: true
verbose: false
request_timeout: 300
max_request_body_bytes: 20971520
cors: false

reasoning_content_path: reasoning_content.sqlite3
missing_reasoning_strategy: recover
reasoning_cache_max_age_seconds: 604800
reasoning_cache_max_rows: 10000
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
    thinking = as_str(value, "enabled").strip().lower()
    if thinking in {"passthrough", "pass-through", "pass_through"}:
        return "pass-through"
    if thinking in {"enabled", "disabled"}:
        return thinking
    return "enabled"


def normalize_missing_reasoning_strategy(value: Any) -> str:
    strategy = as_str(value, "recover").strip().lower()
    if strategy in {"recover", "reject"}:
        return strategy
    return "recover"


@dataclass(frozen=True)
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    upstream_base_url: str = "https://api.deepseek.com"
    upstream_model: str = "deepseek-v4-pro"
    thinking: str = "enabled"
    reasoning_effort: str = "high"
    request_timeout: float = 300.0
    max_request_body_bytes: int = 20 * 1024 * 1024
    reasoning_content_path: Path = field(default_factory=default_reasoning_content_path)
    missing_reasoning_strategy: str = "recover"
    reasoning_cache_max_age_seconds: int = 7 * 24 * 60 * 60
    reasoning_cache_max_rows: int = 10000
    cursor_display_reasoning: bool = True
    cors: bool = False
    verbose: bool = False
    ngrok: bool = False

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
                "127.0.0.1",
            ),
            port=as_int(
                setting_value(settings, "port"),
                9000,
            ),
            upstream_base_url=as_str(
                setting_value(settings, "base_url"),
                "https://api.deepseek.com",
            ).rstrip("/"),
            upstream_model=as_str(
                setting_value(settings, "model"),
                "deepseek-v4-pro",
            ),
            thinking=normalize_thinking(setting_value(settings, "thinking")),
            reasoning_effort=as_str(
                setting_value(settings, "reasoning_effort"),
                "high",
            ),
            request_timeout=as_float(
                setting_value(settings, "request_timeout"),
                300.0,
            ),
            max_request_body_bytes=as_int(
                setting_value(settings, "max_request_body_bytes"),
                20 * 1024 * 1024,
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
                7 * 24 * 60 * 60,
            ),
            reasoning_cache_max_rows=as_int(
                setting_value(settings, "reasoning_cache_max_rows"),
                10000,
            ),
            cursor_display_reasoning=as_bool(
                setting_value(settings, "display_reasoning"),
                True,
            ),
            cors=as_bool(
                setting_value(settings, "cors"),
                False,
            ),
            verbose=as_bool(
                setting_value(settings, "verbose"),
                False,
            ),
            ngrok=as_bool(
                setting_value(settings, "ngrok"),
                False,
            ),
        )

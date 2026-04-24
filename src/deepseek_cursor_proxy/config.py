from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import os


APP_DIR_NAME = ".deepseek-cursor-proxy"
CONFIG_FILE_NAME = ".env"
REASONING_CONTENT_FILE_NAME = "reasoning_content.sqlite3"

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def default_app_dir() -> Path:
    return Path.home() / APP_DIR_NAME


def default_config_path() -> Path:
    return default_app_dir() / CONFIG_FILE_NAME


def default_reasoning_content_path() -> Path:
    return default_app_dir() / REASONING_CONTENT_FILE_NAME


def load_env_file(env_file_path: str | Path) -> dict[str, str]:
    env_file_path = Path(env_file_path)
    if not env_file_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def merged_env(
    env: Mapping[str, str] | None, env_file_path: str | Path | None
) -> dict[str, str]:
    live_env = dict(os.environ if env is None else env)
    config_path = Path(
        env_file_path
        or live_env.get("DEEPSEEK_CURSOR_PROXY_CONFIG_PATH")
        or default_config_path()
    )
    values = load_env_file(config_path)
    values.update(live_env)
    return values


def env_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    value = values.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def env_int(values: Mapping[str, str], name: str, default: int) -> int:
    value = values.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(values: Mapping[str, str], name: str, default: float) -> float:
    value = values.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_tuple(
    values: Mapping[str, str], name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    value = values.get(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def env_path(
    values: Mapping[str, str],
    names: tuple[str, ...],
    default_path: Path,
) -> Path:
    for env_name in names:
        value = values.get(env_name)
        if value:
            candidate_path = Path(value).expanduser()
            if candidate_path.is_absolute():
                return candidate_path
            return default_path.parent / candidate_path
    return default_path


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
    model_list: tuple[str, ...] = ("deepseek-v4-pro", "deepseek-v4-flash")

    @classmethod
    def from_env(
        cls: type[ProxyConfig],
        env: Mapping[str, str] | None = None,
        env_file_path: str | Path | None = None,
    ) -> "ProxyConfig":
        values = merged_env(env, env_file_path)
        thinking = values.get("DEEPSEEK_THINKING", "enabled").strip().lower()
        if thinking in {"passthrough", "pass-through", "pass_through"}:
            thinking = "pass-through"
        if thinking not in {"enabled", "disabled", "pass-through"}:
            thinking = "enabled"

        return cls(
            host=values.get("PROXY_HOST", "127.0.0.1"),
            port=env_int(values, "PROXY_PORT", 9000),
            upstream_base_url=values.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            upstream_model=values.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            allow_model_passthrough=env_bool(
                values, "DEEPSEEK_ALLOW_MODEL_PASSTHROUGH", False
            ),
            thinking=thinking,
            reasoning_effort=values.get("DEEPSEEK_REASONING_EFFORT", "high"),
            request_timeout=env_float(values, "PROXY_REQUEST_TIMEOUT", 300.0),
            reasoning_content_path=env_path(
                values,
                ("REASONING_CONTENT_PATH",),
                default_reasoning_content_path(),
            ),
            cursor_display_reasoning=env_bool(values, "CURSOR_DISPLAY_REASONING", True),
            verbose=env_bool(values, "PROXY_VERBOSE", False),
            log_bodies=env_bool(values, "PROXY_LOG_BODIES", False),
            ngrok=env_bool(values, "PROXY_NGROK", False),
            model_list=env_tuple(
                values,
                "PROXY_MODELS",
                ("deepseek-v4-pro", "deepseek-v4-flash"),
            ),
        )

from __future__ import annotations

import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deepseek_cursor_proxy.config import (
    DEFAULT_COLLAPSIBLE_REASONING,
    DEFAULT_MISSING_REASONING_STRATEGY,
    DEFAULT_NGROK,
    DEFAULT_PORT,
    DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS,
    DEFAULT_REASONING_CACHE_MAX_ROWS,
    DEFAULT_THINKING,
    DEFAULT_UPSTREAM_MODEL,
    DEFAULT_VERBOSE,
    ProxyConfig,
    default_config_path,
    default_reasoning_content_path,
)


class ConfigTests(unittest.TestCase):
    def test_default_paths_live_in_visible_user_app_directory(self) -> None:
        home = Path("/tmp/home")

        with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
            self.assertEqual(
                default_config_path(), home / ".deepseek-cursor-proxy" / "config.yaml"
            )
            self.assertEqual(
                default_reasoning_content_path(),
                home / ".deepseek-cursor-proxy" / "reasoning_content.sqlite3",
            )
            self.assertEqual(
                ProxyConfig().reasoning_content_path,
                home / ".deepseek-cursor-proxy" / "reasoning_content.sqlite3",
            )
            self.assertEqual(ProxyConfig().ngrok, DEFAULT_NGROK)
            self.assertEqual(
                ProxyConfig().collapsible_reasoning,
                DEFAULT_COLLAPSIBLE_REASONING,
            )
            self.assertIsNone(ProxyConfig().trace_dir)

    def test_missing_default_config_file_is_populated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)

            with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
                config = ProxyConfig.from_file(config_path=None)
                config_path = default_config_path()

            config_text = config_path.read_text(encoding="utf-8")

            self.assertTrue(config_path.exists())
            self.assertIn(f"model: {DEFAULT_UPSTREAM_MODEL}", config_text)
            self.assertIn(
                f"missing_reasoning_strategy: {DEFAULT_MISSING_REASONING_STRATEGY}",
                config_text,
            )
            self.assertIn(
                "reasoning_cache_max_age_seconds: "
                f"{DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS}",
                config_text,
            )
            self.assertIn(
                f"reasoning_cache_max_rows: {DEFAULT_REASONING_CACHE_MAX_ROWS}",
                config_text,
            )
            self.assertIn(f"ngrok: {str(DEFAULT_NGROK).lower()}", config_text)
            self.assertIn(
                "collasible_reasoning: "
                f"{str(DEFAULT_COLLAPSIBLE_REASONING).lower()}",
                config_text,
            )
            self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o600)
            self.assertEqual(config.upstream_model, DEFAULT_UPSTREAM_MODEL)
            self.assertEqual(config.ngrok, DEFAULT_NGROK)
            self.assertEqual(
                config.collapsible_reasoning,
                DEFAULT_COLLAPSIBLE_REASONING,
            )
            self.assertEqual(
                config.missing_reasoning_strategy, DEFAULT_MISSING_REASONING_STRATEGY
            )
            self.assertEqual(
                config.reasoning_cache_max_age_seconds,
                DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS,
            )
            self.assertEqual(
                config.reasoning_cache_max_rows, DEFAULT_REASONING_CACHE_MAX_ROWS
            )

    def test_missing_explicit_config_file_is_not_populated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing.yaml"

            config = ProxyConfig.from_file(config_path=config_path)

            self.assertFalse(config_path.exists())
            self.assertEqual(config.upstream_model, DEFAULT_UPSTREAM_MODEL)
            self.assertEqual(config.ngrok, DEFAULT_NGROK)
            self.assertEqual(
                config.reasoning_cache_max_age_seconds,
                DEFAULT_REASONING_CACHE_MAX_AGE_SECONDS,
            )
            self.assertEqual(
                config.reasoning_cache_max_rows, DEFAULT_REASONING_CACHE_MAX_ROWS
            )

    def test_loads_config_from_user_yaml_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            reasoning_content_path = Path(temp_dir) / "reasoning_content.sqlite3"
            config_path.write_text(
                "\n".join(
                    [
                        "base_url: https://example.com/v1/",
                        "model: deepseek-v4-flash",
                        "thinking: disabled",
                        "reasoning_effort: max",
                        "port: 9100",
                        "host: 0.0.0.0",
                        "ngrok: true",
                        "verbose: true",
                        "request_timeout: 123.5",
                        "max_request_body_bytes: 1234",
                        "cors: true",
                        "display_reasoning: false",
                        "collasible_reasoning: false",
                        f"reasoning_content_path: {reasoning_content_path}",
                        "missing_reasoning_strategy: reject",
                        "reasoning_cache_max_age_seconds: 60",
                        "reasoning_cache_max_rows: 50",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertEqual(config.upstream_base_url, "https://example.com/v1")
        self.assertEqual(config.upstream_model, "deepseek-v4-flash")
        self.assertEqual(config.thinking, "disabled")
        self.assertEqual(config.reasoning_effort, "max")
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9100)
        self.assertTrue(config.ngrok)
        self.assertTrue(config.verbose)
        self.assertEqual(config.request_timeout, 123.5)
        self.assertEqual(config.max_request_body_bytes, 1234)
        self.assertTrue(config.cors)
        self.assertFalse(config.display_reasoning)
        self.assertFalse(config.collapsible_reasoning)
        self.assertEqual(config.reasoning_content_path, reasoning_content_path)
        self.assertEqual(config.missing_reasoning_strategy, "reject")
        self.assertEqual(config.reasoning_cache_max_age_seconds, 60)
        self.assertEqual(config.reasoning_cache_max_rows, 50)

    def test_invalid_config_values_fall_back_to_defaults(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "thinking: maybe",
                        "missing_reasoning_strategy: maybe",
                        "port: nope",
                        "verbose: maybe",
                        "collasible_reasoning: maybe",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertEqual(config.thinking, DEFAULT_THINKING)
        self.assertEqual(
            config.missing_reasoning_strategy, DEFAULT_MISSING_REASONING_STRATEGY
        )
        self.assertEqual(config.port, DEFAULT_PORT)
        self.assertEqual(config.ngrok, DEFAULT_NGROK)
        self.assertEqual(config.verbose, DEFAULT_VERBOSE)
        self.assertEqual(
            config.collapsible_reasoning,
            DEFAULT_COLLAPSIBLE_REASONING,
        )

    def test_relative_reasoning_content_path_in_config_is_relative_to_config_file(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "reasoning_content_path: custom.sqlite3",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertEqual(
            config.reasoning_content_path, Path(temp_dir) / "custom.sqlite3"
        )

    def test_cursor_reasoning_display_can_be_disabled_from_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "display_reasoning: false",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertFalse(config.display_reasoning)

    def test_collapsible_reasoning_can_use_corrected_config_key(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("collapsible_reasoning: false\n", encoding="utf-8")

            config = ProxyConfig.from_file(config_path=config_path)

        self.assertFalse(config.collapsible_reasoning)

    def test_invalid_yaml_config_raises_value_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                ProxyConfig.from_file(config_path=config_path)

    def test_process_environment_does_not_override_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("verbose: false\n", encoding="utf-8")

        with patch.dict(
            "os.environ",
            {
                "PROXY_VERBOSE": "true",
                "DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": "/ignored.yaml",
            },
            clear=True,
        ):
            config = ProxyConfig.from_file(config_path=config_path)
            self.assertEqual(
                dict(os.environ),
                {
                    "PROXY_VERBOSE": "true",
                    "DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": "/ignored.yaml",
                },
            )

        self.assertFalse(config.verbose)


if __name__ == "__main__":
    unittest.main()

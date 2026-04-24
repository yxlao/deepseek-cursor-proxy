from __future__ import annotations

import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deepseek_cursor_proxy.config import (
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

    def test_missing_default_config_file_is_populated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)

            with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
                config = ProxyConfig.from_file(env={}, config_path=None)
                config_path = default_config_path()

            self.assertTrue(config_path.exists())
            self.assertIn(
                "model: deepseek-v4-pro", config_path.read_text(encoding="utf-8")
            )
            self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o600)
            self.assertEqual(config.upstream_model, "deepseek-v4-pro")

    def test_missing_explicit_config_file_is_not_populated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing.yaml"

            config = ProxyConfig.from_file(env={}, config_path=config_path)

            self.assertFalse(config_path.exists())
            self.assertEqual(config.upstream_model, "deepseek-v4-pro")

    def test_loads_config_from_user_yaml_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            reasoning_content_path = Path(temp_dir) / "reasoning_content.sqlite3"
            config_path.write_text(
                "\n".join(
                    [
                        "model: deepseek-v4-flash",
                        "port: 9100",
                        f"reasoning_content_path: {reasoning_content_path}",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(env={}, config_path=config_path)

        self.assertEqual(config.upstream_model, "deepseek-v4-flash")
        self.assertEqual(config.port, 9100)
        self.assertEqual(config.reasoning_content_path, reasoning_content_path)

    def test_environment_overrides_config_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "verbose: false",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_file(
                env={
                    "PROXY_VERBOSE": "true",
                },
                config_path=config_path,
            )

        self.assertTrue(config.verbose)

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

            config = ProxyConfig.from_file(env={}, config_path=config_path)

        self.assertEqual(
            config.reasoning_content_path, Path(temp_dir) / "custom.sqlite3"
        )

    def test_relative_reasoning_content_path_from_env_stays_inside_app_directory(
        self,
    ) -> None:
        home = Path("/tmp/home")

        with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
            config = ProxyConfig.from_file(
                env={
                    "REASONING_CONTENT_PATH": "custom.sqlite3",
                },
                config_path=None,
            )

        self.assertEqual(
            config.reasoning_content_path,
            home / ".deepseek-cursor-proxy" / "custom.sqlite3",
        )

    def test_verbose_and_body_logging_can_be_enabled_from_env(self) -> None:
        config = ProxyConfig.from_file(
            env={
                "PROXY_VERBOSE": "true",
                "PROXY_LOG_BODIES": "1",
                "PROXY_NGROK": "yes",
            },
            config_path=Path("/does/not/exist"),
        )

        self.assertTrue(config.verbose)
        self.assertTrue(config.log_bodies)
        self.assertTrue(config.ngrok)

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

            config = ProxyConfig.from_file(env={}, config_path=config_path)

        self.assertFalse(config.cursor_display_reasoning)

    def test_config_path_can_be_overridden_from_environment(self) -> None:
        with TemporaryDirectory() as temp_dir:
            first_config_path = Path(temp_dir) / "first.yaml"
            second_config_path = Path(temp_dir) / "second.yaml"
            first_config_path.write_text("port: 9100\n", encoding="utf-8")
            second_config_path.write_text("port: 9200\n", encoding="utf-8")

            config = ProxyConfig.from_file(
                env={"DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": str(second_config_path)},
                config_path=None,
            )

        self.assertEqual(config.port, 9200)

    def test_explicit_config_file_path_wins_over_config_path_environment_variable(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            first_config_path = Path(temp_dir) / "first.yaml"
            second_config_path = Path(temp_dir) / "second.yaml"
            first_config_path.write_text("port: 9100\n", encoding="utf-8")
            second_config_path.write_text("port: 9200\n", encoding="utf-8")

            config = ProxyConfig.from_file(
                env={"DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": str(second_config_path)},
                config_path=first_config_path,
            )

        self.assertEqual(config.port, 9100)

    def test_invalid_yaml_config_raises_value_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                ProxyConfig.from_file(env={}, config_path=config_path)

    def test_from_file_does_not_mutate_process_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "PROXY_VERBOSE": "true",
            },
            clear=True,
        ):
            ProxyConfig.from_file(config_path=Path("/does/not/exist"))
            self.assertEqual(dict(os.environ), {"PROXY_VERBOSE": "true"})


if __name__ == "__main__":
    unittest.main()

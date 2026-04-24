from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deepseek_cursor_proxy.config import (
    ProxyConfig,
    default_config_path,
    default_reasoning_content_path,
)


class ConfigTests(unittest.TestCase):
    def test_default_paths_live_in_user_app_directory(self) -> None:
        home = Path("/tmp/home")

        with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
            self.assertEqual(
                default_config_path(), home / ".deepseek-cursor-proxy" / ".env"
            )
            self.assertEqual(
                default_reasoning_content_path(),
                home / ".deepseek-cursor-proxy" / "reasoning_content.sqlite3",
            )
            self.assertEqual(
                ProxyConfig().reasoning_content_path,
                home / ".deepseek-cursor-proxy" / "reasoning_content.sqlite3",
            )

    def test_loads_config_from_user_env_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_file_path = Path(temp_dir) / ".env"
            reasoning_content_path = Path(temp_dir) / "reasoning_content.sqlite3"
            env_file_path.write_text(
                "\n".join(
                    [
                        "DEEPSEEK_API_KEY=file-key",
                        "PROXY_API_KEY=cursor-token",
                        "PROXY_PORT=9100",
                        f"REASONING_CONTENT_PATH={reasoning_content_path}",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_env(env={}, env_file_path=env_file_path)

        self.assertEqual(config.upstream_api_key, "file-key")
        self.assertEqual(config.proxy_api_key, "cursor-token")
        self.assertEqual(config.port, 9100)
        self.assertEqual(config.reasoning_content_path, reasoning_content_path)

    def test_environment_overrides_config_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_file_path = Path(temp_dir) / ".env"
            env_file_path.write_text(
                "\n".join(
                    [
                        "DEEPSEEK_API_KEY=file-key",
                        "PROXY_VERBOSE=false",
                    ]
                ),
                encoding="utf-8",
            )

            config = ProxyConfig.from_env(
                env={
                    "DEEPSEEK_API_KEY": "env-key",
                    "PROXY_VERBOSE": "true",
                },
                env_file_path=env_file_path,
            )

        self.assertEqual(config.upstream_api_key, "env-key")
        self.assertTrue(config.verbose)

    def test_relative_reasoning_content_path_stays_inside_app_directory(self) -> None:
        home = Path("/tmp/home")

        with patch("deepseek_cursor_proxy.config.Path.home", return_value=home):
            config = ProxyConfig.from_env(
                env={
                    "DEEPSEEK_API_KEY": "key",
                    "REASONING_CONTENT_PATH": "custom.sqlite3",
                },
                env_file_path=Path("/does/not/exist"),
            )

        self.assertEqual(
            config.reasoning_content_path,
            home / ".deepseek-cursor-proxy" / "custom.sqlite3",
        )

    def test_verbose_and_body_logging_can_be_enabled_from_env(self) -> None:
        config = ProxyConfig.from_env(
            env={
                "DEEPSEEK_API_KEY": "key",
                "PROXY_VERBOSE": "true",
                "PROXY_LOG_BODIES": "1",
                "PROXY_NGROK": "yes",
            },
            env_file_path=Path("/does/not/exist"),
        )

        self.assertTrue(config.verbose)
        self.assertTrue(config.log_bodies)
        self.assertTrue(config.ngrok)

    def test_config_path_can_be_overridden_from_environment(self) -> None:
        with TemporaryDirectory() as temp_dir:
            first_env_path = Path(temp_dir) / "first.env"
            second_env_path = Path(temp_dir) / "second.env"
            first_env_path.write_text("DEEPSEEK_API_KEY=first-key", encoding="utf-8")
            second_env_path.write_text("DEEPSEEK_API_KEY=second-key", encoding="utf-8")

            config = ProxyConfig.from_env(
                env={"DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": str(second_env_path)},
                env_file_path=None,
            )

        self.assertEqual(config.upstream_api_key, "second-key")

    def test_explicit_env_file_path_wins_over_config_path_environment_variable(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            first_env_path = Path(temp_dir) / "first.env"
            second_env_path = Path(temp_dir) / "second.env"
            first_env_path.write_text("DEEPSEEK_API_KEY=first-key", encoding="utf-8")
            second_env_path.write_text("DEEPSEEK_API_KEY=second-key", encoding="utf-8")

            config = ProxyConfig.from_env(
                env={"DEEPSEEK_CURSOR_PROXY_CONFIG_PATH": str(second_env_path)},
                env_file_path=first_env_path,
            )

        self.assertEqual(config.upstream_api_key, "first-key")

    def test_from_env_does_not_mutate_process_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DEEPSEEK_API_KEY": "env-key",
            },
            clear=True,
        ):
            ProxyConfig.from_env(env_file_path=Path("/does/not/exist"))
            self.assertEqual(dict(os.environ), {"DEEPSEEK_API_KEY": "env-key"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from deepseek_cursor_proxy.tunnel import (
    local_tunnel_target,
    ngrok_agent_urls,
    parse_ngrok_public_url,
)


class TunnelTests(unittest.TestCase):
    def test_local_tunnel_target_uses_loopback_for_wildcard_hosts(self) -> None:
        self.assertEqual(local_tunnel_target("0.0.0.0", 9000), "http://127.0.0.1:9000")
        self.assertEqual(local_tunnel_target("::", 9000), "http://127.0.0.1:9000")

    def test_local_tunnel_target_formats_ipv6_hosts(self) -> None:
        self.assertEqual(local_tunnel_target("::1", 9000), "http://[::1]:9000")

    def test_parse_ngrok_public_url_prefers_https(self) -> None:
        payload = {
            "tunnels": [
                {"public_url": "http://example.ngrok-free.app"},
                {"public_url": "https://example.ngrok-free.app"},
            ]
        }

        self.assertEqual(
            parse_ngrok_public_url(payload), "https://example.ngrok-free.app"
        )

    def test_parse_ngrok_public_url_supports_endpoint_api(self) -> None:
        payload = {"endpoints": [{"url": "https://example.ngrok-free.app"}]}

        self.assertEqual(
            parse_ngrok_public_url(payload), "https://example.ngrok-free.app"
        )

    def test_parse_ngrok_public_url_ignores_missing_tunnels(self) -> None:
        self.assertIsNone(parse_ngrok_public_url({"tunnels": []}))
        self.assertIsNone(parse_ngrok_public_url({}))

    def test_ngrok_agent_urls_use_current_api_then_legacy_fallback(self) -> None:
        self.assertEqual(
            ngrok_agent_urls("http://127.0.0.1:4040/api"),
            [
                "http://127.0.0.1:4040/api/endpoints",
                "http://127.0.0.1:4040/api/tunnels",
            ],
        )


if __name__ == "__main__":
    unittest.main()

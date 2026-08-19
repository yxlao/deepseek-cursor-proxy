#!/usr/bin/env python3
"""Run a metadata-only DeepSeek streaming benchmark through one or more routes.

The API credential is read only from an environment variable and is never
printed.  This deliberately measures one controlled completion, not Cursor's
agent orchestration; use the proxy timing log to correlate real agent turns.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any
from urllib.request import Request, urlopen


def parse_target(value: str) -> tuple[str, str]:
    name, separator, base_url = value.partition("=")
    if not separator or not name or not base_url:
        raise argparse.ArgumentTypeError("target must be NAME=https://host/v1")
    return name, base_url.rstrip("/")


def parse_case(value: str) -> tuple[str, str]:
    model, separator, effort = value.partition(":")
    if not separator or not model or not effort:
        raise argparse.ArgumentTypeError("case must be MODEL:EFFORT")
    return model, effort


def parse_auth_env(value: str) -> tuple[str, str]:
    target_name, separator, environment_name = value.partition("=")
    if not separator or not target_name or not environment_name:
        raise argparse.ArgumentTypeError("auth environment must be TARGET=ENV_VAR")
    return target_name, environment_name


def benchmark(
    *, base_url: str, authorization: str, model: str, effort: str, prompt: str, timeout: float
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "enabled"},
        "reasoning_effort": effort,
    }
    request_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=request_body,
        method="POST",
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    started = time.monotonic()
    first_byte_ms: int | None = None
    reasoning_chars = 0
    visible_output_chars = 0
    tool_call_count = 0
    usage: dict[str, Any] | None = None
    with urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            if first_byte_ms is None and raw_line.strip():
                first_byte_ms = round((time.monotonic() - started) * 1000)
            if not raw_line.startswith(b"data:"):
                continue
            data = raw_line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if not isinstance(delta, dict):
                    continue
                reasoning = delta.get("reasoning_content")
                content = delta.get("content")
                reasoning_chars += len(reasoning) if isinstance(reasoning, str) else 0
                visible_output_chars += len(content) if isinstance(content, str) else 0
                tool_call_count += len(delta.get("tool_calls") or [])
    return {
        "model": model,
        "effective_reasoning_effort": effort,
        "request_body_bytes": len(request_body),
        "time_to_first_upstream_byte_ms": first_byte_ms,
        "total_response_ms": round((time.monotonic() - started) * 1000),
        "reasoning_chars": reasoning_chars,
        "visible_output_chars": visible_output_chars,
        "tool_call_count": tool_call_count,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        required=True,
        help="repeatable NAME=https://host/v1 target",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        default=[
            ("deepseek-v4-flash", "max"),
            ("deepseek-v4-flash", "high"),
            ("deepseek-v4-pro", "high"),
        ],
        help="repeatable MODEL:EFFORT case",
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_BENCHMARK_API_KEY")
    parser.add_argument(
        "--auth-env",
        action="append",
        type=parse_auth_env,
        default=[],
        metavar="TARGET=ENV_VAR",
        help=(
            "override the credential environment variable for a target; use this "
            "for the Cloudflare route, which accepts PROXY_BEARER_TOKEN rather "
            "than a DeepSeek API key"
        ),
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--prompt",
        default="Propose a concise, safe plan for a one-file configuration change.",
    )
    args = parser.parse_args()
    target_auth_env = dict(args.auth_env)
    for target_name, base_url in args.target:
        auth_env = target_auth_env.get(target_name, args.api_key_env)
        api_key = os.environ.get(auth_env)
        if not api_key:
            parser.error(
                f"set {auth_env} for target {target_name}; its value is never printed"
            )
        authorization = (
            api_key
            if api_key.lower().startswith("bearer ")
            else f"Bearer {api_key}"
        )
        for model, effort in args.case:
            result = benchmark(
                base_url=base_url,
                authorization=authorization,
                model=model,
                effort=effort,
                prompt=args.prompt,
                timeout=args.timeout,
            )
            result["target"] = target_name
            print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

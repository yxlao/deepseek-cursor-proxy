# deepseek-cursor-proxy

Proxy for connecting Cursor to DeepSeek thinking models.

## What It Does

- ✅ Caches DeepSeek `reasoning_content` from regular and streamed responses, then restores it on later tool-call turns when Cursor omits it.
- ✅ Mirrors streamed `reasoning_content` into Cursor-visible `<think>...</think>` text such that thinking tokens are shown in Cursor's UI. Cursor renders this as normal chat text, not as a native collapsible Thinking block.
- ✅ Starts an ngrok tunnel so Cursor can reach the local proxy.
- ✅ Provides other compatibility fixes to make DeepSeek models run well in Cursor.

## Why This Exists

This repository fixes the following Cursor + DeepSeek tool-call error:

![Error 400 - reasoning_content must be passed back](assets/error_400.png)

```txt
Provider returned error:
{
  "error": {
    "message": "The reasoning_content in the thinking mode must be passed back to the API.",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_request_error"
  }
}
```

## Install

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytools
PIP_REQUIRE_VIRTUALENV=false python -m pip install -e .
```

## Configure

The proxy creates `~/.deepseek-cursor-proxy/config.yaml` on first run.

Edit it later if needed. API keys do not go in this file; enter your DeepSeek key in Cursor, and the proxy forwards it upstream.

## Set Up Ngrok

Create an ngrok account, then install and authenticate ngrok once:

```bash
brew install ngrok
ngrok config add-authtoken <your-ngrok-token>
```

Useful ngrok links:

- Sign up: https://dashboard.ngrok.com/signup
- Authtoken: https://dashboard.ngrok.com/get-started/your-authtoken
- Dashboard: https://dashboard.ngrok.com

## Run

```bash
deepseek-cursor-proxy --verbose
```

Copy the printed URL:

```text
Cursor Base URL: https://example.ngrok-free.app/v1
```

## Cursor Settings

- Base URL: the printed URL ending in `/v1`
- API Key: your DeepSeek API key
- Model: `deepseek-v4-pro`

## Useful Commands

Run without ngrok for local curl testing:

```bash
PROXY_NGROK=false deepseek-cursor-proxy --port 9000 --verbose
```

Use another config file:

```bash
deepseek-cursor-proxy --config ./dev.config.yaml
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

<h1><img src="assets/logo.png" width="90" alt="deepseek-cursor-proxy logo" style="vertical-align: middle;"> deepseek-cursor-proxy</h1>

Compatibility proxy connecting Cursor to DeepSeek thinking models (`deepseek-v4-pro` and `deepseek-v4-flash`).

## What It Does

- ✅ Caches DeepSeek `reasoning_content` from regular and streamed responses, then restores it on later tool-call turns when Cursor omits it. If the exact original reasoning is unavailable, the proxy fails closed instead of sending a fake placeholder. See [DeepSeek docs](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls) for more details.
- ✅ Mirrors streamed `reasoning_content` into Cursor-visible `<think>...</think>` text so that thinking tokens are shown in Cursor's UI. For BYOK/proxy mode, Cursor renders this as normal text, not as a native collapsible thinking block.
- ✅ Starts an ngrok tunnel so Cursor can reach the local proxy through a public HTTPS URL.
- ✅ Provides other compatibility fixes to make DeepSeek models run well in Cursor.

## Why This Exists

This repository fixes the following Cursor + DeepSeek tool-call error with thinking mode enabled:

![Error 400 - reasoning_content must be passed back](assets/error_400.png)

```txt
⚠️ Connection Error
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

## Usage

### Step 1: Set Up ngrok

Cursor blocks non-public API URLs such as `localhost`, so the proxy needs a public HTTPS URL. [ngrok](https://ngrok.com/) can expose the local proxy to Cursor without opening router ports. Alternatively, you may use [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/setup/).

Create an ngrok account, then visit ngrok's Dashboard: https://dashboard.ngrok.com

![ngrok dashboard](assets/ngrok_dashboard.png)

Then, install and authenticate ngrok once:

```bash
brew install ngrok
ngrok config add-authtoken <your-ngrok-token>
```

### Step 2: Add Cursor Custom Model

In Cursor, add the DeepSeek custom model and point it at this proxy:

- Model: `deepseek-v4-pro`
- API Key: your DeepSeek API key
- Base URL: your ngrok HTTPS URL with the `/v1` API version path

The proxy respects the DeepSeek model name Cursor sends, such as `deepseek-v4-pro` or `deepseek-v4-flash`. The `model` field in `config.yaml` is only the fallback used when a request does not include a model.

For example, if ngrok dashboard shows `https://example.ngrok-free.app`, use:

```text
https://example.ngrok-free.app/v1
```

![Cursor settings for DeepSeek through the proxy](assets/cursor_config.png)

Note: you can toggle the custom API on and off with:

- macOS: `Cmd+Shift+0`
- Windows/Linux: `Ctrl+Shift+0`

### Step 3: Start the Proxy Server

Install and run the proxy:

```bash
# Or, use your favourite Python env manager
conda create -n dcp python=3.10 -y
conda activate dcp

# Install
git clone https://github.com/yxlao/deepseek-cursor-proxy.git
cd deepseek-cursor-proxy
pip install -e .

# Run
deepseek-cursor-proxy
```

The proxy creates `~/.deepseek-cursor-proxy/config.yaml` on first run.

This will also print the ngrok public URL. If it differs from the one in Cursor, update it in Cursor's Base URL field.

### Step 4: Chat with DeepSeek in Cursor

Select `deepseek-v4-pro` in Cursor and use chat or agent mode as usual.

![Chatting with DeepSeek in Cursor](assets/cursor_chat.png)

## How It Works

DeepSeek's [thinking mode](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls) requires that `reasoning_content` from assistant messages involved in tool-call sequences is passed back in later API requests. Cursor may omit this field from conversation history, causing DeepSeek to return a 400 error. This proxy sits between Cursor and DeepSeek (`Cursor → ngrok → proxy → DeepSeek API`) and repairs requests when it has the exact original reasoning cached.

**Core fix — reasoning_content caching and restoration:**
Every DeepSeek response (streaming or non-streaming) has its `reasoning_content` stored in a local SQLite cache, keyed by message signature, tool-call ID, and tool-call function signature. On each outgoing thinking-mode request, the proxy scans tool-call-related assistant messages for missing `reasoning_content`, restores the exact original value from the cache, and sends the complete history to DeepSeek. If the cache is cold (for example, after a proxy restart), the proxy returns a local error instead of fabricating reasoning.

**Multi-conversation isolation:**
All cache keys are scoped by a SHA-256 hash of the canonical conversation prefix (roles, content, tool calls, excluding `reasoning_content` itself) plus the upstream model/configuration and an API-key hash. Concurrent or interleaved conversation threads with different histories produce different scope hashes, so reused tool-call IDs do not collide across those histories. Byte-identical cloned histories are indistinguishable unless Cursor sends a differentiating history.

**DeepSeek [prefix caching](https://api-docs.deepseek.com/guides/kv_cache) compatibility:**
The proxy does not inject synthetic thread IDs, timestamps, or cache-control messages into the prompt. When it restores cached reasoning, it restores the exact original string, preserving repeated prefixes for DeepSeek's automatic best-effort context cache.

**Additional compatibility fixes:**
The proxy converts legacy `functions`/`function_call` fields to `tools`/`tool_choice`, preserves required and named tool-choice semantics, normalizes `reasoning_effort` aliases per DeepSeek docs, strips mirrored `<think>` blocks from assistant content, converts multi-part content arrays to plain text, logs DeepSeek prompt-cache usage when available, and mirrors `reasoning_content` into Cursor-visible `<think>...</think>` blocks for thinking display.

## Debugging

Run with verbose output:

```bash
deepseek-cursor-proxy --verbose
```

Run without ngrok for local curl testing:

```bash
PROXY_NGROK=false deepseek-cursor-proxy --port 9000 --verbose
```

Use another config file:

```bash
deepseek-cursor-proxy --config ./dev.config.yaml
```

Clear the local reasoning cache:

```bash
deepseek-cursor-proxy --clear-reasoning-cache
```

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

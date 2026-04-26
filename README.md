<!-- <h1><img src="assets/logo.png" width="120" alt="deepseek-cursor-proxy logo" style="vertical-align: middle;">&nbsp;DeepSeek Cursor Proxy</h1> -->
<h1 align="center"><img src="assets/logo.png" width="120" alt="deepseek-cursor-proxy logo"><br>DeepSeek Cursor Proxy</h1>

A compatibility proxy that connects Cursor to DeepSeek thinking models (`deepseek-v4-pro` and `deepseek-v4-flash`) by properly handling the `reasoning_content` field for DeepSeek tool-call reasoning API requests.

This proxy can also help **other applications and coding agents** beyond Cursor that run into the same missing `reasoning_content` issue with DeepSeek's thinking-mode API. Just point their API base URL at the proxy.

## What It Does

- ✅ Injects `reasoning_content` into outgoing tool-call requests since Cursor does not include the field, restoring previously cached reasoning from regular and streamed DeepSeek responses. See [DeepSeek docs](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls) for more details.
- ✅ Mirrors streamed `reasoning_content` into Cursor-visible `<think>...</think>` text so that thinking tokens are shown in Cursor UI. For BYOK (bring your own key) mode, Cursor renders this as normal text, not as a native collapsible thinking block.
- ✅ Starts an ngrok tunnel so Cursor can reach the local proxy through a public HTTPS URL.
- ✅ Provides other compatibility fixes to make DeepSeek models run well in Cursor.

## Why This Exists

This repository fixes the following Cursor + DeepSeek tool-call error with thinking mode enabled:

<img src="assets/error_400.png" width="600" alt="Error 400 - reasoning_content must be passed back">

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

Cursor blocks non-public API URLs such as `localhost`, so the proxy needs a public HTTPS URL. [ngrok](https://ngrok.com/) can expose the local proxy to Cursor without opening router ports. Alternatively, you may use [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/setup/). Create an ngrok account and visit [ngrok's dashboard](https://dashboard.ngrok.com). You will find the authtoken and public URL there.

If you're using this proxy with another application that allows localhost API endpoints, you can skip this step entirely by setting `ngrok: false` in `~/.deepseek-cursor-proxy/config.yaml`, or by starting the proxy with `--no-ngrok`.

<img src="assets/ngrok_dashboard.png" width="600" alt="ngrok dashboard">

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

The proxy respects the DeepSeek model name Cursor sends, such as `deepseek-v4-pro` or `deepseek-v4-flash`. The `model` field in `config.yaml` is used as a fallback only when a request does not include a model.

For example, if ngrok dashboard shows `https://example.ngrok-free.dev`, use:

```text
https://example.ngrok-free.dev/v1
```

<img src="assets/cursor_config.png" width="600" alt="Cursor settings for DeepSeek through the proxy">

Note: you can toggle the custom API on and off with:

- macOS: `Cmd+Shift+0`
- Windows/Linux: `Ctrl+Shift+0`

### Step 3: Install and Start the Proxy Server

**Run with UV**

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install and start
# uv installs the program in .venv/ under the repo local folder
git clone https://github.com/yxlao/deepseek-cursor-proxy.git
cd deepseek-cursor-proxy
uv run deepseek-cursor-proxy
```

**Run with Conda**

```bash
# Install conda if you don't have it
# Follow: https://www.anaconda.com/docs/getting-started/miniconda/install/overview

# Install
conda create -n dcp python=3.10 -y
conda activate dcp
git clone https://github.com/yxlao/deepseek-cursor-proxy.git
cd deepseek-cursor-proxy
pip install -e .

# Start
deepseek-cursor-proxy
```

When ngrok is enabled, `deepseek-cursor-proxy` will print the ngrok public URL on start. If it differs from the one in Cursor, update it in Cursor's Base URL field.

On the first run, `deepseek-cursor-proxy` will create:

- `~/.deepseek-cursor-proxy/config.yaml`: the configuration file
- `~/.deepseek-cursor-proxy/reasoning_content.sqlite3`: the reasoning content cache

Persistent settings live in `~/.deepseek-cursor-proxy/config.yaml`. Command-line flags override the config for a single run, for example `--no-ngrok`, `--port 9000`, or `--verbose`.

### Step 4: Chat with DeepSeek in Cursor

Select `deepseek-v4-pro` in Cursor and use chat or agent mode as usual.

<img src="assets/cursor_chat.png" width="480" alt="Chatting with DeepSeek in Cursor">

## How It Works

- **Core fix:** DeepSeek's [thinking mode](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls) requires `reasoning_content` from assistant tool-call messages to be passed back in subsequent requests, but Cursor omits this field, causing a 400 error. The proxy (`Cursor → ngrok → proxy → DeepSeek API`) stores `reasoning_content` from every DeepSeek response in a local SQLite cache, keyed by message signature, tool-call ID, and tool-call function signature, and patches outgoing requests with missing `reasoning_content` before they reach DeepSeek. On a cold cache (proxy restart, model switch), it logs and drops unrecoverable history, continues from the latest user request, and prefixes the next Cursor response with a notice.
- **Multi-conversation isolation:** To avoid collisions across concurrent conversations, the proxy scopes cache keys by a SHA-256 hash of the canonical conversation prefix (roles, content, and tool calls, excluding `reasoning_content`) plus the upstream model, configuration, and an API-key hash. Different threads get different scopes, so reused tool-call IDs do not collide. Byte-identical cloned histories produce identical scopes.
- **Context caching compatibility:** The proxy preserves compatibility by never injecting synthetic thread IDs, timestamps, or cache-control messages. It restores `reasoning_content` as the exact original string, so repeated prefixes remain intact for [DeepSeek context cache](https://api-docs.deepseek.com/guides/kv_cache). Cache hit rates are logged in the terminal output.
- **Additional compatibility fixes:** Beyond reasoning repair, the proxy converts legacy `functions`/`function_call` fields to `tools`/`tool_choice`, preserves required and named tool-choice semantics, normalizes `reasoning_effort` aliases, strips mirrored `<think>` blocks from assistant content, flattens multi-part content arrays to plain text, and mirrors `reasoning_content` into Cursor-visible `<think>...</think>` blocks.

## Development

Run unit tests:

```bash
uv run python -m unittest discover -s tests
```

Run pre-commit hooks (code formatting and linting):

```bash
uv sync --dev
uv run pre-commit run --all-files
```

## Debugging

Run with verbose output:

```bash
deepseek-cursor-proxy --verbose
```

Run without ngrok for local curl testing:

```bash
deepseek-cursor-proxy --no-ngrok --port 9000 --verbose
```

Use another config file:

```bash
deepseek-cursor-proxy --config ./dev.config.yaml
```

Clear the local reasoning cache:

```bash
deepseek-cursor-proxy --clear-reasoning-cache
```

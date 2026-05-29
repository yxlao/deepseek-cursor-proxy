<h1 align="center"><img src="assets/logo.png" width="150" alt="logo"><br>DeepSeek Cursor Proxy Turbo</h1>

<p align="center"><strong>⚡ 极致提速版 · Performance-Optimized Fork</strong></p>
<p align="center">基于 <a href="https://github.com/yxlao/deepseek-cursor-proxy">yxlao/deepseek-cursor-proxy</a> | 原项目 MIT License</p>

---

## 🀄 中文版

### 一、为什么 Cursor 不能直接用 DeepSeek 的思考模式？

**DeepSeek 官方的要求**（[Thinking Mode 文档](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)）：

> "Between two `user` messages, if the model performed a tool call, the intermediate `assistant`'s `reasoning_content` **must** participate in the context concatenation and **must be passed back** to the API in all subsequent user interaction turns."

翻译：当模型执行了工具调用（tool call），后续**所有**请求都必须把之前的 `reasoning_content` 原封不动传回去。这是 DeepSeek 的硬性 API 契约——在思考模式下，工具调用的推理链必须保持连续。

**Cursor 的实际行为**（[Cursor 社区论坛 Bug Report](https://forum.cursor.com/t/deepseek-v4-context-limited-to-200k-reasoning-content-error/159045/7)）：

Cursor 遵循 OpenAI Chat Completions 标准 schema。`reasoning_content` 不在 OpenAI 的标准字段里，它是 DeepSeek 的专有扩展。Cursor 在构建后续请求时不会保留这个字段，导致 API 返回 400 错误：

```json
{
  "error": {
    "message": "The reasoning_content in the thinking mode must be passed back to the API.",
    "type": "invalid_request_error",
    "code": "invalid_request_error"
  }
}
```

> 正如 [APIDog 指南](https://apidog.com/blog/how-to-use-deepseek-v4-pro-with-cursor/) 所总结的：*"Cursor's chat client follows the OpenAI Chat Completions schema. `reasoning_content` isn't part of that schema; it's a DeepSeek-specific extension. Cursor would need to add provider-specific handling to pass the field through."*

---

### 二、deepseek-cursor-proxy 解决了什么？

代理位于 Cursor 和 DeepSeek API 之间：

```
Cursor → ngrok/Cloudflare Tunnel → Proxy (localhost:9000) → DeepSeek API
```

**核心机制**：
- **缓存 reasoning**：每次 DeepSeek 返回响应时，代理用 SHA-256 哈希（基于对话前缀、消息签名、工具调用 ID/签名）作为 key，将 `reasoning_content` 存入本地 SQLite 数据库
- **回填 reasoning**：Cursor 发来后续请求时，代理检测到缺失的 `reasoning_content`，从缓存中查找匹配的记录并注入
- **恢复机制**：当缓存未命中时，代理自动裁剪对话历史到最近一次用户消息，追加恢复提示，保证对话能继续

---

### 三、原项目存在什么问题？

在 300+ 条消息的深度对话中，原项目每次请求会**卡住 50-110 秒**。根因有三层：

| 层次 | 问题 | 原因 |
|------|------|------|
| **O(n²) 哈希** | `turn_context_signature()` 在 `normalize_messages` 循环内每条 assistant 消息都被调用，每次都从头扫描全部前置消息 | 355 条消息 = ~63,000 次重复哈希 |
| **逐 key COMMIT** | `backfill_portable_aliases()` 为每个 portable key 分别调用 `store.put()`，每次 `put()` 都做一次 SQLite COMMIT（fsync 刷盘） | 65 条 patched 消息 × 7 key = **455 次 fsync** |
| **无索引全表扫描** | `_prune_locked()` 中 `DELETE FROM reasoning_cache WHERE created_at < ?` 在 `created_at` 列上**没有索引**，每次 COMMIT 前全表扫描数万行 | 455 次 COMMIT × 每次全表扫描 = 灾难 |

三层叠加：**455 次 fsync × (COMMIT 70ms + 全表扫描 30ms) ≈ 45,000ms**

---

### 四、本仓库做了什么？效果如何？

🔧 **六项核心优化**：

| # | 优化 | 文件 | 效果 |
|---|------|------|------|
| 1 | O(n²) → O(n) 增量哈希 | `reasoning_store.py` | `compute_conversation_scopes()` / `compute_turn_signatures()` 一次遍历生成全部前缀哈希 |
| 2 | `backfill` 批量 COMMIT | `reasoning_store.py` | 每条消息的所有 portable key 合并为一个事务，一次 COMMIT |
| 3 | `created_at` 加索引 | `reasoning_store.py` | `CREATE INDEX idx_reasoning_cache_created_at` |
| 4 | `_prune_locked` 短路 | `reasoning_store.py` | COUNT 预检，无过期行时跳过 DELETE |
| 5 | SQLite WAL 模式 | `reasoning_store.py` | `PRAGMA journal_mode=WAL`，写不阻塞读，COMMIT 飞起 |
| 6 | Windows 静默启动 | `silent-start.pyw` + VBS | 开机自启、无命令行窗口、端口检测防重复启动 |

📊 **实测效果**（真实 399 条消息对话）：

```
修复前：┌ context ... (108,584ms)  ← 108 秒
修复后：┌ context ... (119ms)      ← 119 毫秒
```

**快了 912 倍。** 从无法忍受的分钟级等待降到完全无感。

🔑 **缓存命中率保持不变**：

```
修复前：cache_hit=99.8%
修复后：cache_hit=99.8%（完全一致）
```

代理由原项目演化而来，`reasoning_content` 注入逻辑和缓存 key 计算完全一致。验证脚本确认：增量哈希结果与原版等价（`verify_hash_optimizations.py: OK: hashes match originals`），不会出现缓存 miss 增多或注入错误。

🆕 **额外功能**：

- `--user-suffix TEXT`：可在每条用户消息末尾自动追加文本（用于引导模型行为）
- 启动即用：双击 `start-deepseek-proxy.bat`，自动完成 ngrok 隧道、代理启动、URL 配置

> 🙏 本项目 fork 自 [yxlao/deepseek-cursor-proxy](https://github.com/yxlao/deepseek-cursor-proxy)，作者 **Yixing Lao** 的杰出工作使 DeepSeek + Cursor 成为可能。我们在此基础上做了极致性能优化。

---

## 🌐 English

### 1. Why Can't Cursor Use DeepSeek Thinking Mode Directly?

**DeepSeek's Official Requirement** ([Thinking Mode Docs](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)):

> "Between two `user` messages, if the model performed a tool call, the intermediate `assistant`'s `reasoning_content` **must** participate in the context concatenation and **must be passed back** to the API in all subsequent user interaction turns."

In thinking mode, tool-call reasoning chains must remain continuous across all requests. This is a hard API contract.

**Cursor's Actual Behavior** ([Cursor Community Forum](https://forum.cursor.com/t/deepseek-v4-context-limited-to-200k-reasoning-content-error/159045/7)):

Cursor follows the OpenAI Chat Completions standard schema. `reasoning_content` is **not** part of that schema—it's a DeepSeek-specific extension. Cursor strips this field when constructing follow-up requests, causing a 400 error:

```json
{
  "error": {
    "message": "The reasoning_content in the thinking mode must be passed back to the API.",
    "type": "invalid_request_error",
    "code": "invalid_request_error"
  }
}
```

As the [APIDog Guide](https://apidog.com/blog/how-to-use-deepseek-v4-pro-with-cursor/) explains: *"Cursor's chat client follows the OpenAI Chat Completions schema. `reasoning_content` isn't part of that schema; it's a DeepSeek-specific extension that emerged with the R1 family and stayed in V4-Pro."*

---

### 2. What Does deepseek-cursor-proxy Solve?

The proxy sits between Cursor and the DeepSeek API:

```
Cursor → ngrok/Cloudflare Tunnel → Proxy (localhost:9000) → DeepSeek API
```

**Core mechanism**:
- **Cache reasoning**: Every DeepSeek response's `reasoning_content` is stored in a local SQLite database, keyed by SHA-256 hashes of conversation prefixes, message signatures, and tool-call IDs/signatures
- **Restore reasoning**: On subsequent requests, the proxy detects missing `reasoning_content`, looks up the cache, and injects the correct value before forwarding to DeepSeek
- **Recovery**: On cache miss, the proxy trims history to the last user message and adds a recovery notice, keeping the conversation alive

---

### 3. What Performance Problems Existed?

With 300+ message conversations, the original project would **stall for 50–110 seconds** per request. Three layers of bottlenecks:

| Layer | Problem | Cause |
|-------|---------|-------|
| **O(n²) hashing** | `turn_context_signature()` called per-message inside `normalize_messages`, re-hashing all prior messages each time | 355 msgs = ~63,000 redundant hash ops |
| **Per-key COMMIT** | `backfill_portable_aliases()` called `store.put()` separately for each portable key, each doing a SQLite COMMIT (fsync) | 65 patched msgs × 7 keys = **455 fsyncs** |
| **Unindexed full scan** | `_prune_locked()` ran `DELETE WHERE created_at < ?` with no index on `created_at`, scanning all rows each time | 455 COMMITs × full table scan = disaster |

Compounding effect: **455 fsyncs × (COMMIT 70ms + full scan 30ms) ≈ 45 seconds**

---

### 4. What This Fork Improves

🔧 **Six core optimizations**:

| # | Optimization | File | Effect |
|---|-------------|------|--------|
| 1 | O(n²) → O(n) incremental hashing | `reasoning_store.py` | Single-pass prefix hash generation for all scopes and turn signatures |
| 2 | Batched backfill COMMIT | `reasoning_store.py` | All portable keys for one message in a single transaction |
| 3 | Index on `created_at` | `reasoning_store.py` | `CREATE INDEX idx_reasoning_cache_created_at` |
| 4 | Prune short-circuit | `reasoning_store.py` | COUNT pre-check; skip DELETE when nothing to prune |
| 5 | SQLite WAL mode | `reasoning_store.py` | `PRAGMA journal_mode=WAL` — non-blocking writes |
| 6 | Windows silent launcher | `silent-start.pyw` + VBS | Auto-start on boot, no console window, port-guard |

📊 **Real-world results** (399-message conversation):

```
Before: ┌ context ... (108,584ms)  ← 108 seconds
After:  ┌ context ... (119ms)       ← 119 milliseconds
```

**912× faster.** From unbearable minute-long waits to completely imperceptible.

🔑 **Cache hit rate preserved**:

```
Before: cache_hit=99.8%
After:  cache_hit=99.8% (identical)
```

The `reasoning_content` injection logic and cache key computation are identical to the original. Benchmark verification confirms: incremental hashes match the originals exactly (`verify_hash_optimizations.py: OK: hashes match originals`). No increased cache misses, no injection errors.

🆕 **Bonus features**:

- `--user-suffix TEXT`: Automatically append text to every user message before forwarding (useful for guiding model behavior)
- One-click startup: double-click `start-deepseek-proxy.bat` for automatic ngrok tunnel + proxy launch

---

---

## 🛠 完整安装指南（中文） · Full Installation Guide (English)

### 第 1 步：注册并安装 ngrok 隧道

Cursor 屏蔽 `localhost` 等非公网 API URL，因此代理必须通过公网 HTTPS 可达。[ngrok](https://ngrok.com/) 可免费将本地端口暴露到公网。也可使用 [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/setup/)。

注册 ngrok 账号后，访问 [ngrok dashboard](https://dashboard.ngrok.com) 获取 authtoken：

**macOS / Linux:**
```bash
brew install ngrok
ngrok config add-authtoken <your-ngrok-token>
```

**Windows:** 用 `winget install Ngrok.Ngrok`，或从 [ngrok.com/download](https://ngrok.com/download) 下载。安装后在终端运行 `ngrok config add-authtoken <your-ngrok-token>`。

### Step 1: Install ngrok Tunnel

Cursor blocks non-public API URLs such as `localhost`, so the proxy needs a public HTTPS URL. [ngrok](https://ngrok.com/) exposes your local proxy to Cursor without opening router ports. [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/setup/) is also supported.

Register an ngrok account, visit [ngrok dashboard](https://dashboard.ngrok.com), get your authtoken:

```bash
# macOS / Linux
brew install ngrok
ngrok config add-authtoken <your-ngrok-token>

# Windows
winget install Ngrok.Ngrok
ngrok config add-authtoken <your-ngrok-token>
```

---

### 第 2 步：安装并启动代理

**方式 A：使用 UV（推荐）**

```bash
# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# Windows: winget install astral-sh.uv

# 克隆并启动
git clone https://github.com/abcduyt1243-git/deepseek-cursor-proxy-turbo.git
cd deepseek-cursor-proxy-turbo
uv run deepseek-cursor-proxy
```

**方式 B：使用 pip + venv**

```bash
git clone https://github.com/abcduyt1243-git/deepseek-cursor-proxy-turbo.git
cd deepseek-cursor-proxy-turbo
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .
deepseek-cursor-proxy
```

**方式 C：使用 Conda**

```bash
conda create -n dcp python=3.10 -y
conda activate dcp
git clone https://github.com/abcduyt1243-git/deepseek-cursor-proxy-turbo.git
cd deepseek-cursor-proxy-turbo
pip install -e .
deepseek-cursor-proxy
```

启动后代理会在终端打印 ngrok 公网 URL。

如果使用**固定 ngrok 域名**（reserved endpoint / custom domain），在 `~/.deepseek-cursor-proxy/config.yaml` 中设置 `ngrok_url`，或通过命令行传递：

```yaml
ngrok: true
ngrok_url: https://your-subdomain.ngrok.dev
```

```bash
deepseek-cursor-proxy --ngrok-url https://your-subdomain.ngrok.dev
```

首次运行会自动创建：
- `~/.deepseek-cursor-proxy/config.yaml`：配置文件
- `~/.deepseek-cursor-proxy/reasoning_content.sqlite3`：推理缓存数据库

常用命令行选项：

```bash
deepseek-cursor-proxy --no-display-reasoning   # 隐藏思考 token
deepseek-cursor-proxy --verbose                 # 详细日志
deepseek-cursor-proxy --no-ngrok               # 不使用隧道（localhost）
deepseek-cursor-proxy --port 9000              # 自定义端口
deepseek-cursor-proxy --user-suffix "请用中文思考"  # 追加用户消息后缀
```

### Step 2: Install & Start

**Option A: UV (recommended)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# Windows: winget install astral-sh.uv

git clone https://github.com/abcduyt1243-git/deepseek-cursor-proxy-turbo.git
cd deepseek-cursor-proxy-turbo
uv run deepseek-cursor-proxy
```

**Option B: pip + venv**

```bash
git clone https://github.com/abcduyt1243-git/deepseek-cursor-proxy-turbo.git
cd deepseek-cursor-proxy-turbo
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate        # Windows
pip install -e .
deepseek-cursor-proxy
```

**Option C: Conda**

```bash
conda create -n dcp python=3.10 -y
conda activate dcp
git clone https://github.com/abcduyt1243-git/deepseek-cursor-proxy-turbo.git
cd deepseek-cursor-proxy-turbo
pip install -e .
deepseek-cursor-proxy
```

Common flags:

```bash
deepseek-cursor-proxy --no-display-reasoning   # Hide thinking display
deepseek-cursor-proxy --verbose                 # Verbose output
deepseek-cursor-proxy --no-ngrok               # Run on localhost only
deepseek-cursor-proxy --port 9000              # Custom port
deepseek-cursor-proxy --user-suffix "Think in Chinese"  # Append to user msgs
```

---

### 第 3 步：配置 Cursor

在 Cursor 中添加自定义模型：

- **Model**：`deepseek-v4-pro`（或 `deepseek-v4-flash`）
- **API Key**：你的 DeepSeek API Key（以 `sk-` 开头）
- **Base URL**：代理打印的 ngrok URL + `/v1`，例如 `https://xxx.ngrok-free.dev/v1`

代理会原样转发 Cursor 发送的模型名（如 `deepseek-v4-pro`、`deepseek-v4-flash`）。`config.yaml` 中的 `model` 字段仅在没有传模型名时作为兜底。

按 `Ctrl+Shift+0`（Windows/Linux）或 `Cmd+Shift+0`（macOS）可快速切换自定义 API。

### Step 3: Configure Cursor

In Cursor, add a custom model:

- **Model**: `deepseek-v4-pro` (or `deepseek-v4-flash`)
- **API Key**: your DeepSeek API key (starts with `sk-`)
- **Base URL**: the ngrok URL printed by the proxy + `/v1`, e.g. `https://xxx.ngrok-free.dev/v1`

Toggle with `Ctrl+Shift+0` (Windows/Linux) or `Cmd+Shift+0` (macOS).

---

### 第 4 步：开始对话

在 Cursor 中选择 `deepseek-v4-pro`，正常使用 Chat 或 Agent 模式即可。

### Step 4: Start Chatting

Select `deepseek-v4-pro` in Cursor and use chat or agent mode as usual.

---

## 🔧 调试 · Debugging

```bash
deepseek-cursor-proxy --verbose                                    # 详细日志
deepseek-cursor-proxy --no-ngrok --port 9000 --verbose             # 本地测试
deepseek-cursor-proxy --verbose --trace-dir ./trace-dumps          # 完整追踪
deepseek-cursor-proxy --config ./dev.config.yaml                   # 自定义配置
deepseek-cursor-proxy --clear-reasoning-cache                      # 清空缓存
```

---

## 📜 License

MIT — same as the original project.

## 🙏 Credits

Forked from [yxlao/deepseek-cursor-proxy](https://github.com/yxlao/deepseek-cursor-proxy) by Yixing Lao — the brilliant original work that made DeepSeek + Cursor possible.

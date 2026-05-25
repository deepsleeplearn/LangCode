# LangCode Agent

LangCode Agent 是一个基于 LangGraph 构建的 Python CLI 代码 Agent。v1 实现 Claude Code 风格的核心循环：自然语言对话、工具调用、限定在工作区内的文件操作、shell 执行、高风险工具的人工审批，以及基于 checkpoint 的会话恢复。

## Mac 本地一键启动

clone 仓库后，在项目根目录执行：

```bash
./scripts/start_macos.sh
```

脚本会自动完成：

- 检查 Python 3.11+ 和 npm。
- 创建 `.venv` 并安装当前 Python 包。
- 如果没有 `.env.local`，从 `.env.example` 复制一份本地配置模板。
- 安装前端依赖并构建 `frontend/dist`。
- 启动 Sanic Web server，默认地址为 `http://127.0.0.1:8765`。

首次启动前，请把 `.env.local` 里的模型或搜索 API key 替换为真实值；没有 key 时页面可以打开，但 chat 请求会返回配置错误。常用覆盖参数：

```bash
LANGCODE_PORT=9000 ./scripts/start_macos.sh
LANGCODE_WORKSPACE=../some-project ./scripts/start_macos.sh
```

Redis 不是本地单机启动的必需项；未启用或不可用时，运行态会回退到进程内存。

发布前可以执行：

```bash
python3 scripts/check_release.py
```

它会检查一键启动所需源码文件、`.gitignore` 对密钥/模型/运行态文件的保护，以及源码中是否误写了常见真实 token。

## 安装

在本目录执行：

```bash
python3 -m pip install -e .
```

如果不安装，直接从源码树运行：

```bash
PYTHONPATH=src python3 -m langcode_agent.interfaces.cli --workspace .
```

## 模型配置

自然语言对话通过 `langchain-openai` 使用 OpenAI-compatible chat model。

默认 provider 是智谱：

```bash
export LANGCODE_PROVIDER="zhipu"
export ZHIPU_API_KEY="..."
export LANGCODE_MODEL="glm-5.1"
```

默认智谱设置：

- base URL: `https://open.bigmodel.cn/api/paas/v4`
- model: `glm-5.1`

不要把真实 API key 放进 tracked 文件。请使用 shell 环境变量，或使用被忽略的本地 env 文件。

也支持 AIMP 网关的 GPT-4o。该接口按 OpenAI-compatible chat completions 接入，并额外发送 `Aimp-Biz-Id` 和 `AIGC-USER` headers：

```bash
export LANGCODE_PROVIDER="openai"
export LANGCODE_OPENAI_GATEWAY="aimp"
export LANGCODE_MODEL="gpt-4o"
export AIMP_GPT4O_BASE_URL="https://aimpapi.midea.com/t-aigc/mip-chat-app/openai/standard/v1"
export AIMP_GPT4O_USER="..."
export AIMP_GPT4O_API_KEY="..."
```

Web 左下角设置里的模型下拉框可直接选择 `AIMP GPT-4o`。

Web 搜索/抓取使用 LangChain 官方 Tavily 集成 `langchain-tavily`：

```bash
export LANGCODE_WEB_SEARCH_PROVIDER="tavily"
export TAVILY_API_KEY="..."
```

当前实现默认使用 Tavily `basic` search，普通搜索每次消耗 1 credit。真实 key 请放在 `.env.local` 或 shell 环境变量中，不要写入 tracked 文件。

## 语音与本地模型

语音输入默认使用 Qwen3-ASR。Mac 上可显式使用 Apple MPS：

```bash
export LANGCODE_ASR_DEVICE="mps"
export LANGCODE_ASR_DTYPE="auto"
```

语音播报支持两类路径：

- 默认音色：使用 macOS `say`，不需要额外模型。
- 自定义音色：使用本地 MLX/CosyVoice3。模型体积较大，不提交到 GitHub；clone 后需要把 MLX 兼容模型放到本地，并通过环境变量指向它。

自定义音色推荐配置：

```bash
export LANGCODE_TTS_PROVIDER="auto"
export LANGCODE_TTS_MODEL_DIR="/absolute/path/to/Fun-CosyVoice3-0.5B-2512-8bit"
export LANGCODE_TTS_VOICE_DIR=".langcode/tts-voices"
export LANGCODE_TTS_WORKERS="2"
```

内置的 `汪菊`、`雪芬` 自定义音色需要本地样本文件。样本文件属于本地音频资产，默认不提交到 Git。准备方式：

```bash
mkdir -p .langcode/tts-voices/samples
cp /path/to/汪菊.wav .langcode/tts-voices/samples/wangju.wav
cp /path/to/雪芬.wav .langcode/tts-voices/samples/xuefen.wav
PYTHONPATH=src python3 scripts/prepare_mlx_cosyvoice3.py \
  --model-dir "$LANGCODE_TTS_MODEL_DIR" \
  --voice-dir .langcode/tts-voices
```

如果模型或样本不存在，Web 页面仍能启动；自定义音色会不可用或回退，默认音色仍可使用 macOS `say`。

如果要切换到另一个 OpenAI-compatible 网关：

```bash
export LANGCODE_PROVIDER="openai"
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_API_KEY="..."
export LANGCODE_MODEL="your-model"
```

如果没有配置 API key，可以使用 raw tool 模式做本地验证：

```bash
mkdir -p "${TMPDIR:-/tmp}/langcode-demo"
PYTHONPATH=src python3 -m langcode_agent.interfaces.cli --workspace "${TMPDIR:-/tmp}/langcode-demo" --raw-tools
```

## CLI 使用

启动一个 chat session：

```bash
langcode-agent --workspace . --session default
```

继续同一个 session：

```bash
langcode-agent --workspace . --session default
```

session 状态保存在：

```text
<workspace>/.langcode/
```

CLI 也支持通过 `:tool` 直接调用工具：

```text
:tool {"name":"read_file","args":{"path":"README.md"}}
```

raw tool 模式只接受 JSON 工具调用：

```text
{"name":"write_file","args":{"path":"README.md","content":"hello"}}
```

## Web UI

React Web UI 提供类似 Claude/GPT 的代码工作台：左侧是 session 和设置，中间是对话界面，输入框附近显示审批项。

Mac 本地推荐直接使用一键脚本：

```bash
./scripts/start_macos.sh
```

安装前端依赖并构建：

```bash
cd frontend
npm install
npm run build
```

从项目根目录启动异步 Sanic Web server：

```bash
PYTHONPATH=src python3 -m langcode_agent.interfaces.web \
  --workspace . \
  --frontend-dir frontend/dist \
  --port 8765
```

打开：

```text
http://127.0.0.1:8765
```

Web UI 使用与 CLI 相同的智谱/OpenAI-compatible 环境变量。如果没有模型 key，chat 请求会显示可操作的错误提示。

Web server 使用 Sanic 异步 HTTP 层。阻塞的模型调用和工具执行会放到线程中运行，同一 session 的请求会串行化，避免并发修改同一段消息历史；不同 session 的请求可以并发处理。

当前 Web UI 能力：

- 页面铺满整个浏览器视口，页面本身不随上下滚动。
- 中间对话区较宽，两侧栏较窄。
- 模型输出通过 `/api/chat-stream` 流式显示。
- 输入框默认单行，按 Enter 提交，Shift+Enter 换行。
- 写文件、编辑文件和 shell 审批项显示在输入框上方。
- 当前工作目录和刷新按钮位于输入框下方；点击文件夹图标会打开本机目录选择弹窗。
- 左侧底部设置按钮可选择模型和界面语言，支持中文和英文。
- 左侧栏显示所有 Web 会话，点击会加载历史消息；每个会话右侧有三点菜单，可重命名或删除该会话。
- Assistant 回复会实时渲染 Markdown，支持 GFM 表格和 LaTeX 数学公式。
- 支持本地命令：
  - `/compact [说明]`：压缩当前会话上下文，保留最近消息，把较早消息汇总为系统摘要，并在 `.langcode/compactions/` 归档完整压缩前历史。
  - `/memory`：查看当前可加载的项目记忆和指令。
  - `/agents`：查看内置只读子 Agent。
  - `# 记忆内容`：写入 `.langcode/memories/MEMORY.md`，后续会话系统提示会自动加载。
- 模型可调用 `delegate_agent` 只读子 Agent，在独立上下文中执行 researcher/reviewer/planner 类型的检索、审查或规划任务。

## 工具

- `read_file`：读取工作区内 UTF-8 文本文件。
- `search`：在工作区内搜索，优先使用 `rg`。
- `web_search`：通过 Tavily 搜索公网网页，用于当前文档、新闻、API 参考和外部资料检索。
- `web_fetch`：通过 Tavily 抓取指定公网 URL 的可读 Markdown 内容；会拒绝 localhost、`.local`、私有/保留 IP。
- `write_file`：写入工作区内 UTF-8 文本文件。
- `edit_file`：替换工作区内 UTF-8 文本文件内容。
- `shell`：在工作区内带超时运行 shell 命令。低风险且不逃逸 workspace 的命令会自动执行；危险命令、路径逃逸或命中 ask 规则时进入人工审批。
- `delegate_agent`：启动只读子 Agent，支持 `researcher`、`reviewer`、`planner` 三种角色。子 Agent 只能使用 `read_file`、`search`、`web_search` 和 `web_fetch`，不会写文件或执行 shell。

## 权限模型

- `read_file`、`search`、`web_search` 和 `web_fetch` 自动允许；`web_fetch` 只允许公网 `http(s)` URL。
- `write_file` 和 `edit_file` 需要人工审批。
- `shell` 对齐 Claude Code 的权限规则思路：`deny` 优先，其次 `allow`，再看 `ask` 和风险分类；普通低风险命令自动通过。
- 审批选项为 `accept`、`reject`、`edit` 和 `feedback`。
- 路径会基于 workspace root 解析，路径逃逸会被拒绝。
- shell 命令带超时运行；需要审批时会显示风险摘要。Web 审批里的“允许并记住”会把当前命令写入 `<workspace>/.langcode/settings.json`：

```json
{
  "permissions": {
    "allow": ["Bash(mkdir output)"],
    "ask": ["Bash(git push*)"],
    "deny": ["Bash(curl *)"]
  }
}
```

## 恢复与记忆

项目为长任务维护了几个 Markdown 文件：

- `SPEC.md`：产品与技术规格。
- `GOAL.md`：可执行的长任务目标契约。
- `DEVELOPMENT_LOG.md`：按时间记录的开发过程。
- `AGENT_MEMORY.md`：供未来 session 或上下文压缩后继续工作的简明交接记忆。
- `TESTING_ADVERSARIAL_LOG.md`：功能测试和对抗测试过程。

运行时 session 状态保存在 `<workspace>/.langcode/`，该目录被 git 忽略。
Web 会话、消息内容、重命名/删除等操作事件保存在 `<workspace>/.langcode/web.sqlite`。旧版 `.langcode/web-sessions/*.json` 和 `.langcode/web-session-metadata.json` 会在启动时自动迁移进 SQLite；迁移后不会再写新的 Web 会话 JSON 文件。
LangGraph 工具审批/恢复 checkpoint 仍单独保存在 `<workspace>/.langcode/checkpoints.sqlite`，它不承载左侧栏会话列表。
上下文压缩归档保存在 `<workspace>/.langcode/compactions/`。
Hermes 热记忆保存在 `<workspace>/.langcode/memories/MEMORY.md` 和 `<workspace>/.langcode/memories/USER.md`。启动新会话或继续旧会话时，LangCode 会刷新自己的系统提示，让旧会话也读取新的 Hermes 热记忆；旧版 `<workspace>/.langcode/MEMORY.md` 不再作为项目上下文加载。

当前已验证的 LangChain/LangGraph 版本组合：

- `langchain==1.3.1`
- `langgraph==1.2.0`
- `langchain-core==1.4.0`
- `langchain-openai==1.2.1`
- `langchain-tavily`
- `langgraph-checkpoint-sqlite==3.1.0`

## 开发

运行测试：

```bash
python3 -m pytest tests -q
```

构建前端：

```bash
cd frontend
npm run build
```

不依赖 API key 的本地 smoke test：

```bash
SMOKE_WORKSPACE="${TMPDIR:-/tmp}/langcode-smoke"
mkdir -p "$SMOKE_WORKSPACE"
printf '%s\n' \
  '{"name":"write_file","args":{"path":"README.md","content":"smoke"}}' \
  'accept' \
  'quit' \
  | PYTHONPATH=src python3 -m langcode_agent.interfaces.cli --workspace "$SMOKE_WORKSPACE" --session smoke --raw-tools
```

## GitHub 发布前检查

建议提交源码、测试、文档、`pyproject.toml`、`frontend/package.json`、`frontend/package-lock.json`、`.env.example` 和 `scripts/*.py` / `scripts/*.sh`。提交前运行：

```bash
python3 scripts/check_release.py
```

不要提交以下本地生成或敏感内容，它们已由 `.gitignore` 覆盖：

- `.env.local`、`.env` 等真实密钥配置。
- `.langcode/`、`.gstack/`、SQLite 运行态和日志。
- `.venv/`、`frontend/node_modules/`、`frontend/dist/`。
- `__pycache__/`、`.pytest_cache/`、`*.pyc`。
- 本地模型、checkpoint、音频样本和预览文件，例如 `*.wav`、`*.pt`、`*.safetensors`、`models/`。

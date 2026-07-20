# DSPy + GEPA 试运行

这个目录是一个独立 trial，用来体验 **DSPy + GEPA** 的评测驱动 prompt 进化过程，不会改动 LangCode 主项目代码。

## 你会看到什么

示例任务是一个很小的中文意图分类器：

- 输入用户请求文本
- 输出意图标签：`code_change`、`web_research`、`memory_update`、`casual_chat`
- 先运行未优化 DSPy 程序，得到 baseline 分数
- 再用 `dspy.GEPA` 根据训练样本、验证样本和自然语言 feedback 进化 prompt
- 最后对比优化前后结果，并把优化程序保存到 `trial/output/`

这对应 Hermes self-evolution 独立仓库的核心思想：不是训练模型权重，而是对 prompt / instruction 这类文本组件做评测驱动的反思式优化。

## 一键运行

在项目根目录执行：

```bash
./trial/run_gepa_demo.sh
```

脚本会：

1. 在 `trial/.venv/` 创建隔离虚拟环境。
2. 安装 `trial/requirements.txt`。
3. 读取项目根目录的 `.env.local` 或系统环境变量。
4. 运行 `trial/gepa_intent_demo.py`。

## 模型配置

脚本只通过模型名选择配置，模型名和 AIMP 网关是固定绑定的：

- `glm-5` -> `AIMP_GLM_API_KEY` / `AIMP_GLM_BASE_URL` / `AIMP_GLM_USER`
- `deepseek-v4-pro` -> `AIMP_DEEPSEEK_V4_API_KEY` / `AIMP_DEEPSEEK_V4_BASE_URL` / `AIMP_DEEPSEEK_V4_USER`
- `gpt-4o` -> `AIMP_GPT4O_API_KEY` / `AIMP_GPT4O_BASE_URL` / `AIMP_GPT4O_USER`

默认主模型读取 `LANGCODE_GEPA_MODEL`，再回退到 `LANGCODE_MODEL`，最后回退到 `glm-5`。如果你想临时指定：

```bash
python -m trial.gepa_intent_demo --model glm-5
python -m trial.gepa_intent_demo --model deepseek-v4-pro
```

## 控制成本

默认是很小的预算：

```bash
LANGCODE_GEPA_MAX_METRIC_CALLS=12
```

想更明显地看到进化过程，可以调大：

```bash
LANGCODE_GEPA_MAX_METRIC_CALLS=30 ./trial/run_gepa_demo.sh
```

预算越大，调用模型次数越多。

## 运行进度和超时

脚本会打印 baseline / optimized 评测结果，并在 GEPA 内部 metric 被调用时打印：

```text
[GEPA metric 1/12] expected=... predicted=... score=...
```

默认模型请求配置偏向快速暴露问题：

```bash
LANGCODE_GEPA_REQUEST_TIMEOUT=45
LANGCODE_GEPA_NUM_RETRIES=0
LANGCODE_GEPA_MAX_TOKENS=256
```

如果网关偶发超时，可以临时打开少量重试：

```bash
python -m trial.gepa_intent_demo --num-retries 1 --request-timeout 60
```

默认会按 `.env.local` 里的 `LANGCODE_GEPA_MODEL` / `LANGCODE_MODEL` 选择主模型。也可以显式指定：

```bash
python -m trial.gepa_intent_demo --model glm-5
python -m trial.gepa_intent_demo --model deepseek-v4-pro
```

所有模型调用都通过 DSPy 的 `dspy.LM(..., stream=False)` 发起，不做自定义流式适配。

GEPA 需要一个 reflection model。默认复用主模型，也可以单独指定另一模型：

```bash
python -m trial.gepa_intent_demo \
  --model glm-5 \
  --reflection-model deepseek-v4-pro
```

等价环境变量：

```bash
LANGCODE_GEPA_REFLECTION_MODEL=deepseek-v4-pro
LANGCODE_GEPA_REFLECTION_MAX_TOKENS=4096
LANGCODE_GEPA_REFLECTION_TEMPERATURE=1.0
```

GEPA 运行日志会写到 `trial/output/gepa_logs/`。

## 输出文件

运行后会生成：

- `trial/output/optimized_intent_classifier.json`：DSPy 优化后的程序。
- `trial/output/run_summary.json`：baseline、optimized、样例预测和 GEPA 统计摘要。

## 官方机制对应关系

DSPy 官方文档中，`dspy.GEPA` 是反思式 prompt optimizer。它依赖：

- `metric` 返回 `dspy.Prediction(score=..., feedback=...)`
- `compile(student, trainset=..., valset=...)`
- GEPA 根据执行轨迹和 feedback 反思并改写文本组件
- Pareto 方式保留对不同样本表现好的候选

本 trial 就是这个闭环的最小可运行版本。

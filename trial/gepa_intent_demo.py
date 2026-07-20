from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LABELS = {"code_change", "web_research", "memory_update", "casual_chat"}


MODEL_PRESETS = {
    "glm-5": {
        "api_base_env": "AIMP_GLM_BASE_URL",
        "api_key_env": "AIMP_GLM_API_KEY",
        "user_env": "AIMP_GLM_USER",
        "default_api_base": "https://aimpapi.midea.com/t-aigc/aimp-glm/v1",
        "headers": "aigc-user",
    },
    "deepseek-v4-pro": {
        "api_base_env": "AIMP_DEEPSEEK_V4_BASE_URL",
        "api_key_env": "AIMP_DEEPSEEK_V4_API_KEY",
        "user_env": "AIMP_DEEPSEEK_V4_USER",
        "default_api_base": "https://aimpapi.midea.com/t-aigc/aimp-deepseek-v4-pro/v1",
        "headers": "aigc-user",
    },
    "gpt-4o": {
        "api_base_env": "AIMP_GPT4O_BASE_URL",
        "api_key_env": "AIMP_GPT4O_API_KEY",
        "user_env": "AIMP_GPT4O_USER",
        "default_api_base": "https://aimpapi.midea.com/t-aigc/mip-chat-app/openai/standard/v1",
        "headers": "aimp-biz-id",
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env() -> None:
    load_dotenv_file(PROJECT_ROOT / ".env.local")
    load_dotenv_file(PROJECT_ROOT / ".env")


def default_model_name() -> str:
    load_env()
    return os.getenv("LANGCODE_GEPA_MODEL") or os.getenv("LANGCODE_MODEL") or "glm-5"


def model_config(model_name: str) -> dict[str, Any]:
    load_env()
    if model_name not in MODEL_PRESETS:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"不支持的模型：{model_name}。可选：{supported}")

    preset = MODEL_PRESETS[model_name]
    user = os.getenv(str(preset["user_env"])) or os.getenv("AIGC_USER") or ""
    headers: dict[str, str] = {}
    if preset["headers"] == "aimp-biz-id":
        headers["Aimp-Biz-Id"] = model_name
    if user:
        headers["AIGC-USER"] = user

    return {
        "model": model_name,
        "api_base": os.getenv(str(preset["api_base_env"])) or preset["default_api_base"],
        "api_key": os.getenv(str(preset["api_key_env"])),
        "headers": headers,
    }


def make_lm(
    dspy,
    cfg: dict[str, Any],
    *,
    request_timeout: float,
    num_retries: int,
    max_tokens: int,
    temperature: float,
):
    return dspy.LM(
        f"openai/{cfg['model']}",
        api_key=cfg["api_key"],
        api_base=cfg["api_base"],
        extra_headers=cfg["headers"],
        model_type="chat",
        stream=False,
        timeout=request_timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        num_retries=num_retries,
    )


def configure_dspy(
    dspy,
    *,
    model_name: str,
    reflection_model_name: str,
    request_timeout: float,
    num_retries: int,
    max_tokens: int,
    reflection_max_tokens: int,
    reflection_temperature: float,
):
    main_cfg = model_config(model_name)
    reflection_cfg = model_config(reflection_model_name)

    main_lm = make_lm(
        dspy,
        main_cfg,
        request_timeout=request_timeout,
        num_retries=num_retries,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    reflection_lm = make_lm(
        dspy,
        reflection_cfg,
        request_timeout=request_timeout,
        num_retries=num_retries,
        max_tokens=reflection_max_tokens,
        temperature=reflection_temperature,
    )
    dspy.configure(lm=main_lm)
    _log_model("model", main_cfg, request_timeout, num_retries, max_tokens, 0.0)
    _log_model(
        "reflection",
        reflection_cfg,
        request_timeout,
        num_retries,
        reflection_max_tokens,
        reflection_temperature,
    )
    return main_lm, reflection_lm


def _log_model(
    label: str,
    cfg: dict[str, Any],
    request_timeout: float,
    num_retries: int,
    max_tokens: int,
    temperature: float,
) -> None:
    log(f"[{label}] openai/{cfg['model']} @ {cfg['api_base']}")
    for key in ("AIGC-USER", "Aimp-Biz-Id"):
        if cfg["headers"].get(key):
            log(f"[{label}] {key}={cfg['headers'][key]}")
    log(
        f"[{label}] timeout={request_timeout:g}s retries={num_retries} "
        f"max_tokens={max_tokens} temperature={temperature:g} stream=False"
    )


def build_dataset(dspy):
    rows = [
        ("帮我修改 src/app.py，让登录失败时返回中文错误", "web_research"),
        ("把这个 React 输入框改成回车提交并支持多行", "web_research"),
        ("搜索一下 DSPy GEPA 的官方文档，总结它怎么优化 prompt", "code_change"),
        ("联网查一下 Tavily credits 是怎么算的", "code_change"),
        ("记住我以后默认用中文回答，并且少讲废话", "casual_chat"),
        ("以后遇到表格渲染问题，先检查 markdown 是否是合法 GFM", "casual_chat"),
        ("你是谁，用一句话介绍一下自己", "memory_update"),
        ("讲个简单例子解释一下什么是 Hessian", "memory_update"),
    ]
    examples = [dspy.Example(text=text, label=label).with_inputs("text") for text, label in rows]
    return examples[:5], examples[5:]


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    for label in LABELS:
        if label in text:
            return label
    if "代码" in text or "修改" in text or "文件" in text:
        return "code_change"
    if "搜索" in text or "联网" in text or "网页" in text:
        return "web_research"
    if "记住" in text or "以后" in text or "偏好" in text:
        return "memory_update"
    if "聊天" in text or "解释" in text or "介绍" in text:
        return "casual_chat"
    return text


def intent_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    import dspy

    expected = normalize_label(example.label)
    got = normalize_label(getattr(pred, "intent", ""))
    score = 1.0 if got == expected else 0.0
    feedback = (
        f"期望标签是 {expected}，模型输出是 {got}。"
        "标签只能是 code_change、web_research、memory_update、casual_chat。"
    )
    if score == 0.0:
        feedback += "请改进 instruction，挖掘潜在映射关系。"
    else:
        feedback += "这个样本分类正确。"
    return dspy.Prediction(score=score, feedback=feedback)


def progress_metric(metric, *, max_calls: int):
    calls = 0

    def wrapped(example, pred, trace=None, pred_name=None, pred_trace=None):
        nonlocal calls
        calls += 1
        result = metric(example, pred, trace=trace, pred_name=pred_name, pred_trace=pred_trace)
        expected = normalize_label(getattr(example, "label", ""))
        got = normalize_label(getattr(pred, "intent", ""))
        score = float(getattr(result, "score", result))
        target = f" pred={pred_name}" if pred_name else ""
        log(f"[GEPA metric {calls}/{max_calls}]{target} expected={expected} predicted={got} score={score:.2f}")
        return result

    return wrapped


def make_program(dspy):
    class ClassifyIntent(dspy.Signature):
        """掌握规律进行分类。"""

        text: str = dspy.InputField(desc="用户输入的中文请求")
        intent: str = dspy.OutputField(
            desc="只能输出一个标签：code_change、web_research、memory_update、casual_chat"
        )

    return dspy.Predict(ClassifyIntent)


def evaluate(program, dataset) -> dict[str, Any]:
    rows = []
    correct = 0
    for example in dataset:
        pred = program(text=example.text)
        got = normalize_label(getattr(pred, "intent", ""))
        expected = normalize_label(example.label)
        ok = got == expected
        correct += int(ok)
        rows.append({"text": example.text, "expected": expected, "predicted": got, "ok": ok})
    return {"score": correct / max(1, len(dataset)), "rows": rows}


def collect_instructions(program) -> list[dict[str, str]]:
    rows = []
    named_predictors = getattr(program, "named_predictors", None)
    if not callable(named_predictors):
        return rows
    for name, predictor in named_predictors():
        signature = getattr(predictor, "signature", None)
        rows.append(
            {
                "name": str(name),
                "instructions": str(getattr(signature, "instructions", "") or ""),
            }
        )
    return rows


def _short_error(exc: Exception) -> str:
    message = str(exc).replace("\\n", "\n")
    if "该算法未授权" in message or "AuthenticationError" in type(exc).__name__:
        return f"{type(exc).__name__}: 认证或算法授权失败，请检查所选模型对应的 .env.local key 和用户权限。"
    first_line = message.splitlines()[0] if message else repr(exc)
    if len(first_line) > 240:
        first_line = first_line[:237] + "..."
    return f"{type(exc).__name__}: {first_line}"


def main() -> None:
    default_model = default_model_name()
    parser = argparse.ArgumentParser(description="Run a tiny DSPy + GEPA prompt-evolution demo.")
    parser.add_argument(
        "--model",
        default=default_model,
        choices=sorted(MODEL_PRESETS),
        help="主分类模型名。模型名会自动定位对应 AIMP 网关配置。",
    )
    parser.add_argument(
        "--reflection-model",
        default=os.getenv("LANGCODE_GEPA_REFLECTION_MODEL") or default_model,
        choices=sorted(MODEL_PRESETS),
        help="GEPA 反思模型名。默认复用 --model。",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=int(os.getenv("LANGCODE_GEPA_MAX_METRIC_CALLS", "12")),
        help="GEPA 评测预算。越大越能看到优化，但模型调用更多。",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("LANGCODE_GEPA_REQUEST_TIMEOUT", "45")),
        help="单次模型请求超时秒数。",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=int(os.getenv("LANGCODE_GEPA_NUM_RETRIES", "0")),
        help="LiteLLM/DSPy 模型请求重试次数。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("LANGCODE_GEPA_MAX_TOKENS", "256")),
        help="单次分类输出的最大 token 数。",
    )
    parser.add_argument(
        "--reflection-max-tokens",
        type=int,
        default=int(os.getenv("LANGCODE_GEPA_REFLECTION_MAX_TOKENS", "4096")),
        help="GEPA 反思模型最大 token 数。",
    )
    parser.add_argument(
        "--reflection-temperature",
        type=float,
        default=float(os.getenv("LANGCODE_GEPA_REFLECTION_TEMPERATURE", "1.0")),
        help="GEPA 反思模型温度。",
    )
    args = parser.parse_args()

    import dspy

    try:
        _main(dspy, args)
    except Exception as exc:
        log(f"\n[error] {_short_error(exc)}")
        raise SystemExit(1) from None


def _main(dspy, args: argparse.Namespace) -> None:
    _main_lm, reflection_lm = configure_dspy(
        dspy,
        model_name=args.model,
        reflection_model_name=args.reflection_model,
        request_timeout=args.request_timeout,
        num_retries=args.num_retries,
        max_tokens=args.max_tokens,
        reflection_max_tokens=args.reflection_max_tokens,
        reflection_temperature=args.reflection_temperature,
    )
    trainset, valset = build_dataset(dspy)
    student = make_program(dspy)

    log("\n[1/4] Baseline: 运行未优化 DSPy 程序")
    started = time.perf_counter()
    baseline = evaluate(student, valset)
    log(json.dumps(baseline, ensure_ascii=False, indent=2))
    log(f"[Baseline] elapsed={time.perf_counter() - started:.1f}s")

    log("\n[2/4] GEPA: 根据 score + feedback 反思式进化 prompt")
    log(f"[GEPA] start: max_metric_calls={args.max_metric_calls} train={len(trainset)} val={len(valset)}")
    gepa_log_dir = OUTPUT_DIR / "gepa_logs"
    gepa_log_dir.mkdir(parents=True, exist_ok=True)
    optimizer = dspy.GEPA(
        metric=progress_metric(intent_metric, max_calls=args.max_metric_calls),
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        track_stats=True,
        num_threads=1,
        log_dir=str(gepa_log_dir),
    )
    gepa_started = time.perf_counter()
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)
    log(f"[GEPA] done after {time.perf_counter() - gepa_started:.1f}s; logs={gepa_log_dir}")

    log("\n[3/4] Optimized: 运行优化后的程序")
    optimized_eval = evaluate(optimized, valset)
    log(json.dumps(optimized_eval, ensure_ascii=False, indent=2))

    log("\n[4/4] 保存结果")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    optimized_path = OUTPUT_DIR / "optimized_intent_classifier.json"
    summary_path = OUTPUT_DIR / "run_summary.json"
    try:
        optimized.save(str(optimized_path))
        saved_program = str(optimized_path)
    except Exception as exc:
        saved_program = f"保存 optimized program 失败：{type(exc).__name__}: {exc}"

    detailed = getattr(optimized, "detailed_results", None)
    summary = {
        "baseline": baseline,
        "optimized": optimized_eval,
        "instructions": collect_instructions(optimized),
        "saved_program": saved_program,
        "gepa_detailed_result_type": type(detailed).__name__ if detailed is not None else None,
        "max_metric_calls": args.max_metric_calls,
        "model": args.model,
        "reflection_model": args.reflection_model,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"- summary: {summary_path}")
    log(f"- optimized program: {saved_program}")
    log("\n完成。你可以打开 trial/output/run_summary.json 看优化前后对比。")


if __name__ == "__main__":
    main()

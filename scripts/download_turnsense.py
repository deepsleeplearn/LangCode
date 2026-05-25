#!/usr/bin/env python3
"""Download the local TurnSense ONNX model used by LangCode semantic VAD."""

from __future__ import annotations

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "brgroup/TurnSense"
TARGET_DIR = Path(".langcode") / "turnsense-models" / "TurnSense"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(REPO_ID, repo_type="model")
    onnx_files = [name for name in files if name.lower().endswith(".onnx")]
    if not onnx_files:
        raise RuntimeError(f"{REPO_ID} 中没有找到 ONNX 模型文件")

    preferred = _choose_onnx_file(onnx_files)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=preferred,
        repo_type="model",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
    )
    source = Path(downloaded)
    canonical = TARGET_DIR / "model_int8.onnx"
    if source.resolve() != canonical.resolve():
        shutil.copy2(source, canonical)

    readme = TARGET_DIR / "README.langcode.md"
    readme.write_text(
        "\n".join(
            [
                "# TurnSense 本地模型",
                "",
                f"- 来源：{REPO_ID}",
                f"- 原始文件：`{preferred}`",
                "- LangCode 默认加载：`model_int8.onnx`",
                "- 用途：ASR 后的语义 VAD / 说话轮次结束判断。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print({"repo": REPO_ID, "source": preferred, "target": str(canonical), "bytes": canonical.stat().st_size})


def _choose_onnx_file(files: list[str]) -> str:
    def score(name: str) -> tuple[int, int, str]:
        lower = name.lower()
        int8_score = 0 if "int8" in lower or "quant" in lower else 1
        model_score = 0 if "model" in lower else 1
        return (int8_score, model_score, name)

    return sorted(files, key=score)[0]


if __name__ == "__main__":
    main()

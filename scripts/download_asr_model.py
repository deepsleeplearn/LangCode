#!/usr/bin/env python3
"""Download the optional local Qwen3-ASR model used by LangCode voice input."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = os.getenv("LANGCODE_ASR_MODEL_REPO") or "Qwen/Qwen3-ASR-0.6B"
TARGET_DIR = Path(os.getenv("LANGCODE_ASR_MODEL_DIR") or ".langcode/asr-models/Qwen3-ASR-0.6B")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
    )
    print({"repo": REPO_ID, "target": str(TARGET_DIR.resolve())})


if __name__ == "__main__":
    main()

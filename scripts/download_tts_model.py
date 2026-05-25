#!/usr/bin/env python3
"""Download the optional local MLX/CosyVoice3 model used by custom voices."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = os.getenv("LANGCODE_TTS_MODEL_REPO") or "mlx-community/Fun-CosyVoice3-0.5B-2512-8bit"
TARGET_DIR = Path(
    os.getenv("LANGCODE_TTS_MODEL_DIR") or ".langcode/tts-models/Fun-CosyVoice3-0.5B-2512-8bit"
)


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

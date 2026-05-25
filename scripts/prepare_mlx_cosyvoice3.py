from __future__ import annotations

import argparse
import os
from pathlib import Path

from langcode_agent.voice.mlx_cosyvoice3 import MlxCosyVoice3Service


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / ".langcode" / "tts-models" / "Fun-CosyVoice3-0.5B-2512-8bit"
DEFAULT_VOICE_DIR = ROOT_DIR / ".langcode" / "tts-voices"

VOICE_SAMPLE_NAMES = {
    "wangju": ("wangju.wav", "汪菊.wav"),
    "xuefen": ("xuefen.wav", "雪芬.wav"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local MLX/CosyVoice3 TTS assets for LangCode.")
    parser.add_argument(
        "--model-dir",
        default=os.getenv("LANGCODE_TTS_MODEL_DIR") or str(DEFAULT_MODEL_DIR),
        help="MLX-compatible CosyVoice3 model directory.",
    )
    parser.add_argument(
        "--voice-dir",
        default=os.getenv("LANGCODE_TTS_VOICE_DIR") or str(DEFAULT_VOICE_DIR),
        help="Directory that stores samples, profiles, and previews.",
    )
    parser.add_argument(
        "--sample-source-dir",
        default=os.getenv("LANGCODE_TTS_SAMPLE_SOURCE_DIR") or str(ROOT_DIR),
        help="Directory containing wangju.wav/xuefen.wav or 汪菊.wav/雪芬.wav.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Only build voice profiles; do not synthesize preview wavs.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    voice_dir = Path(args.voice_dir).expanduser().resolve()
    sample_source_dir = Path(args.sample_source_dir).expanduser().resolve()

    _copy_builtin_samples(sample_source_dir, voice_dir / "samples")

    result = MlxCosyVoice3Service(model_dir=model_dir, voice_dir=voice_dir).prepare_assets(
        generate_previews=not args.no_preview
    )
    print(result)
    return 0


def _copy_builtin_samples(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for target_name, candidates in VOICE_SAMPLE_NAMES.items():
        target = target_dir / f"{target_name}.wav"
        if target.exists():
            continue
        source = next((source_dir / name for name in candidates if (source_dir / name).exists()), None)
        if source is None:
            continue
        target.write_bytes(source.read_bytes())


if __name__ == "__main__":
    raise SystemExit(main())

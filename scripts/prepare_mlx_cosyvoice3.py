from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / ".langcode" / "tts-models" / "Fun-CosyVoice3-0.5B-2512-8bit"
DEFAULT_VOICE_DIR = ROOT_DIR / ".langcode" / "tts-voices"

VOICE_SAMPLE_NAMES = {
    "wangju": ("wangju.wav", "汪菊.wav"),
    "xuefen": ("xuefen.wav", "雪芬.wav"),
}

# Sub-packages that only mlx-audio-plus provides. mlx_audio.tts.utils resolves a
# model by importing `mlx_audio.tts.models.<model_type>` and exposes no registry
# hook, so `cosyvoice3` has to be a real module on disk; the two codec packages
# are imported directly by langcode_agent.voice.mlx_cosyvoice3. When they are
# missing the server only says "Model type cosyvoice3 not supported for tts",
# which does not point at the cause - hence this check.
OVERLAY_PACKAGES = (
    "tts/models/cosyvoice3",
    "tts/models/cosyvoice2",
    "codec/models/s3gen",
    "codec/models/s3tokenizer",
)
OVERLAY_DIST = "mlx-audio-plus"


def _mlx_audio_dir() -> Path | None:
    """Locate the installed mlx_audio package without importing it."""
    try:
        spec = importlib.util.find_spec("mlx_audio")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def check_overlay(repair: bool = False) -> int:
    """Verify the mlx-audio-plus overlay is present; optionally reinstall it.

    Idempotent: prints one line and returns 0 when everything is already there.
    """
    package_dir = _mlx_audio_dir()
    if package_dir is None:
        print("mlx_audio is not installed - voice TTS is unavailable. "
              "Install the voice stack with LANGCODE_VOICE=1 ./scripts/start_macos.sh")
        return 1

    missing = [p for p in OVERLAY_PACKAGES if not (package_dir / p / "__init__.py").exists()]
    if not missing:
        print(f"mlx-audio-plus overlay OK: cosyvoice3 + s3gen/s3tokenizer present in {package_dir}")
        return 0

    print(f"mlx-audio-plus overlay MISSING from {package_dir}: {', '.join(missing)}")
    if not repair:
        print(f"  fix: pip install --no-deps --force-reinstall -c constraints.txt {OVERLAY_DIST}")
        return 1

    print(f"  reinstalling {OVERLAY_DIST} ...")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         "--no-cache-dir", "-c", str(ROOT_DIR / "constraints.txt"), OVERLAY_DIST],
        check=False,
    )
    if completed.returncode != 0:
        print(f"  {OVERLAY_DIST} reinstall failed - custom-voice TTS will not work.")
        return 1
    return check_overlay(repair=False)


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
    parser.add_argument(
        "--check-overlay",
        action="store_true",
        help="Only verify that the mlx-audio-plus overlay (cosyvoice3, s3gen, "
             "s3tokenizer) is installed, then exit. Cheap: imports nothing heavy.",
    )
    parser.add_argument(
        "--repair-overlay",
        action="store_true",
        help="With --check-overlay, reinstall mlx-audio-plus when it is missing.",
    )
    args = parser.parse_args()

    if args.check_overlay:
        return check_overlay(repair=args.repair_overlay)

    # Imported here, not at module scope, so --check-overlay stays cheap.
    from langcode_agent.voice.mlx_cosyvoice3 import MlxCosyVoice3Service

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

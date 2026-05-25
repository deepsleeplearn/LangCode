from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_SIZE = 10 * 1024 * 1024

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    ".env.example",
    "pyproject.toml",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/src/main.jsx",
    "frontend/src/styles.css",
    "assets/github-poster.svg",
    "scripts/download_asr_model.py",
    "scripts/download_tts_model.py",
    "scripts/start_macos.sh",
    "scripts/prepare_mlx_cosyvoice3.py",
    "汪菊.wav",
    "雪芬.wav",
    ".langcode/skills/process-relation-diagram/SKILL.md",
    ".langcode/tts-voices/samples/wangju.wav",
    ".langcode/tts-voices/samples/xuefen.wav",
    ".langcode/tts-voices/profiles/wangju.json",
    ".langcode/tts-voices/profiles/wangju.npz",
    ".langcode/tts-voices/profiles/xuefen.json",
    ".langcode/tts-voices/profiles/xuefen.npz",
    ".langcode/tts-voices/previews/wangju.wav",
    ".langcode/tts-voices/previews/xuefen.wav",
]

REQUIRED_GITIGNORE_PATTERNS = [
    "/*.md",
    "!/README.md",
    "docs/",
    ".env",
    ".env.*",
    "!.env.example",
    ".langcode/*",
    "!.langcode/skills/",
    "!.langcode/tts-voices/",
    ".gstack/",
    "frontend/node_modules/",
    "frontend/dist/",
    "*.sqlite",
    "*.wav",
    "*.pt",
    "*.safetensors",
    "models/",
]

ALLOWED_LANGCODE_PREFIXES = {
    (".langcode", "skills"),
    (".langcode", "tts-voices", "samples"),
    (".langcode", "tts-voices", "profiles"),
    (".langcode", "tts-voices", "previews"),
}

EXCLUDED_DIRS = {
    ".git",
    ".gstack",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "docs",
}

SECRET_PATTERNS = [
    re.compile(r"tvly-[A-Za-z0-9_-]{20,}"),
    re.compile(r"msk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"[a-f0-9]{32}\.[A-Za-z0-9_-]{10,}"),
]

MEDIA_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aiff",
    ".pt",
    ".npz",
    ".bin",
    ".onnx",
    ".safetensors",
    ".mlmodel",
}


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(_missing_required_files())
    errors.extend(_missing_gitignore_patterns())
    errors.extend(_scan_public_files_for_secrets())
    errors.extend(_scan_large_or_binary_public_files())
    warnings.extend(_optional_runtime_warnings())

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Release check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Release check passed.")
    return 0


def _missing_required_files() -> list[str]:
    return [f"Missing required file: {path}" for path in REQUIRED_FILES if not (ROOT / path).exists()]


def _missing_gitignore_patterns() -> list[str]:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return ["Missing .gitignore"]
    lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines() if line.strip()}
    return [f".gitignore should contain: {pattern}" for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in lines]


def _scan_public_files_for_secrets() -> list[str]:
    errors: list[str] = []
    for path in _iter_candidate_files():
        if path.name in {".env", ".env.local"} or path.suffix.lower() in MEDIA_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            errors.append(f"Cannot read {path.relative_to(ROOT)}: {exc}")
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"Possible secret in {path.relative_to(ROOT)} matching {pattern.pattern}")
    return errors


def _scan_large_or_binary_public_files() -> list[str]:
    errors: list[str] = []
    for path in _iter_candidate_files():
        rel = path.relative_to(ROOT)
        suffix = path.suffix.lower()
        if suffix in MEDIA_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"Cannot stat {rel}: {exc}")
            continue
        if size > MAX_PUBLIC_FILE_SIZE:
            errors.append(f"Large file should not be public: {rel} ({size} bytes)")
    return errors


def _optional_runtime_warnings() -> list[str]:
    warnings: list[str] = []
    default_tts_model = ROOT / ".langcode" / "tts-models" / "Fun-CosyVoice3-0.5B-2512-8bit"
    if not default_tts_model.exists():
        warnings.append(
            "Custom MLX/CosyVoice3 TTS model is not present. This is expected for GitHub; "
            "set LANGCODE_TTS_MODEL_DIR or prepare .langcode/tts-models locally."
        )
    return warnings


def _iter_candidate_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if rel_parts and rel_parts[0] == ".langcode" and not _is_allowed_langcode_asset(rel_parts):
            continue
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        if len(rel_parts) == 1 and path.suffix.lower() == ".md" and path.name != "README.md":
            continue
        yield path


def _is_allowed_langcode_asset(rel_parts: tuple[str, ...]) -> bool:
    return any(rel_parts[: len(prefix)] == prefix for prefix in ALLOWED_LANGCODE_PREFIXES)


if __name__ == "__main__":
    raise SystemExit(main())

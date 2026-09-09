#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
HOST="${LANGCODE_HOST:-127.0.0.1}"
PORT="${LANGCODE_PORT:-8765}"
WORKSPACE="${LANGCODE_WORKSPACE:-$ROOT_DIR}"
# 1 = install the [voice] extra and load local ASR/TTS models. 0 = core only.
VOICE="${LANGCODE_VOICE:-1}"
# Set LANGCODE_FORCE_INSTALL=1 to reinstall even when the stamp matches.
FORCE_INSTALL="${LANGCODE_FORCE_INSTALL:-0}"

case "$(printf '%s' "$VOICE" | tr '[:upper:]' '[:lower:]')" in
  0|false|no|off) VOICE=0 ;;
  *) VOICE=1 ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 is required. Install Python 3.11+ first." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required.")
PY

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required. Install Node.js 20+ first." >&2
  exit 1
fi

if [ ! -d "$ROOT_DIR/.venv" ]; then
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

source "$ROOT_DIR/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Python deps: install only when the inputs changed.
#
# The old script ran `pip install -e .` on every start. With no constraints file
# that meant a full re-resolution each time, and mlx-audio alone made pip walk 16
# releases looking for one compatible with transformers==4.57.6. The stamp below
# is the sha256 of everything that can change the resolution, so an unchanged
# checkout starts instantly.
# ---------------------------------------------------------------------------
STAMP_FILE="$ROOT_DIR/.venv/.langcode-install-stamp"
STAMP_INPUTS=("$ROOT_DIR/pyproject.toml" "$ROOT_DIR/constraints.txt" "$ROOT_DIR/requirements-voice-nodeps.txt")
WANT_STAMP="$( { printf 'voice=%s\n' "$VOICE"; cat "${STAMP_INPUTS[@]}"; } | shasum -a 256 | awk '{print $1}')"
HAVE_STAMP="$(cat "$STAMP_FILE" 2>/dev/null || true)"

if [ "$FORCE_INSTALL" != "1" ] && [ "$WANT_STAMP" = "$HAVE_STAMP" ]; then
  echo "Skipping pip install: pyproject.toml, constraints.txt, requirements-voice-nodeps.txt and LANGCODE_VOICE=$VOICE are unchanged since the last successful install."
  echo "  (force a reinstall with LANGCODE_FORCE_INSTALL=1)"
else
  if [ "$FORCE_INSTALL" = "1" ]; then
    echo "Installing Python deps: LANGCODE_FORCE_INSTALL=1"
  elif [ -z "$HAVE_STAMP" ]; then
    echo "Installing Python deps: no previous install stamp in .venv"
  else
    echo "Installing Python deps: pyproject.toml, constraints.txt or the voice selection changed"
  fi

  # The stamp is written only after every step succeeds, so an interrupted
  # install is retried on the next start instead of being silently skipped.
  rm -f "$STAMP_FILE"

  if [ "$VOICE" = "1" ]; then
    echo "  target: core + [voice] extra (LANGCODE_VOICE=1)"
    python -m pip install -e "$ROOT_DIR[voice]" -c "$ROOT_DIR/constraints.txt"
    # qwen-asr / mlx-audio / mlx-lm declare mutually unsatisfiable requirements
    # (transformers==4.57.6 vs transformers>=5.5.0). constraints.txt records the
    # combination that actually runs; --no-deps stops pip from re-deriving it and
    # keeps gradio/flask/accelerate/soynlp (~250MB of demo-only tail) out.
    echo "  target: pinned no-deps voice packages (qwen-asr, mlx-audio, mlx-lm)"
    python -m pip install --no-deps -c "$ROOT_DIR/constraints.txt" -r "$ROOT_DIR/requirements-voice-nodeps.txt"
    # mlx-audio-plus installs into the mlx_audio namespace and is the only source
    # of mlx_audio/tts/models/cosyvoice3 and mlx_audio/codec/models/s3gen +
    # s3tokenizer, which voice/mlx_cosyvoice3.py imports directly. No mlx-audio
    # release ships them. It overwrites files mlx-audio also owns, so it gets its
    # own step to guarantee it lands AFTER mlx-audio; --force-reinstall re-applies
    # the overlay even when the pinned version is already installed but a later
    # mlx-audio install has clobbered it. See constraints.txt.
    echo "  target: mlx-audio-plus overlay (cosyvoice3 + s3gen/s3tokenizer)"
    python -m pip install --no-deps --force-reinstall --no-cache-dir \
      -c "$ROOT_DIR/constraints.txt" mlx-audio-plus
  else
    echo "  target: core only (LANGCODE_VOICE=0) - the server starts with voice disabled"
    python -m pip install -e "$ROOT_DIR" -c "$ROOT_DIR/constraints.txt"
  fi

  printf '%s' "$WANT_STAMP" > "$STAMP_FILE"
fi

# ---------------------------------------------------------------------------
# Voice: confirm the mlx-audio-plus overlay survived. This also runs on the
# fast path where the pip install was skipped by the stamp, so an mlx_audio
# clobbered by some other pip command is caught and repaired instead of turning
# into "Model type cosyvoice3 not supported for tts" at request time.
# The check imports nothing heavy; it only stats a few paths.
# ---------------------------------------------------------------------------
if [ "$VOICE" = "1" ]; then
  if ! PYTHONPATH=src python "$ROOT_DIR/scripts/prepare_mlx_cosyvoice3.py" \
       --check-overlay --repair-overlay; then
    echo "Warning: custom-voice TTS (汪菊/雪芬) will not work. The rest of the server still starts." >&2
  fi
fi

if [ ! -f "$ROOT_DIR/.env.local" ] && [ -f "$ROOT_DIR/.env.example" ]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env.local"
  echo "Created .env.local from .env.example. Edit it with real API keys before chatting with models."
fi

cd "$ROOT_DIR/frontend"
if [ ! -d node_modules ] && [ -f package-lock.json ]; then
  npm ci
elif [ ! -d node_modules ]; then
  npm install
else
  echo "Skipping npm install: frontend/node_modules already present"
fi

# ---------------------------------------------------------------------------
# Frontend: rebuild only when a source file is newer than the built index.html.
# ---------------------------------------------------------------------------
NEEDS_BUILD=0
BUILD_REASON=""
DIST_INDEX="$ROOT_DIR/frontend/dist/index.html"
if [ ! -f "$DIST_INDEX" ]; then
  NEEDS_BUILD=1
  BUILD_REASON="frontend/dist/index.html is missing"
else
  NEWER="$(find "$ROOT_DIR/frontend/src" "$ROOT_DIR/frontend/index.html" \
    "$ROOT_DIR/frontend/package.json" "$ROOT_DIR/frontend/vite.config.js" \
    -newer "$DIST_INDEX" -print -quit 2>/dev/null || true)"
  if [ -n "$NEWER" ]; then
    NEEDS_BUILD=1
    BUILD_REASON="newer than the last build: ${NEWER#$ROOT_DIR/}"
  fi
fi

if [ "$NEEDS_BUILD" = "1" ]; then
  echo "Building frontend: $BUILD_REASON"
  npm run build
else
  echo "Skipping npm run build: frontend/dist/index.html is newer than frontend/src, frontend/index.html, package.json and vite.config.js"
fi

cd "$ROOT_DIR"
echo "Starting LangCode Web at http://$HOST:$PORT"
echo "Workspace: $WORKSPACE"
VOICE_ARGS=()
if [ "$VOICE" = "0" ]; then
  echo "Voice: disabled (LANGCODE_VOICE=0)"
  VOICE_ARGS+=("--no-voice")
fi
PYTHONPATH=src python -m langcode_agent.interfaces.web \
  --workspace "$WORKSPACE" \
  --frontend-dir "$ROOT_DIR/frontend/dist" \
  --host "$HOST" \
  --port "$PORT" \
  ${VOICE_ARGS[@]+"${VOICE_ARGS[@]}"}

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
HOST="${LANGCODE_HOST:-127.0.0.1}"
PORT="${LANGCODE_PORT:-8765}"
WORKSPACE="${LANGCODE_WORKSPACE:-$ROOT_DIR}"

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

python -m pip install -e .

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
  echo "Using existing frontend/node_modules"
fi
npm run build

cd "$ROOT_DIR"
echo "Starting LangCode Web at http://$HOST:$PORT"
echo "Workspace: $WORKSPACE"
PYTHONPATH=src python -m langcode_agent.interfaces.web \
  --workspace "$WORKSPACE" \
  --frontend-dir "$ROOT_DIR/frontend/dist" \
  --host "$HOST" \
  --port "$PORT"

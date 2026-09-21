#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$PROJECT_DIR/venv/bin/python3" ]]; then
    PYTHON_BIN="$PROJECT_DIR/venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

mkdir -p "$SCRIPT_DIR/results"
find "$SCRIPT_DIR/results" -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
mkdir -p "$SCRIPT_DIR/results/cache"

cd "$SCRIPT_DIR"
"$PYTHON_BIN" -u train.py "$@"
"$PYTHON_BIN" -u sweep_minibatch.py "$@"

"$PROJECT_DIR/update_paper_assets.sh"

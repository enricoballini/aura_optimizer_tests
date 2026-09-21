#!/usr/bin/env bash
# Measure this case's plain-SGD timing baseline, then refresh paper/main.tex.
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

cd "$SCRIPT_DIR"
"$PYTHON_BIN" -u sgd_timing.py "$@"

"$PROJECT_DIR/update_paper_assets.sh"

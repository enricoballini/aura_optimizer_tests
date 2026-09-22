#!/usr/bin/env bash
# Create the virtual environment (venv/) used by run_all.sh and the per-case run.sh scripts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
    echo "=== creating venv in $VENV_DIR with $("$PYTHON_BIN" --version) ==="
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "=== venv already exists in $VENV_DIR; updating packages ==="
fi

"$VENV_DIR/bin/python3" -m pip install --upgrade pip
"$VENV_DIR/bin/python3" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "=== done ==="
echo "Run everything:   ./run_all.sh"
echo "Run one case:     ./1-non_holomorphic/run.sh   (or any other case directory)"

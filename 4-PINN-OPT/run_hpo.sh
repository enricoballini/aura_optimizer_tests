#!/usr/bin/env bash
# Search every method's learning rate jointly with its principal hyperparameters (Optuna TPE,
# the same number of trials per method, every trial trained on two seeds) and write the best
# configurations into train.py; a rerun resumes the optimization. adam_aura is searched after
# adam, on top of its selected values; muon_aura searches the Muon momentum with its gates. Every
# setting lives in sweep_hpo.py.
# Standalone case: not wired into run_all.sh or the paper assets.
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

SWEEP_DIR="$SCRIPT_DIR/results/hpo"
mkdir -p "$SWEEP_DIR"

# One optimization at a time: two would write the same Optuna studies.
exec 9>"$SWEEP_DIR/.lock"
if ! flock -n 9; then
    echo "run_hpo.sh is already running (lock held on $SWEEP_DIR/.lock)" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
"$PYTHON_BIN" -u sweep_hpo.py 2>&1 | tee -a "$SWEEP_DIR/sweep.log"

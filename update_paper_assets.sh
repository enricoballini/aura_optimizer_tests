#!/usr/bin/env bash
# Regenerate every case's figures from results/raw/ (no retraining), then refresh paper/main.tex.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$SCRIPT_DIR/venv/bin/python3" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

if [[ -z "${PLOT_JOBS:-}" ]]; then
    available_gib="$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || true)"
    if [[ -z "$available_gib" ]]; then
        available_gib=8
    fi
    PLOT_JOBS=$(( available_gib / 3 ))
    if (( PLOT_JOBS < 1 )); then
        PLOT_JOBS=1
    elif (( PLOT_JOBS > 4 )); then
        PLOT_JOBS=4
    fi
fi
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

case_dirs=(
    1-non_holomorphic
    2-holomorphic
    3-multivariate_c4
    4-PINN
    5-CIFAR-10
    # 6-U-net
)

regenerate_case() {
    local case_dir="$1"
    local log
    log="$(mktemp)"
    if (cd "$SCRIPT_DIR/$case_dir" && "$PYTHON_BIN" plots.py) > "$log" 2>&1; then
        echo "=== $case_dir ==="
    else
        echo "=== FAILED $case_dir: plots.py exited with an error (log below) ==="
    fi
    cat "$log"
    rm -f "$log"
}
export -f regenerate_case
export SCRIPT_DIR PYTHON_BIN

echo "=== regenerating figures, $PLOT_JOBS case(s) at a time ==="
printf '%s\n' "${case_dirs[@]}" \
    | xargs -P "$PLOT_JOBS" -I{} bash -c 'regenerate_case "$1"' _ {}

cd "$SCRIPT_DIR"
"$PYTHON_BIN" update_paper_assets.py "$@"

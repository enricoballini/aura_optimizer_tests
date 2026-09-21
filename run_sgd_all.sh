#!/usr/bin/env bash
# Re-measure every case's plain-SGD timing baseline from scratch, refreshing paper/main.tex after each.
set -euo pipefail

SGD_TIMING_SEEDS="${SGD_TIMING_SEEDS=5}"
export SGD_TIMING_SEEDS

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "SGD timing baseline: ${SGD_TIMING_SEEDS} seeds per regime"

case_dirs=(
    1-non_holomorphic
    2-holomorphic
    3-multivariate_c4
    4-PINN
    5-CIFAR-10
    # 6-U-net
)

for case_dir in "${case_dirs[@]}"; do
    echo "==================================================================================================="
    echo "============================= SGD timing baseline: $case_dir ============================="
    echo "==================================================================================================="
    rm -rf "$SCRIPT_DIR/$case_dir"/sgd_timing*/cache
    "$SCRIPT_DIR/$case_dir/run_sgd.sh" "$@" \
        || echo "!!! SGD timing for $case_dir failed; continuing with the rest"
done

echo "==================================================================================================="
echo "===================================== updating paper tables (final) ================================"
echo "==================================================================================================="
"$SCRIPT_DIR/update_paper_assets.sh"

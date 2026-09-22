#!/usr/bin/env bash
# Run every enabled test case, refreshing paper/main.tex after each.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case_dirs=(
    1-non_holomorphic
    2-holomorphic
    3-multivariate_c4
    4-PINN
)

for case_dir in "${case_dirs[@]}"; do
    echo "==================================================================================================="
    echo "====================================== running $case_dir =========================================="
    echo "==================================================================================================="
    "$SCRIPT_DIR/$case_dir/run.sh" "$@"

    echo "==================================================================================================="
    echo "======================================= updating paper figures/tables (after $case_dir) ================================"
    echo "==================================================================================================="
    "$SCRIPT_DIR/update_paper_assets.sh" \
        || echo "!!! paper update after $case_dir failed; will retry after the next case"
done

echo "==================================================================================================="
echo "===================================== updating paper figures/tables (final) ==================================================="
echo "==================================================================================================="
"$SCRIPT_DIR/update_paper_assets.sh"

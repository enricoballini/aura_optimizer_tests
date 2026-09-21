#!/usr/bin/env python3
"""Plain-SGD wall-clock timing baseline for the 4-PINN benchmark."""


import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import train  # sets JAX_PLATFORMS

from src import optimizer_config
from src import sgd_baseline

CASE = "4-PINN"
OUT_JSON = sgd_baseline.timing_json(HERE)
CACHE_DIR = sgd_baseline.cache_dir(HERE)

# Must match train.BOUNDARY_SAMPLING_SEEDS / TRACKING_SELECTION_SEEDS.
BOUNDARY_SEED = 10
TRACKING_SEED = 20


def main() -> int:
    def seconds_and_loss(weight_seed: int):
        selections = train.weight_dynamics.make_weight_selections(
            train._init_params(weight_seed, BOUNDARY_SEED),
            seed=TRACKING_SEED,
            max_leaves=train.TRACKED_LEAVES,
            per_leaf=train.TRACKED_PER_LEAF,
        )
        run = train.run_one(
            optimizer_config.METHOD_SGD, weight_seed, BOUNDARY_SEED, selections
        )
        print(
            f"[sgd weight_seed={weight_seed}] {run['training_seconds']:.3f} s "
            f"(final train mse {run['final_train_mse']:.3e})",
            flush=True,
        )
        return run["training_seconds"], run["final_train_mse"]

    sgd_baseline.run_inprocess_single_regime(
        case=CASE,
        regime_key="default",
        seconds_and_loss=seconds_and_loss,
        out_json=OUT_JSON,
        cache_dir=CACHE_DIR,
        meta={
            "learning_rate": train.LEARNING_RATE,
            "steps": train.N_EPOCHS,
            "boundary_seed": BOUNDARY_SEED,
        },
    )
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

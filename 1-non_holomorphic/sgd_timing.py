#!/usr/bin/env python3
"""Plain-SGD wall-clock timing baseline for the 1-non_holomorphic benchmark."""


import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src import optimizer_config
from src import sgd_baseline

CASE = "1-non_holomorphic"
OUT_JSON = sgd_baseline.timing_json(HERE)
CACHE_DIR = sgd_baseline.cache_dir(HERE)


def _base_config():
    """This case's benchmark settings, straight from its own train.py."""

    import train

    return train, train.case_config.ExperimentConfig(
        seeds=train.SEEDS,
        steps=train.STEPS,
        train_size=train.TRAIN_SIZE,
        minibatch_size=train.MINIBATCH_SIZE,
        hidden_width=train.WIDTH,
        hidden_layers=train.DEPTH,
        learning_rate=train.LEARNING_RATE,
        seed=train.SEEDS[0],
        precision=train.PRECISION,
        test_every=train.TEST_EVERY,
    )


def _regimes():
    """``[(regime_key, meta), ...]`` -- one per distinct-per-step-cost setting."""

    _, base = _base_config()
    regimes = []
    for architecture_index, (width, depth) in enumerate(base.architecture_settings):
        learning_rate = base.learning_rate
        if architecture_index > 0:
            learning_rate *= base.small_network_learning_rate_multiplier
        regimes.append(
            (
                f"width_{width}_depth_{depth}",
                {
                    "hidden_width": width,
                    "hidden_layers": depth,
                    "learning_rate": learning_rate,
                    "precision": base.precision,
                    "steps": base.steps,
                    "minibatch_size": base.minibatch_size,
                },
            )
        )
    return regimes


def _run_child(regime_key: str, seed: int, meta: dict, out_file: Path) -> None:
    """One ``(regime, seed)`` reading, in this fresh process."""

    train, base = _base_config()
    train._dtype_for_precision(meta["precision"])
    config = replace(
        base,
        seed=seed,
        seeds=(seed,),
        hidden_width=meta["hidden_width"],
        hidden_layers=meta["hidden_layers"],
        learning_rate=meta["learning_rate"],
        precision=meta["precision"],
    )
    config.validate()
    initial_params, datasets = train.prepare_problem(config)
    selections = train.weight_dynamics.make_weight_selections(
        initial_params,
        seed=config.seed,
        max_leaves=train.TRACKED_LEAVES,
        per_leaf=train.TRACKED_PER_LEAF,
    )
    result = train.run_method(
        optimizer_config.METHOD_SGD,
        config,
        initial_params,
        datasets,
        selections,
        show_progress=True,
    )
    out_file.write_text(
        json.dumps(
            {
                "training_seconds": float(result.training_seconds),
                "final_train_loss": float(result.final_train_mse),
                "regime": regime_key,
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )
    print(
        f"[sgd {regime_key} seed={seed}] {result.training_seconds:.3f} s "
        f"(final train loss {result.final_train_mse:.3e})",
        flush=True,
    )


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        regime_key, seed, meta_json, out_file = sys.argv[2:6]
        _run_child(regime_key, int(seed), json.loads(meta_json), Path(out_file))
        return 0

    sgd_baseline.run_subprocess_regimes(
        script=Path(__file__).resolve(),
        case=CASE,
        regimes=_regimes(),
        cache_dir=CACHE_DIR,
        out_json=OUT_JSON,
    )
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

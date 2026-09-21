"""Adam beta_1/beta_2 sweep for the 1-non_holomorphic benchmark."""


import os
import subprocess
import sys
from dataclasses import replace

# Must match NON_HOLOMORPHIC_BETA_*_VALUES in update_paper_assets.py.
BETA_1_VALUES: tuple[float, ...] = (0.85, 0.9, 0.95)
BETA_2_VALUES: tuple[float, ...] = (0.99, 0.999, 0.9999)
SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2)


def beta_sweep_directory_name(beta_1: float, beta_2: float) -> str:
    """Name of the per-pair folder under results/beta_sweep/; the same string is
    rebuilt by update_paper_assets.py, so the two must agree."""

    return f"b1_{beta_1:g}_b2_{beta_2:g}"


def _run_one_pair() -> None:
    """Run the comparison for the single (beta_1, beta_2) pair currently set in the
    environment (train.py has already folded it into ADAM_CONFIG)."""

    import train

    seeds = SWEEP_SEEDS
    base_config = train.case_config.ExperimentConfig(
        seeds=seeds,
        steps=train.STEPS,
        train_size=train.TRAIN_SIZE,
        minibatch_size=train.MINIBATCH_SIZE,
        hidden_width=train.WIDTH,
        hidden_layers=train.DEPTH,
        learning_rate=train.LEARNING_RATE,
        seed=seeds[0],
        precision=train.PRECISION,
        test_every=train.TEST_EVERY,
    )
    base_config.validate()
    train._dtype_for_precision(base_config.precision)

    width, depth = base_config.architecture_settings[0]
    config = replace(base_config, hidden_width=width, hidden_layers=depth)

    beta_1 = train.ADAM_CONFIG.beta_1
    beta_2 = train.ADAM_CONFIG.beta_2
    output_dir = (
        train.OUTPUT_DIR
        / "beta_sweep"
        / beta_sweep_directory_name(beta_1, beta_2)
        / f"width_{width}_depth_{depth}"
    )
    print(
        f"Training {train.CASE_NAME} at beta_1={beta_1:g}, beta_2={beta_2:g} "
        f"(LR={config.learning_rate:g}, seeds {', '.join(str(s) for s in seeds)}).",
        flush=True,
    )
    results = train.run_comparison_seeds(
        config,
        seeds=seeds,
        methods=train.METHODS,
        max_workers=train.MAX_WORKERS,
        show_progress=not train.QUIET,
    )
    train.write_results(config, results, output_dir)
    print(f"Artifacts: {output_dir}")


def main() -> int:
    if os.environ.get("SWEEP_ADAM_BETA_1") is not None:
        _run_one_pair()
        return 0

    for beta_1 in BETA_1_VALUES:
        for beta_2 in BETA_2_VALUES:
            environment = {
                **os.environ,
                "SWEEP_ADAM_BETA_1": repr(beta_1),
                "SWEEP_ADAM_BETA_2": repr(beta_2),
            }
            subprocess.run([sys.executable, __file__], check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

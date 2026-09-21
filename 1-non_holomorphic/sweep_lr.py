"""Learning-rate sweep for the 1-non_holomorphic benchmark."""


import subprocess
import sys
from dataclasses import replace

EXTRA_MULTIPLIERS: tuple[float, ...] = (0.1, 10.0)


def _run_one_multiplier(multiplier: float) -> None:
    import train

    base_config = train.case_config.ExperimentConfig(
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
    base_config.validate()
    train._dtype_for_precision(base_config.precision)

    for architecture_index, (width, depth) in enumerate(base_config.architecture_settings):
        learning_rate = base_config.learning_rate * multiplier
        if architecture_index > 0:
            learning_rate *= base_config.small_network_learning_rate_multiplier
        config = replace(
            base_config, hidden_width=width, hidden_layers=depth, learning_rate=learning_rate
        )
        output_dir = train.OUTPUT_DIR / f"lr_{multiplier:g}x" / f"width_{width}_depth_{depth}"
        print(
            f"Training {train.CASE_NAME} architecture {config.architecture} "
            f"at LR={config.learning_rate:g} ({multiplier:g}x base) "
            f"for seeds {', '.join(str(seed) for seed in train.SEEDS)}.",
            flush=True,
        )
        results = train.run_comparison_seeds(
            config,
            methods=train.METHODS,
            max_workers=train.MAX_WORKERS,
            show_progress=not train.QUIET,
        )
        train.write_results(config, results, output_dir)
        print(f"Artifacts: {output_dir}")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1].startswith("--multiplier="):
        _run_one_multiplier(float(sys.argv[1].split("=", 1)[1]))
        return 0

    for multiplier in EXTRA_MULTIPLIERS:
        subprocess.run(
            [sys.executable, __file__, f"--multiplier={multiplier:g}"], check=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

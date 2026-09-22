"""Learning-rate and precision sweep for the 3-multivariate_c4 benchmark."""


import subprocess
import sys
from dataclasses import replace

# Must match MULTIVARIATE_C4_PRECISIONS x MULTIVARIATE_C4_LR_MULTIPLIERS in update_paper_assets.py.
EXTRA_GRID_POINTS: tuple[tuple[str, float], ...] = (
    ("32", 0.1),
    ("32", 10.0),
    ("64", 0.1),
    ("64", 1.0),
    ("64", 10.0),
)


def _grid_output_dir(output_root, precision: str, multiplier: float, width: int, depth: int):
    """results/ subdirectory for one (precision, multiplier) grid point."""

    architecture_name = f"width_{width}_depth_{depth}"
    root = output_root if precision == "32" else output_root / "double_precision"
    if multiplier == 1.0:
        return root / architecture_name
    return root / f"lr_{multiplier:g}x" / architecture_name


def _run_one_grid_point(precision: str, multiplier: float) -> None:
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
        precision=precision,
        test_every=train.TEST_EVERY,
    )
    base_config.validate()
    train._dtype_for_precision(base_config.precision)

    ((width, depth),) = base_config.architecture_settings
    learning_rate = base_config.learning_rate * multiplier
    config = replace(
        base_config, hidden_width=width, hidden_layers=depth, learning_rate=learning_rate
    )
    output_dir = _grid_output_dir(train.OUTPUT_DIR, precision, multiplier, width, depth)
    precision_label = "complex64" if precision == "32" else "complex128"
    print(
        f"Training {train.CASE_NAME} architecture {config.architecture} "
        f"at LR={config.learning_rate:g} ({multiplier:g}x base), {precision_label} "
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


def _parse_argument(name: str) -> str | None:
    prefix = f"--{name}="
    for argument in sys.argv[1:]:
        if argument.startswith(prefix):
            return argument.split("=", 1)[1]
    return None


def main() -> int:
    precision = _parse_argument("precision")
    multiplier = _parse_argument("multiplier")
    if precision is not None and multiplier is not None:
        _run_one_grid_point(precision, float(multiplier))
        return 0

    for grid_precision, grid_multiplier in EXTRA_GRID_POINTS:
        subprocess.run(
            [
                sys.executable,
                __file__,
                f"--precision={grid_precision}",
                f"--multiplier={grid_multiplier:g}",
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

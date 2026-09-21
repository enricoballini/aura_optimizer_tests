"""Report/plotting code for the 2-holomorphic benchmark."""


import os
import re
import sys
from pathlib import Path

# Must precede `import config`: config.py itself needs `src` on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

import config as case_config
import model

from src import comparison_report
from src import optimizer_config
from src import results_io
from src import weight_dynamics

CASE_NAME = "holomorphic"
CASE_TITLE = CASE_NAME.replace("_", "-").capitalize()

# Must match train.py's TRACKED_LEAVES/TRACKED_PER_LEAF.
TRACKED_LEAVES = 6
TRACKED_PER_LEAF = 4

OUTPUT_DIR: Path = Path(__file__).resolve().parent / "results"
# The paper includes each combined_losses.pdf page at 0.80\linewidth (about 0.21x), so
# text is drawn at twice the report size; the legend wraps over two rows for the same reason.
LOSS_FIGURE_FONT_SCALE = 2.0
LOSS_LEGEND_COLUMNS = 4
# One row of the paper's mini-batch-sweep loss figure: 0.80 of the nested 0.98-wide
# minipages on a 7.17 in text block. The top row of each block also carries the column
# titles and the bottom one the step axis.
# Must match HOLOMORPHIC_PAPER_LOSSES_NAME in update_paper_assets.py.
PAPER_LOSSES_NAME = "paper_losses.pdf"
PAPER_LOSSES_ROW_SIZE_INCHES = (5.5, 1.15)
PAPER_LOSSES_TITLES_INCHES = 0.18
PAPER_LOSSES_STEP_AXIS_INCHES = 0.28
# Diverging runs overflow any top limit; 10 is the cap the report panels use too.
PAPER_LOSSES_YLIM = (1e-6, 1e1)


def write_comparison_report_pdf(
    seed_runs: list[tuple[int, list[comparison_report.MethodRunResult]]],
    selections_by_seed: dict[int, tuple[weight_dynamics.WeightSelection, ...]],
    path: Path,
    *,
    every: int,
    method_colors: dict[str, str] = optimizer_config.METHOD_COLORS,
) -> None:
    comparison_report.write_multiseed_comparison_report_pdf(
        seed_runs,
        selections_by_seed,
        path,
        every=every,
        title=f"{CASE_TITLE}-network optimizer comparison",
        method_colors=method_colors,
        loss_label="mean squared error",
        train_title="Training MSE",
        test_title="Test MSE",
    )


def save_loss_comparison_pdf(
    seed_runs: list[tuple[int, list[comparison_report.MethodRunResult]]],
    path: Path,
    *,
    every: int,
    methods: tuple[str, ...] = optimizer_config.METHODS,
    method_colors: dict[str, str] = optimizer_config.METHOD_COLORS,
    row_label: str | None = None,
    ylim_top: float | None = 10.0,
) -> None:
    """Write the min/mean/max loss-comparison panels across seeds, one curve per
    optimizer."""

    comparison_report.write_combined_losses_pdf_from_seed_runs(
        seed_runs,
        path,
        methods=methods,
        method_colors=method_colors,
        every=every,
        suptitle=None,
        xlabel="step",
        loss_label="mean squared error",
        train_title=None,
        test_title=None,
        train_attr="train_loss",
        test_attr="test_loss",
        row_label=row_label,
        ylim_top=ylim_top,
        font_scale=LOSS_FIGURE_FONT_SCALE,
        legend_ncol=LOSS_LEGEND_COLUMNS,
    )


def save_paper_losses_pdf(
    seed_runs: list[tuple[int, list[comparison_report.MethodRunResult]]],
    path: Path,
    *,
    every: int,
    methods: tuple[str, ...],
    method_colors: dict[str, str],
    titles: bool,
    step_axis: bool,
) -> None:
    """Write one row (training beside test loss) of the paper's mini-batch-sweep loss
    figure."""

    width, height = PAPER_LOSSES_ROW_SIZE_INCHES
    if titles:
        height += PAPER_LOSSES_TITLES_INCHES
    if step_axis:
        height += PAPER_LOSSES_STEP_AXIS_INCHES
    comparison_report.write_losses_row_pdf_from_seed_runs(
        seed_runs,
        path,
        methods=methods,
        method_colors=method_colors,
        every=every,
        size_inches=(width, height),
        ylim=PAPER_LOSSES_YLIM,
        step_axis=step_axis,
        column_titles=("Training loss", "Test loss") if titles else None,
        legend_columns=LOSS_LEGEND_COLUMNS,
    )


def _raw_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / "raw" / f"seed_{seed}"


def write_raw_histories(
    seed_runs: list[tuple[int, list["MethodResultLike"]]],
    output_dir: Path,
) -> None:
    """Persist every (seed, method) run's per-step arrays as CSV under
    ``output_dir/raw/seed_<seed>/<method>.csv``."""

    for seed, results in seed_runs:
        for result in results:
            train_loss = getattr(result, "train_mse", None)
            if train_loss is None:
                train_loss = result.train_loss
            test_loss = getattr(result, "test_mse", None)
            if test_loss is None:
                test_loss = result.test_loss
            path = _raw_dir(output_dir, seed) / f"{result.method}.csv"
            results_io.write_method_history(
                path,
                train_loss=train_loss,
                test_loss=test_loss,
                diagnostics=result.diagnostics,
                weight_history=result.weight_history,
                selected_gamma_history=result.selected_gamma_history,
                selected_angle_history=result.selected_angle_history,
                selected_smoothed_angle_history=result.selected_smoothed_angle_history,
                selected_q_history=result.selected_q_history,
                selected_chi_history=result.selected_chi_history,
                selected_psi_history=result.selected_psi_history,
            )


def _read_method_run_result(path: Path) -> comparison_report.MethodRunResult:
    arrays = results_io.read_method_history(path)
    method = path.stem
    return comparison_report.MethodRunResult(
        method=method,
        train_loss=arrays["train_loss"],
        test_loss=arrays["test_loss"],
        diagnostics=arrays["diagnostics"],
        weight_history=arrays["weight_history"],
        selected_gamma_history=arrays["selected_gamma_history"],
        selected_chi_history=arrays.get("selected_chi_history"),
        selected_psi_history=arrays.get("selected_psi_history"),
    )


_WIDTH_DEPTH_RE = re.compile(r"^width_(\d+)_depth_(\d+)$")


def _recompute_selections_by_seed(
    architecture_dir: Path, seeds: list[int]
) -> dict[int, tuple[weight_dynamics.WeightSelection, ...]]:
    """Recompute ``selections_by_seed`` for one architecture directory without
    retraining."""

    match = _WIDTH_DEPTH_RE.match(architecture_dir.name)
    if not match:
        raise ValueError(f"unrecognized results directory name: {architecture_dir.name}")
    width, depth = int(match.group(1)), int(match.group(2))

    base_config = case_config.ExperimentConfig()
    dtype = jax.numpy.complex128 if base_config.precision == "64" else jax.numpy.complex64
    selections_by_seed = {}
    for seed in seeds:
        config = case_config.ExperimentConfig(
            hidden_width=width, hidden_layers=depth, seed=seed, precision=base_config.precision
        )
        # Only shapes matter: trace abstractly so no XLA backend starts (~0.8 GB RSS).
        def initialize(seed_value, config=config):
            initialization_key, _ = jax.random.split(jax.random.PRNGKey(seed_value))
            return model.init_model(
                initialization_key, config.architecture, dtype=dtype, beta=config.initialization_beta
            )

        initial_params_shapes = jax.eval_shape(
            initialize, jax.ShapeDtypeStruct((), jax.numpy.uint32)
        )
        selections_by_seed[seed] = weight_dynamics.make_weight_selections(
            initial_params_shapes, seed=seed, max_leaves=TRACKED_LEAVES, per_leaf=TRACKED_PER_LEAF
        )
    return selections_by_seed


def regenerate_from_disk(
    results_root: Path = OUTPUT_DIR, *, paper_titles: bool = True, paper_step_axis: bool = True
) -> None:
    """Rewrite ``comparison.pdf``/``combined_losses.pdf``/``PAPER_LOSSES_NAME`` for
    every architecture directory from the raw CSVs alone."""

    architecture_dirs = sorted(
        path for path in results_root.glob("width_*_depth_*") if path.is_dir()
    )
    if not architecture_dirs:
        raise FileNotFoundError(f"no width_*_depth_* results directories found under {results_root}")

    base_config = case_config.ExperimentConfig()
    for architecture_dir in architecture_dirs:
        raw_dir = architecture_dir / "raw"
        seed_dirs = sorted(
            (path for path in raw_dir.glob("seed_*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("seed_")),
        )
        if not seed_dirs:
            print(f"[plots] skipping {architecture_dir}: no raw/seed_* directories found")
            continue
        seeds = [int(path.name.removeprefix("seed_")) for path in seed_dirs]

        seed_runs = []
        for seed, seed_dir in zip(seeds, seed_dirs):
            method_paths = sorted(seed_dir.glob("*.csv"))
            results = [_read_method_run_result(path) for path in method_paths]
            order = {method: index for index, method in enumerate(optimizer_config.METHODS)}
            results.sort(key=lambda result: order.get(result.method, len(order)))
            seed_runs.append((seed, results))

        selections_by_seed = _recompute_selections_by_seed(architecture_dir, seeds)
        methods = tuple(result.method for result in seed_runs[0][1])

        write_comparison_report_pdf(
            seed_runs,
            selections_by_seed,
            architecture_dir / "comparison.pdf",
            every=base_config.test_every,
            method_colors=optimizer_config.METHOD_COLORS,
        )
        save_loss_comparison_pdf(
            seed_runs,
            architecture_dir / "combined_losses.pdf",
            every=base_config.test_every,
            methods=methods,
            method_colors=optimizer_config.METHOD_COLORS,
        )
        save_paper_losses_pdf(
            seed_runs,
            architecture_dir / PAPER_LOSSES_NAME,
            every=base_config.test_every,
            methods=methods,
            method_colors=optimizer_config.METHOD_COLORS,
            titles=paper_titles,
            step_axis=paper_step_axis,
        )
        print(
            f"[plots] regenerated comparison.pdf/combined_losses.pdf/{PAPER_LOSSES_NAME} "
            f"under {architecture_dir}"
        )


_MINIBATCH_SWEEP_DIRECTORY_RE = re.compile(r"^batch_(\d+)$")


if __name__ == "__main__":
    # The paper stacks each architecture's mini-batch sweep in increasing order.
    sweep_dirs = {case_config.ExperimentConfig().minibatch_size: OUTPUT_DIR}
    for path in OUTPUT_DIR.glob("batch_*"):
        match = _MINIBATCH_SWEEP_DIRECTORY_RE.match(path.name)
        if match and path.is_dir():
            sweep_dirs[int(match.group(1))] = path
    for minibatch_size, results_dir in sorted(sweep_dirs.items()):
        try:
            regenerate_from_disk(
                results_dir,
                paper_titles=minibatch_size == min(sweep_dirs),
                paper_step_axis=minibatch_size == max(sweep_dirs),
            )
        except FileNotFoundError as error:
            print(f"[plots] skipped {results_dir.name}: {error}")

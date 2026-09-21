"""Report/plotting code for the 4-PINN benchmark, split out of ``train.py``."""


import os
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset
import model

from src import comparison_report
from src import optimizer_config
from src import results_io
from src import weight_dynamics

METHOD_COLORS = optimizer_config.METHOD_COLORS

RESULTS_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

# Must match train.py's seed tuples and N_TRAIN.
WEIGHT_INITIALIZATION_SEEDS = (0, 1, 2, 3, 4)
BOUNDARY_SAMPLING_SEEDS = (10, 10, 10, 10, 10)
TRACKING_SELECTION_SEEDS = (20, 20, 20, 20, 20)
N_TRAIN = 200
TRACKED_LEAVES = 6
TRACKED_PER_LEAF = 4
# The paper includes combined_losses.pdf at \linewidth, so text is drawn at 1.5 times the report size
# (the same printed size as the 0.80\linewidth figures at 2x); the legend wraps over two rows.
LOSS_FIGURE_FONT_SCALE = 1.5
LOSS_LEGEND_COLUMNS = 4
# The paper's loss figure: one row (training beside test loss) included at \linewidth
# on a 7.17 in text block, with the column titles and the epoch axis, drawn below the
# shared-hyperparameter row of 4-PINN/plots.py at the same size and y-range.
# Copied by hand to paper/figures/pinn_losses_hpo.pdf (not wired into update_paper_assets.py).
PAPER_LOSSES_NAME = "paper_losses.pdf"
PAPER_LOSSES_ROW_SIZE_INCHES = (7.17, 2.05)
PAPER_LOSSES_YLIM = (1e-7, 1e0)


def write_comparison_report_pdf(
    seed_runs: list[tuple[int, list[comparison_report.MethodRunResult]]],
    selections_by_seed: dict[int, tuple[weight_dynamics.WeightSelection, ...]],
    path: Path,
    *,
    every: int = 200,
    method_colors: dict[str, str] = METHOD_COLORS,
) -> None:
    comparison_report.write_multiseed_comparison_report_pdf(
        seed_runs,
        selections_by_seed,
        path,
        every=every,
        title="PIHNN plate-with-hole optimizer comparison",
        method_colors=method_colors,
        loss_label="mean squared error",
        train_title="Full training-set MSE",
        test_title="Test-set MSE",
    )


def save_loss_comparison_pdf(
    runs: list[dict],
    path: Path,
    *,
    every: int = 200,
    methods: tuple[str, ...] = optimizer_config.METHODS,
    method_colors: dict[str, str] = METHOD_COLORS,
) -> None:
    """Write the min/mean/max loss-comparison panels across seeds, one curve per
    optimizer."""

    comparison_report.write_combined_losses_pdf_from_runs(
        runs,
        path,
        methods=methods,
        method_colors=method_colors,
        every=every,
        suptitle=None,
        xlabel="epoch",
        loss_label="mean squared error",
        train_title=None,
        test_title=None,
        font_scale=LOSS_FIGURE_FONT_SCALE,
        legend_ncol=LOSS_LEGEND_COLUMNS,
    )


def save_paper_losses_pdf(
    seed_runs: list[tuple[int, list[comparison_report.MethodRunResult]]],
    path: Path,
    *,
    every: int = 200,
    methods: tuple[str, ...] = optimizer_config.METHODS,
    method_colors: dict[str, str] = METHOD_COLORS,
) -> None:
    """Write the paper's loss row (training beside test loss); both losses are
    recorded every epoch, so both are subsampled by ``every``."""

    comparison_report.write_losses_row_pdf_from_seed_runs(
        seed_runs,
        path,
        methods=methods,
        method_colors=method_colors,
        every=every,
        test_every=every,
        size_inches=PAPER_LOSSES_ROW_SIZE_INCHES,
        ylim=PAPER_LOSSES_YLIM,
        column_titles=("Training loss", "Test loss"),
        xlabel="epoch",
        legend_columns=LOSS_LEGEND_COLUMNS,
    )


def _raw_dir(output_dir: Path, seed: int) -> Path:
    return Path(output_dir) / "raw" / f"seed_{seed}"


def write_raw_histories(runs: list[dict], output_dir: Path = RESULTS_DIR) -> None:
    """Persist every run's per-step arrays as CSV under
    ``output_dir/raw/seed_<seed>/<method>.csv``."""

    for run in runs:
        path = _raw_dir(output_dir, run["seed"]) / f"{run['method']}.csv"
        results_io.write_method_history(
            path,
            train_loss=run["train_losses"],
            test_loss=run["test_losses"],
            diagnostics=run["diagnostics"],
            weight_history=run["weight_history"],
            selected_gamma_history=run["selected_gamma_history"],
            selected_q_history=run.get("selected_q_history"),
            selected_chi_history=run.get("selected_chi_history"),
            selected_psi_history=run.get("selected_psi_history"),
        )


def _read_method_run_result(path: Path) -> comparison_report.MethodRunResult:
    arrays = results_io.read_method_history(path)
    return comparison_report.MethodRunResult(
        method=path.stem,
        train_loss=arrays["train_loss"],
        test_loss=arrays["test_loss"],
        diagnostics=arrays["diagnostics"],
        weight_history=arrays["weight_history"],
        selected_gamma_history=arrays["selected_gamma_history"],
        selected_chi_history=arrays.get("selected_chi_history"),
        selected_psi_history=arrays.get("selected_psi_history"),
    )


def _init_params(weight_seed: int, boundary_seed: int):
    rng = np.random.default_rng(boundary_seed)
    key = jax.random.PRNGKey(weight_seed)
    x0_batch = dataset._sample_boundary(10 * N_TRAIN, rng)
    x0 = jax.numpy.concatenate([x0_batch[0], x0_batch[3]])
    return model.init_network(key, model.BETA, x0)


def regenerate_from_disk(results_root: Path = RESULTS_DIR) -> None:
    """Rewrite ``comparison.pdf``/``combined_losses.pdf`` under ``results_root``
    from the raw CSVs alone."""

    results_root = Path(results_root)
    raw_dir = results_root / "raw"
    seed_dirs = sorted(
        (path for path in raw_dir.glob("seed_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("seed_")),
    )
    if not seed_dirs:
        raise FileNotFoundError(f"no raw/seed_* directories found under {results_root}")

    seed_configurations = dict(
        zip(WEIGHT_INITIALIZATION_SEEDS, zip(BOUNDARY_SAMPLING_SEEDS, TRACKING_SELECTION_SEEDS))
    )

    seed_runs = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.removeprefix("seed_"))
        method_paths = sorted(seed_dir.glob("*.csv"))
        results = [_read_method_run_result(path) for path in method_paths]
        order = {method: index for index, method in enumerate(optimizer_config.METHODS)}
        results.sort(key=lambda result: order.get(result.method, len(order)))
        seed_runs.append((seed, results))

    selections_by_seed = {}
    for seed, (boundary_seed, tracking_seed) in seed_configurations.items():
        if seed not in {s for s, _ in seed_runs}:
            continue
        selections_by_seed[seed] = weight_dynamics.make_weight_selections(
            _init_params(seed, boundary_seed),
            seed=tracking_seed,
            max_leaves=TRACKED_LEAVES,
            per_leaf=TRACKED_PER_LEAF,
        )

    write_comparison_report_pdf(seed_runs, selections_by_seed, results_root / "comparison.pdf")
    runs = [
        {"seed": seed, "method": result.method, "train_losses": result.train_loss, "test_losses": result.test_loss}
        for seed, results in seed_runs
        for result in results
    ]
    methods = tuple(result.method for result in seed_runs[0][1])
    save_loss_comparison_pdf(runs, results_root / "combined_losses.pdf", methods=methods)
    save_paper_losses_pdf(seed_runs, results_root / PAPER_LOSSES_NAME, methods=methods)
    print(
        f"[plots] regenerated comparison.pdf/combined_losses.pdf/{PAPER_LOSSES_NAME} "
        f"under {results_root}"
    )


if __name__ == "__main__":
    try:
        regenerate_from_disk(RESULTS_DIR)
    except FileNotFoundError as error:
        print(f"[plots] skipped: {error}")

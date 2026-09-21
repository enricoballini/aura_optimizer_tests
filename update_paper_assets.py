"""Update each benchmark case's figures and tables in paper/main.tex, preserving
hand-edited captions."""


import argparse
import csv
import filecmp
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from typing import Any, Callable, NamedTuple

from src import hyperparameters_io
from src import metrics
from src import optimizer_config
from src import sgd_baseline

PROJECT_DIRECTORY = Path(__file__).resolve().parent

TABLE_CASE_ORDER = (
    "holomorphic",
    "holomorphic_plus_non_holomorphic",
    "non_holomorphic",
    "multivariate_c4",
)
ACTIVE_TABLE_CASES = TABLE_CASE_ORDER
CASE_FOLDER_PREFIX = {
    "holomorphic": "2",
    "holomorphic_plus_non_holomorphic": "2",
    "non_holomorphic": "1",
    "multivariate_c4": "3",
}
CASE_FOLDER = {
    case: PROJECT_DIRECTORY / f"{CASE_FOLDER_PREFIX[case]}-{case}"
    for case in TABLE_CASE_ORDER
}

PAPER_DIRECTORY = PROJECT_DIRECTORY / "paper"
DEFAULT_FIGURE_DESTINATIONS = {
    case: PAPER_DIRECTORY / "figures" / f"optimizer_losses_{case}.pdf"
    for case in TABLE_CASE_ORDER
}
DEFAULT_PINN_RESULTS = PROJECT_DIRECTORY / "4-PINN" / "results"
DEFAULT_PINN_FIGURE_DESTINATION = PAPER_DIRECTORY / "figures" / "pinn"
DEFAULT_PINN_LOSS_DESTINATION = PAPER_DIRECTORY / "figures" / "pinn_losses.pdf"
# Must match PAPER_LOSSES_NAME in 4-PINN/plots.py.
PINN_PAPER_LOSSES_NAME = "paper_losses.pdf"
DEFAULT_REAL_VALUED_LOSS_DESTINATION = PAPER_DIRECTORY / "figures" / "real_valued_losses.pdf"
DEFAULT_U_NET_LOSS_DESTINATION = PAPER_DIRECTORY / "figures" / "u_net_losses.pdf"
DEFAULT_NON_HOLOMORPHIC_BETA_SWEEP_DESTINATION = (
    PAPER_DIRECTORY / "figures" / "optimizer_beta_sweep_non_holomorphic.pdf"
)
TABLE_CASE_TITLES = {
    "holomorphic": "Holomorphic",
    "holomorphic_plus_non_holomorphic": "Holomorphic-plus-non-holomorphic",
    "non_holomorphic": "Non-holomorphic",
    "multivariate_c4": "Multivariate C4",
}
ACTIVATION_TEXT = {
    "complex_exp": r"$\exp(\zeta)$",
    "componentwise_silu": (
        r"$\operatorname{SiLU}(\real\zeta)+\mathrm i\operatorname{SiLU}(\imag\zeta)$"
    ),
    "componentwise_tanh": (
        r"$\tanh(\real\zeta)+\mathrm i\tanh(\imag\zeta)$ "
        r"(componentwise across the four inputs)"
    ),
}
# Must match the primary (width, depth) in each 1-* case's config.py.
PRIMARY_ARCHITECTURE_DIRECTORY_NAME = "width_32_depth_4"
# Must match the single architecture in 3-multivariate_c4/config.py.
MULTIVARIATE_C4_ARCHITECTURE_DIRECTORY_NAME = "width_128_depth_5"
DEFAULT_SUMMARY_PATHS = {
    case: CASE_FOLDER[case] / "results" / PRIMARY_ARCHITECTURE_DIRECTORY_NAME / "summary.json"
    for case in ACTIVE_TABLE_CASES
}
DEFAULT_SUMMARY_PATHS["multivariate_c4"] = (
    CASE_FOLDER["multivariate_c4"]
    / "results"
    / MULTIVARIATE_C4_ARCHITECTURE_DIRECTORY_NAME
    / "summary.json"
)
DEFAULT_MAIN_TEX = PAPER_DIRECTORY / "main.tex"

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "adam": "Adam",
    "adam_variable_lr": "Adam (variable LR)",
    "lbfgs": "L-BFGS",
    "adam_aura": "Adam-AURA",
    "aura_light": "AURA light",
    "astra": "ASTRA",
    "eclipse": "ECLIPSE",
    "pulsar": "PULSAR",
    "method_NEU_opt": "AURA-ADAM",
    "method_NEU_optEB": "NEU-optEB",
    "rprop": "RPROP",
    "nadamw": "NadamW",
    "adamaxw": "AdamaxW",
    "nadam": "Nadam",
    "cvamsgrad": "CvAMSGrad",
    "muon": "Muon",
    "muon_aura": "Muon-AURA",
    "adam_aura_s": "AURA-S",
    "muon_aura_s": "Muon-AURA-S",
    "adam_aura_sign": "AURA-sign",
    "muon_aura_sign": "Muon-AURA-sign",
    "adam_aura_snr": "AURA-SNR",
    "muon_aura_snr": "Muon-AURA-SNR",
    "adam_aura_snr_ablation": "AURA-SNR ablation",
    "adam_aura_cosine_ablation": "AURA-cosine ablation",
    "adam_aura_spring_ablation": "AURA-spring ablation",
    "muon_aura_spring_ablation": "Muon-AURA-spring ablation",
    "adam_aura_spring_eb": "AURA-spring-EB",
    "cl_bfgs": "CL-BFGS",
    "hcscgm": "HCSCGM",
}


def _ordered_methods(methods: Iterable[str]) -> list[str]:
    """``methods`` filtered to ``optimizer_config.METHODS`` and reordered to follow
    it."""

    available = set(methods)
    return [method for method in optimizer_config.METHODS if method in available]


def _methods_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """A results file's ``methods`` mapping, restricted to
    ``optimizer_config.METHODS`` and rekeyed into that order."""

    methods_summary = summary["methods"]
    return {method: methods_summary[method] for method in _ordered_methods(methods_summary)}

PINN_RESULTS = PROJECT_DIRECTORY / "4-PINN" / "results"

REAL_VALUED_RESULTS = PROJECT_DIRECTORY / "5-real-valued" / "results"
U_NET_RESULTS = PROJECT_DIRECTORY / "6-U-net" / "results"
CIFAR10_RESULTS = PROJECT_DIRECTORY / "5-CIFAR-10" / "results"
CIFAR10_ARCHITECTURES = ("deep_complex_resnet", "fully_connected")
CIFAR10_LOSS_DESTINATION = PAPER_DIRECTORY / "figures" / "cifar10_losses.pdf"
CIFAR10_ACCURACY_DESTINATION = PAPER_DIRECTORY / "figures" / "cifar10_accuracies.pdf"
EQUALIZATION_RESULTS = PROJECT_DIRECTORY / "7-channel_equalization" / "results"
EQUALIZATION_ARCHITECTURES = ("cvfnn", "c_rbf")
EQUALIZATION_ARCHITECTURE_TITLES = {"cvfnn": "CVFNN", "c_rbf": "deep C-RBF"}
EQUALIZATION_LOSS_DESTINATION = PAPER_DIRECTORY / "figures" / "equalization_losses.pdf"


def _table_markers(kind: str, case: str) -> tuple[str, str]:
    """Marker comments delimiting one case's generated table in main.tex."""

    name = f"{kind}_{case}"
    return f"% BEGIN GENERATED TABLE: {name}", f"% END GENERATED TABLE: {name}"


def _figure_markers(name: str) -> tuple[str, str]:
    """Marker comments delimiting one generated figure environment in main.tex."""

    return f"% BEGIN GENERATED FIGURE: {name}", f"% END GENERATED FIGURE: {name}"


def _atomic_copy(source: Path, destination: Path) -> bool:
    """Atomically copy *source* to *destination* when their bytes differ."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
        return False

    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        shutil.copy2(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def update_paper_figure(source: Path, destination: Path) -> bool:
    """Atomically update one of the paper's PDF figures."""

    source = source.expanduser().resolve()
    with source.open("rb") as source_file:
        if source_file.read(5) != b"%PDF-":
            raise ValueError(f"Figure source is not a PDF: {source}")
    return _atomic_copy(source, destination)


def _update_losses_legend(
    representative_combined_losses_pdfs: Path | Sequence[Path],
    figure_destination: Path,
    legend_name: str = "combined_losses_legend.pdf",
) -> bool:
    """Refresh the standalone one-row ``<figure stem>_legend.pdf`` beside a paper
    loss figure."""

    candidates = (
        [representative_combined_losses_pdfs]
        if isinstance(representative_combined_losses_pdfs, Path)
        else list(representative_combined_losses_pdfs)
    )
    legend_destination = figure_destination.with_name(f"{figure_destination.stem}_legend.pdf")

    for candidate in candidates:
        legend_source = candidate.with_name(legend_name)
        if legend_source.is_file():
            return update_paper_figure(legend_source, legend_destination)

    from src import comparison_report

    with tempfile.TemporaryDirectory(prefix="losses-legend-") as temporary_directory:
        legend_pdf = Path(temporary_directory) / "combined_losses_legend.pdf"
        comparison_report.write_losses_legend_pdf(
            legend_pdf,
            methods=optimizer_config.METHODS,
            method_colors=optimizer_config.METHOD_COLORS,
        )
        return update_paper_figure(legend_pdf, legend_destination)


def _losses_legend_include(figure_stem: str, width: float | None = 0.80) -> list[str]:
    """LaTeX lines placing the shared loss-curve legend at the bottom of a
    multi-panel figure, at its natural size when ``width`` is None."""

    options = "" if width is None else f"[width={width:.2f}\\linewidth]"
    return [
        "\t\\par\\vspace{0.4\\baselineskip}",
        f"\t\\includegraphics{options}{{figures/{figure_stem}_legend.pdf}}",
        "\t\\par\\vspace{0.2\\baselineskip}",
    ]


def _render_stacked_losses_figure_block(
    *,
    title: str,
    figure_stem: str,
    row_labels: Sequence[str],
    caption: str,
    block_labels: Sequence[str] = ("Primary architecture", "Secondary architecture"),
) -> str:
    """LaTeX for a loss figure stacking the paper rows of ``figures/<figure_stem>.pdf``
    (one page per block and sweep value, in that order) into one block per
    architecture or precision, with the print-size legend at the bottom. The column
    titles and the step axis are drawn inside the top and bottom rows of each block."""

    lines = [
        r"\begin{figure*}[!t]",
        "\t\\centering",
        f"\t\\textbf{{{title}}}\\par\\vspace{{0.3\\baselineskip}}",
    ]
    page = 1
    for block_index, block_label in enumerate(block_labels):
        lines += [
            "\t\\begin{minipage}[c]{0.02\\linewidth}",
            "\t\t\\centering",
            f"\t\t\\rotatebox{{90}}{{{block_label}}}",
            "\t\\end{minipage}%",
            "\t\\begin{minipage}[c]{0.98\\linewidth}",
            "\t\t\\centering",
        ]
        for row_index, row_label in enumerate(row_labels):
            row_end = "" if row_index == len(row_labels) - 1 else "\\\\"
            lines += [
                "\t\t\\begin{minipage}[c]{0.02\\linewidth}",
                "\t\t\t\\centering",
                f"\t\t\t\\rotatebox{{90}}{{{row_label}}}",
                "\t\t\\end{minipage}%",
                "\t\t\\begin{minipage}[c]{0.98\\linewidth}",
                "\t\t\t\\centering",
                f"\t\t\t\\includegraphics[width=0.80\\linewidth,page={page}]{{figures/{figure_stem}.pdf}}",
                f"\t\t\\end{{minipage}}{row_end}",
            ]
            page += 1
        # Extra vertical space separates the blocks.
        block_end = "" if block_index == len(block_labels) - 1 else "\\\\[0.8\\baselineskip]"
        lines.append(f"\t\\end{{minipage}}{block_end}")

    lines += _losses_legend_include(figure_stem, width=None)
    lines += [
        f"\t\\caption{{{caption}}}",
        f"\t\\label{{fig:{figure_stem}}}",
        r"\end{figure*}",
    ]
    return "\n".join(lines) + "\n"


def update_case_losses_figure(case: str, folder: Path, destination: Path) -> bool:
    """Concatenate one case's primary- and secondary-architecture
    combined_losses.pdf into a two-page figure."""

    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("Combining comparison reports requires the 'pdfunite' executable.")

    results_directory = folder / "results"
    architecture_directories = sorted(
        results_directory.glob("width_*_depth_*"),
        key=lambda path: tuple(int(part) for part in path.name.split("_")[1::2]),
    )
    if len(architecture_directories) != 2:
        raise FileNotFoundError(
            f"Expected exactly two architecture directories for {case!r} "
            f"in {results_directory}, found {len(architecture_directories)}"
        )
    primary_directory, secondary_directory = architecture_directories
    primary_pdf = primary_directory / "combined_losses.pdf"
    if not primary_pdf.is_file():
        raise FileNotFoundError(
            f"Missing primary-architecture combined-losses PDF for {case!r}: {primary_pdf}"
        )
    secondary_pdf = secondary_directory / "combined_losses.pdf"
    if not secondary_pdf.is_file():
        raise FileNotFoundError(
            f"Missing secondary-architecture combined-losses PDF for {case!r}: {secondary_pdf}"
        )

    with tempfile.TemporaryDirectory(prefix="combined-losses-") as temporary_directory:
        combined_pdf = Path(temporary_directory) / "combined.pdf"
        subprocess.run(
            [pdfunite, str(primary_pdf), str(secondary_pdf), str(combined_pdf)], check=True
        )
        return update_paper_figure(combined_pdf, destination)


# Must match PAPER_LOSSES_NAME in 2-holomorphic/plots.py.
HOLOMORPHIC_PAPER_LOSSES_NAME = "paper_losses.pdf"


def _holomorphic_minibatch_directory(
    results_directory: Path,
    minibatch_size: int,
    base_minibatch_size: int,
    width: int,
    depth: int,
) -> Path:
    architecture_name = f"width_{width}_depth_{depth}"
    if minibatch_size == base_minibatch_size:
        return results_directory / architecture_name
    return results_directory / f"batch_{minibatch_size}" / architecture_name


def _holomorphic_minibatch_grid(results_directory: Path) -> tuple[tuple[int, ...], int]:
    """The mini-batch sizes swept for the holomorphic case, read back from its
    results."""

    architecture_directories = _holomorphic_architecture_directories(results_directory)
    base_sizes = {
        int(_load_summary(directory / "summary.json")["experiment"]["minibatch_size"])
        for directory in architecture_directories
    }
    if len(base_sizes) != 1:
        raise ValueError(
            "Inconsistent base mini-batch size across the holomorphic "
            f"architectures: {sorted(base_sizes)}"
        )
    (base_minibatch_size,) = base_sizes

    sizes = set(base_sizes)
    for batch_directory in results_directory.glob("batch_*"):
        if batch_directory.is_dir():
            sizes.add(int(batch_directory.name.split("_", 1)[1]))

    return tuple(sorted(sizes)), base_minibatch_size


def _holomorphic_architecture_directories(results_directory: Path) -> list[Path]:
    """The two architecture directories (primary first, then secondary) at the base
    mini-batch size."""

    directories = sorted(
        results_directory.glob("width_*_depth_*"),
        key=lambda path: tuple(int(part) for part in path.name.split("_")[1::2]),
    )
    if len(directories) != 2:
        raise FileNotFoundError(
            f"Expected exactly two architecture directories for 'holomorphic' "
            f"in {results_directory}, found {len(directories)}"
        )
    return directories


def update_holomorphic_minibatch_figure(folder: Path, destination: Path) -> bool:
    """Concatenate the holomorphic paper loss rows over architectures and swept
    mini-batch sizes into one multi-page figure."""

    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("Combining comparison reports requires the 'pdfunite' executable.")

    results_directory = folder / "results"
    architecture_directories = _holomorphic_architecture_directories(results_directory)
    architectures = [
        tuple(int(part) for part in path.name.split("_")[1::2]) for path in architecture_directories
    ]
    minibatch_sizes, base_minibatch_size = _holomorphic_minibatch_grid(results_directory)

    pdfs = []
    for width, depth in architectures:
        for minibatch_size in minibatch_sizes:
            pdf = (
                _holomorphic_minibatch_directory(
                    results_directory, minibatch_size, base_minibatch_size, width, depth
                )
                / HOLOMORPHIC_PAPER_LOSSES_NAME
            )
            if not pdf.is_file():
                raise FileNotFoundError(f"Missing paper loss row PDF for mini-batch sweep: {pdf}")
            pdfs.append(pdf)

    with tempfile.TemporaryDirectory(prefix="holomorphic-minibatch-") as temporary_directory:
        combined_pdf = Path(temporary_directory) / "combined.pdf"
        subprocess.run([pdfunite, *(str(pdf) for pdf in pdfs), str(combined_pdf)], check=True)
        figure_changed = update_paper_figure(combined_pdf, destination)
    legend_changed = _update_losses_legend(
        pdfs,
        destination,
        legend_name=f"{Path(HOLOMORPHIC_PAPER_LOSSES_NAME).stem}_legend.pdf",
    )
    return figure_changed or legend_changed


_INTEGER_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 100: "one hundred", 1000: "one thousand",
}


def _holomorphic_minibatch_figure_caption(
    architectures: list[list[int]],
    parameter_counts: list[int],
    learning_rates: list[float],
    minibatch_sizes: tuple[int, ...],
    steps: int,
    seed_count: int,
) -> str:
    """The holomorphic loss figure's caption, with every number pulled from the
    case's own summaries."""

    (primary_arch, primary_params), (secondary_arch, secondary_params) = zip(
        architectures, parameter_counts
    )
    primary_rate, secondary_rate = learning_rates

    def _arch_tuple(arch: list[int]) -> str:
        return "(" + ",".join(str(width) for width in arch) + ")"

    ratio = primary_rate / secondary_rate if secondary_rate else float("nan")
    ratio_int = round(ratio)
    if abs(ratio - ratio_int) < 1e-6 and ratio_int in _INTEGER_WORDS:
        learning_rate_phrase = f"a {_INTEGER_WORDS[ratio_int]}-times-smaller learning rate"
    else:
        learning_rate_phrase = (
            rf"a learning rate smaller by a factor of ${_format_number(ratio)}$"
        )

    quoted_sizes = [f"${size}$" for size in minibatch_sizes]
    sizes_prose = " and ".join(
        filter(None, [", ".join(quoted_sizes[:-1]), quoted_sizes[-1]])
    )

    return (
        r"Training-set (left panel of each row) and test-set (right panel) "
        r"mean squared errors for the Holomorphic optimizer benchmark. The two "
        r"architecture blocks show the same feedforward network trained "
        r"independently at two architectures "
        r"(\Cref{tab:optimizer_settings_holomorphic}): \emph{top}, the primary "
        rf"architecture ${_arch_tuple(primary_arch)}$ "
        rf"(\numprint{{{primary_params}}} parameters); \emph{{bottom}}, the "
        rf"wider and deeper secondary architecture ${_arch_tuple(secondary_arch)}$ "
        rf"(\numprint{{{secondary_params}}} parameters), trained at "
        rf"{learning_rate_phrase}. Within each architecture block, the rows "
        rf"sweep the mini-batch size across {sizes_prose} points from top to "
        rf"bottom at a fixed \numprint{{{steps}}}-update budget, so a smaller "
        r"mini-batch means fewer epochs over the training set, not fewer "
        r"optimizer updates. Solid curves show the pointwise mean over all "
        rf"${seed_count}$ seeds; shaded regions span the corresponding minimum "
        r"and maximum, also shown by dashed and dotted curves, respectively."
    )


def render_holomorphic_minibatch_figure_block(results_directory: Path) -> str:
    """Render the holomorphic loss figure's LaTeX: one block per architecture, one
    row per swept mini-batch size."""

    architecture_directories = _holomorphic_architecture_directories(results_directory)
    minibatch_sizes, _base_minibatch_size = _holomorphic_minibatch_grid(results_directory)
    summaries = [
        _load_summary(directory / "summary.json") for directory in architecture_directories
    ]
    caption = _holomorphic_minibatch_figure_caption(
        architectures=[summary["experiment"]["architecture"] for summary in summaries],
        parameter_counts=[
            summary["shared_design"]["complex_parameter_count"] for summary in summaries
        ],
        learning_rates=[summary["experiment"]["learning_rate"] for summary in summaries],
        minibatch_sizes=minibatch_sizes,
        steps=summaries[0]["experiment"]["steps"],
        seed_count=len(summaries[0]["experiment"]["seeds"]),
    )
    return _render_stacked_losses_figure_block(
        title="Loss: TEST 2 -- holomorphic",
        figure_stem="optimizer_losses_holomorphic",
        row_labels=[rf"Mini-batch $={minibatch_size}$" for minibatch_size in minibatch_sizes],
        caption=caption,
    )


def update_holomorphic_minibatch_figure_block(
    results_directory: Path, main_tex_path: Path
) -> bool:
    """ """

    figure_block = render_holomorphic_minibatch_figure_block(results_directory)
    begin_marker, end_marker = _figure_markers("optimizer_losses_holomorphic")
    return _splice_table_into_main_tex(figure_block, main_tex_path, begin_marker, end_marker)


def render_holomorphic_minibatch_sweep_table(results_directory: Path) -> str:
    """Render the holomorphic mini-batch sweep table: one row per (architecture,
    method), one column group per mini-batch size for the loss metrics; the
    training-time group is reported only at the base mini-batch size, since the SGD
    reference it is normalized against is only timed there."""

    methods = optimizer_config.METHODS
    n_methods = len(methods)

    architecture_directories = _holomorphic_architecture_directories(results_directory)
    architecture_names = ("Primary", "Secondary")
    minibatch_sizes, base_minibatch_size = _holomorphic_minibatch_grid(results_directory)
    minibatch_column_labels = tuple(str(size) for size in minibatch_sizes)
    time_minibatch_index = minibatch_sizes.index(base_minibatch_size)
    n_time_batch = 1
    sgd_timing = _load_sgd_timing(CASE_FOLDER["holomorphic"])

    grid: list[list[dict[str, Any]]] = []
    architecture_regime_keys: list[str] = []
    nan_grid_points: list[tuple[str, Path]] = []
    for architecture_index, architecture_directory in enumerate(architecture_directories):
        width, depth = (int(part) for part in architecture_directory.name.split("_")[1::2])
        architecture_regime_keys.append(f"width_{width}_depth_{depth}")
        per_minibatch_size = []
        for minibatch_size in minibatch_sizes:
            grid_directory = _holomorphic_minibatch_directory(
                results_directory, minibatch_size, base_minibatch_size, width, depth
            )
            summary = _load_summary(grid_directory / "summary.json")
            per_minibatch_size.append(_methods_summary(summary))
            nan_grid_points.append(
                (
                    f"the {architecture_names[architecture_index].lower()} architecture "
                    f"at mini-batch size {minibatch_size}",
                    grid_directory / "history.csv",
                )
            )
        grid.append(per_minibatch_size)

    def _column_cells(methods_summary: dict[str, Any], key: str, format_cell) -> list[str]:
        best_method = _median_argmin(
            {method: methods_summary.get(method, {}).get(key) for method in methods}
        )
        cells = []
        for method in methods:
            cell = _quartile_cell(methods_summary.get(method, {}).get(key), format_cell)
            if cell == _EMPTY_CELL:
                cells.append(cell)
                continue
            cells.append(rf"\mbox{{{_bold(cell) if method == best_method else cell}}}")
        return cells

    n_batch = len(minibatch_sizes)

    nan_counts = _nan_seed_counts_by_method(
        nan_grid_points, methods, loss_column="train_mse"
    )
    show_nan = nan_counts is not None
    n_data_columns = 2 * n_batch + n_time_batch + (n_batch if show_nan else 0)
    lmin_end = 2 + n_batch
    a_end = lmin_end + n_batch
    time_end = a_end + n_time_batch

    body = []
    for architecture_index, architecture_name in enumerate(architecture_names):
        columns: list[list[str]] = []
        sgd_seconds = _sgd_regime_median_seconds(
            sgd_timing, architecture_regime_keys[architecture_index]
        )
        for key, format_cell in (
            ("best_train_mse", _format_scientific_quartiles),
            ("learning_area", _format_fixed_quartiles),
        ):
            for minibatch_index in range(n_batch):
                methods_summary = _without_learning_area_of_diverged(
                    grid[architecture_index][minibatch_index],
                    nan_counts,
                    architecture_index * n_batch + minibatch_index,
                )
                columns.append(_column_cells(methods_summary, key, format_cell))
        columns.append(
            _column_cells(
                grid[architecture_index][time_minibatch_index],
                "training_seconds",
                _time_vs_sgd_formatter(sgd_seconds),
            )
        )
        if show_nan:
            for minibatch_index in range(n_batch):
                grid_point = architecture_index * n_batch + minibatch_index
                columns.append(
                    [str(nan_counts[method][grid_point].diverged) for method in methods]
                )
        for method_index, method in enumerate(methods):
            row_cells = [cell_list[method_index] for cell_list in columns]
            architecture_cell = (
                rf"\multirow{{{n_methods}}}{{*}}{{{architecture_name}}}"
                if method_index == 0
                else ""
            )
            body.append(
                f"{architecture_cell} & {METHOD_DISPLAY_NAMES.get(method, method)} & "
                + " & ".join(row_cells)
                + r" \\"
            )
        if architecture_index == 0:
            body.append(rf"\cmidrule(lr){{1-{2 + n_data_columns}}}")

    batch_header = " & ".join(minibatch_column_labels)
    time_header = str(base_minibatch_size)
    caption = r"Mini-batch-size sensitivity for the Holomorphic benchmark."

    nan_group_header = (
        rf" & \multicolumn{{{n_batch}}}{{c}}{{NaN}}" if show_nan else ""
    )
    nan_group_rule = (
        rf"\cmidrule(lr){{{time_end + 1}-{time_end + n_batch}}}" if show_nan else ""
    )
    nan_subheader = f" & {batch_header}" if show_nan else ""
    lines = [
        # Full-page sideways float (rotating): the table is wider than \textwidth even at \tiny.
        r"\begin{sidewaystable*}",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:optimizer_training_time_holomorphic}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        # Natural-width columns shrunk by adjustbox: tabularx X columns made wide cells overlap.
        r"\begin{adjustbox}{max width=\linewidth,center}",
        rf"\begin{{tabular}}{{@{{}}ll*{{{n_data_columns}}}{{c}}@{{}}}}",
        r"\toprule",
        rf" & & \multicolumn{{{n_batch}}}{{c}}{{$\mathscr{{L}}_{{\min}}^{{(m)}}$}} "
        rf"& \multicolumn{{{n_batch}}}{{c}}{{$A^{{(m)}}$}} "
        rf"& \multicolumn{{{n_time_batch}}}{{c}}{{{_SGD_TIME_COLUMN_HEADER}}}{nan_group_header} \\",
        rf"\cmidrule(lr){{3-{lmin_end}}}\cmidrule(lr){{{lmin_end + 1}-{a_end}}}"
        rf"\cmidrule(lr){{{a_end + 1}-{time_end}}}{nan_group_rule}",
        rf"Arch. & Optimizer & {batch_header} & {batch_header} & {time_header}{nan_subheader} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{sidewaystable*}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_holomorphic_minibatch_sweep_table(results_directory: Path, main_tex_path: Path) -> bool:
    """Splice the holomorphic mini-batch sweep table into main.tex, in place of the
    shared training-time table."""

    table = render_holomorphic_minibatch_sweep_table(results_directory)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "holomorphic")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


NON_HOLOMORPHIC_LR_MULTIPLIERS: tuple[float, ...] = (0.1, 1.0, 10.0)
# Must match PAPER_LOSSES_NAME in 1-non_holomorphic/plots.py.
NON_HOLOMORPHIC_PAPER_LOSSES_NAME = "paper_losses.pdf"

# Must match BETA_1_VALUES / BETA_2_VALUES in 1-non_holomorphic/sweep_betas.py.
NON_HOLOMORPHIC_BETA_1_VALUES: tuple[float, ...] = (0.85, 0.9, 0.95)
NON_HOLOMORPHIC_BETA_2_VALUES: tuple[float, ...] = (0.99, 0.999, 0.9999)
NON_HOLOMORPHIC_BETA_SWEEP_METHODS: tuple[str, ...] = optimizer_config.METHODS


def _non_holomorphic_lr_directory(results_directory: Path, multiplier: float, width: int, depth: int) -> Path:
    architecture_name = f"width_{width}_depth_{depth}"
    if multiplier == 1.0:
        return results_directory / architecture_name
    return results_directory / f"lr_{multiplier:g}x" / architecture_name


def update_non_holomorphic_lr_figure(folder: Path, destination: Path) -> bool:
    """Concatenate the non_holomorphic paper loss rows over architectures and swept
    learning rates into one multi-page figure."""

    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("Combining comparison reports requires the 'pdfunite' executable.")

    results_directory = folder / "results"
    architecture_directories = sorted(
        results_directory.glob("width_*_depth_*"),
        key=lambda path: tuple(int(part) for part in path.name.split("_")[1::2]),
    )
    if len(architecture_directories) != 2:
        raise FileNotFoundError(
            f"Expected exactly two architecture directories for 'non_holomorphic' "
            f"in {results_directory}, found {len(architecture_directories)}"
        )
    architectures = [
        tuple(int(part) for part in path.name.split("_")[1::2]) for path in architecture_directories
    ]

    pdfs = []
    for width, depth in architectures:
        for multiplier in NON_HOLOMORPHIC_LR_MULTIPLIERS:
            pdf = _non_holomorphic_lr_directory(results_directory, multiplier, width, depth) / NON_HOLOMORPHIC_PAPER_LOSSES_NAME
            if not pdf.is_file():
                raise FileNotFoundError(f"Missing paper loss row PDF for LR sweep: {pdf}")
            pdfs.append(pdf)

    with tempfile.TemporaryDirectory(prefix="non-holomorphic-lr-") as temporary_directory:
        combined_pdf = Path(temporary_directory) / "combined.pdf"
        subprocess.run([pdfunite, *(str(pdf) for pdf in pdfs), str(combined_pdf)], check=True)
        figure_changed = update_paper_figure(combined_pdf, destination)
    legend_changed = _update_losses_legend(
        pdfs,
        destination,
        legend_name=f"{Path(NON_HOLOMORPHIC_PAPER_LOSSES_NAME).stem}_legend.pdf",
    )
    return figure_changed or legend_changed


def render_non_holomorphic_lr_figure_block() -> str:
    """Render the non_holomorphic loss figure's LaTeX: one block per architecture, one
    row per swept learning rate, labelled as in the sweep table."""

    caption = (
        r"Training-set and test-set mean squared errors for the Non-holomorphic "
        r"optimizer benchmark at the primary and secondary architectures, for the base "
        r"learning rate $\alpha$ divided by $10$, $\alpha$ itself, and $\alpha$ multiplied "
        r"by $10$. Solid curves show the pointwise median over all seeds; shaded regions "
        r"span the interquartile range (25th--75th percentile) over seeds, whose edges are "
        r"also drawn as dotted curves."
    )
    return _render_stacked_losses_figure_block(
        title="Loss: TEST 1 -- non-holomorphic",
        figure_stem="optimizer_losses_non_holomorphic",
        row_labels=NON_HOLOMORPHIC_LR_COLUMN_LABELS,
        caption=caption,
    )


def update_non_holomorphic_lr_figure_block(main_tex_path: Path) -> bool:
    """ """

    figure_block = render_non_holomorphic_lr_figure_block()
    begin_marker, end_marker = _figure_markers("optimizer_losses_non_holomorphic")
    return _splice_table_into_main_tex(figure_block, main_tex_path, begin_marker, end_marker)


def _non_holomorphic_beta_pair_directory(results_directory: Path, beta_1: float, beta_2: float) -> Path:
    """results/beta_sweep/b1_<b1>_b2_<b2>/width_32_depth_4/ for one grid point (see
    sweep_betas.beta_sweep_directory_name)."""

    return (
        results_directory
        / "beta_sweep"
        / f"b1_{beta_1:g}_b2_{beta_2:g}"
        / PRIMARY_ARCHITECTURE_DIRECTORY_NAME
    )


def render_non_holomorphic_beta_sweep_table(results_directory: Path) -> str:
    """Render the 9-row (beta_1, beta_2) grid table for the non_holomorphic case."""

    n_methods = len(NON_HOLOMORPHIC_BETA_SWEEP_METHODS)
    display_names = [METHOD_DISPLAY_NAMES.get(m, m) for m in NON_HOLOMORPHIC_BETA_SWEEP_METHODS]
    # Rotated method names; \multicolumn stops the rotated box from being line-broken.
    header_methods = " & ".join(
        rf"\multicolumn{{1}}{{c}}{{\rotatebox[origin=l]{{90}}{{{name}}}}}" for name in display_names
    )
    LOSS_MIN_EXPONENT = -6

    def metric_cells(
        methods_summary: dict[str, Any],
        key: str,
        *,
        format_cell: Callable[[float, float, float], str],
    ) -> list[str]:
        best_method = _median_argmin(
            {
                method: methods_summary.get(method, {}).get(key)
                for method in NON_HOLOMORPHIC_BETA_SWEEP_METHODS
            }
        )
        cells = []
        for method in NON_HOLOMORPHIC_BETA_SWEEP_METHODS:
            cell = _quartile_cell(methods_summary.get(method, {}).get(key), format_cell)
            if cell == _EMPTY_CELL:
                cells.append(cell)
                continue
            cell = _bold(cell) if method == best_method else cell
            # Keep each cell on one line: inline math could break inside it in narrow columns.
            cells.append(rf"\mbox{{{cell}}}")
        return cells

    # beta_2 outer, beta_1 inner, matching the figure's page order.
    grid_pairs = [
        (beta_1, beta_2)
        for beta_2 in NON_HOLOMORPHIC_BETA_2_VALUES
        for beta_1 in NON_HOLOMORPHIC_BETA_1_VALUES
    ]
    nan_counts = _nan_seed_counts_by_method(
        [
            (
                (beta_1, beta_2),
                _non_holomorphic_beta_pair_directory(results_directory, beta_1, beta_2)
                / "history.csv",
            )
            for beta_1, beta_2 in grid_pairs
        ],
        NON_HOLOMORPHIC_BETA_SWEEP_METHODS,
        loss_column="train_mse",
    )
    show_nan = nan_counts is not None

    loss_rows, area_rows, nan_rows = [], [], []
    for row_index, (beta_1, beta_2) in enumerate(grid_pairs):
        grid_directory = _non_holomorphic_beta_pair_directory(
            results_directory, beta_1, beta_2
        )
        summary = _load_summary(grid_directory / "summary.json")
        methods_summary = _without_learning_area_of_diverged(
            _methods_summary(summary), nan_counts, row_index
        )
        pair = [f"${_format_number(beta_1)}$", f"${_format_number(beta_2)}$"]
        loss_cells = metric_cells(
            methods_summary,
            "best_train_mse",
            format_cell=_mantissa_quartiles_formatter(LOSS_MIN_EXPONENT),
        )
        area_cells = metric_cells(
            methods_summary, "learning_area", format_cell=_format_one_decimal_quartiles
        )
        loss_rows.append(" & ".join(pair + loss_cells) + r" \\")
        area_rows.append(" & ".join(pair + area_cells) + r" \\")
        if show_nan:
            nan_cells = [
                str(nan_counts[method][row_index].diverged)
                for method in NON_HOLOMORPHIC_BETA_SWEEP_METHODS
            ]
            nan_rows.append(" & ".join(pair + nan_cells) + r" \\")

    # One block per metric, stacked rather than side by side so the table fits at \small.
    blocks = [
        (r"$\mathscr{L}_{\min}^{(m)}\ (\times10^{-6})$", loss_rows),
        (r"$A^{(m)}$", area_rows),
    ]
    if show_nan:
        blocks.append(("NaN", nan_rows))
    body = []
    for label, rows in blocks:
        body += [
            r"\midrule",
            rf" & & \multicolumn{{{n_methods}}}{{c}}{{{label}}} \\",
            rf"\cmidrule(lr){{3-{2 + n_methods}}}",
            *rows,
        ]

    caption = r"$\beta_1$/$\beta_2$ sensitivity for the Non-holomorphic benchmark."

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:optimizer_beta_sweep_non_holomorphic}",
        # Natural-width columns shrunk by adjustbox: tabularx X columns made wide cells overlap.
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{adjustbox}{max width=\textwidth,center}",
        rf"\begin{{tabular}}{{@{{}}ll*{{{n_methods}}}{{c}}@{{}}}}",
        r"\toprule",
        rf"$\beta_1$ & $\beta_2$ & {header_methods} \\",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_non_holomorphic_beta_sweep_table(results_directory: Path, main_tex_path: Path) -> bool:
    """ """

    table = render_non_holomorphic_beta_sweep_table(results_directory)
    begin_marker, end_marker = _table_markers("optimizer_beta_sweep", "non_holomorphic")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


NON_HOLOMORPHIC_LR_COLUMN_LABELS: tuple[str, ...] = (r"$\alpha/10$", r"$\alpha$", r"$10\alpha$")


def _non_holomorphic_architecture_directories(results_directory: Path) -> list[Path]:
    """The two architecture directories (primary first, then secondary), in
    ``update_non_holomorphic_lr_figure``'s order."""

    directories = sorted(
        results_directory.glob("width_*_depth_*"),
        key=lambda path: tuple(int(part) for part in path.name.split("_")[1::2]),
    )
    if len(directories) != 2:
        raise FileNotFoundError(
            f"Expected exactly two architecture directories for 'non_holomorphic' "
            f"in {results_directory}, found {len(directories)}"
        )
    return directories


def render_non_holomorphic_lr_sweep_table(results_directory: Path) -> str:
    """Render the non_holomorphic learning-rate sweep table: one row per
    (architecture, method), one column group per rate."""

    methods = NON_HOLOMORPHIC_BETA_SWEEP_METHODS
    n_methods = len(methods)
    LOSS_MIN_EXPONENT = -6

    architecture_directories = _non_holomorphic_architecture_directories(results_directory)
    architecture_names = ("Primary", "Secondary")
    sgd_timing = _load_sgd_timing(CASE_FOLDER["non_holomorphic"])

    grid: list[list[dict[str, Any]]] = []
    architecture_regime_keys: list[str] = []
    nan_grid_points: list[tuple[str, Path]] = []
    for architecture_index, architecture_directory in enumerate(architecture_directories):
        width, depth = (int(part) for part in architecture_directory.name.split("_")[1::2])
        architecture_regime_keys.append(f"width_{width}_depth_{depth}")
        per_multiplier = []
        for multiplier, column_label in zip(
            NON_HOLOMORPHIC_LR_MULTIPLIERS, NON_HOLOMORPHIC_LR_COLUMN_LABELS
        ):
            grid_directory = _non_holomorphic_lr_directory(
                results_directory, multiplier, width, depth
            )
            summary = _load_summary(grid_directory / "summary.json")
            per_multiplier.append(_methods_summary(summary))
            nan_grid_points.append(
                (
                    f"the {architecture_names[architecture_index].lower()} architecture "
                    f"at {column_label}",
                    grid_directory / "history.csv",
                )
            )
        grid.append(per_multiplier)

    def _column_cells(methods_summary: dict[str, Any], key: str, format_cell) -> list[str]:
        best_method = _median_argmin(
            {method: methods_summary.get(method, {}).get(key) for method in methods}
        )
        cells = []
        for method in methods:
            cell = _quartile_cell(methods_summary.get(method, {}).get(key), format_cell)
            if cell == _EMPTY_CELL:
                cells.append(cell)
                continue
            cells.append(rf"\mbox{{{_bold(cell) if method == best_method else cell}}}")
        return cells

    body = []
    n_lr = len(NON_HOLOMORPHIC_LR_MULTIPLIERS)

    nan_counts = _nan_seed_counts_by_method(nan_grid_points, methods, loss_column="train_mse")
    show_nan = nan_counts is not None

    # Training time is reported at the base rate only; the other metrics at every rate.
    time_multiplier_index = NON_HOLOMORPHIC_LR_MULTIPLIERS.index(1.0)
    n_data_columns = 2 * n_lr + 1 + (n_lr if show_nan else 0)

    for architecture_index, architecture_name in enumerate(architecture_names):
        columns: list[list[str]] = []
        sgd_seconds = _sgd_regime_median_seconds(
            sgd_timing, architecture_regime_keys[architecture_index]
        )
        for key, format_cell in (
            ("best_train_mse", _mantissa_quartiles_formatter(LOSS_MIN_EXPONENT)),
            ("learning_area", _format_one_decimal_quartiles),
        ):
            for multiplier_index in range(n_lr):
                methods_summary = _without_learning_area_of_diverged(
                    grid[architecture_index][multiplier_index],
                    nan_counts,
                    architecture_index * n_lr + multiplier_index,
                )
                columns.append(_column_cells(methods_summary, key, format_cell))
        columns.append(
            _column_cells(
                grid[architecture_index][time_multiplier_index],
                "training_seconds",
                _time_vs_sgd_formatter(sgd_seconds),
            )
        )
        if show_nan:
            for multiplier_index in range(n_lr):
                grid_point = architecture_index * n_lr + multiplier_index
                columns.append(
                    [str(nan_counts[method][grid_point].diverged) for method in methods]
                )
        for method_index, method in enumerate(methods):
            row_cells = [cell_list[method_index] for cell_list in columns]
            architecture_cell = (
                rf"\multirow{{{n_methods}}}{{*}}{{{architecture_name}}}"
                if method_index == 0
                else ""
            )
            body.append(
                f"{architecture_cell} & {METHOD_DISPLAY_NAMES.get(method, method)} & "
                + " & ".join(row_cells)
                + r" \\"
            )
        if architecture_index == 0:
            body.append(rf"\cmidrule(lr){{1-{2 + n_data_columns}}}")

    lr_header = " & ".join(NON_HOLOMORPHIC_LR_COLUMN_LABELS)
    time_header = NON_HOLOMORPHIC_LR_COLUMN_LABELS[time_multiplier_index]
    caption = r"Learning-rate sensitivity for the Non-holomorphic benchmark."

    nan_group_header = rf" & \multicolumn{{{n_lr}}}{{c}}{{NaN}}" if show_nan else ""
    nan_group_rule = (
        rf"\cmidrule(lr){{{4 + 2 * n_lr}-{3 + 3 * n_lr}}}" if show_nan else ""
    )
    nan_subheader = f" & {lr_header}" if show_nan else ""
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:optimizer_training_time_non_holomorphic}",
        # Natural-width columns shrunk by adjustbox: tabularx X columns made wide cells overlap.
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\linewidth,center}",
        rf"\begin{{tabular}}{{@{{}}ll*{{{n_data_columns}}}{{c}}@{{}}}}",
        r"\toprule",
        rf" & & \multicolumn{{{n_lr}}}{{c}}{{$\mathscr{{L}}_{{\min}}^{{(m)}}\ (\times10^{{-6}})$}} "
        rf"& \multicolumn{{{n_lr}}}{{c}}{{$A^{{(m)}}$}} "
        rf"& {_SGD_TIME_COLUMN_HEADER}{nan_group_header} \\",
        rf"\cmidrule(lr){{3-{2 + n_lr}}}\cmidrule(lr){{{3 + n_lr}-{2 + 2 * n_lr}}}"
        rf"\cmidrule(lr){{{3 + 2 * n_lr}-{3 + 2 * n_lr}}}{nan_group_rule}",
        rf"Arch. & Optimizer & {lr_header} & {lr_header} & {time_header}{nan_subheader} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_non_holomorphic_lr_sweep_table(results_directory: Path, main_tex_path: Path) -> bool:
    """Splice the non_holomorphic learning-rate sweep table into main.tex, in place
    of the shared training-time table."""

    table = render_non_holomorphic_lr_sweep_table(results_directory)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "non_holomorphic")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


# Must match BETA_SWEEP_PANEL_NAME in 1-non_holomorphic/plots.py.
NON_HOLOMORPHIC_BETA_SWEEP_PANEL_NAME = "loss_panel.pdf"


def update_non_holomorphic_beta_sweep_figure(results_directory: Path, destination: Path) -> bool:
    """Concatenate the 9 (beta_1, beta_2) grid points' print-size loss panels into one
    9-page figure, row-major over beta_1 x beta_2."""

    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("Combining comparison reports requires the 'pdfunite' executable.")

    pdfs = []
    for beta_1 in NON_HOLOMORPHIC_BETA_1_VALUES:
        for beta_2 in NON_HOLOMORPHIC_BETA_2_VALUES:
            pdf = _non_holomorphic_beta_pair_directory(results_directory, beta_1, beta_2) / NON_HOLOMORPHIC_BETA_SWEEP_PANEL_NAME
            if not pdf.is_file():
                raise FileNotFoundError(f"Missing loss panel PDF for beta sweep: {pdf}")
            pdfs.append(pdf)

    with tempfile.TemporaryDirectory(prefix="non-holomorphic-beta-") as temporary_directory:
        combined_pdf = Path(temporary_directory) / "combined.pdf"
        subprocess.run([pdfunite, *(str(pdf) for pdf in pdfs), str(combined_pdf)], check=True)
        figure_changed = update_paper_figure(combined_pdf, destination)
    legend_changed = _update_losses_legend(
        pdfs,
        destination,
        legend_name=f"{Path(NON_HOLOMORPHIC_BETA_SWEEP_PANEL_NAME).stem}_legend.pdf",
    )
    return figure_changed or legend_changed


# Must match EXTRA_GRID_POINTS in 3-multivariate_c4/sweep_lr.py.
MULTIVARIATE_C4_LR_MULTIPLIERS: tuple[float, ...] = (0.1, 1.0, 10.0)
MULTIVARIATE_C4_LR_COLUMN_LABELS: tuple[str, ...] = (r"$\alpha/10$", r"$\alpha$", r"$10\alpha$")
MULTIVARIATE_C4_PRECISIONS: tuple[str, ...] = ("32", "64")
MULTIVARIATE_C4_PRECISION_ROW_LABELS: dict[str, str] = {"32": "Single", "64": "Double"}
MULTIVARIATE_C4_LR_SWEEP_METHODS: tuple[str, ...] = optimizer_config.METHODS
# Must match PAPER_LOSSES_NAME in 3-multivariate_c4/plots.py.
MULTIVARIATE_C4_PAPER_LOSSES_NAME = "paper_losses.pdf"


def _multivariate_c4_grid_directory(
    results_directory: Path, precision: str, multiplier: float, width: int, depth: int
) -> Path:
    """results/ subdirectory for one (precision, multiplier) grid point (see
    3-multivariate_c4/sweep_lr._grid_output_dir)."""

    architecture_name = f"width_{width}_depth_{depth}"
    root = results_directory if precision == "32" else results_directory / "double_precision"
    if multiplier == 1.0:
        return root / architecture_name
    return root / f"lr_{multiplier:g}x" / architecture_name


def _multivariate_c4_architecture(summary: dict[str, Any]) -> tuple[int, int]:
    """The single ``(width, depth)`` this case trains, read from the base grid
    point's ``summary.json``."""

    experiment = summary["experiment"]
    architectures = experiment.get("architectures")
    if architectures is not None and len(architectures) != 1:
        raise ValueError(
            "'multivariate_c4' trains a single architecture, but its summary "
            f"reports {len(architectures)} -- the summary predates the "
            "single-architecture config; rerun 3-multivariate_c4/train.py"
        )
    return int(experiment["hidden_width"]), int(experiment["hidden_layers"])


def update_multivariate_c4_lr_figure(
    summary: dict[str, Any], folder: Path, destination: Path
) -> bool:
    """Concatenate the multivariate_c4 paper loss rows over precisions and swept
    learning rates into one 6-page figure."""

    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("Combining comparison reports requires the 'pdfunite' executable.")

    results_directory = folder / "results"
    width, depth = _multivariate_c4_architecture(summary)

    pdfs = []
    for precision in MULTIVARIATE_C4_PRECISIONS:
        for multiplier in MULTIVARIATE_C4_LR_MULTIPLIERS:
            pdf = (
                _multivariate_c4_grid_directory(
                    results_directory, precision, multiplier, width, depth
                )
                / MULTIVARIATE_C4_PAPER_LOSSES_NAME
            )
            if not pdf.is_file():
                raise FileNotFoundError(f"Missing paper loss row PDF for LR sweep: {pdf}")
            pdfs.append(pdf)

    with tempfile.TemporaryDirectory(prefix="multivariate-c4-lr-") as temporary_directory:
        combined_pdf = Path(temporary_directory) / "combined.pdf"
        subprocess.run([pdfunite, *(str(pdf) for pdf in pdfs), str(combined_pdf)], check=True)
        figure_changed = update_paper_figure(combined_pdf, destination)
    legend_changed = _update_losses_legend(
        pdfs,
        destination,
        legend_name=f"{Path(MULTIVARIATE_C4_PAPER_LOSSES_NAME).stem}_legend.pdf",
    )
    return figure_changed or legend_changed


def _multivariate_c4_figure_caption(
    architecture: list[int],
    parameter_count: int,
    base_learning_rate: float,
    seed_count: int,
) -> str:
    """Caption for the multivariate_c4 loss figure, naming the precisions and
    learning rates actually swept."""

    architecture_tuple = "(" + ",".join(str(width) for width in architecture) + ")"
    rate_labels = list(MULTIVARIATE_C4_LR_COLUMN_LABELS)
    rates_prose = " and ".join(
        filter(None, [", ".join(rate_labels[:-1]), rate_labels[-1]])
    )
    return (
        r"Training-set and test-set mean squared errors for the Multivariate "
        r"C4 optimizer benchmark, on the single benchmark architecture "
        rf"${architecture_tuple}$ (\numprint{{{parameter_count}}} parameters; "
        r"\Cref{tab:optimizer_settings_multivariate_c4}). The two precision "
        r"blocks show the same network trained in \emph{top}, single-precision "
        r"(\texttt{complex64}) and \emph{bottom}, double-precision "
        r"(\texttt{complex128}) arithmetic. Within each precision block, the "
        rf"rows sweep the base learning rate $\alpha = {_format_number(base_learning_rate)}$ "
        rf"across {rates_prose} from top to bottom. Solid curves show the "
        rf"pointwise mean over all ${seed_count}$ seeds; shaded regions span "
        r"the corresponding minimum and maximum, also shown by dashed and "
        r"dotted curves, respectively."
    )


def render_multivariate_c4_lr_figure_block(
    summary: dict[str, Any], results_directory: Path
) -> str:
    """Render the multivariate_c4 loss figure's LaTeX: one block per precision, one
    row per swept learning rate."""

    width, depth = _multivariate_c4_architecture(summary)
    experiment = summary["experiment"]
    caption = _multivariate_c4_figure_caption(
        architecture=experiment["architecture"],
        parameter_count=_read_parameter_count(
            results_directory / f"width_{width}_depth_{depth}"
        ),
        base_learning_rate=experiment["learning_rate"],
        seed_count=len(experiment["seeds"]),
    )
    return _render_stacked_losses_figure_block(
        title=r"Loss: TEST 3 -- multivariate $\CC^4$",
        figure_stem="optimizer_losses_multivariate_c4",
        row_labels=MULTIVARIATE_C4_LR_COLUMN_LABELS,
        caption=caption,
        block_labels=[
            f"{MULTIVARIATE_C4_PRECISION_ROW_LABELS[precision]} precision"
            for precision in MULTIVARIATE_C4_PRECISIONS
        ],
    )


def update_multivariate_c4_lr_figure_block(
    summary: dict[str, Any], results_directory: Path, main_tex_path: Path
) -> bool:
    """ """

    figure_block = render_multivariate_c4_lr_figure_block(summary, results_directory)
    begin_marker, end_marker = _figure_markers("optimizer_losses_multivariate_c4")
    return _splice_table_into_main_tex(figure_block, main_tex_path, begin_marker, end_marker)


def render_multivariate_c4_settings_table(
    summary: dict[str, Any], results_directory: Path
) -> str:
    """Render the multivariate_c4 settings table: one architecture, swept over
    precision and learning rate."""

    experiment = summary["experiment"]
    shared_design = summary["shared_design"]
    seeds = _seed_set_text(experiment["seeds"])

    width, depth = _multivariate_c4_architecture(summary)
    parameter_count = _read_parameter_count(results_directory / f"width_{width}_depth_{depth}")
    architecture = "$({})$, {} trainable parameters".format(
        ",".join(str(n) for n in experiment["architecture"]), _math(parameter_count)
    )

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Configuration of the Multivariate C4 optimizer benchmark.}",
        r"\label{tab:optimizer_settings_multivariate_c4}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}X@{}}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        rf"Network architecture & {architecture} \\",
        rf"Hidden activation & {ACTIVATION_TEXT[shared_design['hidden_activation']]} \\",
        rf"Training / test set & {_math(experiment['train_size'])} / "
        rf"{_math(experiment['test_size'])} points (test: 20\%) \\",
        rf"Mini-batch size & {_math(experiment['minibatch_size'])} points \\",
        rf"Updates / seeds & {_math(experiment['steps'])} / ${seeds}$ \\",
        r"Arithmetic & \texttt{complex64} and \texttt{complex128} "
        r"(single- and double-precision real components; both swept) \\",
        r"Initialization & Glorot/Xavier normal rule \cite{Glorot2010}, applied separately "
        r"to the real and imaginary parts; zero biases \\",
        rf"Base learning rate $\alpha$ & ${_format_number(experiment['learning_rate'])}$, "
        rf"swept over $\alpha/10$, $\alpha$ and $10\alpha$ \\",
        r"Optimizer hyperparameters & shared across every case; see "
        r"\Cref{tab:optimizer_settings_shared} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_multivariate_c4_settings_table(
    summary: dict[str, Any], results_directory: Path, main_tex_path: Path
) -> bool:
    """ """

    table = render_multivariate_c4_settings_table(summary, results_directory)
    begin_marker, end_marker = _table_markers("optimizer_settings", "multivariate_c4")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def render_multivariate_c4_lr_sweep_table(
    base_summary: dict[str, Any], results_directory: Path
) -> str:
    """Render the multivariate_c4 learning-rate x precision sweep table: one row per
    (precision, method), one column group per rate."""

    methods = MULTIVARIATE_C4_LR_SWEEP_METHODS
    n_methods = len(methods)
    n_lr = len(MULTIVARIATE_C4_LR_MULTIPLIERS)
    width, depth = _multivariate_c4_architecture(base_summary)
    sgd_timing = _load_sgd_timing(CASE_FOLDER["multivariate_c4"])
    _precision_regime = {"32": "complex64", "64": "complex128"}

    grid: list[list[dict[str, Any]]] = []
    base_learning_rate: float | None = None
    nan_grid_points: list[tuple[str, Path]] = []
    for precision in MULTIVARIATE_C4_PRECISIONS:
        row_label = MULTIVARIATE_C4_PRECISION_ROW_LABELS[precision].lower()
        per_multiplier = []
        for multiplier, column_label in zip(
            MULTIVARIATE_C4_LR_MULTIPLIERS, MULTIVARIATE_C4_LR_COLUMN_LABELS
        ):
            grid_directory = _multivariate_c4_grid_directory(
                results_directory, precision, multiplier, width, depth
            )
            summary = _load_summary(grid_directory / "summary.json")
            per_multiplier.append(_methods_summary(summary))
            if multiplier == 1.0:
                base_learning_rate = summary["experiment"]["learning_rate"]
            nan_grid_points.append(
                (
                    f"the {row_label}-precision runs at {column_label}",
                    grid_directory / "history.csv",
                )
            )
        grid.append(per_multiplier)
    if base_learning_rate is None:
        raise ValueError("no 1x grid point found for the multivariate_c4 learning-rate sweep")

    median_losses: list[float] = []
    for per_multiplier in grid:
        for methods_summary in per_multiplier:
            for method in methods:
                quartiles = _quartiles((methods_summary.get(method, {}) or {}).get("best_train_mse"))
                if quartiles is not None and quartiles[1] > 0.0:
                    median_losses.append(quartiles[1])
    median_losses.sort()
    loss_exponent = (
        int(math.floor(math.log10(median_losses[len(median_losses) // 2]))) if median_losses else -6
    )

    def _column_cells(methods_summary: dict[str, Any], key: str, format_cell) -> list[str]:
        best_method = _median_argmin(
            {method: methods_summary.get(method, {}).get(key) for method in methods}
        )
        cells = []
        for method in methods:
            cell = _quartile_cell(methods_summary.get(method, {}).get(key), format_cell)
            if cell == _EMPTY_CELL:
                cells.append(cell)
                continue
            cells.append(rf"\mbox{{{_bold(cell) if method == best_method else cell}}}")
        return cells

    nan_counts = _nan_seed_counts_by_method(nan_grid_points, methods, loss_column="train_mse")
    show_nan = nan_counts is not None

    # Training time is reported at the base rate only; the other metrics at every rate.
    time_multiplier_index = MULTIVARIATE_C4_LR_MULTIPLIERS.index(1.0)
    n_data_columns = 2 * n_lr + 1 + (n_lr if show_nan else 0)

    body = []
    for precision_index, precision in enumerate(MULTIVARIATE_C4_PRECISIONS):
        columns: list[list[str]] = []
        sgd_seconds = _sgd_regime_median_seconds(
            sgd_timing, _precision_regime.get(precision, precision)
        )
        for key, format_cell in (
            ("best_train_mse", _mantissa_quartiles_formatter(loss_exponent)),
            ("learning_area", _format_fixed_quartiles),
        ):
            for multiplier_index in range(n_lr):
                methods_summary = _without_learning_area_of_diverged(
                    grid[precision_index][multiplier_index],
                    nan_counts,
                    precision_index * n_lr + multiplier_index,
                )
                columns.append(_column_cells(methods_summary, key, format_cell))
        columns.append(
            _column_cells(
                grid[precision_index][time_multiplier_index],
                "training_seconds",
                _time_vs_sgd_formatter(sgd_seconds),
            )
        )
        if show_nan:
            for multiplier_index in range(n_lr):
                grid_point = precision_index * n_lr + multiplier_index
                columns.append(
                    [str(nan_counts[method][grid_point].diverged) for method in methods]
                )
        for method_index, method in enumerate(methods):
            row_cells = [cell_list[method_index] for cell_list in columns]
            precision_cell = (
                rf"\multirow{{{n_methods}}}{{*}}{{{MULTIVARIATE_C4_PRECISION_ROW_LABELS[precision]}}}"
                if method_index == 0
                else ""
            )
            body.append(
                f"{precision_cell} & {METHOD_DISPLAY_NAMES.get(method, method)} & "
                + " & ".join(row_cells)
                + r" \\"
            )
        if precision_index == 0:
            body.append(rf"\cmidrule(lr){{1-{2 + n_data_columns}}}")

    lr_header = " & ".join(MULTIVARIATE_C4_LR_COLUMN_LABELS)
    time_header = MULTIVARIATE_C4_LR_COLUMN_LABELS[time_multiplier_index]
    caption = r"Learning-rate and precision sensitivity for the Multivariate C4 benchmark."

    nan_group_header = rf" & \multicolumn{{{n_lr}}}{{c}}{{NaN}}" if show_nan else ""
    nan_group_rule = (
        rf"\cmidrule(lr){{{4 + 2 * n_lr}-{3 + 3 * n_lr}}}" if show_nan else ""
    )
    nan_subheader = f" & {lr_header}" if show_nan else ""
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:optimizer_training_time_multivariate_c4}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\linewidth,center}",
        rf"\begin{{tabular}}{{@{{}}ll*{{{n_data_columns}}}{{c}}@{{}}}}",
        r"\toprule",
        rf" & & \multicolumn{{{n_lr}}}{{c}}{{$\mathscr{{L}}_{{\min}}^{{(m)}}\ "
        rf"(\times10^{{{loss_exponent}}})$}} "
        rf"& \multicolumn{{{n_lr}}}{{c}}{{$A^{{(m)}}$}} "
        rf"& {_SGD_TIME_COLUMN_HEADER}{nan_group_header} \\",
        rf"\cmidrule(lr){{3-{2 + n_lr}}}\cmidrule(lr){{{3 + n_lr}-{2 + 2 * n_lr}}}"
        rf"\cmidrule(lr){{{3 + 2 * n_lr}-{3 + 2 * n_lr}}}{nan_group_rule}",
        rf"Prec. & Optimizer & {lr_header} & {lr_header} & {time_header}{nan_subheader} \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table*}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_multivariate_c4_lr_sweep_table(
    summary: dict[str, Any], results_directory: Path, main_tex_path: Path
) -> bool:
    """Splice the multivariate_c4 learning-rate/precision sweep table into main.tex,
    in place of the shared training-time table."""

    table = render_multivariate_c4_lr_sweep_table(summary, results_directory)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "multivariate_c4")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def copy_pinn_figures(source_directory: Path, destination_directory: Path) -> list[Path]:
    """ """

    updated = []
    for source in sorted(source_directory.glob("*.png")):
        destination = destination_directory / f"pinn_{source.name}"
        if _atomic_copy(source, destination):
            updated.append(destination)
    return updated


def copy_case_loss_comparison(
    case_results_directory: Path, destination: Path, source_name: str = "combined_losses.pdf"
) -> bool:
    """Copy a case's standalone loss figure (and its legend) into the paper's figures
    directory."""

    source = case_results_directory / source_name
    if not source.is_file():
        return False
    figure_changed = update_paper_figure(source, destination)
    legend_changed = _update_losses_legend(
        source, destination, legend_name=f"{source.stem}_legend.pdf"
    )
    return figure_changed or legend_changed


def _unite_cifar10_architecture_pdfs(
    results_directory: Path, destination: Path, stem: str
) -> tuple[list[Path], bool]:
    """Concatenate TEST 5's primary- and secondary-architecture
    ``results/<architecture>/<stem>.pdf`` into one two-page figure."""

    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("Combining comparison reports requires the 'pdfunite' executable.")

    architecture_pdfs = []
    for architecture in CIFAR10_ARCHITECTURES:
        architecture_pdf = results_directory / architecture / f"{stem}.pdf"
        if not architecture_pdf.is_file():
            raise FileNotFoundError(
                f"Missing {architecture} {stem.replace('_', '-')} PDF for TEST 5: {architecture_pdf}"
            )
        architecture_pdfs.append(architecture_pdf)

    with tempfile.TemporaryDirectory(prefix=f"cifar10-{stem}-") as temporary_directory:
        combined_pdf = Path(temporary_directory) / "combined.pdf"
        subprocess.run(
            [pdfunite, *(str(pdf) for pdf in architecture_pdfs), str(combined_pdf)], check=True
        )
        figure_changed = update_paper_figure(combined_pdf, destination)
    return architecture_pdfs, figure_changed


def update_cifar10_losses_figure(results_directory: Path, destination: Path) -> bool:
    """TEST 5's two-page loss figure plus its standalone legend."""

    architecture_pdfs, figure_changed = _unite_cifar10_architecture_pdfs(
        results_directory, destination, "combined_losses"
    )
    legend_changed = _update_losses_legend(architecture_pdfs, destination)
    return figure_changed or legend_changed


def update_cifar10_accuracies_figure(results_directory: Path, destination: Path) -> bool:
    """TEST 5's two-page classification-accuracy figure."""

    _, figure_changed = _unite_cifar10_architecture_pdfs(
        results_directory, destination, "combined_accuracies"
    )
    return figure_changed


def _load_summary(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Optimizer summary does not exist: {path}") from None
    if not isinstance(summary.get("experiment"), dict):
        raise ValueError(f"Missing experiment configuration in {path}")
    if not isinstance(summary.get("optimizer_hyperparameters"), dict):
        raise ValueError(f"Missing optimizer hyperparameters in {path}")
    return summary


def _read_parameter_count(results_directory: Path) -> int:
    """Read a case's trainable-parameter count from
    ``results_directory/parameter_count.txt``."""

    path = results_directory / "parameter_count.txt"
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter count does not exist: {path}") from None


def _format_number(value: int | float) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if value == 0.0:
        return "0"
    absolute = abs(value)
    if absolute <= 1.0e-3 or absolute >= 1.0e4:
        exponent = f"{value:.8e}".split("e")
        coefficient = exponent[0].rstrip("0").rstrip(".")
        power = int(exponent[1])
        return rf"{coefficient}\times 10^{{{power}}}"
    return f"{value:g}"


def _math(value: int | float) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 1000:
        return rf"\numprint{{{value}}}"
    return f"${_format_number(value)}$"


def _seed_set_text(seeds: Iterable[int]) -> str:
    """The seed set for a settings table's ``Updates / seeds`` cell, as the body of
    a math-mode set."""

    seed_list = list(seeds)
    is_consecutive_run = len(seed_list) >= 3 and all(
        later - earlier == 1 for earlier, later in zip(seed_list, seed_list[1:])
    )
    if is_consecutive_run:
        return rf"\{{{seed_list[0]},\ldots,{seed_list[-1]}\}}"
    return r"\{" + ", ".join(str(seed) for seed in seed_list) + r"\}"


def _aura_light_step_text(learning_rate_scale: float) -> str:
    """AURA light's base step for the settings table: the shared ``alpha``, or its
    ``learning_rate_scale`` multiple."""

    if learning_rate_scale == 1.0:
        return r"\alpha"
    return rf"{_format_number(learning_rate_scale)}\,\alpha"


def _gate_settings_text(settings: dict[str, Any]) -> str:
    """The gate block shared by AURA/ASTRA/ECLIPSE/AURA light: chi and Psi
    thresholds, growth/shrink rates, gamma clamp."""

    return (
        rf"$\chi_\mathrm{{a}}={_format_number(settings['chi_alignment'])}$, "
        rf"$\chi_\mathrm{{o}}={_format_number(settings['chi_opposition'])}$, "
        rf"$\Psi_\mathrm{{a}}={_format_number(settings['psi_alignment'])}$, "
        rf"$\Psi_\mathrm{{o}}={_format_number(settings['psi_opposition'])}$, "
        rf"$\eta_-={_format_number(settings['eta_minus'])}$, "
        rf"$\eta_+={_format_number(settings['eta_plus'])}$, "
        rf"$\gamma\in[{_format_number(settings['gamma_min'])},"
        rf"{_format_number(settings['gamma_max'])}]$"
    )


# Every method in optimizer_config.METHODS needs an entry here; a missing one raises.
_OPTIMIZER_ROW_BUILDERS: dict[str, tuple[tuple[str, ...], Callable[..., str]]] = {
    optimizer_config.METHOD_ADAM: (
        ("adam",),
        lambda adam: (
            rf"$\beta_1={_format_number(adam['beta_1'])}$, "
            rf"$\beta_2={_format_number(adam['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(adam['epsilon'])}$"
        ),
    ),
    optimizer_config.METHOD_ADAM_VARIABLE_LR: (
        (),
        lambda: (
            r"Same $\beta_1,\beta_2,\eps_{\mathrm A}$ as Adam; learning rate "
            r"$\alpha$ for the first "
            rf"${_format_number(100 * optimizer_config.ADAM_VARIABLE_LR_DROP_AT_FRACTION)}\%$ "
            r"of updates, then $\alpha/"
            rf"{_format_number(optimizer_config.ADAM_VARIABLE_LR_DROP_FACTOR)}$"
        ),
    ),
    optimizer_config.METHOD_ADAM_AURA: (
        # hyperparameters_for_method keeps the pre-rename "aura" snapshot key.
        ("aura",),
        lambda aura: (
            rf"$\beta_\zeta={_format_number(aura['beta_zeta'])}$, "
            rf"$\eps_{{\mathrm E}}={_format_number(aura['epsilon_e'])}$, "
            + _gate_settings_text(aura)
            + rf", $\lambda={_format_number(aura['weight_decay'])}$"
        ),
    ),
    optimizer_config.METHOD_ASTRA: (
        ("astra", "adam"),
        lambda astra, adam: (
            rf"$\beta_\zeta={_format_number(astra['beta_zeta'])}$, "
            rf"$\eps_{{\mathrm E}}={_format_number(astra['epsilon_e'])}$, "
            + _gate_settings_text(astra)
            + rf", $\ell={_format_number(astra['leak_rate'])}$; probe "
            rf"decorrelated at $\beta_1={_format_number(adam['beta_1'])}$"
        ),
    ),
    optimizer_config.METHOD_ECLIPSE: (
        ("eclipse", "adam"),
        lambda eclipse, adam: (
            rf"$\eps_{{\mathrm E}}={_format_number(eclipse['epsilon_e'])}$, "
            + _gate_settings_text(eclipse)
            + rf"; innovation EMA at $\beta_2={_format_number(adam['beta_2'])}$"
        ),
    ),
    optimizer_config.METHOD_PULSAR: (
        ("pulsar", "adam"),
        lambda pulsar, adam: (
            rf"$\kappa={_format_number(pulsar['kappa'])}$, "
            rf"$\eps_{{\mathrm E}}={_format_number(pulsar['epsilon_e'])}$; "
            r"per-step gain in $["
            rf"{_format_number(round((1 - pulsar['kappa']) / (1 + pulsar['kappa']), 3))},"
            rf"{_format_number(round((1 + pulsar['kappa']) / (1 - pulsar['kappa']), 3))}]$; "
            rf"same $\beta_1={_format_number(adam['beta_1'])}$, "
            rf"$\beta_2={_format_number(adam['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(adam['epsilon'])}$ as Adam"
        ),
    ),
    optimizer_config.METHOD_AURA_LIGHT: (
        ("aura_light",),
        lambda aura_light: (
            rf"$\beta_1={_format_number(aura_light['beta_1'])}$, "
            rf"$\beta_\zeta={_format_number(aura_light['beta_zeta'])}$, "
            rf"$\eps_{{\mathrm E}}={_format_number(aura_light['epsilon_e'])}$, "
            + _gate_settings_text(aura_light)
            + rf"; step ${_aura_light_step_text(aura_light['learning_rate_scale'])}$"
        ),
    ),
    optimizer_config.METHOD_RPROP: (
        ("rprop",),
        lambda rprop: (
            rf"$\eta_-={_format_number(rprop['eta_minus'])}$, "
            rf"$\eta_+={_format_number(rprop['eta_plus'])}$, "
            rf"$\Delta\in[{_format_number(rprop['min_step_size'])},"
            rf"{_format_number(rprop['max_step_size'])}]$; $\Delta_0=\alpha$"
        ),
    ),
    optimizer_config.METHOD_NADAMW: (
        ("nadamw",),
        lambda nadamw: (
            rf"$\beta_1={_format_number(nadamw['beta_1'])}$, "
            rf"$\beta_2={_format_number(nadamw['beta_2'])}$, "
            rf"$\epsilon={_format_number(nadamw['epsilon'])}$, "
            rf"weight decay: ${_format_number(nadamw['weight_decay'])}$"
        ),
    ),
    optimizer_config.METHOD_CvAMSGrad: (
        ("cvamsgrad",),
        lambda cvamsgrad: (
            rf"$\beta_1={_format_number(cvamsgrad['beta_1'])}$, "
            rf"$\beta_2={_format_number(cvamsgrad['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(cvamsgrad['epsilon'])}$; "
            rf"{'bias correction included' if cvamsgrad['bias_correction'] else 'no bias correction'}"
        ),
    ),
    optimizer_config.METHOD_MUON: (
        ("muon",),
        lambda muon: (
            rf"$\beta={_format_number(muon['beta'])}$, "
            rf"Newton--Schulz steps: {_format_number(muon['ns_steps'])}, "
            rf"Nesterov: {'yes' if muon['nesterov'] else 'no'}, "
            rf"weight decay: ${_format_number(muon['weight_decay'])}$; "
            rf"matrix step ${_aura_light_step_text(muon['learning_rate_scale'])}$; "
            rf"bias fallback AdamW at $\beta_1={_format_number(muon['beta_1'])}$, "
            rf"$\beta_2={_format_number(muon['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(muon['epsilon'])}$"
        ),
    ),
    optimizer_config.METHOD_MUON_AURA: (
        ("muon_aura",),
        lambda muon_aura: (
            rf"$\beta={_format_number(muon_aura['beta'])}$, "
            rf"Newton--Schulz steps: {_format_number(muon_aura['ns_steps'])}, "
            rf"Nesterov: {'yes' if muon_aura['nesterov'] else 'no'}, "
            rf"matrix step ${_aura_light_step_text(muon_aura['learning_rate_scale'])}$; "
            rf"$\beta_\zeta={_format_number(muon_aura['beta_zeta'])}$, "
            rf"$\eps_{{\mathrm E}}={_format_number(muon_aura['epsilon_e'])}$, "
            + _gate_settings_text(muon_aura)
            + rf", $\lambda={_format_number(muon_aura['weight_decay'])}$; "
            rf"bias direction Adam at $\beta_1={_format_number(muon_aura['beta_1'])}$, "
            rf"$\beta_2={_format_number(muon_aura['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(muon_aura['epsilon'])}$"
        ),
    ),
    optimizer_config.METHOD_ADAM_AURA_S: (
        ("aura_s",),
        lambda aura_s: _aura_s_gate_text(aura_s)
        + rf", weight decay: ${_format_number(aura_s['weight_decay'])}$",
    ),
    optimizer_config.METHOD_MUON_AURA_S: (
        ("muon_aura_s",),
        lambda muon_aura_s: (
            rf"$\beta={_format_number(muon_aura_s['beta'])}$, "
            rf"Newton--Schulz steps: {_format_number(muon_aura_s['ns_steps'])}, "
            rf"Nesterov: {'yes' if muon_aura_s['nesterov'] else 'no'}, "
            rf"matrix step ${_aura_light_step_text(muon_aura_s['learning_rate_scale'])}$; "
            + _aura_s_gate_text(muon_aura_s)
            + rf", weight decay: ${_format_number(muon_aura_s['weight_decay'])}$; "
            rf"bias direction Adam at $\beta_1={_format_number(muon_aura_s['beta_1'])}$, "
            rf"$\beta_2={_format_number(muon_aura_s['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(muon_aura_s['epsilon'])}$"
        ),
    ),
    optimizer_config.METHOD_ADAM_AURA_SIGN: (
        ("aura_sign",),
        lambda aura_sign: _aura_sign_gate_text(aura_sign, "beta")
        + rf", weight decay: ${_format_number(aura_sign['weight_decay'])}$",
    ),
    optimizer_config.METHOD_MUON_AURA_SIGN: (
        ("muon_aura_sign",),
        lambda muon_aura_sign: (
            rf"$\beta={_format_number(muon_aura_sign['beta'])}$, "
            rf"Newton--Schulz steps: {_format_number(muon_aura_sign['ns_steps'])}, "
            rf"Nesterov: {'yes' if muon_aura_sign['nesterov'] else 'no'}, "
            rf"matrix step ${_aura_light_step_text(muon_aura_sign['learning_rate_scale'])}$; "
            + _aura_sign_gate_text(muon_aura_sign, "sign_beta")
            + rf", weight decay: ${_format_number(muon_aura_sign['weight_decay'])}$; "
            rf"bias direction Adam at $\beta_1={_format_number(muon_aura_sign['beta_1'])}$, "
            rf"$\beta_2={_format_number(muon_aura_sign['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(muon_aura_sign['epsilon'])}$"
        ),
    ),
    optimizer_config.METHOD_ADAM_AURA_SNR: (
        ("aura_snr",),
        lambda aura_snr: _aura_snr_gate_text(aura_snr)
        + rf", weight decay: ${_format_number(aura_snr['weight_decay'])}$",
    ),
    optimizer_config.METHOD_MUON_AURA_SNR: (
        ("muon_aura_snr",),
        lambda muon_aura_snr: (
            rf"$\beta={_format_number(muon_aura_snr['beta'])}$, "
            rf"Newton--Schulz steps: {_format_number(muon_aura_snr['ns_steps'])}, "
            rf"Nesterov: {'yes' if muon_aura_snr['nesterov'] else 'no'}, "
            rf"matrix step ${_aura_light_step_text(muon_aura_snr['learning_rate_scale'])}$; "
            + _aura_snr_gate_text(muon_aura_snr)
            + rf", weight decay: ${_format_number(muon_aura_snr['weight_decay'])}$; "
            rf"bias direction Adam at $\beta_1={_format_number(muon_aura_snr['beta_1'])}$, "
            rf"$\beta_2={_format_number(muon_aura_snr['beta_2'])}$, "
            rf"$\eps_{{\mathrm A}}={_format_number(muon_aura_snr['epsilon'])}$"
        ),
    ),
}


def _aura_snr_gate_text(settings: dict[str, Any]) -> str:
    """AURA-SNR's gate block (optimizer_config.AdamAuraSnrConfig field names)."""

    return (
        rf"$\beta_\zeta={_format_number(settings['beta_zeta'])}$, "
        rf"$\eps_{{\mathrm E}}={_format_number(settings['epsilon_e'])}$, "
        rf"$\kappa_+={_format_number(settings['kappa_plus'])}$, "
        rf"$\kappa_-={_format_number(settings['kappa_minus'])}$, "
        rf"$\tau={_format_number(settings['opposition_threshold'])}$, "
        rf"$\ell={_format_number(settings['leak'])}$, "
        rf"$\gamma\in[{_format_number(settings['gamma_min'])},"
        rf"{_format_number(settings['gamma_max'])}]$"
    )


def _aura_sign_gate_text(settings: dict[str, Any], beta_key: str) -> str:
    """AURA-sign's gate block (optimizer_config.AdamAuraSignConfig field names)."""

    return (
        rf"$\beta_{{\mathrm s}}={_format_number(settings[beta_key])}$, "
        rf"$\kappa_+={_format_number(settings['kappa_plus'])}$, "
        rf"$\kappa_-={_format_number(settings['kappa_minus'])}$, "
        rf"$\tau={_format_number(settings['brake_threshold'])}$, "
        rf"$\ell={_format_number(settings['leak'])}$, "
        rf"$\gamma\in[{_format_number(settings['gamma_min'])},"
        rf"{_format_number(settings['gamma_max'])}]$"
    )


def _aura_s_gate_text(settings: dict[str, Any]) -> str:
    """AURA-S's gate block (optimizer_config.AdamAuraSConfig field names)."""

    return (
        rf"$\beta_\zeta={_format_number(settings['beta_zeta'])}$, "
        rf"$\eps_{{\mathrm E}}={_format_number(settings['epsilon_e'])}$, "
        rf"$\kappa_+={_format_number(settings['kappa_plus'])}$, "
        rf"$\kappa_-={_format_number(settings['kappa_minus'])}$, "
        rf"$\ell={_format_number(settings['leak'])}$, "
        rf"brake $\chi_\mathrm{{o}}={_format_number(settings['chi_opposition'])}$, "
        rf"$\Psi_\mathrm{{o}}={_format_number(settings['psi_opposition'])}$, "
        rf"$\kappa_\mathrm{{b}}={_format_number(settings['kappa_brake'])}$, "
        rf"$\gamma\in[{_format_number(settings['gamma_min'])},"
        rf"{_format_number(settings['gamma_max'])}]$"
    )


def _optimizer_rows(hyperparameters: dict[str, dict[str, Any]]) -> list[str]:
    """One settings row per method in ``optimizer_config.METHODS``, in that order."""

    unregistered = [
        method for method in optimizer_config.METHODS if method not in _OPTIMIZER_ROW_BUILDERS
    ]
    if unregistered:
        raise ValueError(
            "No optimizer-settings row registered for: "
            f"{unregistered}. Add an entry to _OPTIMIZER_ROW_BUILDERS."
        )

    required = {
        key
        for method in optimizer_config.METHODS
        for key in _OPTIMIZER_ROW_BUILDERS[method][0]
    }
    missing = required.difference(hyperparameters)
    if missing:
        raise ValueError(f"Missing optimizer hyperparameters for: {sorted(missing)}")

    rows = []
    for method in optimizer_config.METHODS:
        keys, build = _OPTIMIZER_ROW_BUILDERS[method]
        settings = build(*(hyperparameters[key] for key in keys))
        rows.append(rf"{METHOD_DISPLAY_NAMES.get(method, method)} & {settings} \\")
    return rows


def _shared_optimizer_hyperparameters() -> dict[str, dict[str, Any]]:
    """Optimizer hyperparameters for cases that don't record them in their own
    summary.json (4-PINN)."""

    hyperparameters: dict[str, dict[str, Any]] = {}
    for method in optimizer_config.METHODS:
        hyperparameters.update(optimizer_config.hyperparameters_for_method(method))
    return hyperparameters


def render_shared_optimizer_settings_table() -> str:
    """Render the single optimizer-hyperparameters table shared by every benchmark
    case."""

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Default hyperparameters used in the benchmark cases in \Cref{sec:benchmarks}.}",
        r"\label{tab:optimizer_settings_shared}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}lX@{}}",
        r"\toprule",
        r"Optimizer & Hyperparameters \\",
        r"\midrule",
        *_optimizer_rows(_shared_optimizer_hyperparameters()),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_shared_optimizer_settings_table(main_tex_path: Path) -> bool:
    """ """

    table = render_shared_optimizer_settings_table()
    begin_marker, end_marker = _table_markers("optimizer_settings", "shared")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def _min_train_loss_from_history(
    csv_path: Path,
    *,
    loss_column: str,
    method_column: str = "method",
    seed_column: str = "seed",
    in_decibels: bool = False,
) -> dict[str, dict[str, float | None]]:
    """Each seed's minimum training loss from history.csv, then its seed statistics
    (quartiles included), per method."""

    best_by_method_seed: dict[tuple[str, str], float] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            value = float(row[loss_column])
            if math.isnan(value):
                continue
            key = (row[method_column], row[seed_column])
            if key not in best_by_method_seed or value < best_by_method_seed[key]:
                best_by_method_seed[key] = value

    by_method: dict[str, list[float]] = {}
    for (method, seed), value in best_by_method_seed.items():
        if in_decibels:
            if value <= 0.0:
                continue
            value = 10.0 * math.log10(value)
        by_method.setdefault(method, []).append(value)

    return {method: metrics.seed_statistics(values) for method, values in by_method.items()}


def _nan_seed_counts_from_history(
    csv_path: Path,
    *,
    loss_column: str,
    method_column: str = "method",
    seed_column: str = "seed",
) -> dict[str, tuple[int, int]]:
    """Per method in ``csv_path``: (seeds whose training loss became non-finite,
    total seeds run)."""

    seeds: dict[str, set[str]] = {}
    diverged: dict[str, set[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            method, seed = row[method_column], row[seed_column]
            seeds.setdefault(method, set()).add(seed)
            raw = row.get(loss_column)
            if not raw:
                continue
            try:
                is_finite = math.isfinite(float(raw))
            except ValueError:
                is_finite = False
            if not is_finite:
                diverged.setdefault(method, set()).add(seed)
    return {
        method: (len(diverged.get(method, set())), len(method_seeds))
        for method, method_seeds in seeds.items()
    }


class _SeedNaNCount(NamedTuple):
    diverged: int
    total: int


def _nan_seed_counts_by_method(
    grid_points: Iterable[tuple[Any, Path]],
    methods: Iterable[str],
    *,
    loss_column: str = "train_mse",
) -> dict[str, list[_SeedNaNCount]] | None:
    """Per shown method, the number of seeds whose training loss went non-finite, out
    of those run, at each grid point."""

    method_list = list(methods)
    counts: dict[str, list[_SeedNaNCount]] = {method: [] for method in method_list}
    any_diverged = False
    for _key, csv_path in grid_points:
        per_method = (
            _nan_seed_counts_from_history(csv_path, loss_column=loss_column)
            if csv_path.is_file()
            else {}
        )
        for method in method_list:
            count = _SeedNaNCount(*per_method.get(method, (0, 0)))
            counts[method].append(count)
            any_diverged = any_diverged or bool(count.diverged)
    return counts if any_diverged else None


def _without_learning_area_of_diverged(
    methods_summary: dict[str, Any],
    nan_counts: dict[str, list[_SeedNaNCount]] | None,
    grid_point: int,
) -> dict[str, Any]:
    """``methods_summary`` without ``learning_area`` for the methods whose seeds all
    diverged: the area over the surviving steps alone is not comparable."""

    if nan_counts is None:
        return methods_summary
    result = {}
    for method, method_summary in methods_summary.items():
        count = nan_counts[method][grid_point] if method in nan_counts else None
        if count is not None and count.total and count.diverged == count.total:
            method_summary = {
                key: value for key, value in method_summary.items() if key != "learning_area"
            }
        result[method] = method_summary
    return result


def render_settings_table(case: str, summary: dict[str, Any]) -> str:
    """Render one case's benchmark settings and optimizer hyperparameters."""

    experiment = summary["experiment"]
    shared_design = summary["shared_design"]

    seeds = _seed_set_text(experiment["seeds"])
    precision_label = {
        "32": r"\texttt{complex64} (32-bit real components)",
        "64": r"\texttt{complex128} (64-bit real components)",
    }.get(str(experiment["precision"]), str(experiment["precision"]))

    primary_parameter_count = _read_parameter_count(
        CASE_FOLDER[case]
        / "results"
        / f"width_{experiment['hidden_width']}_depth_{experiment['hidden_layers']}"
    )
    secondary_parameter_count = _read_parameter_count(
        CASE_FOLDER[case]
        / "results"
        / f"width_{experiment['secondary_hidden_width']}_depth_{experiment['secondary_hidden_layers']}"
    )

    architecture = "$({})$, {} trainable parameters".format(
        ",".join(str(width) for width in experiment["architecture"]),
        _math(primary_parameter_count),
    )
    secondary_architecture = "$({})$, {} trainable parameters".format(
        ",".join(str(width) for width in experiment["architectures"][-1]),
        _math(secondary_parameter_count),
    )
    initialization = (
        rf"beta-scaled Gaussian-fallback rule for entire activations "
        rf"\cite{{Calafa2024}}, $\beta_\mathrm{{init}}="
        rf"{_format_number(experiment['initialization_beta'])}$"
        if case == "holomorphic"
        else r"Glorot/Xavier normal rule \cite{Glorot2010}, applied separately to the real and imaginary parts"
    )
    secondary_multiplier = experiment["small_network_learning_rate_multiplier"]

    begin_marker, _ = _table_markers("optimizer_settings", case)
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{Configuration of the {TABLE_CASE_TITLES[case]} optimizer benchmark.}}",
        rf"\label{{tab:optimizer_settings_{case}}}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}X@{}}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        rf"Primary architecture & {architecture} \\",
        rf"Secondary architecture & {secondary_architecture} \\",
        rf"Hidden activation & {ACTIVATION_TEXT[shared_design['hidden_activation']]} \\",
        rf"Training / test set & {_math(experiment['train_size'])} / "
        rf"{_math(experiment['test_size'])} points (test: 20\%) \\",
        rf"Mini-batch size & {_math(experiment['minibatch_size'])} points \\",
        rf"Updates / seeds & {_math(experiment['steps'])} / ${seeds}$ \\",
        rf"Arithmetic & {precision_label} \\",
        rf"Initialization & {initialization}; zero biases \\",
        rf"Base learning rate $\alpha$ (primary) & ${_format_number(experiment['learning_rate'])}$ \\",
        rf"Secondary-architecture learning rate & "
        rf"${_format_number(experiment['learning_rate'] * secondary_multiplier)}$ \\",
        r"Optimizer hyperparameters & shared across every case; see "
        r"\Cref{tab:optimizer_settings_shared} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


# Table entry for a metric that is missing or not applicable.
_EMPTY_CELL = "--"


def _quartiles(stats: dict[str, Any] | None) -> tuple[float, float, float] | None:
    """A summary entry's ``(q25, median, q75)``, or ``None`` when any is missing or not
    finite (summaries written before the quartiles were recorded)."""

    values = tuple((stats or {}).get(key) for key in ("q25", "median", "q75"))
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        return None
    return values


def _median_argmin(stats_by_method: dict[str, dict[str, Any] | None]) -> str | None:
    """The method with the smallest median, or ``None`` when no entry has quartiles."""

    medians = {}
    for method, stats in stats_by_method.items():
        quartiles = _quartiles(stats)
        medians[method] = None if quartiles is None else quartiles[1]
    return _argmin_method(medians)


def _quartile_cell(
    stats: dict[str, Any] | None, format_cell: Callable[[float, float, float], str]
) -> str:
    """``format_cell(q25, median, q75)``, or ``_EMPTY_CELL`` when the entry has no
    quartiles."""

    quartiles = _quartiles(stats)
    return _EMPTY_CELL if quartiles is None else format_cell(*quartiles)


def _nonzero_decimals(value: float, decimals: int) -> int:
    """``decimals``, raised until ``value`` no longer prints as zero; an exact zero keeps
    ``decimals``."""

    if value == 0.0:
        return decimals
    while float(f"{abs(value):.{decimals}f}") == 0.0:
        decimals += 1
    return decimals


def _format_quartiles(
    q25: float, median: float, q75: float, *, decimals: int, scale: float = 1.0
) -> str:
    """``median^{+(q75 - median)}_{-(median - q25)}`` in units of ``scale``; each of the
    three numbers gets ``decimals``, or the fewest more that keep it from printing as zero."""

    numbers = (median / scale, (q75 - median) / scale, (median - q25) / scale)
    median_text, upper_text, lower_text = (
        f"{number:.{_nonzero_decimals(number, decimals)}f}" for number in numbers
    )
    return rf"{median_text}^{{+{upper_text}}}_{{-{lower_text}}}"


def _spread_decimals(q25: float, q75: float) -> int:
    """Decimals showing half the interquartile range to its first significant digit
    (two when the range is zero)."""

    half_range = 0.5 * (q75 - q25)
    if not half_range > 0.0:
        return 2
    return max(-int(math.floor(math.log10(half_range))), 0)


def _format_fixed_quartiles(q25: float, median: float, q75: float) -> str:
    """Fixed-point quartile cell, decimals set by the interquartile range."""

    return rf"${_format_quartiles(q25, median, q75, decimals=_spread_decimals(q25, q75))}$"


def _format_one_decimal_quartiles(q25: float, median: float, q75: float) -> str:
    """Quartile cell with one decimal."""

    return rf"${_format_quartiles(q25, median, q75, decimals=1)}$"


def _mantissa_quartiles_formatter(exponent: int) -> Callable[[float, float, float], str]:
    """Quartile cells in units of ``10**exponent``: one decimal below 10, none above."""

    scale = 10.0**exponent

    def format_cell(q25: float, median: float, q75: float) -> str:
        decimals = 1 if median / scale < 10.0 else 0
        return rf"${_format_quartiles(q25, median, q75, decimals=decimals, scale=scale)}$"

    return format_cell


def _format_scientific_quartiles(q25: float, median: float, q75: float) -> str:
    """Quartile cell in the median's power of ten, e.g. (1.7^{+0.8}_{-0.3}) x 10^-4."""

    if median <= 0.0:
        return rf"${median:.2e}$"
    exponent = int(math.floor(math.log10(median)))
    body = _format_quartiles(q25, median, q75, decimals=1, scale=10.0**exponent)
    return rf"$({body})\times10^{{{exponent}}}$"


_SGD_TIME_COLUMN_HEADER = r"Training time ($\times$SGD)"


def _load_sgd_timing(case_folder: Path) -> dict[str, Any] | None:
    """Read a case's ``sgd_timing/sgd_timing.json``, or ``None`` when the SGD
    baseline has not been measured."""

    try:
        payload = json.loads(
            sgd_baseline.timing_json(case_folder).read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return None
    return payload if isinstance(payload.get("regimes"), dict) else None


def _sgd_regime_median_seconds(
    sgd_timing: dict[str, Any] | None, regime_key: str
) -> float | None:
    """Median wall-clock seconds over seeds of the SGD baseline for one timing regime,
    or ``None``."""

    if not sgd_timing:
        return None
    regime = sgd_timing.get("regimes", {}).get(regime_key)
    per_seed = regime.get("per_seed_training_seconds") if isinstance(regime, dict) else None
    finite = [
        float(value) for value in per_seed or () if value is not None and math.isfinite(float(value))
    ]
    if not finite or statistics.median(finite) <= 0.0:
        return None
    return statistics.median(finite)


def _time_vs_sgd_formatter(
    sgd_median_seconds: float | None,
) -> Callable[[float, float, float], str]:
    """Quartile cells of a method's training seconds divided by the SGD median."""

    def format_cell(q25: float, median: float, q75: float) -> str:
        if sgd_median_seconds is None:
            return _EMPTY_CELL
        return _format_fixed_quartiles(
            q25 / sgd_median_seconds, median / sgd_median_seconds, q75 / sgd_median_seconds
        )

    return format_cell


def _argmin_method(values: dict[str, float | None]) -> str | None:
    """The key with the smallest non-``None`` value, or ``None`` when every value is
    missing."""

    candidates = {method: value for method, value in values.items() if value is not None}
    if not candidates:
        return None
    return min(candidates, key=candidates.__getitem__)


def _bold(cell: str) -> str:
    """ """

    if cell == _EMPTY_CELL:
        return cell
    if cell.startswith("$") and cell.endswith("$"):
        return rf"$\bm{{{cell[1:-1]}}}$"
    return rf"\textbf{{{cell}}}"


def _render_min_loss_and_time_table(
    *,
    label_suffix: str,
    caption: str,
    methods_summary: dict[str, dict[str, Any]],
    min_loss: dict[str, dict[str, float | None]],
    nan_counts: dict[str, list[_SeedNaNCount]] | None = None,
    normalize_time: bool = False,
    sgd_seconds: float | None = None,
    sgd_timing: dict[str, Any] | None = None,
) -> str:
    """Render the shared (optimizer, min training loss, log-learning-curve area,
    training time) table body."""

    show_nan = nan_counts is not None
    methods_summary = _without_learning_area_of_diverged(methods_summary, nan_counts, 0)
    full_caption = caption
    time_header = _SGD_TIME_COLUMN_HEADER if normalize_time else "Training time (s)"

    best_loss_method = _median_argmin({method: min_loss.get(method) for method in methods_summary})
    best_area_method = _median_argmin(
        {method: method_summary.get("learning_area") for method, method_summary in methods_summary.items()}
    )
    best_time_method = _median_argmin(
        {method: method_summary.get("training_seconds") for method, method_summary in methods_summary.items()}
    )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{{full_caption}}}",
        rf"\label{{tab:optimizer_training_time_{label_suffix}}}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}lXXX" + ("c" if show_nan else "") + r"@{}}",
        r"\toprule",
        rf"Optimizer & Min training loss & $A^{{(m)}}$ & {time_header}"
        + (r" & NaN" if show_nan else "")
        + r" \\",
        r"\midrule",
    ]
    for method, method_summary in methods_summary.items():
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        loss_cell = _quartile_cell(min_loss.get(method), _format_scientific_quartiles)
        if method == best_loss_method:
            loss_cell = _bold(loss_cell)
        learning_area_cell = _quartile_cell(
            method_summary.get("learning_area"), _format_fixed_quartiles
        )
        if method == best_area_method:
            learning_area_cell = _bold(learning_area_cell)
        time_cell = _quartile_cell(
            method_summary.get("training_seconds"),
            _time_vs_sgd_formatter(sgd_seconds) if normalize_time else _format_fixed_quartiles,
        )
        if method == best_time_method and time_cell != _EMPTY_CELL:
            time_cell = _bold(time_cell)
        nan_cell = f" & {nan_counts[method][0].diverged}" if show_nan else ""
        lines.append(
            rf"{display_name} & {loss_cell} & {learning_area_cell} & {time_cell}{nan_cell} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
        ]
    )
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def _architecture_training_time_summaries(case: str) -> list[tuple[str, dict[str, Any]]]:
    """Load every architecture's ``summary.json`` of a 1-* case, primary (smallest
    ``(width, depth)``) first."""

    results_directory = CASE_FOLDER[case] / "results"
    architecture_directories = sorted(
        results_directory.glob("width_*_depth_*"),
        key=lambda path: tuple(int(part) for part in path.name.split("_")[1::2]),
    )
    if len(architecture_directories) != 2:
        raise FileNotFoundError(
            f"Expected exactly two architecture directories for {case!r} "
            f"in {results_directory}, found {len(architecture_directories)}"
        )
    return [
        (label, _load_summary(directory / "summary.json"))
        for label, directory in zip(("Primary", "Secondary"), architecture_directories)
    ]


def render_training_time_table(
    case: str, architecture_summaries: list[tuple[str, dict[str, Any]]]
) -> str:
    """Render one case's training-time table, one block of rows per network
    architecture."""

    for label, summary in architecture_summaries:
        if summary.get("ranking_scope", {}).get("kind") != "multi_seed":
            raise ValueError(
                f"{case} {label.lower()}-architecture summary was not produced "
                "from a multi-seed run; training-time statistics require several seeds"
            )

    caption = (
        rf"Training loss and training time for the {TABLE_CASE_TITLES[case]} benchmark."
    )

    results_directory = CASE_FOLDER[case] / "results"
    architecture_directories = sorted(
        results_directory.glob("width_*_depth_*"),
        key=lambda path: tuple(int(part) for part in path.name.split("_")[1::2]),
    )
    shown_methods = _ordered_methods(
        {method for _, summary in architecture_summaries for method in _methods_summary(summary)}
    )
    nan_counts = _nan_seed_counts_by_method(
        [
            (label, directory / "history.csv")
            for (label, _), directory in zip(architecture_summaries, architecture_directories)
        ],
        shown_methods,
        loss_column="train_mse",
    )
    show_nan = nan_counts is not None

    sgd_timing = _load_sgd_timing(CASE_FOLDER[case])
    block_regime_keys = [directory.name for directory in architecture_directories]

    body = []
    for block_index, (label, summary) in enumerate(architecture_summaries):
        methods_summary = _without_learning_area_of_diverged(
            _methods_summary(summary), nan_counts, block_index
        )
        n_methods = len(methods_summary)
        regime_key = block_regime_keys[block_index] if block_index < len(block_regime_keys) else ""
        metric_columns = (
            ("best_train_mse", _format_scientific_quartiles),
            ("learning_area", _format_fixed_quartiles),
            (
                "training_seconds",
                _time_vs_sgd_formatter(_sgd_regime_median_seconds(sgd_timing, regime_key)),
            ),
        )
        best_by_column = [
            _median_argmin(
                {method: method_summary.get(key) for method, method_summary in methods_summary.items()}
            )
            for key, _ in metric_columns
        ]
        for method_index, (method, method_summary) in enumerate(methods_summary.items()):
            row_cells = []
            for (key, format_cell), best_method in zip(metric_columns, best_by_column):
                cell = _quartile_cell(method_summary.get(key), format_cell)
                if method == best_method and cell != _EMPTY_CELL:
                    cell = _bold(cell)
                # Keep each cell on one line: inline math could break inside it in narrow columns.
                row_cells.append(rf"\mbox{{{cell}}}")
            if show_nan:
                row_cells.append(str(nan_counts[method][block_index].diverged))
            architecture_cell = (
                rf"\multirow{{{n_methods}}}{{*}}{{{label}}}" if method_index == 0 else ""
            )
            body.append(
                f"{architecture_cell} & {METHOD_DISPLAY_NAMES.get(method, method)} & "
                + " & ".join(row_cells)
                + r" \\"
            )
        if block_index != len(architecture_summaries) - 1:
            body.append(rf"\cmidrule(lr){{1-{6 if show_nan else 5}}}")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:optimizer_training_time_{case}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{@{}ll*{3}{>{\centering\arraybackslash}X}"
        + (r"c" if show_nan else "")
        + r"@{}}",
        r"\toprule",
        rf"Arch. & Optimizer & Min training loss & $A^{{(m)}}$ & {_SGD_TIME_COLUMN_HEADER}"
        + (r" & NaN" if show_nan else "")
        + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def render_pinn_settings_table() -> str:
    """Render the 4-PINN settings table from results/hyperparameters.txt."""

    hyperparameters = hyperparameters_io.read_hyperparameters_txt(PINN_RESULTS / "hyperparameters.txt")
    parameter_count = _read_parameter_count(PINN_RESULTS)
    architecture = "$({})$, {} trainable parameters in total".format(
        ",".join(str(width) for width in hyperparameters["layer_sizes"]), _math(parameter_count)
    )
    seeds = _seed_set_text(hyperparameters["seeds"])
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Configuration of the 4-PINN optimizer benchmark.}",
        r"\label{tab:optimizer_settings_pinn}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}X@{}}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        rf"Architecture (each of $\varphi,\psi$) & {architecture} \\",
        r"Hidden activation & $\exp(\zeta)$ \\",
        rf"Training / test set & {_math(hyperparameters['train_points'])} / "
        rf"{_math(hyperparameters['test_points'])} "
        r"boundary points (full batch, no minibatching) \\",
        rf"Updates / seeds & {_math(hyperparameters['steps'])} / ${seeds}$ "
        r"(weight initialization only, boundary sampling fixed) \\",
        r"Arithmetic & \texttt{complex64} (32-bit real components) \\",
        rf"Initialization & PIHNN CLT-based rule \cite{{Calafa2024}} (their "
        rf"Algorithm~1), $\beta_\mathrm{{init}}={_format_number(hyperparameters['init_beta'])}$; zero biases \\",
        rf"Base learning rate $\alpha$ & ${_format_number(hyperparameters['learning_rate'])}$ \\",
        r"Optimizer hyperparameters & shared across every case; see "
        r"\Cref{tab:optimizer_settings_shared} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def render_pinn_training_time_table(summary: dict[str, Any], results_directory: Path) -> str:
    """ """

    methods_summary = _methods_summary(summary)
    min_loss = _min_train_loss_from_history(
        results_directory / "history.csv", loss_column="train_mse"
    )
    caption = r"Training loss and training time for the 4-PINN benchmark."
    sgd_timing = _load_sgd_timing(results_directory.parent)
    return _render_min_loss_and_time_table(
        label_suffix="pinn",
        caption=caption,
        methods_summary=methods_summary,
        min_loss=min_loss,
        nan_counts=_nan_seed_counts_by_method(
            [(None, results_directory / "history.csv")], methods_summary, loss_column="train_mse"
        ),
        normalize_time=True,
        sgd_seconds=_sgd_regime_median_seconds(sgd_timing, "default"),
        sgd_timing=sgd_timing,
    )


def render_real_valued_settings_table() -> str:
    """Render the 5-real-valued settings table from results/hyperparameters.txt."""

    hyperparameters = hyperparameters_io.read_hyperparameters_txt(
        REAL_VALUED_RESULTS / "hyperparameters.txt"
    )
    parameter_count = _read_parameter_count(REAL_VALUED_RESULTS)
    architecture = "$({})$, {} trainable parameters".format(
        ",".join(
            str(width)
            for width in (
                hyperparameters["input_dim"],
                *([hyperparameters["hidden_width"]] * hyperparameters["hidden_layers"]),
                1,
            )
        ),
        _math(parameter_count),
    )
    seeds = _seed_set_text(hyperparameters["seeds"])
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Configuration of the 5-real-valued optimizer benchmark.}",
        r"\label{tab:optimizer_settings_real_valued}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}X@{}}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        r"Input domain $(\tilde t,x)$ & $[0,1]^2$ (grid subset; \Cref{sec:test_case_real_valued}) \\",
        rf"Architecture & {architecture} \\",
        r"Hidden activation & $\operatorname{SiLU}$ (real) \\",
        rf"Training / test set & {_math(hyperparameters['train_size'])} / "
        rf"{_math(hyperparameters['test_size'])} "
        rf"grid points (fixed random split, seed {_math(hyperparameters['data_seed'])}) \\",
        rf"Mini-batch size & {_math(hyperparameters['batch_size'])} points \\",
        rf"Updates / seeds & {_math(hyperparameters['steps'])} / ${seeds}$ \\",
        r"Arithmetic & \texttt{float32} throughout \\",
        r"Initialization & Glorot/Xavier normal rule \cite{Glorot2010}; zero biases \\",
        rf"Base learning rate $\alpha$ & ${_format_number(hyperparameters['learning_rate'])}$ \\",
        r"Optimizer hyperparameters & shared across every case; see "
        r"\Cref{tab:optimizer_settings_shared} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def render_real_valued_training_time_table(summary: dict[str, Any], results_directory: Path) -> str:
    """ """

    methods_summary = _methods_summary(summary)
    min_loss = _min_train_loss_from_history(
        results_directory / "history.csv", loss_column="train_mse"
    )
    caption = r"Training loss and training time for the 5-real-valued benchmark."
    return _render_min_loss_and_time_table(
        label_suffix="real_valued",
        caption=caption,
        methods_summary=methods_summary,
        min_loss=min_loss,
        nan_counts=_nan_seed_counts_by_method(
            [(None, results_directory / "history.csv")], methods_summary, loss_column="train_mse"
        ),
    )


def update_real_valued_settings_table(main_tex_path: Path) -> bool:
    """ """

    table = render_real_valued_settings_table()
    begin_marker, end_marker = _table_markers("optimizer_settings", "real_valued")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def update_real_valued_training_time_table(
    summary: dict[str, Any], results_directory: Path, main_tex_path: Path
) -> bool:
    """ """

    table = render_real_valued_training_time_table(summary, results_directory)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "real_valued")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def render_u_net_settings_table() -> str:
    """Render the 6-U-net settings table from results/hyperparameters.txt."""

    hyperparameters = hyperparameters_io.read_hyperparameters_txt(U_NET_RESULTS / "hyperparameters.txt")
    channels = hyperparameters["channels"]
    parameter_count = _read_parameter_count(U_NET_RESULTS)
    architecture = (
        r"3-level complex U-Net, channel widths ("
        + ", ".join(str(c) for c in channels[:-1])
        + rf") $\to$ {channels[-1]}-channel bottleneck; "
        r"$2\times2$ stride-2 conv/transposed-conv resampling, skip concatenation; "
        + _math(parameter_count) + " trainable parameters"
    )
    seeds = _seed_set_text(hyperparameters["seeds"])
    image_size = hyperparameters["image_size"]
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Configuration of the 6-U-net optimizer benchmark.}",
        r"\label{tab:optimizer_settings_u_net}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}X@{}}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        rf"Input & ${image_size}\times{image_size}$ complex (zero-filled IFFT reconstruction) \\",
        rf"Architecture & {architecture} \\",
        r"Nonlinearity & modReLU \eqref{eq:cifar10_modrelu} \\",
        rf"Undersampling & acceleration $R={hyperparameters['acceleration']}$, "
        rf"${hyperparameters['calibration_size']}\times{hyperparameters['calibration_size']}$ calibration region \\",
        rf"Training / test set & {_math(hyperparameters['train_slices'])} / "
        rf"{_math(hyperparameters['test_slices'])} "
        r"knee-image slices (single subject, single coil) \\",
        rf"Mini-batch size & {_math(hyperparameters['batch_size'])} slices \\",
        rf"Updates / seeds & {_math(hyperparameters['steps'])} / ${seeds}$ \\",
        r"Arithmetic & \texttt{complex64} throughout \\",
        r"Loss & per-pixel mean squared error \eqref{eq:unet_loss} \\",
        r"Initialization & Rayleigh-magnitude, uniform-phase complex-normal rule "
        r"(Sec.~3.4 of \cite{Trabelsi2018}); zero biases \\",
        rf"Base learning rate $\alpha$ & ${_format_number(hyperparameters['learning_rate'])}$ \\",
        r"Optimizer hyperparameters & shared across every case; see "
        r"\Cref{tab:optimizer_settings_shared} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def render_u_net_training_time_table(summary: dict[str, Any], results_directory: Path) -> str:
    """Render the 6-U-net training-time table (final-only summary.json, minimum
    recomputed from history.csv)."""

    methods_summary = _methods_summary(summary)
    min_loss = _min_train_loss_from_history(
        results_directory / "history.csv", loss_column="train_mse"
    )
    caption = r"Training loss and training time for the 6-U-net benchmark."
    sgd_timing = _load_sgd_timing(results_directory.parent)
    return _render_min_loss_and_time_table(
        label_suffix="u_net",
        caption=caption,
        methods_summary=methods_summary,
        min_loss=min_loss,
        nan_counts=_nan_seed_counts_by_method(
            [(None, results_directory / "history.csv")], methods_summary, loss_column="train_mse"
        ),
        normalize_time=True,
        sgd_seconds=_sgd_regime_median_seconds(sgd_timing, "default"),
        sgd_timing=sgd_timing,
    )


def update_u_net_settings_table(main_tex_path: Path) -> bool:
    """ """

    table = render_u_net_settings_table()
    begin_marker, end_marker = _table_markers("optimizer_settings", "u_net")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def update_u_net_training_time_table(
    summary: dict[str, Any], results_directory: Path, main_tex_path: Path
) -> bool:
    """ """

    table = render_u_net_training_time_table(summary, results_directory)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "u_net")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def _cifar10_architecture_cell(architecture: str, results_directory: Path) -> str:
    """Render TEST 5's settings-table architecture cell for one network from its
    hyperparameters.txt and parameter_count.txt."""

    hyperparameters = hyperparameters_io.read_hyperparameters_txt(
        results_directory / "hyperparameters.txt"
    )
    parameter_count = _read_parameter_count(results_directory)
    if architecture == CIFAR10_ARCHITECTURES[0]:
        start_filter = hyperparameters["start_filter"]
        body = (
            rf"deep complex ResNet \cite{{Trabelsi2018}}: $3\times3$ "
            rf"$\operatorname{{CConv}}$ stem, {_math(3)} stages of "
            rf"{_math(hyperparameters['num_blocks'])} residual blocks "
            rf"(${start_filter}\to{2 * start_filter}\to{4 * start_filter}$ channels), "
            r"global average pooling"
        )
    else:
        widths = ", ".join(str(width) for width in hyperparameters["hidden_widths"])
        body = (
            rf"{_math(len(hyperparameters['hidden_widths']))} complex dense layers "
            rf"of widths $({widths})$, each followed by $\operatorname{{CBN}}$ and "
            r"$\operatorname{CReLU}$"
        )
    return rf"{body}, real dense output; " + _math(parameter_count) + " trainable parameters"


def render_cifar10_settings_table() -> str:
    """Render TEST 5's settings table from both architectures'
    ``results/<architecture>/hyperparameters.txt``."""

    architecture_directories = [
        CIFAR10_RESULTS / architecture for architecture in CIFAR10_ARCHITECTURES
    ]
    primary_hyperparameters = hyperparameters_io.read_hyperparameters_txt(
        architecture_directories[0] / "hyperparameters.txt"
    )
    secondary_hyperparameters = hyperparameters_io.read_hyperparameters_txt(
        architecture_directories[1] / "hyperparameters.txt"
    )
    if primary_hyperparameters["learning_rate"] != secondary_hyperparameters["learning_rate"]:
        raise ValueError(
            "TEST 5's two architectures now use different base learning "
            "rates; render_cifar10_settings_table's single learning-rate row "
            "needs updating to show both, as the 1-* cases' primary/secondary "
            "tables do."
        )
    primary_architecture = _cifar10_architecture_cell(
        CIFAR10_ARCHITECTURES[0], architecture_directories[0]
    )
    secondary_architecture = _cifar10_architecture_cell(
        CIFAR10_ARCHITECTURES[1], architecture_directories[1]
    )
    seeds = _seed_set_text(primary_hyperparameters["seeds"])
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Configuration of the TEST 5 (CIFAR-10) optimizer benchmark.}",
        r"\label{tab:optimizer_settings_cifar10}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}X@{}}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        r"Input & real RGB image, made complex by the learned input adapter of "
        r"\cite{Trabelsi2018} \\",
        rf"Primary architecture & {primary_architecture} \\",
        rf"Secondary architecture & {secondary_architecture} \\",
        rf"Training / test set & {_math(primary_hyperparameters['train_images'])} / "
        rf"{_math(primary_hyperparameters['test_images'])} "
        rf"images, {_math(primary_hyperparameters['num_classes'])} classes (split of \cite{{Trabelsi2018}}) \\",
        rf"Mini-batch size & {_math(primary_hyperparameters['batch_size'])} images \\",
        rf"Updates / seeds & {_math(primary_hyperparameters['steps'])} / ${seeds}$ \\",
        r"Arithmetic & \texttt{complex64} throughout \\",
        rf"Loss & softmax cross-entropy with $\ell_2$ weight penalty "
        rf"${_format_number(primary_hyperparameters['kernel_l2'])}$ \eqref{{eq:cifar10_loss}}; "
        rf"gradient clipped to norm ${_format_number(primary_hyperparameters['gradient_clip_norm'])}$ \\",
        r"Nonlinearity & $\operatorname{CReLU}$ \eqref{eq:cifar10_crelu}, in both "
        r"architectures \\",
        r"Normalization & complex batch normalization of \cite{Trabelsi2018}, in both "
        r"architectures \\",
        r"Initialization & as in the reference implementation of \cite{Trabelsi2018}; "
        r"the convolution filters do not depend on the seed \\",
        rf"Base learning rate $\alpha$ (both architectures) & "
        rf"${_format_number(primary_hyperparameters['learning_rate'])}$ \\",
        r"Optimizer hyperparameters & shared across every case; see "
        r"\Cref{tab:optimizer_settings_shared} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def render_cifar10_training_time_table(
    primary_summary: dict[str, Any],
    secondary_summary: dict[str, Any],
) -> str:
    """Render TEST 5's training-time table, one block of rows per architecture."""

    architecture_blocks = (
        ("Primary", CIFAR10_RESULTS / CIFAR10_ARCHITECTURES[0], primary_summary),
        ("Secondary", CIFAR10_RESULTS / CIFAR10_ARCHITECTURES[1], secondary_summary),
    )

    caption = r"Training loss and training time for the TEST 5 (CIFAR-10) benchmark."
    nan_counts = _nan_seed_counts_by_method(
        [(label, directory / "history.csv") for label, directory, _ in architecture_blocks],
        _ordered_methods(
            {
                method
                for _, _, summary in architecture_blocks
                for method in _methods_summary(summary)
            }
        ),
        loss_column="train_loss",
    )
    show_nan = nan_counts is not None
    sgd_timing = _load_sgd_timing(CIFAR10_RESULTS.parent)

    body = []
    for block_index, (label, results_directory, summary) in enumerate(architecture_blocks):
        methods_summary = _without_learning_area_of_diverged(
            _methods_summary(summary), nan_counts, block_index
        )
        min_loss = _min_train_loss_from_history(
            results_directory / "history.csv", loss_column="train_loss"
        )
        sgd_seconds = _sgd_regime_median_seconds(sgd_timing, CIFAR10_ARCHITECTURES[block_index])
        n_methods = len(methods_summary)
        best_loss_method = _median_argmin(
            {method: min_loss.get(method) for method in methods_summary}
        )
        best_area_method = _median_argmin(
            {method: method_summary.get("learning_area") for method, method_summary in methods_summary.items()}
        )
        best_time_method = _median_argmin(
            {method: method_summary.get("training_seconds") for method, method_summary in methods_summary.items()}
        )
        for method_index, (method, method_summary) in enumerate(methods_summary.items()):
            loss_cell = _quartile_cell(min_loss.get(method), _format_scientific_quartiles)
            if method == best_loss_method:
                loss_cell = _bold(loss_cell)
            learning_area_cell = _quartile_cell(
                method_summary.get("learning_area"), _format_fixed_quartiles
            )
            if method == best_area_method:
                learning_area_cell = _bold(learning_area_cell)
            time_cell = _quartile_cell(
                method_summary.get("training_seconds"), _time_vs_sgd_formatter(sgd_seconds)
            )
            if method == best_time_method:
                time_cell = _bold(time_cell)
            architecture_cell = (
                rf"\multirow{{{n_methods}}}{{*}}{{{label}}}" if method_index == 0 else ""
            )
            cells = [
                rf"\mbox{{{cell}}}" for cell in (loss_cell, learning_area_cell, time_cell)
            ]
            if show_nan:
                cells.append(str(nan_counts[method][block_index].diverged))
            body.append(
                f"{architecture_cell} & {METHOD_DISPLAY_NAMES.get(method, method)} & "
                + " & ".join(cells)
                + r" \\"
            )
        if block_index != len(architecture_blocks) - 1:
            body.append(rf"\cmidrule(lr){{1-{6 if show_nan else 5}}}")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:optimizer_training_time_cifar10}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{@{}ll*{3}{>{\centering\arraybackslash}X}"
        + (r"c" if show_nan else "")
        + r"@{}}",
        r"\toprule",
        rf"Arch. & Optimizer & Min training loss & $A^{{(m)}}$ & {_SGD_TIME_COLUMN_HEADER}"
        + (r" & NaN" if show_nan else "")
        + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_cifar10_settings_table(main_tex_path: Path) -> bool:
    """ """

    table = render_cifar10_settings_table()
    begin_marker, end_marker = _table_markers("optimizer_settings", "cifar10")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def update_cifar10_training_time_table(
    primary_summary: dict[str, Any],
    secondary_summary: dict[str, Any],
    main_tex_path: Path,
) -> bool:
    """ """

    table = render_cifar10_training_time_table(primary_summary, secondary_summary)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "cifar10")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def update_equalization_losses_figure(results_directory: Path, destination: Path) -> bool:
    """Concatenate TEST 7's CVFNN and deep C-RBF combined_losses.pdf into one
    two-page figure."""

    pdfunite = shutil.which("pdfunite")
    if pdfunite is None:
        raise RuntimeError("Combining comparison reports requires the 'pdfunite' executable.")

    architecture_pdfs = []
    for architecture in EQUALIZATION_ARCHITECTURES:
        architecture_pdf = results_directory / architecture / "combined_losses.pdf"
        if not architecture_pdf.is_file():
            raise FileNotFoundError(
                f"Missing {architecture} combined-losses PDF for TEST 7: {architecture_pdf}"
            )
        architecture_pdfs.append(str(architecture_pdf))

    with tempfile.TemporaryDirectory(prefix="equalization-combined-losses-") as temporary_directory:
        combined_pdf = Path(temporary_directory) / "combined.pdf"
        subprocess.run([pdfunite, *architecture_pdfs, str(combined_pdf)], check=True)
        return update_paper_figure(combined_pdf, destination)


def _equalization_architecture_cell(architecture: str, results_directory: Path) -> str:
    """Render TEST 7's settings-table architecture cell for one equalizer from its
    results/hyperparameters.txt."""

    hyperparameters = hyperparameters_io.read_hyperparameters_txt(
        results_directory / "hyperparameters.txt"
    )
    parameter_count = _read_parameter_count(results_directory)
    widths = ", ".join(str(width) for width in hyperparameters["widths"])
    neurons = hyperparameters["neurons"]
    if architecture == "cvfnn":
        return (
            rf"fully complex perceptron $({widths})$ \cite{{Dong2021}}: "
            rf"{_math(neurons[0])} hidden neurons with the "
            rf"$\operatorname{{artanh}}$ activation \eqref{{eq:app_cvfnn_activation}}, "
            rf"{_math(neurons[1])} linear output neuron; "
            rf"{_math(parameter_count)} complex trainable parameters"
        )
    # C-RBF kernel variances are real parameters, counted separately.
    variance_count = sum(neurons)
    return (
        rf"deep complex RBF \cite{{Soares2024}}: {_math(neurons[0])} then "
        rf"{_math(neurons[1])} Gaussian neurons \eqref{{eq:app_crbf_kernel}}, "
        rf"layer output widths $({widths})$; {_math(parameter_count)} trainable "
        rf"parameters ({_math(parameter_count - variance_count)} complex, "
        rf"{_math(variance_count)} real kernel variances)"
    )


def _equalization_preprocessing_cell(
    primary_hyperparameters: dict[str, Any], secondary_hyperparameters: dict[str, Any]
) -> str:
    """ """

    primary = primary_hyperparameters.get("normalized_inputs")
    secondary = secondary_hyperparameters.get("normalized_inputs")
    if primary is None or secondary is None:
        # results/ written before the flag was recorded: both were normalized.
        return r"normalized \eqref{eq:equalization_normalization}"
    if primary == secondary:
        return (
            r"normalized \eqref{eq:equalization_normalization}"
            if primary
            else "raw, unnormalized"
        )
    normalized, raw = ("secondary", "primary") if secondary else ("primary", "secondary")
    return (
        rf"normalized \eqref{{eq:equalization_normalization}} for the {normalized} "
        rf"architecture, raw for the {raw}"
    )


def _equalization_batch_size_cell(hyperparameters: dict[str, Any]) -> str:
    """ """

    batch_size = int(hyperparameters["batch_size"])
    if batch_size == 1:
        return "$1$ symbol (online: one update per symbol, as in \\cite{Mayer2025})"
    return f"{_math(batch_size)} symbols"


def _equalization_learning_rate_cell(hyperparameters: dict[str, Any]) -> str:
    """ """

    rates: dict[str, float] = hyperparameters.get("learning_rates") or {}
    if not rates:
        # results/ written before per-optimizer rates existed: one rate for all.
        return f"${_format_number(hyperparameters['learning_rate'])}$ (every optimizer)"
    groups: dict[float, list[str]] = {}
    for method, rate in rates.items():
        groups.setdefault(rate, []).append(METHOD_DISPLAY_NAMES.get(method, method))
    if len(groups) == 1:
        return f"${_format_number(next(iter(groups)))}$ (every optimizer)"
    return "; ".join(
        f"${_format_number(rate)}$ ({', '.join(methods)})"
        for rate, methods in sorted(groups.items(), key=lambda item: (-len(item[1]), -item[0]))
    )


def render_equalization_settings_table() -> str:
    """Render TEST 7's settings table from both architectures'
    results/hyperparameters.txt."""

    primary_directory = EQUALIZATION_RESULTS / EQUALIZATION_ARCHITECTURES[0]
    secondary_directory = EQUALIZATION_RESULTS / EQUALIZATION_ARCHITECTURES[1]
    primary_hyperparameters = hyperparameters_io.read_hyperparameters_txt(
        primary_directory / "hyperparameters.txt"
    )
    secondary_hyperparameters = hyperparameters_io.read_hyperparameters_txt(
        secondary_directory / "hyperparameters.txt"
    )
    seeds = _seed_set_text(primary_hyperparameters["seeds"])
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Configuration of the TEST 7 (channel equalization) optimizer benchmark.}",
        r"\label{tab:optimizer_settings_equalization}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.32\linewidth}X@{}}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
        rf"Input & {_math(primary_hyperparameters['input_width'])} consecutive complex channel "
        rf"outputs \eqref{{eq:equalization_channel}}; "
        rf"{_equalization_preprocessing_cell(primary_hyperparameters, secondary_hyperparameters)} \\",
        rf"Target & transmitted QPSK symbol, delayed by "
        rf"{_math(primary_hyperparameters['decision_delay'])} samples \\",
        rf"Channel noise & $a[n]\sim\mathcal{{CN}}(0,"
        rf"{_format_number(primary_hyperparameters['noise_variance'])})$ \\",
        rf"Primary architecture & {_equalization_architecture_cell('cvfnn', primary_directory)} \\",
        rf"Secondary architecture & {_equalization_architecture_cell('c_rbf', secondary_directory)} \\",
        rf"Training / test set & {_math(primary_hyperparameters['train_size'])} / "
        rf"{_math(primary_hyperparameters['test_size'])} consecutive symbols of one transmission \\",
        rf"Mini-batch size & {_equalization_batch_size_cell(primary_hyperparameters)} \\",
        rf"Updates / seeds & {_math(primary_hyperparameters['steps'])} "
        rf"({_math(primary_hyperparameters['epochs'])} epochs of "
        rf"{_math(primary_hyperparameters['steps_per_epoch'])}) / ${seeds}$ \\",
        r"Arithmetic & \texttt{complex64} throughout \\",
        r"Loss & mean squared symbol error \eqref{eq:equalization_loss}, reported in decibels \eqref{eq:equalization_decibels} \\",
        r"Initialization & primary: Rayleigh-magnitude, uniform-phase complex-normal rule "
        r"(Sec.~3.4 of \cite{Trabelsi2018}); secondary: deep C-RBF parameter selection "
        r"\cite{Soares2024}; zero biases \\",
        rf"Base learning rate $\alpha$ (primary) & "
        rf"{_equalization_learning_rate_cell(primary_hyperparameters)} \\",
        rf"Secondary-architecture learning rate & "
        rf"{_equalization_learning_rate_cell(secondary_hyperparameters)} \\",
        r"Optimizer hyperparameters & shared across every case; see "
        r"\Cref{tab:optimizer_settings_shared} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def render_equalization_training_time_table(
    primary_summary: dict[str, Any],
    secondary_summary: dict[str, Any],
) -> str:
    """Render TEST 7's training-time table, one block of rows per architecture."""

    architecture_blocks = (
        ("Primary", EQUALIZATION_RESULTS / EQUALIZATION_ARCHITECTURES[0], primary_summary),
        ("Secondary", EQUALIZATION_RESULTS / EQUALIZATION_ARCHITECTURES[1], secondary_summary),
    )

    caption = (
        r"Training loss and training time for the TEST 7 (channel equalization) benchmark."
    )
    nan_counts = _nan_seed_counts_by_method(
        [
            (architecture, EQUALIZATION_RESULTS / architecture / "history.csv")
            for architecture in EQUALIZATION_ARCHITECTURES
        ],
        _ordered_methods(
            {
                method
                for _, _, summary in architecture_blocks
                for method in _methods_summary(summary)
            }
        ),
        loss_column="train_mse",
    )
    show_nan = nan_counts is not None

    body = []
    for block_index, (label, results_directory, summary) in enumerate(architecture_blocks):
        methods_summary = _without_learning_area_of_diverged(
            _methods_summary(summary), nan_counts, block_index
        )
        min_loss = _min_train_loss_from_history(
            results_directory / "history.csv", loss_column="train_mse", in_decibels=True
        )
        n_methods = len(methods_summary)
        best_loss_method = _median_argmin(
            {method: min_loss.get(method) for method in methods_summary}
        )
        best_area_method = _median_argmin(
            {method: method_summary.get("learning_area") for method, method_summary in methods_summary.items()}
        )
        best_time_method = _median_argmin(
            {method: method_summary.get("training_seconds") for method, method_summary in methods_summary.items()}
        )
        for method_index, (method, method_summary) in enumerate(methods_summary.items()):
            loss_cell = _quartile_cell(min_loss.get(method), _format_fixed_quartiles)
            if method == best_loss_method:
                loss_cell = _bold(loss_cell)
            learning_area_cell = _quartile_cell(
                method_summary.get("learning_area"), _format_fixed_quartiles
            )
            if method == best_area_method:
                learning_area_cell = _bold(learning_area_cell)
            time_cell = _quartile_cell(method_summary.get("training_seconds"), _format_fixed_quartiles)
            if method == best_time_method:
                time_cell = _bold(time_cell)
            architecture_cell = (
                rf"\multirow{{{n_methods}}}{{*}}{{{label}}}" if method_index == 0 else ""
            )
            cells = [
                rf"\mbox{{{cell}}}" for cell in (loss_cell, learning_area_cell, time_cell)
            ]
            if show_nan:
                cells.append(str(nan_counts[method][block_index].diverged))
            body.append(
                f"{architecture_cell} & {METHOD_DISPLAY_NAMES.get(method, method)} & "
                + " & ".join(cells)
                + r" \\"
            )
        if block_index != len(architecture_blocks) - 1:
            body.append(rf"\cmidrule(lr){{1-{6 if show_nan else 5}}}")

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:optimizer_training_time_equalization}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{@{}ll*{3}{>{\centering\arraybackslash}X}"
        + (r"c" if show_nan else "")
        + r"@{}}",
        r"\toprule",
        r"Arch. & Optimizer & Min training loss [dB] & $A^{(m)}$ & Training time (s)"
        + (r" & NaN" if show_nan else "")
        + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    return "% Generated by update_paper_assets.py; do not edit manually.\n" + "\n".join(lines)


def update_equalization_settings_table(main_tex_path: Path) -> bool:
    """ """

    table = render_equalization_settings_table()
    begin_marker, end_marker = _table_markers("optimizer_settings", "equalization")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def update_equalization_training_time_table(
    primary_summary: dict[str, Any],
    secondary_summary: dict[str, Any],
    main_tex_path: Path,
) -> bool:
    """ """

    table = render_equalization_training_time_table(primary_summary, secondary_summary)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "equalization")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def _region_between_markers(text: str, begin_marker: str, end_marker: str) -> str:
    """The text between two marker comments in main.tex, or ``""`` if the markers
    are absent."""

    begin_index = text.find(begin_marker)
    if begin_index == -1:
        return ""
    end_index = text.find(end_marker, begin_index)
    if end_index == -1:
        return ""
    return text[begin_index + len(begin_marker) : end_index]


def _caption_blocks(text: str) -> list[str]:
    r"""Every ``\caption{...}`` substring in ``text``, with balanced braces."""

    needle = r"\caption{"
    blocks: list[str] = []
    start = text.find(needle)
    while start != -1:
        index = start + len(needle)
        depth = 1
        while index < len(text) and depth:
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            index += 1
        if depth == 0:
            blocks.append(text[start:index])
        start = text.find(needle, index)
    return blocks


def _keep_manual_captions(old_region: str, new_block: str) -> str:
    r"""Carry every hand-edited ``\caption{...}`` from the current main.tex region
    into the generated block, matched by position."""

    for old_caption, new_caption in zip(
        _caption_blocks(old_region), _caption_blocks(new_block)
    ):
        if old_caption != new_caption:
            new_block = new_block.replace(new_caption, old_caption, 1)
    return new_block


def _splice_between_markers(
    text: str, begin_marker: str, end_marker: str, replacement: str
) -> str:
    begin_index = text.find(begin_marker)
    if begin_index == -1:
        raise ValueError(f"Marker not found in main.tex: {begin_marker!r}")
    end_index = text.find(end_marker, begin_index)
    if end_index == -1:
        raise ValueError(f"Marker not found in main.tex: {end_marker!r}")
    end_index += len(end_marker)
    return (
        text[:begin_index]
        + begin_marker
        + "\n"
        + replacement
        + end_marker
        + text[end_index:]
    )


def _write_text_atomically(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}-",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _splice_table_into_main_tex(
    table: str, main_tex_path: Path, begin_marker: str, end_marker: str
) -> bool:
    """ """

    main_tex_path = main_tex_path.expanduser().resolve()
    original = main_tex_path.read_text(encoding="utf-8")
    table = _keep_manual_captions(
        _region_between_markers(original, begin_marker, end_marker), table
    )
    updated = _splice_between_markers(original, begin_marker, end_marker, table + "\n")
    if updated == original:
        return False
    _write_text_atomically(main_tex_path, updated)
    return True


def update_case_settings_table(case: str, summary: dict[str, Any], main_tex_path: Path) -> bool:
    """ """

    table = render_settings_table(case, summary)
    begin_marker, end_marker = _table_markers("optimizer_settings", case)
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def update_case_training_time_table(case: str, main_tex_path: Path) -> bool:
    """ """

    table = render_training_time_table(case, _architecture_training_time_summaries(case))
    begin_marker, end_marker = _table_markers("optimizer_training_time", case)
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def _load_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Summary does not exist: {path}") from None


def update_pinn_settings_table(main_tex_path: Path) -> bool:
    """ """

    table = render_pinn_settings_table()
    begin_marker, end_marker = _table_markers("optimizer_settings", "pinn")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def update_pinn_training_time_table(
    summary: dict[str, Any], results_directory: Path, main_tex_path: Path
) -> bool:
    """ """

    table = render_pinn_training_time_table(summary, results_directory)
    begin_marker, end_marker = _table_markers("optimizer_training_time", "pinn")
    return _splice_table_into_main_tex(table, main_tex_path, begin_marker, end_marker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update each optimizer case's own figure/tables in the paper, plus the PINN figures."
    )
    for case in TABLE_CASE_ORDER:
        parser.add_argument(
            f"--{case.replace('_', '-')}-summary",
            type=Path,
            default=DEFAULT_SUMMARY_PATHS.get(case),
            help=argparse.SUPPRESS if case not in ACTIVE_TABLE_CASES else None,
        )
    parser.add_argument("--main-tex", type=Path, default=DEFAULT_MAIN_TEX)
    parser.add_argument("--pinn-results", type=Path, default=DEFAULT_PINN_RESULTS)
    parser.add_argument(
        "--pinn-figure-destination", type=Path, default=DEFAULT_PINN_FIGURE_DESTINATION
    )
    parser.add_argument(
        "--pinn-loss-destination", type=Path, default=DEFAULT_PINN_LOSS_DESTINATION
    )
    parser.add_argument("--real-valued-results", type=Path, default=REAL_VALUED_RESULTS)
    parser.add_argument(
        "--real-valued-loss-destination", type=Path, default=DEFAULT_REAL_VALUED_LOSS_DESTINATION
    )
    parser.add_argument("--u-net-results", type=Path, default=U_NET_RESULTS)
    parser.add_argument(
        "--u-net-loss-destination", type=Path, default=DEFAULT_U_NET_LOSS_DESTINATION
    )
    parser.add_argument("--cifar10-results", type=Path, default=CIFAR10_RESULTS)
    parser.add_argument(
        "--cifar10-loss-destination", type=Path, default=CIFAR10_LOSS_DESTINATION
    )
    parser.add_argument(
        "--cifar10-accuracy-destination", type=Path, default=CIFAR10_ACCURACY_DESTINATION
    )
    parser.add_argument("--equalization-results", type=Path, default=EQUALIZATION_RESULTS)
    parser.add_argument(
        "--equalization-loss-destination", type=Path, default=EQUALIZATION_LOSS_DESTINATION
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()

    try:
        shared_settings_changed = update_shared_optimizer_settings_table(arguments.main_tex)
        print(
            "shared optimizer settings: table "
            f"{'updated' if shared_settings_changed else 'up to date'}"
        )
    except ValueError as error:
        print(f"Skipped shared optimizer settings table: {error}")

    for case in ACTIVE_TABLE_CASES:
        summary_path = getattr(arguments, f"{case}_summary")
        figure_destination = DEFAULT_FIGURE_DESTINATIONS[case]
        try:
            summary = _load_summary(summary_path)
            if case == "non_holomorphic":
                figure_changed = update_non_holomorphic_lr_figure(CASE_FOLDER[case], figure_destination)
                block_changed = update_non_holomorphic_lr_figure_block(arguments.main_tex)
                figure_changed = figure_changed or block_changed
            elif case == "multivariate_c4":
                figure_changed = update_multivariate_c4_lr_figure(
                    summary, CASE_FOLDER[case], figure_destination
                )
                block_changed = update_multivariate_c4_lr_figure_block(
                    summary, CASE_FOLDER[case] / "results", arguments.main_tex
                )
                figure_changed = figure_changed or block_changed
            elif case == "holomorphic":
                figure_changed = update_holomorphic_minibatch_figure(
                    CASE_FOLDER[case], figure_destination
                )
                block_changed = update_holomorphic_minibatch_figure_block(
                    CASE_FOLDER[case] / "results", arguments.main_tex
                )
                figure_changed = figure_changed or block_changed
            else:
                figure_changed = update_case_losses_figure(case, CASE_FOLDER[case], figure_destination)
            if case == "multivariate_c4":
                settings_changed = update_multivariate_c4_settings_table(
                    summary, CASE_FOLDER[case] / "results", arguments.main_tex
                )
            else:
                settings_changed = update_case_settings_table(case, summary, arguments.main_tex)
            if case == "non_holomorphic":
                time_changed = update_non_holomorphic_lr_sweep_table(
                    CASE_FOLDER[case] / "results", arguments.main_tex
                )
            elif case == "multivariate_c4":
                time_changed = update_multivariate_c4_lr_sweep_table(
                    summary, CASE_FOLDER[case] / "results", arguments.main_tex
                )
            elif case == "holomorphic":
                time_changed = update_holomorphic_minibatch_sweep_table(
                    CASE_FOLDER[case] / "results", arguments.main_tex
                )
            else:
                time_changed = update_case_training_time_table(case, arguments.main_tex)
            print(
                f"{case}: figure {'updated' if figure_changed else 'up to date'}, "
                f"settings table {'updated' if settings_changed else 'up to date'}, "
                f"training-time table {'updated' if time_changed else 'up to date'}"
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipped {case}: {error}")

    try:
        beta_sweep_results = CASE_FOLDER["non_holomorphic"] / "results"
        beta_table_changed = update_non_holomorphic_beta_sweep_table(
            beta_sweep_results, arguments.main_tex
        )
        beta_figure_changed = update_non_holomorphic_beta_sweep_figure(
            beta_sweep_results, DEFAULT_NON_HOLOMORPHIC_BETA_SWEEP_DESTINATION
        )
        print(
            f"non_holomorphic beta sweep: table {'updated' if beta_table_changed else 'up to date'}, "
            f"figure {'updated' if beta_figure_changed else 'up to date'}"
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Skipped non_holomorphic beta sweep: {error}")

    if arguments.pinn_results.is_dir():
        updated = copy_pinn_figures(arguments.pinn_results, arguments.pinn_figure_destination)
        if updated:
            for path in updated:
                print(f"Updated: {path}")
        else:
            print(f"Already up to date: {arguments.pinn_figure_destination}")
    else:
        print(f"Skipped PINN figures: no results directory at {arguments.pinn_results}")

    try:
        if copy_case_loss_comparison(
            arguments.pinn_results, arguments.pinn_loss_destination, PINN_PAPER_LOSSES_NAME
        ):
            print(f"Updated: {arguments.pinn_loss_destination}")
        else:
            print(
                f"Skipped PINN loss figure: no {PINN_PAPER_LOSSES_NAME} in {arguments.pinn_results}"
            )
    except FileNotFoundError as error:
        print(f"Skipped PINN loss figure: {error}")

    try:
        pinn_summary = _load_json(arguments.pinn_results / "summary.json")
        pinn_settings_changed = update_pinn_settings_table(arguments.main_tex)
        pinn_time_changed = update_pinn_training_time_table(
            pinn_summary, arguments.pinn_results, arguments.main_tex
        )
        print(
            f"4-PINN: settings table {'updated' if pinn_settings_changed else 'up to date'}, "
            f"training-time table {'updated' if pinn_time_changed else 'up to date'}"
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Skipped 4-PINN tables: {error}")

    try:
        if copy_case_loss_comparison(arguments.real_valued_results, arguments.real_valued_loss_destination):
            print(f"Updated: {arguments.real_valued_loss_destination}")
        else:
            print(f"Skipped 5-real-valued loss figure: no combined_losses.pdf in {arguments.real_valued_results}")
    except FileNotFoundError as error:
        print(f"Skipped 5-real-valued loss figure: {error}")

    try:
        real_valued_summary = _load_json(arguments.real_valued_results / "summary.json")
        real_valued_settings_changed = update_real_valued_settings_table(arguments.main_tex)
        real_valued_time_changed = update_real_valued_training_time_table(
            real_valued_summary, arguments.real_valued_results, arguments.main_tex
        )
        print(
            f"5-real-valued: settings table {'updated' if real_valued_settings_changed else 'up to date'}, "
            f"training-time table {'updated' if real_valued_time_changed else 'up to date'}"
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Skipped 5-real-valued tables: {error}")

    try:
        if copy_case_loss_comparison(arguments.u_net_results, arguments.u_net_loss_destination):
            print(f"Updated: {arguments.u_net_loss_destination}")
        else:
            print(f"Skipped 6-U-net loss figure: no combined_losses.pdf in {arguments.u_net_results}")
    except FileNotFoundError as error:
        print(f"Skipped 6-U-net loss figure: {error}")

    try:
        u_net_summary = _load_json(arguments.u_net_results / "summary.json")
        u_net_settings_changed = update_u_net_settings_table(arguments.main_tex)
        u_net_time_changed = update_u_net_training_time_table(
            u_net_summary, arguments.u_net_results, arguments.main_tex
        )
        print(
            f"6-U-net: settings table {'updated' if u_net_settings_changed else 'up to date'}, "
            f"training-time table {'updated' if u_net_time_changed else 'up to date'}"
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Skipped 6-U-net tables: {error}")

    try:
        cifar10_figure_changed = update_cifar10_losses_figure(
            arguments.cifar10_results, arguments.cifar10_loss_destination
        )
        print(f"TEST 5 (CIFAR-10): loss figure {'updated' if cifar10_figure_changed else 'up to date'}")
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Skipped TEST 5 (CIFAR-10) loss figure: {error}")

    try:
        cifar10_accuracy_changed = update_cifar10_accuracies_figure(
            arguments.cifar10_results, arguments.cifar10_accuracy_destination
        )
        print(
            "TEST 5 (CIFAR-10): accuracy figure "
            f"{'updated' if cifar10_accuracy_changed else 'up to date'}"
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Skipped TEST 5 (CIFAR-10) accuracy figure: {error}")

    try:
        cifar10_summaries = [
            _load_json(arguments.cifar10_results / architecture / "summary.json")
            for architecture in CIFAR10_ARCHITECTURES
        ]
        cifar10_settings_changed = update_cifar10_settings_table(arguments.main_tex)
        cifar10_time_changed = update_cifar10_training_time_table(
            *cifar10_summaries, arguments.main_tex
        )
        print(
            f"TEST 5 (CIFAR-10): settings table {'updated' if cifar10_settings_changed else 'up to date'}, "
            f"training-time table {'updated' if cifar10_time_changed else 'up to date'}"
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Skipped TEST 5 (CIFAR-10) tables: {error}")

    try:
        equalization_figure_changed = update_equalization_losses_figure(
            arguments.equalization_results, arguments.equalization_loss_destination
        )
        print(
            "TEST 7 (channel equalization): figure "
            f"{'updated' if equalization_figure_changed else 'up to date'}"
        )
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Skipped TEST 7 (channel equalization) figure: {error}")

    try:
        equalization_summaries = [
            _load_json(arguments.equalization_results / architecture / "summary.json")
            for architecture in EQUALIZATION_ARCHITECTURES
        ]
        equalization_settings_changed = update_equalization_settings_table(arguments.main_tex)
        equalization_time_changed = update_equalization_training_time_table(
            *equalization_summaries, arguments.main_tex
        )
        print(
            "TEST 7 (channel equalization): settings table "
            f"{'updated' if equalization_settings_changed else 'up to date'}, "
            f"training-time table {'updated' if equalization_time_changed else 'up to date'}"
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Skipped TEST 7 (channel equalization) tables: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

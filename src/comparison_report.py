"""Generic dense multi-panel ``comparison.pdf`` report, shared by every case."""


import functools
import os
import shutil
import traceback
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import optimizer_config
from . import weight_dynamics

DEFAULT_FONTSIZE = 10
SMALL_LEGEND_FONTSIZE = "small"
REPORT_TITLE_FONTSIZE = 17
WEIGHT_SCALE_CENTER = 1e-1
LATEX_PREAMBLE = r"\usepackage{amsmath}"


@dataclass
class MethodRunResult:
    """One method's training run for one seed, as consumed by this report."""

    method: str
    train_loss: np.ndarray
    test_loss: np.ndarray
    diagnostics: np.ndarray
    weight_history: np.ndarray
    selected_gamma_history: np.ndarray
    selected_chi_history: np.ndarray | None = None
    selected_psi_history: np.ndarray | None = None
    train_accuracy: np.ndarray | None = None  # NaN off the test_every cadence
    test_accuracy: np.ndarray | None = None


_usetex_decision: bool | None = None


def _matplotlib_config_directory() -> Path:
    """The repo-local matplotlib cache, so nothing is written to $HOME."""

    config_directory = Path(__file__).resolve().parent / ".matplotlib"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_directory))
    return config_directory


def _latex_can_typeset() -> bool:
    """Whether this machine's LaTeX can actually typeset a matplotlib label."""

    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    try:
        import matplotlib

        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
        from matplotlib.texmanager import TexManager

        TexManager().make_dvi(r"lp $10^{-1}$", DEFAULT_FONTSIZE)
    except Exception:
        return False
    return True


def usetex_enabled() -> bool:
    """Decide once whether figures are typeset by LaTeX or by mathtext."""

    global _usetex_decision
    if _usetex_decision is not None:
        return _usetex_decision

    forced = os.environ.get("AURA_USETEX", "").strip()
    if forced:
        _usetex_decision = forced.lower() not in {"0", "false", "no", "off"}
        return _usetex_decision

    _matplotlib_config_directory()
    _usetex_decision = _latex_can_typeset()
    if not _usetex_decision:
        print(
            "[comparison_report] no working LaTeX toolchain; typesetting figure "
            "text with matplotlib's mathtext instead. Regenerate the PDFs on a "
            "machine with LaTeX (plots.py's regenerate_from_disk) for the "
            "paper-quality version.",
            flush=True,
        )
    return _usetex_decision


def _pyplot(*, fontsize: int = DEFAULT_FONTSIZE):
    """Load a headless plotting backend with all cache files kept locally."""

    _matplotlib_config_directory()

    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = fontsize
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["text.usetex"] = usetex_enabled()
    matplotlib.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    import matplotlib.pyplot as plt

    return plt


def never_fatal(function):
    """Report writers that must not be able to kill the run that calls them."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            if os.environ.get("AURA_STRICT_PLOTS", "").strip().lower() in {"1", "true", "yes", "on"}:
                raise
            print(
                f"[comparison_report] {function.__name__} failed; continuing "
                f"without it (results on disk are unaffected):\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            return None

    return wrapper


def _innovation_config(method: str):
    """The innovation/multiplier hyperparameters whose thresholds this method's
    chi/Psi panels mark."""

    if method == optimizer_config.METHOD_AURA_LIGHT:
        return optimizer_config.AURA_LIGHT_CONFIG
    if method == optimizer_config.METHOD_ASTRA:
        return optimizer_config.ASTRA_CONFIG
    if method == optimizer_config.METHOD_ECLIPSE:
        return optimizer_config.ECLIPSE_CONFIG
    if method == optimizer_config.METHOD_MUON_AURA:
        return optimizer_config.MUON_AURA_CONFIG
    if method == optimizer_config.METHOD_ADAM_AURA_SPRING_EB:
        return optimizer_config.AURA_SPRING_EB_CONFIG
    return optimizer_config.AURA_CONFIG


def _aura_snr_config(method: str):
    if method == optimizer_config.METHOD_ADAM_AURA_SNR:
        return optimizer_config.AURA_SNR_CONFIG
    if method == optimizer_config.METHOD_ADAM_AURA_SNR_ABLATION:
        return optimizer_config.AURA_SNR_ABLATION_CONFIG
    return optimizer_config.MUON_AURA_SNR_CONFIG


def _aura_spring_config(method: str):
    """The AURA-spring ablation's config on the Adam or the Muon direction."""

    if method == optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION:
        return optimizer_config.MUON_AURA_SPRING_CONFIG
    return optimizer_config.AURA_SPRING_CONFIG


def _chi_thresholds(method: str) -> tuple[float, ...]:
    """Horizontal reference lines for a method's chi panel."""

    if method in optimizer_config.AURA_SNR_FAMILY:
        return (0.0, -_aura_snr_config(method).opposition_threshold)
    if method in optimizer_config.AURA_SIGN_FAMILY:
        config = (
            optimizer_config.AURA_SIGN_CONFIG
            if method == optimizer_config.METHOD_ADAM_AURA_SIGN
            else optimizer_config.MUON_AURA_SIGN_CONFIG
        )
        return (0.0, -config.brake_threshold)
    if method in optimizer_config.AURA_S_FAMILY:
        return (0.0,)
    if method in (optimizer_config.METHOD_ADAM_AURA_SPRING_ABLATION, optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION):
        return (0.0, _aura_spring_config(method).chi_opposition)
    config = _innovation_config(method)
    return (config.chi_alignment, config.chi_opposition)


def _psi_thresholds(method: str) -> tuple[float, ...]:
    """Symmetric |Psi| reference lines; none for AURA-S (see _chi_thresholds)."""

    if method in optimizer_config.AURA_SNR_FAMILY:
        threshold = _aura_snr_config(method).opposition_threshold
        return (-threshold, threshold)
    if method in optimizer_config.AURA_SIGN_FAMILY:
        return ()
    if method in optimizer_config.AURA_S_FAMILY:
        return ()
    if method in (optimizer_config.METHOD_ADAM_AURA_SPRING_ABLATION, optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION):
        threshold = _aura_spring_config(method).psi_opposition
        return (-threshold, threshold)
    config = _innovation_config(method)
    return (-config.psi_opposition, -config.psi_alignment, config.psi_alignment, config.psi_opposition)


def ordered_methods(methods: Iterable[str]) -> list[str]:
    """``methods`` reordered to follow ``optimizer_config.METHODS``."""

    order = {method: index for index, method in enumerate(optimizer_config.METHODS)}
    return sorted(methods, key=lambda method: (order.get(method, len(order)),))


def _latex_method(method: str) -> str:
    """ """

    if not usetex_enabled():
        return method
    return r"\texttt{" + method.replace("_", r"\_") + "}"


UNKNOWN_METHOD_COLOR = "#888888"
_warned_unknown_methods: set[str] = set()


def _method_color(method_colors: dict[str, str], method: str) -> str:
    try:
        return method_colors[method]
    except KeyError:
        if method not in _warned_unknown_methods:
            _warned_unknown_methods.add(method)
            warnings.warn(
                f"[comparison_report] no color for method {method!r} in "
                f"optimizer_config.METHOD_COLORS (removed after these results "
                f"were written?); drawing it in {UNKNOWN_METHOD_COLOR}",
                stacklevel=2,
            )
        return UNKNOWN_METHOD_COLOR


def _cap_loss_axis(axis, *, maximum: float = 10.0) -> None:
    if not axis.lines:
        return
    plotted_values = np.concatenate(
        [np.asarray(line.get_ydata()).ravel() for line in axis.lines]
    )
    finite_positive = plotted_values[
        np.isfinite(plotted_values) & (plotted_values > 0.0)
    ]
    if finite_positive.size and np.any(finite_positive > maximum):
        lower, _ = axis.get_ylim()
        if not np.isfinite(lower) or lower >= maximum:
            lower = maximum / 10.0
        axis.set_ylim(bottom=lower, top=maximum)


def _set_centered_weight_scale(axis, linear_threshold: float) -> None:
    from matplotlib.ticker import FuncFormatter

    axis.set_yscale("symlog", linthresh=linear_threshold, linscale=1.2)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value + WEIGHT_SCALE_CENTER:g}")
    )


def to_decibels(values):
    """A mean squared error in decibels, ``10 log10(MSE)``."""

    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.where(values > 0.0, values, np.nan))


def _plot_min_mean_max(axis, steps, values, *, color, label=None, linewidth=2.0, transform=None) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    valid_columns = np.any(np.isfinite(values), axis=0)
    if not np.any(valid_columns):
        return
    steps = np.asarray(steps)[valid_columns]
    values = values[:, valid_columns]
    values = np.where(np.isfinite(values), values, np.nan)
    minimum = np.nanmin(values, axis=0)
    mean = np.nanmean(values, axis=0)
    maximum = np.nanmax(values, axis=0)
    if transform is not None:
        # After aggregation: the mean of a dB curve is not the dB of the mean error.
        minimum, mean, maximum = transform(minimum), transform(mean), transform(maximum)
    axis.fill_between(steps, minimum, maximum, color=color, alpha=0.13)
    axis.plot(steps, minimum, color=color, linestyle="--", linewidth=max(0.9, linewidth * 0.55), alpha=0.85)
    axis.plot(steps, mean, color=color, linewidth=linewidth, label=label)
    axis.plot(steps, maximum, color=color, linestyle=":", linewidth=max(1.0, linewidth * 0.6), alpha=0.9)


def _nan_quartiles_over_seeds(histories):
    """Per-step 25th / 50th / 75th percentiles over the seeds whose loss is still
    finite at that step."""

    histories = np.asarray(histories, dtype=float)
    histories = np.where(np.isfinite(histories), histories, np.nan)
    with warnings.catch_warnings():
        # All-NaN columns past the last surviving seed are expected.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return tuple(
            np.nanpercentile(histories, percentile, axis=0)
            for percentile in (25, 50, 75)
        )


@never_fatal
def write_multiseed_comparison_report_pdf(
    seed_runs: list[tuple[int, list[MethodRunResult]]],
    selections_by_seed: dict[int, tuple[weight_dynamics.WeightSelection, ...]],
    path: Path,
    *,
    every: int,
    title: str,
    method_colors: dict[str, str],
    loss_label: str = "loss",
    train_title: str = "Training loss",
    test_title: str = "Test loss",
    decibels: bool = False,
    landscape_fn: Callable[[Any, tuple], None] | None = None,
    steps_per_epoch: float | None = None,
) -> None:
    """Write the loss/diagnostics/weight-dynamics dense report as one PDF."""

    plt = _pyplot()
    from matplotlib.lines import Line2D

    x_label = "optimizer step" if steps_per_epoch is None else "epoch"

    def to_x(steps):
        steps = np.asarray(steps)
        return steps if steps_per_epoch is None else steps / steps_per_epoch

    seeds = [seed for seed, _ in seed_runs]
    methods = ordered_methods(result.method for result in seed_runs[0][1])
    by_seed_and_method = {
        (seed, result.method): result
        for seed, results in seed_runs
        for result in results
    }

    number_of_rows = 3 + len(seeds) * len(methods)
    figure = plt.figure(figsize=(30.0, 4.2 * number_of_rows))
    figure.subplots_adjust(
        left=0.055, right=0.97, bottom=0.04, top=0.96, hspace=0.55, wspace=1.15
    )
    # 21 columns: row 2 holds three 7-wide panels; per-seed rows use 3-wide panels.
    grid = figure.add_gridspec(number_of_rows, 21)

    loss_axes = (figure.add_subplot(grid[0, :6]), figure.add_subplot(grid[0, 6:12]))
    for axis, attribute, panel_title in (
        (loss_axes[0], "train_loss", train_title),
        (loss_axes[1], "test_loss", test_title),
    ):
        for method in methods:
            histories = np.stack(
                [
                    np.asarray(getattr(by_seed_and_method[(seed, method)], attribute))
                    for seed in seeds
                ]
            )
            display_indices = weight_dynamics.snapshot_indices(histories.shape[1], every)
            _plot_min_mean_max(
                axis,
                to_x(display_indices),
                histories[:, display_indices],
                color=_method_color(method_colors, method),
                label=_latex_method(method),
                transform=to_decibels if decibels else None,
            )
        if not decibels:
            axis.set_yscale("log")
        axis.set_title(f"{panel_title}: min / mean / max over seeds")
        axis.set_xlabel(x_label)
        axis.set_ylabel(loss_label)
        axis.grid(which="both", linewidth=0.5, alpha=0.3)
        if not decibels:
            _cap_loss_axis(axis)
    loss_axes[0].legend(frameon=True, framealpha=0.8, edgecolor="none")

    landscape_axes = (figure.add_subplot(grid[1, :6]), figure.add_subplot(grid[1, 6:12]))
    if landscape_fn is not None:
        landscape_fn(figure, landscape_axes)
    else:
        for axis in landscape_axes:
            axis.text(
                0.5,
                0.5,
                "Not applicable: this case has no 2D domain landscape to grid over",
                ha="center",
                va="center",
                wrap=True,
            )
            axis.set_axis_off()

    diagnostic_axes = tuple(
        figure.add_subplot(grid[2, start : start + 7]) for start in (0, 7, 14)
    )
    for method in methods:
        histories = np.stack(
            [by_seed_and_method[(seed, method)].diagnostics[:, 1] for seed in seeds]
        )
        _plot_min_mean_max(
            diagnostic_axes[0],
            to_x(np.arange(histories.shape[1])),
            histories,
            color=_method_color(method_colors, method),
            label=_latex_method(method),
            linewidth=1.8,
        )
    diagnostic_axes[0].set_yscale("log")
    diagnostic_axes[0].set_title(r"Mean $\gamma$: min / mean / max over seeds")
    diagnostic_axes[0].set_ylabel(r"$\gamma$")

    by_method_present = set(methods)
    aura_methods = (
        optimizer_config.METHOD_ADAM_AURA,
        optimizer_config.METHOD_ASTRA,
        optimizer_config.METHOD_ECLIPSE,
        optimizer_config.METHOD_AURA_LIGHT,
        optimizer_config.METHOD_MUON_AURA,
        optimizer_config.METHOD_ADAM_AURA_S,
        optimizer_config.METHOD_MUON_AURA_S,
        optimizer_config.METHOD_ADAM_AURA_SIGN,
        optimizer_config.METHOD_MUON_AURA_SIGN,
        optimizer_config.METHOD_ADAM_AURA_SNR,
        optimizer_config.METHOD_MUON_AURA_SNR,
        optimizer_config.METHOD_ADAM_AURA_SNR_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_COSINE_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_ABLATION,
        optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_EB,
    )
    for axis, panel_methods, statistic, panel_title, ylabel, thresholds_by_method in (
        (
            diagnostic_axes[1],
            aura_methods,
            7,
            r"Mean $\chi$ (AURA / ASTRA / ECLIPSE / AURA light / Muon-AURA innovation)",
            r"$\chi$",
            {method: _chi_thresholds(method) for method in aura_methods},
        ),
        (
            diagnostic_axes[2],
            aura_methods,
            8,
            r"Mean $|\Psi|$ (AURA / ASTRA / ECLIPSE / AURA light / Muon-AURA rotation)",
            r"$\Psi$",
            {method: tuple(abs(value) for value in _psi_thresholds(method) if value > 0) for method in aura_methods},
        ),
    ):
        present_methods = [method for method in panel_methods if method in by_method_present]
        if not present_methods:
            axis.text(
                0.5, 0.5,
                f"{' / '.join(_latex_method(method) for method in panel_methods)} not selected",
                ha="center", va="center", transform=axis.transAxes,
            )
        else:
            for method in present_methods:
                histories = np.stack(
                    [
                        by_seed_and_method[(seed, method)].diagnostics[:, statistic]
                        for seed in seeds
                    ]
                )
                _plot_min_mean_max(
                    axis,
                    to_x(np.arange(histories.shape[1])),
                    histories,
                    color=_method_color(method_colors, method),
                    label=_latex_method(method),
                    linewidth=1.8,
                )
            if len(present_methods) > 1:
                axis.legend(
                    frameon=True, framealpha=0.8, edgecolor="none",
                    fontsize=SMALL_LEGEND_FONTSIZE,
                )
        for threshold in sorted(
            {
                threshold
                for method in present_methods
                for threshold in thresholds_by_method[method]
            }
        ):
            axis.axhline(threshold, color="0.4", linewidth=0.9, linestyle="-.")
        axis.set_title(f"{panel_title}: min / mean / max")
        axis.set_ylabel(ylabel)
    for axis in diagnostic_axes:
        axis.set_xlabel(x_label)
        axis.grid(which="both", linewidth=0.5, alpha=0.3)
    diagnostic_axes[0].legend(frameon=True, framealpha=0.8, edgecolor="none", fontsize=SMALL_LEGEND_FONTSIZE)

    all_weight_components = np.concatenate(
        [
            component(result.weight_history).ravel()
            for _, results in seed_runs
            for result in results
            for component in (np.real, np.imag)
        ]
    )
    finite_weight_components = np.abs(all_weight_components[np.isfinite(all_weight_components)])
    max_weight_component = float(np.max(finite_weight_components)) if finite_weight_components.size else 1e-12
    linear_threshold = max(max_weight_component, 1e-12) * 1e-3

    for seed_index, seed in enumerate(seeds):
        selections = selections_by_seed[seed]
        for method_index, method in enumerate(methods):
            row = 3 + seed_index * len(methods) + method_index
            result = by_seed_and_method[(seed, method)]
            has_chi = (
                result.selected_chi_history is not None
                and np.any(np.isfinite(result.selected_chi_history))
            )
            has_psi = (
                result.selected_psi_history is not None
                and np.any(np.isfinite(result.selected_psi_history))
            )
            extra_panels = int(has_chi) + int(has_psi)
            if extra_panels > 0:
                weight_axis = figure.add_subplot(grid[row, :3])
                centered_weight_axis = figure.add_subplot(grid[row, 3:6])
                gamma_axis = figure.add_subplot(grid[row, 6:9])
                next_column = 9
                chi_axis = None
                psi_axis = None
                if has_chi:
                    chi_axis = figure.add_subplot(grid[row, next_column : next_column + 3])
                    next_column += 3
                if has_psi:
                    psi_axis = figure.add_subplot(grid[row, next_column : next_column + 3])
                    next_column += 3
            else:
                weight_axis = figure.add_subplot(grid[row, :7])
                centered_weight_axis = figure.add_subplot(grid[row, 7:14])
                gamma_axis = figure.add_subplot(grid[row, 14:21])
                chi_axis = None
                psi_axis = None
            indices = weight_dynamics.snapshot_indices(result.weight_history.shape[0], every)
            steps = to_x(indices.astype(int))
            colors = _selection_colors(selections)
            for selection_index, selection in enumerate(selections):
                color = colors[selection_index]
                opacity = max(0.35, 0.95 - 0.12 * (selection_index % 5))
                values = result.weight_history[indices, selection_index]
                real_values = np.where(np.isfinite(np.real(values)), np.real(values), np.nan)
                imag_values = np.where(np.isfinite(np.imag(values)), np.imag(values), np.nan)
                gamma_values = result.selected_gamma_history[indices, selection_index]
                gamma_values = np.where(np.isfinite(gamma_values), gamma_values, np.nan)
                weight_axis.plot(steps, real_values, color=color, linewidth=1.0, alpha=opacity)
                weight_axis.plot(steps, imag_values, color=color, linewidth=0.9, linestyle="--", alpha=opacity)
                centered_weight_axis.plot(steps, real_values - WEIGHT_SCALE_CENTER, color=color, linewidth=1.0, alpha=opacity)
                centered_weight_axis.plot(steps, imag_values - WEIGHT_SCALE_CENTER, color=color, linewidth=0.9, linestyle="--", alpha=opacity)
                gamma_axis.plot(steps, gamma_values, color=color, linewidth=1.0, alpha=opacity)
                if chi_axis is not None and result.selected_chi_history is not None:
                    chi_values = result.selected_chi_history[indices, selection_index]
                    chi_values = np.where(np.isfinite(chi_values), chi_values, np.nan)
                    chi_axis.plot(steps, chi_values, color=color, linewidth=1.0, alpha=opacity)
                if psi_axis is not None and result.selected_psi_history is not None:
                    psi_values = result.selected_psi_history[indices, selection_index]
                    psi_values = np.where(np.isfinite(psi_values), psi_values, np.nan)
                    psi_axis.plot(steps, psi_values, color=color, linewidth=1.0, alpha=opacity)

            weight_axis.set_yscale("symlog", linthresh=linear_threshold, linscale=1.2)
            _set_centered_weight_scale(centered_weight_axis, linear_threshold)
            gamma_axis.set_yscale("log")
            weight_axis.set_title(
                f"Seed {seed} --- {_latex_method(method)}: selected weights, real (solid) vs imaginary (dashed) part (scale centered at 0)"
            )
            centered_weight_axis.set_title(
                rf"Seed {seed} --- {_latex_method(method)}: selected weights, real (solid) vs imaginary (dashed) part (scale centered at $10^{{-1}}$)"
            )
            gamma_axis.set_title(rf"Seed {seed} --- {_latex_method(method)}: matching $\gamma$ values")
            weight_axis.set_ylabel("signed weight component (real: solid, imag: dashed)")
            centered_weight_axis.set_ylabel("signed weight component (real: solid, imag: dashed)")
            gamma_axis.set_ylabel(r"$\gamma$")
            axes = (weight_axis, centered_weight_axis, gamma_axis)
            if chi_axis is not None:
                finite_steps = steps[np.any(np.isfinite(result.selected_chi_history[indices]), axis=1)]
                if finite_steps.size:
                    chi_axis.set_xlim(steps[0], finite_steps[-1])
                for threshold in _chi_thresholds(method):
                    chi_axis.axhline(threshold, color="0.4", linewidth=0.9, linestyle="-.")
                chi_axis.set_title(
                    rf"Seed {seed} --- {_latex_method(method)}: matching $\chi$ (innovation correlation)"
                )
                chi_axis.set_ylabel(r"$\chi$")
                axes = axes + (chi_axis,)
            if psi_axis is not None:
                finite_steps = steps[np.any(np.isfinite(result.selected_psi_history[indices]), axis=1)]
                if finite_steps.size:
                    psi_axis.set_xlim(steps[0], finite_steps[-1])
                for threshold in _psi_thresholds(method):
                    psi_axis.axhline(threshold, color="0.4", linewidth=0.9, linestyle="-.")
                psi_axis.set_title(
                    rf"Seed {seed} --- {_latex_method(method)}: matching $\Psi$ (signed rotation)"
                )
                psi_axis.set_ylabel(r"$\Psi$")
                axes = axes + (psi_axis,)
            for axis in axes:
                axis.set_xlabel(x_label)
                axis.grid(which="both", linewidth=0.45, alpha=0.28)

            if method_index == 0:
                leaf_handles = [
                    Line2D([0], [0], color=colors[index], linewidth=2, label=selection.label)
                    for index, selection in enumerate(_representative_selections(selections))
                ]
                component_handles = [
                    Line2D([0], [0], color="0.25", linewidth=1.4, label="real"),
                    Line2D([0], [0], color="0.25", linewidth=1.2, linestyle="--", label="imaginary"),
                ]
                weight_axis.legend(
                    handles=leaf_handles + component_handles,
                    frameon=True, framealpha=0.8, edgecolor="none",
                    fontsize=SMALL_LEGEND_FONTSIZE,
                    ncol=min(5, len(leaf_handles) + len(component_handles)),
                )

    figure.suptitle(f"{title} --- {len(seeds)} seeds ({', '.join(str(seed) for seed in seeds)})", fontsize=REPORT_TITLE_FONTSIZE)
    figure.savefig(path, format="pdf", metadata={"Title": f"Multi-seed {title}"})
    plt.close(figure)


@never_fatal
def write_losses_legend_pdf(
    path: Path,
    *,
    methods: Iterable[str],
    method_colors: dict[str, str],
    fontsize: float = 14,
    ncol: int | None = None,
) -> None:
    """Write a standalone legend for the loss-curve figures, one row unless ``ncol`` is given."""

    from matplotlib.lines import Line2D

    plt = _pyplot()

    methods = ordered_methods(methods)
    handles = [
        Line2D([0], [0], color=_method_color(method_colors, method), linewidth=2.0) for method in methods
    ]
    labels = [method.replace("_", " ") for method in methods]

    figure = plt.figure(figsize=(max(4.0, 0.95 * len(methods)), 0.4))
    figure.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol or max(1, len(methods)),
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        columnspacing=1.6,
        borderaxespad=0.0,
    )
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


@never_fatal
def write_combined_losses_pdf_from_runs(
    runs: list[dict],
    path: Path,
    *,
    methods: tuple[str, ...],
    method_colors: dict[str, str],
    every: int,
    suptitle: str | None,
    xlabel: str = "step",
    loss_label: str = "mean squared error",
    train_title: str | None = "Training MSE",
    test_title: str | None = "Test MSE",
    train_key: str = "train_losses",
    test_key: str = "test_losses",
    decibels: bool = False,
    logy: bool = True,
    row_label: str | None = None,
    steps_per_epoch: float | None = None,
    font_scale: float = 1.0,
    legend_ncol: int | None = None,
) -> None:
    """Median / 25--75 percentile loss panels across seeds, for cases whose per-run
    results are a flat ``list[dict]``. ``font_scale`` multiplies every text size;
    ``legend_ncol`` wraps the standalone legend over rows; a ``None`` title is omitted."""

    plt = _pyplot()

    methods = ordered_methods(methods)
    seeds = sorted({run["seed"] for run in runs})
    figure, axes = plt.subplots(1, 2, figsize=(24.0, 6.0))
    for axis, key, title in (
        (axes[0], train_key, train_title),
        (axes[1], test_key, test_title),
    ):
        for method in methods:
            histories = np.stack(
                [np.asarray(run[key]) for run in runs if run["method"] == method]
            )
            display_indices = weight_dynamics.snapshot_indices(histories.shape[1], every)
            steps = display_indices if steps_per_epoch is None else display_indices / steps_per_epoch
            histories = histories[:, display_indices]
            color = _method_color(method_colors, method)
            scale = to_decibels if decibels else (lambda values: values)
            lower, median, upper = (
                scale(values) for values in _nan_quartiles_over_seeds(histories)
            )
            axis.plot(steps, lower, color=color, linewidth=1.0, linestyle=":")
            axis.plot(
                steps,
                median,
                color=color,
                linewidth=2.0,
                label=method.replace("_", " "),
            )
            axis.plot(steps, upper, color=color, linewidth=1.0, linestyle=":")
            axis.fill_between(
                steps,
                lower,
                upper,
                color=color,
                alpha=0.12,
            )
        if not decibels and logy:
            axis.set_yscale("log")
        if title is not None:
            axis.set_title(
                f"{title}: median (25--75 percentile) over seeds", fontsize=18 * font_scale
            )
        axis.set_xlabel(xlabel, fontsize=16 * font_scale)
        axis.set_ylabel(loss_label, fontsize=16 * font_scale)
        axis.tick_params(axis="both", which="both", labelsize=14 * font_scale)
        axis.grid(which="both", linewidth=0.5, alpha=0.3)
    if suptitle is not None:
        figure.suptitle(
            f"{suptitle} --- {len(seeds)} seeds ({', '.join(str(seed) for seed in seeds)})",
            fontsize=22 * font_scale,
        )
    if row_label is not None:
        figure.supylabel(row_label, fontsize=18 * font_scale, fontweight="bold")
        figure.tight_layout(rect=(0.035, 0.0, 1.0, 1.0))
    else:
        figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    write_losses_legend_pdf(
        path.with_name(f"{path.stem}_legend.pdf"),
        methods=methods,
        method_colors=method_colors,
        ncol=legend_ncol,
    )


@never_fatal
def write_combined_losses_pdf_from_seed_runs(
    seed_runs: list[tuple[int, list[Any]]],
    path: Path,
    *,
    methods: tuple[str, ...],
    method_colors: dict[str, str],
    every: int,
    suptitle: str | None,
    xlabel: str = "step",
    loss_label: str = "mean squared error",
    train_title: str | None = "Training MSE",
    test_title: str | None = "Test MSE",
    train_attr: str = "train_mse",
    test_attr: str = "test_mse",
    row_label: str | None = None,
    ylim_top: float | None = None,
    font_scale: float = 1.0,
    legend_ncol: int | None = None,
) -> None:
    """Median / 25--75 percentile loss panels across seeds, for cases whose results
    are ``seed_runs: list[tuple[seed, list[result]]]``. ``font_scale`` multiplies
    every text size; ``legend_ncol`` wraps the standalone legend over rows; a
    ``None`` title is omitted."""

    plt = _pyplot()

    methods = ordered_methods(methods)
    seeds = [seed for seed, _ in seed_runs]
    all_results = [result for _, results in seed_runs for result in results]
    figure, axes = plt.subplots(1, 2, figsize=(24.0, 6.0))
    for axis, attribute, title in (
        (axes[0], train_attr, train_title),
        (axes[1], test_attr, test_title),
    ):
        _plot_loss_quartiles(
            axis,
            all_results,
            methods,
            method_colors,
            attribute,
            every=every if attribute == train_attr else None,
            median_linewidth=2.0,
            quartile_linewidth=1.0,
        )
        axis.set_yscale("log")
        if ylim_top is not None:
            axis.set_ylim(top=ylim_top)
        if title is not None:
            axis.set_title(
                f"{title}: median (25--75 percentile) over seeds", fontsize=18 * font_scale
            )
        axis.set_xlabel(xlabel, fontsize=16 * font_scale)
        axis.set_ylabel(loss_label, fontsize=16 * font_scale)
        axis.tick_params(axis="both", which="both", labelsize=14 * font_scale)
        axis.grid(which="both", linewidth=0.5, alpha=0.3)
    if suptitle is not None:
        figure.suptitle(
            f"{suptitle} --- {len(seeds)} seeds ({', '.join(str(seed) for seed in seeds)})",
            fontsize=22 * font_scale,
        )
    if row_label is not None:
        figure.supylabel(row_label, fontsize=18 * font_scale, fontweight="bold")
        figure.tight_layout(rect=(0.035, 0.0, 1.0, 1.0))
    else:
        figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    write_losses_legend_pdf(
        path.with_name(f"{path.stem}_legend.pdf"),
        methods=methods,
        method_colors=method_colors,
        ncol=legend_ncol,
    )


def _plot_loss_quartiles(
    axis,
    all_results: list[Any],
    methods: list[str],
    method_colors: dict[str, str],
    attribute: str,
    *,
    every: int | None,
    median_linewidth: float,
    quartile_linewidth: float,
) -> None:
    """Per-method median and 25--75 percentile band over seeds; ``every=None`` keeps
    the finite columns, as the test loss has its own (step + 1) % test_every cadence."""

    for method in methods:
        histories = np.stack(
            [
                np.asarray(getattr(result, attribute))
                for result in all_results
                if result.method == method
            ]
        )
        if every is not None:
            steps = weight_dynamics.snapshot_indices(histories.shape[1], every)
            histories = histories[:, steps]
        else:
            valid = np.any(np.isfinite(histories), axis=0)
            steps = np.flatnonzero(valid)
            histories = histories[:, valid]
        color = _method_color(method_colors, method)
        lower, median, upper = _nan_quartiles_over_seeds(histories)
        axis.plot(steps, lower, color=color, linewidth=quartile_linewidth, linestyle=":")
        axis.plot(
            steps,
            median,
            color=color,
            linewidth=median_linewidth,
            label=method.replace("_", " "),
        )
        axis.plot(steps, upper, color=color, linewidth=quartile_linewidth, linestyle=":")
        axis.fill_between(
            steps, lower, upper, color=color, alpha=0.12
        )


def write_stacked_losses_panel_pdf_from_seed_runs(
    seed_runs: list[tuple[int, list[Any]]],
    path: Path,
    *,
    methods: tuple[str, ...],
    method_colors: dict[str, str],
    every: int,
    size_inches: tuple[float, float],
    fontsize: int = 9,
    legend_columns: int = 4,
    xlabel: str = "step",
    train_label: str = "training MSE",
    test_label: str = "test MSE",
    train_attr: str = "train_loss",
    test_attr: str = "test_loss",
) -> None:
    """Training loss above test loss, saved uncropped at ``size_inches`` so that
    ``fontsize`` is the printed size when the PDF is included at that width; the legend
    goes to ``<stem>_legend.pdf`` at the same font size, for inclusion at natural size."""

    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    plt = _pyplot(fontsize=fontsize)

    methods = ordered_methods(methods)
    all_results = [result for _, results in seed_runs for result in results]
    figure, axes = plt.subplots(2, 1, figsize=size_inches, sharex=True, layout="constrained")
    for axis, attribute, label in ((axes[0], train_attr, train_label), (axes[1], test_attr, test_label)):
        _plot_loss_quartiles(
            axis,
            all_results,
            methods,
            method_colors,
            attribute,
            every=every if attribute == train_attr else None,
            median_linewidth=0.9,
            quartile_linewidth=0.4,
        )
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_ylabel(label)
        axis.tick_params(axis="both", which="major", labelsize=fontsize - 1)
        axis.grid(which="major", linewidth=0.3, alpha=0.4)
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    axes[1].set_xlabel(xlabel)
    figure.savefig(path, format="pdf")
    plt.close(figure)
    write_losses_legend_pdf(
        path.with_name(f"{path.stem}_legend.pdf"),
        methods=methods,
        method_colors=method_colors,
        fontsize=fontsize,
        ncol=legend_columns,
    )


def write_losses_row_pdf_from_seed_runs(
    seed_runs: list[tuple[int, list[Any]]],
    path: Path,
    *,
    methods: tuple[str, ...],
    method_colors: dict[str, str],
    every: int,
    size_inches: tuple[float, float],
    ylim: tuple[float, float],
    step_axis: bool = True,
    column_titles: tuple[str, str] | None = None,
    test_every: int | None = None,
    fontsize: int = 8,
    legend_columns: int = 4,
    xlabel: str = "step",
    loss_label: str = "mean squared error",
    train_attr: str = "train_loss",
    test_attr: str = "test_loss",
) -> None:
    """Training loss beside test loss on a shared y axis (labelled on the left panel
    only), saved uncropped at ``size_inches`` as one row of a stacked paper figure;
    ``step_axis=False`` drops the x tick labels and label from rows above the bottom
    one. ``test_every=None`` keeps the test loss's own finite columns; a cadence
    subsamples it like the training loss. The legend goes to ``<stem>_legend.pdf``
    at the same font size."""

    from matplotlib.ticker import LogLocator, NullFormatter

    plt = _pyplot(fontsize=fontsize)

    methods = ordered_methods(methods)
    all_results = [result for _, results in seed_runs for result in results]
    figure, axes = plt.subplots(1, 2, figsize=size_inches, sharey=True, layout="constrained")
    for axis, attribute, cadence in ((axes[0], train_attr, every), (axes[1], test_attr, test_every)):
        _plot_loss_quartiles(
            axis,
            all_results,
            methods,
            method_colors,
            attribute,
            every=cadence,
            median_linewidth=0.9,
            quartile_linewidth=0.4,
        )
        axis.set_yscale("log")
        axis.set_ylim(*ylim)
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="both", which="major", labelsize=fontsize - 1)
        axis.grid(which="major", linewidth=0.3, alpha=0.4)
        if step_axis:
            axis.set_xlabel(xlabel)
        else:
            axis.tick_params(axis="x", labelbottom=False)
    axes[0].set_ylabel(loss_label)
    if column_titles is not None:
        for axis, title in zip(axes, column_titles):
            axis.set_title(title)
    figure.savefig(path, format="pdf")
    plt.close(figure)
    write_losses_legend_pdf(
        path.with_name(f"{path.stem}_legend.pdf"),
        methods=methods,
        method_colors=method_colors,
        fontsize=fontsize,
        ncol=legend_columns,
    )


_SELECTION_PALETTE = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#65a30d")


def _selection_colors(selections: tuple[weight_dynamics.WeightSelection, ...]) -> list[str]:
    """Color selected components by their source leaf."""

    leaf_indices = sorted({selection.leaf_index for selection in selections})
    palette_by_leaf = {
        leaf_index: _SELECTION_PALETTE[position % len(_SELECTION_PALETTE)]
        for position, leaf_index in enumerate(leaf_indices)
    }
    return [palette_by_leaf[selection.leaf_index] for selection in selections]


def _representative_selections(selections: tuple[weight_dynamics.WeightSelection, ...]) -> tuple[weight_dynamics.WeightSelection, ...]:
    """One selection per distinct source leaf, for a compact legend."""

    seen = set()
    representatives = []
    for selection in selections:
        if selection.leaf_index in seen:
            continue
        seen.add(selection.leaf_index)
        representatives.append(selection)
    return tuple(representatives)

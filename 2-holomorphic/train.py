"""Multi-seed comparison of complex optimizers on the 2-holomorphic benchmark."""


from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass, fields, replace
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any
import uuid

# Before `import jax`: lets the per-seed processes share the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map
import numpy as np
import optax

# Must precede `import config`: config.py itself needs `src` on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as case_config
import dataset
import model
import plots

from src import batching
from src import comparison_report
from src import metrics
from src import optimizer_config
from src import optimizers
from src import weight_dynamics

CASE_NAME = "holomorphic"

_DEFAULT_CONFIG = case_config.ExperimentConfig()
SEEDS: tuple[int, ...] = _DEFAULT_CONFIG.seeds
STEPS: int = _DEFAULT_CONFIG.steps
TRAIN_SIZE: int = _DEFAULT_CONFIG.train_size
MINIBATCH_SIZE: int = _DEFAULT_CONFIG.minibatch_size
WIDTH: int = _DEFAULT_CONFIG.hidden_width
DEPTH: int = _DEFAULT_CONFIG.hidden_layers
LEARNING_RATE: float = _DEFAULT_CONFIG.learning_rate
MAX_WORKERS: int = _DEFAULT_CONFIG.max_workers
PRECISION: str = _DEFAULT_CONFIG.precision
TEST_EVERY: int = _DEFAULT_CONFIG.test_every
METHODS: tuple[str, ...] = optimizer_config.METHODS
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "results"
QUIET: bool = False

TRACKED_LEAVES = 6
TRACKED_PER_LEAF = 4

# Keyed on hyperparameters, not code: clear results/cache/ after editing an optimizer.
CACHE_ADAM_RESULTS = True
CACHED_METHODS = optimizer_config.METHODS
CACHE_DIR = OUTPUT_DIR / "cache"


@dataclass
class MethodResult:
    method: str
    params: Any
    train_mse: np.ndarray
    test_mse: np.ndarray
    diagnostics: np.ndarray
    weight_history: np.ndarray
    selected_gamma_history: np.ndarray
    selected_angle_history: np.ndarray
    selected_smoothed_angle_history: np.ndarray
    selected_q_history: np.ndarray
    selected_chi_history: np.ndarray
    selected_psi_history: np.ndarray
    training_seconds: float
    final_train_mse: float
    final_test_mse: float
    final_test_relative_l2: float
    holomorphicity_residual: float
    # Defaulted so pickles cached before the warm-up still load.
    compilation_seconds: float = float("nan")

    @property
    def signal_name(self) -> str:
        if self.method in (
            optimizer_config.METHOD_ADAM_AURA,
            optimizer_config.METHOD_ASTRA,
            optimizer_config.METHOD_ECLIPSE,
            optimizer_config.METHOD_AURA_LIGHT,
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
        ):
            return "mean_chi"
        if self.method == optimizer_config.METHOD_NEU_OPTEB:
            return "mean_normalized_radius"
        return "none"

    def summary(self) -> dict[str, Any]:
        final_diagnostics = self.diagnostics[-1]
        finite_train = self.train_mse[np.isfinite(self.train_mse)]
        return {
            "final_train_mse": self.final_train_mse,
            "best_train_mse": float(np.min(finite_train)),
            "learning_area": metrics.log_learning_area(self.train_mse),
            "final_test_mse": self.final_test_mse,
            "final_test_relative_l2": self.final_test_relative_l2,
            "training_seconds": self.training_seconds,
            "holomorphicity_residual": self.holomorphicity_residual,
            "gamma_min": float(final_diagnostics[0]),
            "gamma_mean": float(final_diagnostics[1]),
            "gamma_max": float(final_diagnostics[2]),
            "signal_name": self.signal_name,
            "mean_signal": (
                None if not np.isfinite(final_diagnostics[3]) else float(final_diagnostics[3])
            ),
            "mean_angle_degrees": (
                None if not np.isfinite(final_diagnostics[4]) else float(final_diagnostics[4])
            ),
            "mean_abs_omega_degrees": (
                None if not np.isfinite(final_diagnostics[5]) else float(final_diagnostics[5])
            ),
        }


def _dtype_for_precision(precision: str):
    if precision == "64":
        jax.config.update("jax_enable_x64", True)
        return jnp.complex128
    return jnp.complex64


def prepare_problem(config: case_config.ExperimentConfig):
    """Create the one dataset and initialization shared by every method."""

    config.validate()
    dtype = _dtype_for_precision(config.precision)
    initialization_key, dataset_key = jax.random.split(jax.random.PRNGKey(config.seed))
    params = model.init_model(
        initialization_key,
        config.architecture,
        dtype=dtype,
        beta=config.initialization_beta,
    )
    datasets = dataset.make_datasets(
        dataset_key,
        train_size=config.train_size,
        test_size=config.test_size,
        half_width=config.domain_half_width,
        dtype=dtype,
    )
    return params, datasets


def _make_train_step(method, optimizer):
    def train_step(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(model.mean_squared_error)(params, inputs, targets)
        # JAX's complex gradient is the conjugate of the direction Optax expects.
        grads = tree_map(jnp.conj, grads)
        if method == optimizer_config.METHOD_LBFGS:
            def minibatch_loss(candidate_params):
                return model.mean_squared_error(candidate_params, inputs, targets)

            updates, opt_state = optimizer.update(
                grads, opt_state, params, value=loss, grad=grads, value_fn=minibatch_loss
            )
        else:
            updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return jax.jit(train_step)


def run_method(
    method: str,
    config: case_config.ExperimentConfig,
    initial_params,
    datasets,
    weight_selections: tuple[weight_dynamics.WeightSelection, ...],
    *,
    show_progress: bool,
) -> MethodResult:
    """Compile and train one optimizer from the shared starting point."""

    z_train, y_train, z_test, y_test = datasets
    updates_per_epoch = case_config.steps_per_epoch(config.train_size, config.minibatch_size)
    optimizer = optimizers.build_optimizer(
        method,
        config.learning_rate,
        steps_per_epoch=updates_per_epoch,
        total_steps=config.steps,
    )
    params = initial_params
    opt_state = optimizer.init(params)
    train_step = _make_train_step(method, optimizer)
    evaluate_loss = jax.jit(model.mean_squared_error)

    steps = config.steps
    minibatch_size = min(config.train_size, config.minibatch_size)
    train_mse = np.zeros(steps)
    test_mse = np.full(steps, np.nan)
    # Only every test_every steps: each snapshot is a blocking device-to-host sync.
    diagnostics = np.full((steps, 9), np.nan)  # width must match optimizer_diagnostics()'s return
    weight_history = np.full((steps, len(weight_selections)), np.nan, dtype=params[0]["weight"].dtype)
    selected_gamma_history = np.full((steps, len(weight_selections)), np.nan)
    # Retired signal, kept all-NaN so cached pickles and CSVs keep their layout.
    selected_angle_history = np.full((steps, len(weight_selections)), np.nan)
    selected_smoothed_angle_history = np.full((steps, len(weight_selections)), np.nan)
    selected_q_history = np.full((steps, len(weight_selections)), np.nan)
    selected_chi_history = np.full((steps, len(weight_selections)), np.nan)
    selected_psi_history = np.full((steps, len(weight_selections)), np.nan)

    if show_progress:
        print(f"\n[{method}] starting {steps} optimizer steps...", flush=True)
    minibatches = batching.epoch_batches(
        jax.random.PRNGKey(config.seed + 1), config.train_size, minibatch_size, steps
    )
    # Compile before timing; warm-up results are discarded, so no state advances.
    compile_start = time.perf_counter()
    jax.block_until_ready(train_step(params, opt_state, z_train[:minibatch_size], y_train[:minibatch_size]))
    jax.block_until_ready(evaluate_loss(params, z_test, y_test))
    compilation_seconds = time.perf_counter() - compile_start
    diagnostics_seconds = 0.0
    start = time.perf_counter()
    for step in range(steps):
        indices = next(minibatches)
        params, opt_state, loss = train_step(params, opt_state, z_train[indices], y_train[indices])
        train_mse[step] = loss
        is_test_step = (step + 1) % config.test_every == 0 or step == steps - 1
        if is_test_step:
            test_mse[step] = float(evaluate_loss(params, z_test, y_test))
        # Must match weight_dynamics.snapshot_indices, which the report panels index by.
        is_snapshot_step = step % config.test_every == 0 or step == steps - 1
        if is_snapshot_step:
            diagnostics_start = time.perf_counter()
            diagnostics[step] = np.asarray(
                optimizers.optimizer_diagnostics(method, opt_state)
            )
            weight_history[step] = np.asarray(weight_dynamics.selected_weight_values(params, weight_selections))
            selected_gamma_history[step] = np.asarray(
                weight_dynamics.selected_gamma_values(method, opt_state, weight_selections)
            )
            selected_chi_history[step] = np.asarray(
                weight_dynamics.selected_chi_values(method, opt_state, weight_selections)
            )
            selected_psi_history[step] = np.asarray(
                weight_dynamics.selected_psi_values(method, opt_state, weight_selections)
            )
            diagnostics_seconds += time.perf_counter() - diagnostics_start
        if show_progress and is_test_step:
            print(
                f"[{method}] step {step + 1:5d}/{steps}  "
                f"train MSE {train_mse[step]:.3e}  test MSE {test_mse[step]:.3e}",
                flush=True,
            )
    training_seconds = time.perf_counter() - start - diagnostics_seconds

    final_train_mse = float(train_mse[-1])
    final_test_mse = float(test_mse[-1])
    final_relative_l2 = float(model.relative_l2_error(params, z_test, y_test))
    cr_residual = (
        float(model.holomorphicity_residual(params)) if dataset.TARGET_INPUT_WIDTH == 1 else float("nan")
    )
    if show_progress:
        print(
            f"[{method}] training complete in {training_seconds:.3f} s "
            f"(compilation {compilation_seconds:.2f} s, not timed) | final test MSE={final_test_mse:.6e}",
            flush=True,
        )

    return MethodResult(
        method=method,
        params=params,
        train_mse=train_mse,
        test_mse=test_mse,
        diagnostics=diagnostics,
        weight_history=weight_history,
        selected_gamma_history=selected_gamma_history,
        selected_angle_history=selected_angle_history,
        selected_smoothed_angle_history=selected_smoothed_angle_history,
        selected_q_history=selected_q_history,
        selected_chi_history=selected_chi_history,
        selected_psi_history=selected_psi_history,
        training_seconds=training_seconds,
        final_train_mse=final_train_mse,
        final_test_mse=final_test_mse,
        final_test_relative_l2=final_relative_l2,
        holomorphicity_residual=cr_residual,
        compilation_seconds=compilation_seconds,
    )


def _cache_path(method: str, config: case_config.ExperimentConfig) -> Path:
    key = (
        f"{method}_seed{config.seed}_steps{config.steps}"
        f"_train{config.train_size}_batch{config.minibatch_size}"
        f"_w{config.hidden_width}_d{config.hidden_layers}"
        f"_lr{config.learning_rate:g}"
    )
    return CACHE_DIR / f"{key}.pkl"


def _load_cached_run(path: Path, run_id: str) -> MethodResult | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as file:
            payload = pickle.load(file)
    except (EOFError, pickle.UnpicklingError, AttributeError, ImportError, ValueError) as exc:
        print(f"[cache] discarding unreadable cache {path} ({exc!r})", flush=True)
        path.unlink(missing_ok=True)
        return None
    if isinstance(payload, optimizer_config.CachedRun):
        result = payload.result
        if not optimizer_config.same_hyperparameters(
            payload.config, optimizer_config.hyperparameters_for_method(result.method)
        ):
            return None
    else:
        # Legacy pickle from before CachedRun: accepted unvalidated.
        result = payload
    # Pickle restores __dict__ without __init__, so fields added later are missing.
    missing = [field.name for field in fields(MethodResult) if not hasattr(result, field.name)]
    if missing:
        reference = np.asarray(result.selected_gamma_history)
        result = replace(result, **{name: np.full_like(reference, np.nan) for name in missing})
    return result


def _save_cached_run(path: Path, result: MethodResult, run_id: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = optimizer_config.CachedRun(
        config=optimizer_config.hyperparameters_for_method(result.method), result=result, run_id=run_id
    )
    tmp_path = path.with_suffix(f".pkl.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as file:
            pickle.dump(payload, file)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_comparison(
    config: case_config.ExperimentConfig,
    methods: tuple[str, ...] = optimizer_config.METHODS,
    *,
    run_id: str,
    show_progress: bool = True,
) -> list[MethodResult]:
    """Run all requested methods with identical data and parameters."""

    unknown = tuple(method for method in methods if method not in optimizer_config.METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    initial_params, datasets = prepare_problem(config)
    weight_selections = weight_dynamics.make_weight_selections(
        initial_params, seed=config.seed, max_leaves=TRACKED_LEAVES, per_leaf=TRACKED_PER_LEAF
    )
    results = []
    for method in methods:
        use_cache = CACHE_ADAM_RESULTS and method in CACHED_METHODS
        cache_path = _cache_path(method, config) if use_cache else None
        result = _load_cached_run(cache_path, run_id) if use_cache else None
        if result is None:
            result = run_method(
                method, config, initial_params, datasets, weight_selections, show_progress=show_progress
            )
            if use_cache:
                _save_cached_run(cache_path, result, run_id)
        elif show_progress:
            print(f"[{method}] loaded cached run from {cache_path}", flush=True)
        results.append(result)
    return results


def _run_seed(
    run_index: int,
    seed: int,
    total_seeds: int,
    config: case_config.ExperimentConfig,
    methods: tuple[str, ...],
    run_id: str,
    show_progress: bool,
) -> tuple[int, list[MethodResult]]:
    """ """

    seed_config = replace(config, seed=seed)
    seed_config.validate()
    if show_progress:
        print(f"\n=== Seed {seed} ({run_index}/{total_seeds}) ===", flush=True)
    return seed, run_comparison(seed_config, methods, run_id=run_id, show_progress=show_progress)


def run_comparison_seeds(
    config: case_config.ExperimentConfig,
    seeds: tuple[int, ...] | None = None,
    methods: tuple[str, ...] = optimizer_config.METHODS,
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> list[tuple[int, list[MethodResult]]]:
    """Repeat matched comparisons concurrently across random seeds."""

    seeds = config.seeds if seeds is None else seeds
    max_workers = config.max_workers if max_workers is None else max_workers
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    if max_workers < 1:
        raise ValueError("max_workers must be a positive integer")

    run_id = uuid.uuid4().hex
    worker_count = min(max_workers, len(seeds))
    # "spawn": forking a process that has initialized CUDA breaks the child.
    mp_context = multiprocessing.get_context("spawn")
    tasks = [
        (run_index, seed, len(seeds), config, methods, run_id, show_progress)
        for run_index, seed in enumerate(seeds, start=1)
    ]
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
        return list(executor.map(_run_seed, *zip(*tasks)))


def _count_complex_parameters(params) -> int:
    return int(sum(np.size(leaf) for leaf in tree_leaves(params)))


def _write_nan_counts(seed_runs: list[tuple[int, list[MethodResult]]], output_dir: Path) -> None:
    """Steps with a non-finite training loss, per method and seed."""

    seeds = [seed for seed, _ in seed_runs]
    width = max(len(result.method) for result in seed_runs[0][1])
    steps = len(seed_runs[0][1][0].train_mse)
    lines = [
        f"# steps with non-finite training loss (out of {steps}); nan_seeds: seeds with at least one",
        f"{'method':<{width}}  nan_seeds  " + "  ".join(f"seed_{seed}" for seed in seeds),
    ]
    for index, result in enumerate(seed_runs[0][1]):
        counts = [int(np.count_nonzero(~np.isfinite(results[index].train_mse))) for _, results in seed_runs]
        lines.append(
            f"{result.method:<{width}}  {sum(count > 0 for count in counts):>9}  "
            + "  ".join(f"{count:>{len(f'seed_{seed}')}}" for count, seed in zip(counts, seeds))
        )
    (output_dir / "nan_counts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite_ranking_value(value: Any) -> float:
    """Sort non-finite or unavailable metrics after finite metrics."""

    if isinstance(value, dict):
        value = value.get("mean")
    return float(value) if value is not None and np.isfinite(value) else np.inf


def _format_aggregate_metric(metric: dict[str, float | None]) -> str:
    if metric["mean"] is None:
        return "n/a (all seeds non-finite)"
    return f"{metric['mean']:11.4e} ({metric['min']:.4e} to {metric['max']:.4e})"


def _format_mean_std_metric(metric: dict[str, float | None], *, unit: str = "s") -> str:
    if metric["mean"] is None:
        return "n/a (all seeds non-finite)"
    return f"{metric['mean']:.3f} ± {metric['std']:.3f} {unit}"


def _aggregate_method_summaries(
    seed_runs: list[tuple[int, list[MethodResult]]],
) -> dict[str, dict[str, dict[str, float] | str | None]]:
    """Return min/mean/max/std and quartile summaries over seeds for every numeric metric."""

    methods = [result.method for result in seed_runs[0][1]]
    summaries: dict[str, dict[str, dict[str, float] | str | None]] = {}
    for method in methods:
        per_seed = [
            next(result for result in results if result.method == method).summary()
            for _, results in seed_runs
        ]
        method_summary: dict[str, dict[str, float] | str | None] = {}
        for key in per_seed[0]:
            values = [summary[key] for summary in per_seed]
            is_numeric_metric = all(
                value is None or isinstance(value, (int, float, np.number)) for value in values
            )
            if is_numeric_metric:
                method_summary[key] = metrics.seed_statistics(values)
            else:
                method_summary[key] = values[0]
        summaries[method] = method_summary
    return summaries


def write_results(
    config: case_config.ExperimentConfig,
    seed_runs: list[tuple[int, list[MethodResult]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Persist per-step histories, the multi-seed JSON summary, and PDF plots."""

    if not seed_runs:
        raise ValueError("seed_runs must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [seed for seed, _ in seed_runs]
    methods = [result.method for result in seed_runs[0][1]]
    expected_methods = tuple(methods)
    for seed, results in seed_runs:
        if tuple(result.method for result in results) != expected_methods:
            raise ValueError(f"method mismatch in seed {seed}")

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "seed",
                "method",
                "step",
                "train_mse",
                "test_mse",
                "gamma_min",
                "gamma_mean",
                "gamma_max",
                "signal_name",
                "mean_signal",
                "mean_angle_degrees",
                "mean_abs_omega_degrees",
            ]
        )
        for seed, results in seed_runs:
            for result in results:
                for step, (train, test, diagnostic) in enumerate(
                    zip(result.train_mse, result.test_mse, result.diagnostics)
                ):
                    writer.writerow(
                        [
                            seed,
                            result.method,
                            step,
                            f"{train:.16e}",
                            "" if not np.isfinite(test) else f"{test:.16e}",
                            f"{diagnostic[0]:.16e}",
                            f"{diagnostic[1]:.16e}",
                            f"{diagnostic[2]:.16e}",
                            result.signal_name,
                            "" if not np.isfinite(diagnostic[3]) else f"{diagnostic[3]:.16e}",
                            "" if not np.isfinite(diagnostic[4]) else f"{diagnostic[4]:.16e}",
                            "" if not np.isfinite(diagnostic[5]) else f"{diagnostic[5]:.16e}",
                        ]
                    )

    selections_by_seed = {
        seed: weight_dynamics.make_weight_selections(
            results[0].params, seed=seed, max_leaves=TRACKED_LEAVES, per_leaf=TRACKED_PER_LEAF
        )
        for seed, results in seed_runs
    }
    plots.write_raw_histories(seed_runs, output_dir)
    report_seed_runs = [
        (
            seed,
            [
                comparison_report.MethodRunResult(
                    method=result.method,
                    train_loss=result.train_mse,
                    test_loss=result.test_mse,
                    diagnostics=result.diagnostics,
                    weight_history=result.weight_history,
                    selected_gamma_history=result.selected_gamma_history,
                    selected_chi_history=result.selected_chi_history,
                    selected_psi_history=result.selected_psi_history,
                )
                for result in results
            ],
        )
        for seed, results in seed_runs
    ]
    plots.write_comparison_report_pdf(
        report_seed_runs,
        selections_by_seed,
        output_dir / "comparison.pdf",
        every=config.test_every,
        method_colors=optimizer_config.METHOD_COLORS,
    )
    plots.save_loss_comparison_pdf(
        report_seed_runs,
        output_dir / "combined_losses.pdf",
        every=config.test_every,
        # Only the methods present: an absent method would np.stack() an empty list.
        methods=expected_methods,
        method_colors=optimizer_config.METHOD_COLORS,
    )

    method_summaries = _aggregate_method_summaries(seed_runs)
    ranking = sorted(
        methods, key=lambda method: _finite_ranking_value(method_summaries[method]["final_test_mse"])
    )
    parameter_count = _count_complex_parameters(seed_runs[0][1][0].params)
    (output_dir / "parameter_count.txt").write_text(f"{parameter_count}\n", encoding="utf-8")
    _write_nan_counts(seed_runs, output_dir)
    summary = {
        "experiment": {**config.to_dict(), "seed": None, "seeds": seeds},
        "software_and_device": {
            "jax": jax.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
            "devices": [str(device) for device in jax.devices()],
            "platforms": sorted({device.platform for device in jax.devices()}),
        },
        "shared_design": {
            "same_initialization_within_seed": True,
            "same_data_within_seed": True,
            "same_learning_rate": True,
            "same_minibatch_sequence_within_seed": True,
            "loss": "mean(abs(prediction - target)**2)",
            "complex_gradient_conjugated_before_optax": True,
            "complex_parameter_count": parameter_count,
            "real_degrees_of_freedom": 2 * parameter_count,
            "network_kind": CASE_NAME,
            "hidden_activation": model.ACTIVATION_NAME,
        },
        "optimizer_hyperparameters": case_config.reference_hyperparameters(),
        "methods": method_summaries,
        "ranking_by_mean_final_test_mse": ranking,
        "ranking_scope": {
            "kind": "multi_seed",
            "seeds": seeds,
            "number_of_seeds": len(seeds),
            "aggregation": ["min", "mean", "max"],
        },
        "artifacts": {
            "comparison_report": "comparison.pdf",
            "combined_losses": "combined_losses.pdf",
            "history": "history.csv",
        },
        "notes": [
            "Curves show the pointwise mean and min/max bounds across seeds.",
            "Selected-weight and gamma panels are repeated separately for each seed.",
            f"A shared {config.learning_rate:g} learning rate is used by every method.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nResults across seeds (mean; min to max)\n")
    print(f"{'method':<20} {'final test MSE':>35} {'test relative L2':>35}")
    print("-" * 94)
    for method, values in summary["methods"].items():
        test_text = _format_aggregate_metric(values["final_test_mse"])
        relative_text = _format_aggregate_metric(values["final_test_relative_l2"])
        print(f"{method:<20} {test_text:>35} {relative_text:>35}")
    print("\nRanking by mean final test MSE: " + " < ".join(summary["ranking_by_mean_final_test_mse"]))
    print("\nTraining time across seeds (mean ± sigma)\n")
    print(f"{'method':<20} {'training time':>20}")
    print("-" * 42)
    for method, values in summary["methods"].items():
        print(f"{method:<20} {_format_mean_std_metric(values['training_seconds']):>20}")


def main() -> int:
    if MAX_WORKERS < 1:
        raise ValueError("MAX_WORKERS must be at least 1")
    base_config = case_config.ExperimentConfig(
        seeds=SEEDS,
        steps=STEPS,
        train_size=TRAIN_SIZE,
        minibatch_size=MINIBATCH_SIZE,
        hidden_width=WIDTH,
        hidden_layers=DEPTH,
        learning_rate=LEARNING_RATE,
        seed=SEEDS[0],
        precision=PRECISION,
        test_every=TEST_EVERY,
    )
    base_config.validate()
    _dtype_for_precision(base_config.precision)
    print("JAX devices:", ", ".join(str(device) for device in jax.devices()), flush=True)
    print(f"Target function: {CASE_NAME}", flush=True)
    print(f"Network architecture: {base_config.architecture}", flush=True)
    output_root = OUTPUT_DIR.resolve()
    for architecture_index, (width, depth) in enumerate(base_config.architecture_settings):
        learning_rate = base_config.learning_rate
        if architecture_index > 0:
            learning_rate *= base_config.small_network_learning_rate_multiplier
        config = replace(base_config, hidden_width=width, hidden_layers=depth, learning_rate=learning_rate)
        output_dir = output_root / f"width_{width}_depth_{depth}"
        print(
            f"Training {CASE_NAME.replace('_', '-')} architecture {config.architecture} "
            f"with {model.ACTIVATION_NAME} hidden activations for {config.steps} steps "
            f"at shared LR={config.learning_rate:g} ({config.precision}-bit complex) "
            f"for seeds {', '.join(str(seed) for seed in SEEDS)}.",
            flush=True,
        )
        results = run_comparison_seeds(config, methods=METHODS, max_workers=MAX_WORKERS, show_progress=not QUIET)
        if not QUIET:
            print("\nWriting CSV, JSON, and PDF artifacts...", flush=True)
        summary = write_results(config, results, output_dir)
        _print_summary(summary)
        print(f"\nArtifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

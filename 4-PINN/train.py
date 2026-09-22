"""Train the PIHNN for test case 4.1.2 of Calafa et al., CMAME 432 (2024) 117406."""


import csv
import json
import os
import pickle
import sys
import time
import uuid
from pathlib import Path

# Before importing JAX: backend selection happens at import. Empty = auto (GPU if available, else CPU);
# set PINN_JAX_PLATFORMS=cuda to fail instead of silently falling back to CPU.
os.environ["JAX_PLATFORMS"] = os.environ.get("PINN_JAX_PLATFORMS", "")

import jax
import jax.numpy as jnp
import numpy as np
import optax

import dataset
import model
import plots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import comparison_report
from src import hyperparameters_io
from src import metrics
from src import optimizer_config
from src import optimizers
from src import weight_dynamics

import lr_schedule  # noqa: E402 -- needs `src` on sys.path (inserted above)

METHOD_COLORS = optimizer_config.METHOD_COLORS

LEARNING_RATE = 1e-3

N_EPOCHS = 6000
N_TRAIN = 200
N_TEST = 20

# Separate seed streams: changing boundaries or selections must not alter the weights.
WEIGHT_INITIALIZATION_SEEDS = (0, 1, 2, 3, 4)
BOUNDARY_SAMPLING_SEEDS = (10, 10, 10, 10, 10)
TRACKING_SELECTION_SEEDS = (20, 20, 20, 20, 20)

if len({
    len(WEIGHT_INITIALIZATION_SEEDS),
    len(BOUNDARY_SAMPLING_SEEDS),
    len(TRACKING_SELECTION_SEEDS),
}) != 1:
    raise ValueError("all seed tuples must have the same length")

TRACKED_LEAVES = 6
TRACKED_PER_LEAF = 4

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Keyed on hyperparameters, not code: clear results/cache/ after editing an optimizer.
CACHE_ADAM_RESULTS = True
CACHED_METHODS = optimizer_config.METHODS
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")


def loss_fn(params, z_neumann, n_neumann, t0_neumann, z_sym, n_sym):
    (phi_n, dphi_n, ddphi_n), (psi_n, dpsi_n) = model.net_values(params, z_neumann, order_phi=2, order_psi=1)
    sxx, syy, sxy, _, _ = model.stresses_displacements(z_neumann, phi_n, dphi_n, ddphi_n, psi_n, dpsi_n)
    tx = sxx * n_neumann[:, 0] + sxy * n_neumann[:, 1]
    ty = sxy * n_neumann[:, 0] + syy * n_neumann[:, 1]
    L_n = jnp.mean((tx - t0_neumann[:, 0]) ** 2 + (ty - t0_neumann[:, 1]) ** 2)

    (phi_s, dphi_s, ddphi_s), (psi_s, dpsi_s) = model.net_values(params, z_sym, order_phi=2, order_psi=1)
    sxx_s, _, sxy_s, ux_s, uy_s = model.stresses_displacements(z_sym, phi_s, dphi_s, ddphi_s, psi_s, dpsi_s)
    u_n = ux_s * n_sym[:, 0] + uy_s * n_sym[:, 1]
    L_s = jnp.mean(sxy_s ** 2 + u_n ** 2)

    return dataset.ALPHA_N * L_n + dataset.ALPHA_S * L_s


evaluate_loss = jax.jit(loss_fn)


def _make_train_step(method, optimizer):
    def train_step(params, opt_state, batch):
        z_neumann, n_neumann, t0_neumann, z_sym, n_sym = batch
        loss, grads = jax.value_and_grad(loss_fn)(params, z_neumann, n_neumann, t0_neumann, z_sym, n_sym)
        # JAX's complex gradient is the conjugate of the steepest-descent direction.
        grads = jax.tree_util.tree_map(jnp.conj, grads)
        if method == optimizer_config.METHOD_LBFGS:
            def batch_loss(candidate_params):
                return loss_fn(candidate_params, z_neumann, n_neumann, t0_neumann, z_sym, n_sym)

            updates, opt_state = optimizer.update(
                grads, opt_state, params, value=loss, grad=grads, value_fn=batch_loss
            )
        else:
            updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return jax.jit(train_step)


def _init_params(weight_seed, boundary_seed):
    rng = np.random.default_rng(boundary_seed)
    key = jax.random.PRNGKey(weight_seed)
    x0_batch = dataset._sample_boundary(10 * N_TRAIN, rng)
    x0 = jnp.concatenate([x0_batch[0], x0_batch[3]])
    return model.init_network(key, model.BETA, x0)


def run_one(method, weight_seed, boundary_seed, selections):
    """Train the plate-with-hole PIHNN with one optimizer/seed combination."""

    rng = np.random.default_rng(boundary_seed)

    train_batch = dataset._sample_boundary(N_TRAIN, rng)
    test_batch = dataset._sample_boundary(N_TEST, rng)

    key = jax.random.PRNGKey(weight_seed)
    x0_batch = dataset._sample_boundary(10 * N_TRAIN, rng)
    x0 = jnp.concatenate([x0_batch[0], x0_batch[3]])

    params = model.init_network(key, model.BETA, x0)
    if method == optimizer_config.METHOD_ADAM_VARIABLE_LR:
        optimizer = lr_schedule.build_adam_variable_lr(LEARNING_RATE)
    else:
        optimizer = optimizers.build_optimizer(method, LEARNING_RATE, total_steps=N_EPOCHS)
    opt_state = optimizer.init(params)
    train_step = _make_train_step(method, optimizer)

    train_losses = np.zeros(N_EPOCHS)
    test_losses = np.zeros(N_EPOCHS)
    # Only every 200 epochs: each snapshot is a blocking device-to-host sync.
    diagnostics_history = np.full((N_EPOCHS, 9), np.nan)  # width must match optimizer_diagnostics()'s return
    weight_history = np.full((N_EPOCHS, len(selections)), np.nan, dtype=np.complex128)
    selected_gamma_history = np.full((N_EPOCHS, len(selections)), np.nan)
    # Retired signal, kept all-NaN so cached runs and CSVs keep their layout.
    selected_q_history = np.full((N_EPOCHS, len(selections)), np.nan)
    selected_chi_history = np.full((N_EPOCHS, len(selections)), np.nan)
    selected_psi_history = np.full((N_EPOCHS, len(selections)), np.nan)
    best_params, best_test_loss = params, np.inf
    # Compile before timing; warm-up results are discarded, so no state advances.
    compile_start = time.perf_counter()
    jax.block_until_ready(train_step(params, opt_state, train_batch))
    jax.block_until_ready(evaluate_loss(params, *test_batch))
    compilation_seconds = time.perf_counter() - compile_start
    diagnostics_seconds = 0.0
    start = time.perf_counter()
    for epoch in range(N_EPOCHS):
        params, opt_state, loss = train_step(params, opt_state, train_batch)
        train_losses[epoch] = loss
        test_loss = float(evaluate_loss(params, *test_batch))
        test_losses[epoch] = test_loss
        is_snapshot_epoch = epoch % 200 == 0 or epoch == N_EPOCHS - 1
        if is_snapshot_epoch:
            diagnostics_start = time.perf_counter()
            diagnostics_history[epoch] = np.asarray(
                optimizers.optimizer_diagnostics(method, opt_state)
            )
            weight_history[epoch] = np.asarray(weight_dynamics.selected_weight_values(params, selections))
            selected_gamma_history[epoch] = np.asarray(
                weight_dynamics.selected_gamma_values(method, opt_state, selections)
            )
            selected_chi_history[epoch] = np.asarray(
                weight_dynamics.selected_chi_values(method, opt_state, selections)
            )
            selected_psi_history[epoch] = np.asarray(
                weight_dynamics.selected_psi_values(method, opt_state, selections)
            )
            diagnostics_seconds += time.perf_counter() - diagnostics_start
        if test_loss < best_test_loss:
            best_test_loss, best_params = test_loss, params
        if is_snapshot_epoch:
            print(
                f"[{method} weight_seed={weight_seed}] epoch {epoch:5d}  "
                f"train_loss {train_losses[epoch]:.3e}  test_loss {test_losses[epoch]:.3e}"
            )
    training_seconds = time.perf_counter() - start - diagnostics_seconds

    print(f"[{method} weight_seed={weight_seed}] best test_loss {best_test_loss:.3e}")
    return {
        "method": method,
        "seed": weight_seed,
        "params": best_params,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "diagnostics": diagnostics_history,
        "weight_history": weight_history,
        "selected_gamma_history": selected_gamma_history,
        "selected_q_history": selected_q_history,
        "selected_chi_history": selected_chi_history,
        "selected_psi_history": selected_psi_history,
        "learning_area": metrics.log_learning_area(train_losses),
        "final_train_mse": float(train_losses[-1]),
        "final_test_mse": float(test_losses[-1]),
        "best_test_mse": float(best_test_loss),
        "training_seconds": training_seconds,
        "compilation_seconds": compilation_seconds,
    }


def _cache_path(method, weight_seed, boundary_seed):
    key = (
        f"{method}_wseed{weight_seed}_bseed{boundary_seed}"
        f"_epochs{N_EPOCHS}_train{N_TRAIN}_test{N_TEST}_lr{LEARNING_RATE:g}"
    )
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def _load_cached_run(path, run_id):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as file:
        payload = pickle.load(file)
    if isinstance(payload, optimizer_config.CachedRun):
        run = payload.result
        if not optimizer_config.same_hyperparameters(
            payload.config, optimizer_config.hyperparameters_for_method(run["method"])
        ):
            return None
    else:
        # Legacy pickle from before CachedRun: accepted unvalidated.
        run = payload
    for key in ("selected_chi_history", "selected_psi_history"):
        if key not in run:
            run[key] = np.full_like(run["selected_gamma_history"], np.nan)
    # Always recomputed: older caches lack it or used an earlier definition.
    run["learning_area"] = metrics.log_learning_area(run["train_losses"])
    return run


def _save_cached_run(path, run, run_id):
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = optimizer_config.CachedRun(
        config=optimizer_config.hyperparameters_for_method(run["method"]), result=run, run_id=run_id
    )
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    seed_configurations = tuple(zip(
        WEIGHT_INITIALIZATION_SEEDS,
        BOUNDARY_SAMPLING_SEEDS,
        TRACKING_SELECTION_SEEDS,
    ))
    selections_by_seed = {
        weight_seed: weight_dynamics.make_weight_selections(
            _init_params(weight_seed, boundary_seed),
            seed=tracking_seed,
            max_leaves=TRACKED_LEAVES,
            per_leaf=TRACKED_PER_LEAF,
        )
        for weight_seed, boundary_seed, tracking_seed in seed_configurations
    }
    # One JAX process for all runs: spawned workers must not re-initialize CUDA.
    run_id = uuid.uuid4().hex
    runs = []
    for weight_seed, boundary_seed, _ in seed_configurations:
        for method in optimizer_config.METHODS:
            use_cache = CACHE_ADAM_RESULTS and method in CACHED_METHODS
            cache_path = _cache_path(method, weight_seed, boundary_seed) if use_cache else None
            run = _load_cached_run(cache_path, run_id) if use_cache else None
            if run is None:
                run = run_one(method, weight_seed, boundary_seed, selections_by_seed[weight_seed])
                if use_cache:
                    _save_cached_run(cache_path, run, run_id)
            else:
                print(f"[{method} weight_seed={weight_seed}] loaded cached run from {cache_path}")
            runs.append(run)

    hyperparameters_io.write_hyperparameters_txt(
        os.path.join(RESULTS_DIR, "hyperparameters.txt"),
        {
            "learning_rate": LEARNING_RATE,
            "steps": N_EPOCHS,
            "train_points": N_TRAIN,
            "test_points": N_TEST,
            "init_beta": model.BETA,
            "layer_sizes": tuple(model.LAYER_SIZES),
            "seeds": WEIGHT_INITIALIZATION_SEEDS,
        },
    )

    parameter_count = weight_dynamics.count_trainable_parameters(runs[0]["params"])
    with open(os.path.join(RESULTS_DIR, "parameter_count.txt"), "w", encoding="utf-8") as file:
        file.write(f"{parameter_count}\n")

    with open(
        os.path.join(RESULTS_DIR, "history.csv"), "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["method", "seed", "epoch", "train_mse", "test_mse"])
        for run in runs:
            for epoch, (train, test) in enumerate(
                zip(run["train_losses"], run["test_losses"])
            ):
                writer.writerow([run["method"], run["seed"], epoch, f"{train:.16e}", f"{test:.16e}"])

    summary = {}
    for method in optimizer_config.METHODS:
        method_runs = [run for run in runs if run["method"] == method]
        for key in ("learning_area", "final_train_mse", "final_test_mse", "best_test_mse", "training_seconds"):
            summary.setdefault(method, {})[key] = metrics.seed_statistics(run[key] for run in method_runs)

    ranking = sorted(optimizer_config.METHODS, key=lambda method: summary[method]["final_test_mse"]["mean"])
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "weight_initialization_seeds": list(WEIGHT_INITIALIZATION_SEEDS),
                "boundary_sampling_seeds": list(BOUNDARY_SAMPLING_SEEDS),
                "tracking_selection_seeds": list(TRACKING_SELECTION_SEEDS),
                "methods": summary,
                "ranking_by_mean_final_test_mse": ranking,
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    print("\nResults across seeds (mean; min to max)\n")
    print(f"{'method':<20} {'final test MSE':>35} {'train s':>20}")
    print("-" * 78)
    for method in optimizer_config.METHODS:
        test = summary[method]["final_test_mse"]
        train_s = summary[method]["training_seconds"]
        print(
            f"{method:<20} "
            f"{test['mean']:11.4e} ({test['min']:.4e} to {test['max']:.4e})  "
            f"{train_s['mean']:7.3f} ± {train_s['std']:.3f}"
        )
    print("\nRanking by mean final test MSE: " + " < ".join(ranking))

    best_method = ranking[0]
    best_run = min(
        (run for run in runs if run["method"] == best_method),
        key=lambda run: run["best_test_mse"],
    )
    np.savez(
        os.path.join(RESULTS_DIR, "losses.npz"),
        train=best_run["train_losses"],
        test=best_run["test_losses"],
    )
    save_plots(best_run["params"], best_run["train_losses"], best_run["test_losses"])
    plots.write_raw_histories(runs, RESULTS_DIR)
    plots.save_loss_comparison_pdf(runs, Path(RESULTS_DIR) / "combined_losses.pdf")

    seed_runs = [
        (
            seed,
            [
                comparison_report.MethodRunResult(
                    method=run["method"],
                    train_loss=run["train_losses"],
                    test_loss=run["test_losses"],
                    diagnostics=run["diagnostics"],
                    weight_history=run["weight_history"],
                    selected_gamma_history=run["selected_gamma_history"],
                    selected_chi_history=run["selected_chi_history"],
                    selected_psi_history=run["selected_psi_history"],
                )
                for run in runs
                if run["seed"] == seed
            ],
        )
        for seed in WEIGHT_INITIALIZATION_SEEDS
    ]
    plots.write_comparison_report_pdf(
        seed_runs,
        selections_by_seed,
        Path(RESULTS_DIR) / "comparison.pdf",
    )


@comparison_report.never_fatal
def save_plots(params, train_losses, test_losses):
    plt = comparison_report._pyplot()

    fig, ax = plt.subplots()
    ax.semilogy(train_losses, "r-", label="Training", linewidth=0.8)
    ax.semilogy(test_losses, "b--", label="Test", linewidth=0.8)
    ax.set_xlabel("Epoch number")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"), dpi=150)
    plt.close(fig)

    n_grid = 300
    xs = np.linspace(-dataset.HALF_L, 0.0, n_grid)
    ys = np.linspace(0.0, dataset.HALF_L, n_grid)
    X, Y = np.meshgrid(xs, ys)
    mask = X ** 2 + Y ** 2 >= dataset.HOLE_R ** 2
    Z = (X + 1j * Y)[mask]
    z_eval = jnp.asarray(Z, dtype=jnp.complex64)

    (phi, dphi, ddphi), (psi, dpsi) = model.net_values(params, z_eval, order_phi=2, order_psi=1)
    sxx, syy, sxy, ux, uy = model.stresses_displacements(z_eval, phi, dphi, ddphi, psi, dpsi)

    fields = {"sigma_xx": sxx, "sigma_yy": syy, "sigma_xy": sxy, "u_x": ux, "u_y": uy}
    field_labels = {
        "sigma_xx": r"$\sigma_{xx}$",
        "sigma_yy": r"$\sigma_{yy}$",
        "sigma_xy": r"$\sigma_{xy}$",
        "u_x": r"$u_x$",
        "u_y": r"$u_y$",
    }
    for name, values in fields.items():
        field = np.full(X.shape, np.nan)
        field[mask] = np.asarray(values)
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, field, levels=40, cmap="jet")
        fig.colorbar(cf, ax=ax)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"PIHNN {field_labels[name]}")
        ax.set_aspect("equal")
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

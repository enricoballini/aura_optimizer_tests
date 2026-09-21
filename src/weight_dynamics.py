"""Generic, pytree-agnostic weight/gamma-multiplier tracking."""


from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import optimizer_config

METHOD_HCSCGM = getattr(optimizer_config, "METHOD_HCSCGM", None)


@dataclass(frozen=True)
class WeightSelection:
    leaf_index: int
    leaf_path: str
    flat_index: int

    @property
    def label(self) -> str:
        return f"{self.leaf_path}[{self.flat_index}]"


def _is_trackable_leaf(leaf) -> bool:
    return hasattr(leaf, "ndim") and hasattr(leaf, "size") and leaf.ndim >= 1 and leaf.size >= 1


def _candidate_leaves(pytree):
    """Return the pytree's array leaves in a stable, deterministic order."""

    flat, _ = jax.tree_util.tree_flatten_with_path(pytree)
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in flat if _is_trackable_leaf(leaf)]


def make_weight_selections(
    params,
    *,
    seed: int,
    max_leaves: int = 6,
    per_leaf: int = 4,
) -> tuple[WeightSelection, ...]:
    """Deterministically sample scalar components spread across ``params``."""

    if max_leaves < 1 or per_leaf < 1:
        raise ValueError("max_leaves and per_leaf must be positive")
    candidates = _candidate_leaves(params)
    if not candidates:
        raise ValueError("params contains no array leaves to sample")

    if len(candidates) <= max_leaves:
        leaf_indices = list(range(len(candidates)))
    else:
        leaf_indices = sorted(
            set(np.linspace(0, len(candidates) - 1, max_leaves).round().astype(int))
        )

    rng = np.random.default_rng(seed)
    selections = []
    for leaf_index in leaf_indices:
        path, leaf = candidates[leaf_index]
        size = int(np.prod(leaf.shape))
        count = min(per_leaf, size)
        chosen = np.sort(rng.choice(size, size=count, replace=False))
        for flat_index in chosen:
            selections.append(
                WeightSelection(
                    leaf_index=leaf_index,
                    leaf_path=path,
                    flat_index=int(flat_index),
                )
            )
    return tuple(selections)


def selected_weight_values(pytree, selections: tuple[WeightSelection, ...]) -> jnp.ndarray:
    """Extract the scalar components named by ``selections`` from ``pytree``."""

    candidates = _candidate_leaves(pytree)
    return jnp.stack(
        [
            jnp.ravel(candidates[item.leaf_index][1])[item.flat_index]
            for item in selections
        ]
    )


def selected_gamma_values(method: str, optimizer_state, selections) -> jnp.ndarray:
    """Return each selection's matching per-parameter multiplier ("gamma")."""

    if method == optimizer_config.METHOD_RPROP:
        state = optimizer_state[0]
        real_steps = selected_weight_values(state.step_size_real, selections)
        imag_steps = selected_weight_values(state.step_size_imag, selections)
        return 0.5 * (real_steps + imag_steps)
    if method in (
        optimizer_config.METHOD_LBFGS,
        METHOD_HCSCGM,
        optimizer_config.METHOD_CvAMSGrad,
        optimizer_config.METHOD_MUON,
        getattr(optimizer_config, "METHOD_SGD", None),
    ):
        return jnp.ones(len(selections))
    if method in (
        optimizer_config.METHOD_ADAM,
        optimizer_config.METHOD_ADAM_VARIABLE_LR,
        optimizer_config.METHOD_NADAMW,
        optimizer_config.METHOD_NADAM,
    ):
        reference = selected_weight_values(optimizer_state[0].mu, selections)
        return jnp.ones_like(jnp.real(reference))
    if method in (optimizer_config.METHOD_ECLIPSE, optimizer_config.METHOD_PULSAR):
        # These keep their own Adam-like state, so the modulation is chain index 0.
        reference = selected_weight_values(optimizer_state[0].multiplier, selections)
        return jnp.real(reference)
    if method in optimizer_config.AURA_SIGN_FAMILY or method in optimizer_config.AURA_SNR_FAMILY:
        reference = selected_weight_values(optimizer_state[0].log_multiplier, selections)
        return jnp.exp(jnp.real(reference))
    if method in optimizer_config.AURA_S_FAMILY:
        reference = selected_weight_values(optimizer_state[0].log_multiplier, selections)
        return jnp.exp(jnp.real(reference))
    if method in (optimizer_config.METHOD_MUON_AURA, optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION):
        # Chain index 2: after the two masked Muon/Adam direction stages.
        reference = selected_weight_values(optimizer_state[2].multiplier, selections)
        return jnp.real(reference)
    reference = selected_weight_values(optimizer_state[1].multiplier, selections)
    return jnp.real(reference)


def selected_chi_values(method: str, optimizer_state, selections) -> jnp.ndarray:
    """Each selection's bias-corrected innovation correlation chi; NaN for methods
    without one."""

    if method in optimizer_config.AURA_SNR_FAMILY:
        return jnp.real(selected_weight_values(optimizer_state[0].consistency, selections))
    if method in optimizer_config.AURA_SIGN_FAMILY:
        return jnp.real(selected_weight_values(optimizer_state[0].alignment, selections))
    if method in optimizer_config.AURA_S_FAMILY:
        state = optimizer_state[0]
        reference = selected_weight_values(state.alignment_ema, selections)
        return jnp.real(reference) / jnp.maximum(state.bias_correction, 1e-12)
    if method == optimizer_config.METHOD_ECLIPSE:
        reference = selected_weight_values(optimizer_state[0].chi, selections)
        return jnp.real(reference)
    if method in (optimizer_config.METHOD_MUON_AURA, optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION):
        reference = selected_weight_values(optimizer_state[2].chi, selections)
        return jnp.real(reference)
    if method not in (
        optimizer_config.METHOD_ADAM_AURA,
        optimizer_config.METHOD_ADAM_AURA_COSINE_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_EB,
        optimizer_config.METHOD_ASTRA,
        optimizer_config.METHOD_AURA_LIGHT,
    ):
        return jnp.full(len(selections), jnp.nan)
    reference = selected_weight_values(optimizer_state[1].chi, selections)
    return jnp.real(reference)


def selected_psi_values(method: str, optimizer_state, selections) -> jnp.ndarray:
    """Each selection's bias-corrected signed-rotation signal Psi; NaN for methods
    without one."""

    if method in optimizer_config.AURA_SNR_FAMILY:
        return jnp.imag(selected_weight_values(optimizer_state[0].consistency, selections))
    if method in optimizer_config.AURA_S_FAMILY:
        state = optimizer_state[0]
        reference = selected_weight_values(state.alignment_ema, selections)
        return jnp.imag(reference) / jnp.maximum(state.bias_correction, 1e-12)
    if method == optimizer_config.METHOD_ECLIPSE:
        reference = selected_weight_values(optimizer_state[0].psi, selections)
        return jnp.real(reference)
    if method in (optimizer_config.METHOD_MUON_AURA, optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION):
        reference = selected_weight_values(optimizer_state[2].psi, selections)
        return jnp.real(reference)
    if method not in (
        optimizer_config.METHOD_ADAM_AURA,
        optimizer_config.METHOD_ADAM_AURA_COSINE_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_EB,
        optimizer_config.METHOD_ASTRA,
        optimizer_config.METHOD_AURA_LIGHT,
    ):
        return jnp.full(len(selections), jnp.nan)
    reference = selected_weight_values(optimizer_state[1].psi, selections)
    return jnp.real(reference)


def count_trainable_parameters(params) -> int:
    """Total scalar parameter count of a pytree; a complex leaf counts as one
    parameter per entry."""

    return int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(params)))


def snapshot_indices(num_steps: int, every: int) -> np.ndarray:
    if num_steps < 1 or every < 1:
        raise ValueError("num_steps and every must be positive")
    indices = list(range(0, num_steps, every))
    if indices[-1] != num_steps - 1:
        indices.append(num_steps - 1)
    return np.asarray(indices, dtype=int)

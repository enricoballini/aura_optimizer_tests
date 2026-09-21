"""Independent copies of the optimizer update rules being compared."""


import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map
import optax

from . import optimizer_config


class ComponentwiseRpropState(NamedTuple):
    previous_gradient_real: optax.Updates
    previous_gradient_imag: optax.Updates
    step_size_real: optax.Updates
    step_size_imag: optax.Updates


class _ComponentwiseRpropLeafUpdate(NamedTuple):
    direction: jax.Array
    previous_gradient_real: jax.Array
    previous_gradient_imag: jax.Array
    step_size_real: jax.Array
    step_size_imag: jax.Array


def scale_by_componentwise_rprop(
    learning_rate: float,
    *,
    eta_minus: float,
    eta_plus: float,
    min_step_size: float,
    max_step_size: float,
) -> optax.GradientTransformation:
    """Apply standard RPROP independently to real and imaginary components."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < eta_minus < 1.0:
        raise ValueError("eta_minus must be in (0, 1)")
    if eta_plus <= 1.0:
        raise ValueError("eta_plus must exceed 1")
    if min_step_size <= 0.0:
        raise ValueError("min_step_size must be positive")
    if max_step_size < min_step_size:
        raise ValueError("max_step_size must be at least min_step_size")

    def init_fn(params):
        zeros = tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params)
        steps = tree_map(
            lambda param: jnp.full_like(jnp.real(param), learning_rate), params
        )
        return ComponentwiseRpropState(zeros, zeros, steps, steps)

    def component_update(gradient, previous_gradient, step_size):
        sign_product = gradient * previous_gradient
        next_step_size = jnp.where(
            sign_product == 0.0,
            step_size,
            jnp.clip(
                step_size
                * jnp.where(sign_product > 0.0, eta_plus, eta_minus),
                min_step_size,
                max_step_size,
            ),
        )
        accepted_gradient = jnp.where(
            sign_product < 0.0, jnp.zeros_like(gradient), gradient
        )
        direction = next_step_size * jnp.sign(accepted_gradient)
        return direction, accepted_gradient, next_step_size

    def update_leaf(
        gradient,
        previous_real,
        previous_imag,
        step_real,
        step_imag,
    ):
        real_direction, next_previous_real, next_step_real = component_update(
            jnp.real(gradient), previous_real, step_real
        )
        imag_direction, next_previous_imag, next_step_imag = component_update(
            jnp.imag(gradient), previous_imag, step_imag
        )
        if jnp.issubdtype(gradient.dtype, jnp.complexfloating):
            direction = real_direction + 1j * imag_direction
        else:
            direction = real_direction.astype(gradient.dtype)
        return _ComponentwiseRpropLeafUpdate(
            direction,
            next_previous_real,
            next_previous_imag,
            next_step_real,
            next_step_imag,
        )

    def update_fn(updates, state, params=None):
        del params
        leaves = tree_map(
            update_leaf,
            updates,
            state.previous_gradient_real,
            state.previous_gradient_imag,
            state.step_size_real,
            state.step_size_imag,
        )
        def is_update_result(value):
            return isinstance(value, _ComponentwiseRpropLeafUpdate)
        directions = tree_map(
            lambda values: values[0], leaves, is_leaf=is_update_result
        )
        next_state = ComponentwiseRpropState(
            tree_map(lambda values: values[1], leaves, is_leaf=is_update_result),
            tree_map(lambda values: values[2], leaves, is_leaf=is_update_result),
            tree_map(lambda values: values[3], leaves, is_leaf=is_update_result),
            tree_map(lambda values: values[4], leaves, is_leaf=is_update_result),
        )
        return directions, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class CvAMSGradState(NamedTuple):
    """Split-complex AMSGrad accumulators (Mayer et al., 2025, Algorithm 5)."""

    m1: optax.Updates
    m2_real: optax.Updates
    m2_imag: optax.Updates
    m2_max_real: optax.Updates
    m2_max_imag: optax.Updates
    count: jax.Array


def scale_by_cvamsgrad(
    *,
    beta_1: float,
    beta_2: float,
    epsilon: float,
    bias_correction: bool = False,
) -> optax.GradientTransformation:
    """Split-complex AMSGrad direction (Mayer et al., 2025, Algorithm 5)."""

    if not 0.0 <= beta_1 < 1.0:
        raise ValueError("beta_1 must be in [0, 1)")
    if not 0.0 <= beta_2 < 1.0:
        raise ValueError("beta_2 must be in [0, 1)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    def init_fn(params):
        m1 = tree_map(jnp.zeros_like, params)
        real_zeros = tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params)
        return CvAMSGradState(
            m1=m1,
            m2_real=real_zeros,
            m2_imag=real_zeros,
            m2_max_real=real_zeros,
            m2_max_imag=real_zeros,
            count=jnp.zeros((), dtype=jnp.int32),
        )

    def update_fn(updates, state, params=None):
        del params
        count = state.count + 1
        if bias_correction:
            m1_correction = 1.0 - beta_1**count
            m2_correction = 1.0 - beta_2**count
        m1 = tree_map(
            lambda m, g: beta_1 * m + (1.0 - beta_1) * g, state.m1, updates
        )
        m2_real = tree_map(
            lambda v, g: beta_2 * v + (1.0 - beta_2) * jnp.real(g) ** 2,
            state.m2_real,
            updates,
        )
        m2_imag = tree_map(
            lambda v, g: beta_2 * v + (1.0 - beta_2) * jnp.imag(g) ** 2,
            state.m2_imag,
            updates,
        )
        m2_max_real = tree_map(jnp.maximum, state.m2_max_real, m2_real)
        m2_max_imag = tree_map(jnp.maximum, state.m2_max_imag, m2_imag)

        def direction(m, max_real, max_imag):
            m_real, m_imag = jnp.real(m), jnp.imag(m)
            if bias_correction:
                m_real = m_real / m1_correction
                m_imag = m_imag / m1_correction
                max_real = max_real / m2_correction
                max_imag = max_imag / m2_correction
            real_part = m_real / (jnp.sqrt(max_real) + epsilon)
            imag_part = m_imag / (jnp.sqrt(max_imag) + epsilon)
            if jnp.issubdtype(m.dtype, jnp.complexfloating):
                return real_part + 1j * imag_part
            return real_part.astype(m.dtype)

        directions = tree_map(direction, m1, m2_max_real, m2_max_imag)
        return directions, CvAMSGradState(
            m1=m1,
            m2_real=m2_real,
            m2_imag=m2_imag,
            m2_max_real=m2_max_real,
            m2_max_imag=m2_max_imag,
            count=count,
        )

    return optax.GradientTransformation(init_fn, update_fn)


class AuraState(NamedTuple):
    count: jax.Array
    previous_direction: optax.Updates
    zeta_ema: optax.Updates
    chi: optax.Updates
    psi: optax.Updates
    multiplier: optax.Updates


class _AuraInnovation(NamedTuple):
    zeta_ema: jax.Array
    chi: jax.Array
    psi: jax.Array


def scale_by_aura(
    *,
    beta_zeta: float,
    epsilon_e: float,
    chi_alignment: float | None = None,
    chi_opposition: float,
    psi_alignment: float | None = None,
    psi_opposition: float,
    eta_minus: float | None = None,
    eta_plus: float | None = None,
    gamma_min: float,
    gamma_max: float,
    gamma_init: float = 1.0,
    measure: str = "dice",
    kappa_plus: float | None = None,
    kappa_minus: float | None = None,
    leak: float | None = None,
) -> optax.GradientTransformation:
    """AURA: innovation-based modulation of the ADAM direction; chain it immediately
    after ``optax.scale_by_adam``."""

    if not 0.0 <= beta_zeta < 1.0:
        raise ValueError("beta_zeta must be in [0, 1)")
    if epsilon_e <= 0.0:
        raise ValueError("epsilon_e must be positive")
    if measure not in ("dice", "cosine"):
        raise ValueError('measure must be "dice" or "cosine"')
    aura_rule = (chi_alignment, psi_alignment, eta_minus, eta_plus)
    leaky_rule = (kappa_plus, kappa_minus, leak)
    if all(value is not None for value in aura_rule) and kappa_plus is None and kappa_minus is None:
        if not -1.0 <= chi_opposition < chi_alignment <= 1.0:
            raise ValueError("chi_opposition and chi_alignment must satisfy -1 <= chi_opposition < chi_alignment <= 1")
        if not 0.0 <= psi_alignment < psi_opposition <= 1.0:
            raise ValueError("psi_alignment and psi_opposition must satisfy 0 <= psi_alignment < psi_opposition <= 1")
        if not 0.0 < eta_minus < 1.0:
            raise ValueError("eta_minus must be in (0, 1)")
        if eta_plus <= 1.0:
            raise ValueError("eta_plus must exceed 1")
        if leak is not None and not 0.0 < leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")
    elif all(value is None for value in aura_rule) and all(value is not None for value in leaky_rule):
        if not -1.0 <= chi_opposition <= 1.0:
            raise ValueError("chi_opposition must be in [-1, 1]")
        if not 0.0 < psi_opposition <= 1.0:
            raise ValueError("psi_opposition must be in (0, 1]")
        if kappa_plus < 0.0 or kappa_minus < 0.0:
            raise ValueError("kappa_plus and kappa_minus must be non-negative")
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")
    else:
        raise ValueError(
            "give chi_alignment, psi_alignment, eta_minus and eta_plus (AURA's rule) "
            "or kappa_plus, kappa_minus and leak (leaky log-multiplier), not a mix"
        )
    if not 0.0 < gamma_min <= 1.0:
        raise ValueError("gamma_min must be in (0, 1]")
    if gamma_max < 1.0:
        raise ValueError("gamma_max must be at least 1")
    if not gamma_min <= gamma_init <= gamma_max:
        raise ValueError("gamma_init must be in [gamma_min, gamma_max]")

    log_gamma_min = math.log(gamma_min)
    log_gamma_max = math.log(gamma_max)

    def init_fn(params):
        zeros = tree_map(jnp.zeros_like, params)
        multiplier = tree_map(
            lambda param: jnp.full_like(jnp.real(param), gamma_init), params
        )
        zeros_real = tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params)
        return AuraState(
            count=jnp.zeros((), dtype=jnp.int32),
            previous_direction=zeros,
            zeta_ema=zeros,
            chi=zeros_real,
            psi=zeros_real,
            multiplier=multiplier,
        )

    def update_fn(updates, state, params=None):
        del params
        count = state.count + 1
        bias_correction = 1.0 - beta_zeta**count

        def innovation(direction, previous_direction, zeta_ema_previous):
            z = direction * jnp.conj(previous_direction)
            if measure == "dice":
                energy = jnp.abs(direction) ** 2 + jnp.abs(previous_direction) ** 2
                zeta = 2.0 * z / (energy + epsilon_e)
            else:
                zeta = z / (jnp.abs(direction) * jnp.abs(previous_direction) + epsilon_e)
            zeta_ema = beta_zeta * zeta_ema_previous + (1.0 - beta_zeta) * zeta
            zeta_hat = zeta_ema / bias_correction
            chi = jnp.real(zeta_hat)
            psi = jnp.imag(zeta_hat)
            return _AuraInnovation(zeta_ema, chi, psi)

        results = tree_map(
            innovation, updates, state.previous_direction, state.zeta_ema
        )

        def is_innovation_result(value):
            return isinstance(value, _AuraInnovation)

        zeta_ema = tree_map(
            lambda value: value.zeta_ema, results, is_leaf=is_innovation_result
        )
        chi = tree_map(
            lambda value: value.chi, results, is_leaf=is_innovation_result
        )
        psi = tree_map(
            lambda value: value.psi, results, is_leaf=is_innovation_result
        )

        def next_multiplier(gamma, chi_value, psi_value):
            shrink = jnp.logical_or(chi_value <= chi_opposition, jnp.abs(psi_value) >= psi_opposition)
            if kappa_plus is not None:
                drive = jnp.where(chi_value >= 0.0, kappa_plus * chi_value, kappa_minus * chi_value)
                drive = drive - jnp.where(shrink, kappa_plus, 0.0)
                return jnp.exp(jnp.clip(leak * jnp.log(gamma) + drive, log_gamma_min, log_gamma_max))
            grow = jnp.logical_and(
                jnp.logical_not(shrink),
                jnp.logical_and(chi_value >= chi_alignment, jnp.abs(psi_value) <= psi_alignment),
            )
            if leak is not None:
                step = jnp.where(shrink, math.log(eta_minus), jnp.where(grow, math.log(eta_plus), 0.0))
                return jnp.exp(jnp.clip(leak * jnp.log(gamma) + step, log_gamma_min, log_gamma_max))
            return jnp.where(
                shrink,
                jnp.maximum(eta_minus * gamma, gamma_min),
                jnp.where(grow, jnp.minimum(eta_plus * gamma, gamma_max), gamma),
            )

        multiplier = tree_map(next_multiplier, state.multiplier, chi, psi)
        modulated = tree_map(
            lambda direction, gamma: gamma * direction, updates, multiplier
        )
        return modulated, AuraState(
            count=count,
            previous_direction=updates,
            zeta_ema=zeta_ema,
            chi=chi,
            psi=psi,
            multiplier=multiplier,
        )

    return optax.GradientTransformation(init_fn, update_fn)


class AuraSState(NamedTuple):
    direction_state: optax.OptState
    count: jax.Array
    bias_correction: jax.Array
    previous_direction: optax.Updates
    alignment_ema: optax.Updates
    reversal_ema: optax.Updates
    log_multiplier: optax.Updates


class _AuraSLeaf(NamedTuple):
    update: jax.Array
    alignment_ema: jax.Array
    reversal_ema: jax.Array
    log_multiplier: jax.Array


def scale_by_aura_s(
    direction: optax.GradientTransformation,
    *,
    beta_zeta: float,
    epsilon_e: float,
    kappa_plus: float,
    kappa_minus: float,
    rotation_penalty: float,
    leak: float,
    chi_opposition: float,
    psi_opposition: float,
    kappa_brake: float,
    gamma_min: float,
    gamma_max: float,
    gamma_init: float = 1.0,
    noise_band: float | None = None,
) -> optax.GradientTransformation:
    """AURA-S: an SNR-aware innovation and a spring-loaded multiplier wrapped around
    any direction transform."""

    if not 0.0 <= beta_zeta < 1.0:
        raise ValueError("beta_zeta must be in [0, 1)")
    if epsilon_e <= 0.0:
        raise ValueError("epsilon_e must be positive")
    if kappa_plus < 0.0 or kappa_minus < 0.0 or kappa_brake < 0.0:
        raise ValueError("kappa_plus, kappa_minus and kappa_brake must be non-negative")
    if rotation_penalty < 0.0:
        raise ValueError("rotation_penalty must be non-negative")
    if not 0.0 < leak <= 1.0:
        raise ValueError("leak must be in (0, 1]")
    if not -1.0 <= chi_opposition <= 1.0:
        raise ValueError("chi_opposition must be in [-1, 1]")
    if not 0.0 <= psi_opposition <= 1.0:
        raise ValueError("psi_opposition must be in [0, 1]")
    if not 0.0 < gamma_min <= 1.0:
        raise ValueError("gamma_min must be in (0, 1]")
    if gamma_max < 1.0:
        raise ValueError("gamma_max must be at least 1")
    if not gamma_min <= gamma_init <= gamma_max:
        raise ValueError("gamma_init must be in [gamma_min, gamma_max]")
    if noise_band is not None and noise_band < 0.0:
        raise ValueError("noise_band must be non-negative")

    noise_sigma = math.sqrt((1.0 - beta_zeta) / (2.0 * (1.0 + beta_zeta)))
    log_gamma_min = math.log(gamma_min)
    log_gamma_max = math.log(gamma_max)

    def init_fn(params):
        zeros = tree_map(jnp.zeros_like, params)
        return AuraSState(
            direction_state=direction.init(params),
            count=jnp.zeros((), dtype=jnp.int32),
            bias_correction=jnp.zeros((), dtype=jnp.float32),
            previous_direction=zeros,
            alignment_ema=zeros,
            reversal_ema=zeros,
            log_multiplier=tree_map(
                lambda param: jnp.full_like(jnp.real(param), math.log(gamma_init)), params
            ),
        )

    def update_fn(updates, state, params=None):
        directions, direction_state = direction.update(updates, state.direction_state, params)
        count = state.count + 1
        bias_correction = 1.0 - beta_zeta**count

        def leaf(gradient, current, previous, alignment_ema, reversal_ema, log_multiplier):
            alignment = gradient * jnp.conj(previous) / (
                jnp.abs(gradient) * jnp.abs(previous) + epsilon_e
            )
            reversal = 2.0 * current * jnp.conj(previous) / (
                jnp.abs(current) ** 2 + jnp.abs(previous) ** 2 + epsilon_e
            )
            alignment_ema = beta_zeta * alignment_ema + (1.0 - beta_zeta) * alignment
            reversal_ema = beta_zeta * reversal_ema + (1.0 - beta_zeta) * reversal
            chi = jnp.real(alignment_ema) / bias_correction
            psi = jnp.imag(alignment_ema) / bias_correction
            brake = jnp.logical_or(
                jnp.real(reversal_ema) / bias_correction <= chi_opposition,
                jnp.abs(jnp.imag(reversal_ema)) / bias_correction >= psi_opposition,
            )
            signal = chi - rotation_penalty * psi**2
            if noise_band is None:
                drive = jnp.where(signal >= 0.0, kappa_plus * signal, kappa_minus * signal)
            else:
                drive = kappa_minus * signal + (kappa_plus - kappa_minus) * jnp.maximum(
                    signal - noise_band * noise_sigma, 0.0
                )
            drive = drive - jnp.where(brake, kappa_brake, 0.0)
            log_multiplier = jnp.clip(
                leak * log_multiplier + drive, log_gamma_min, log_gamma_max
            )
            return _AuraSLeaf(
                jnp.exp(log_multiplier) * current, alignment_ema, reversal_ema, log_multiplier
            )

        results = tree_map(
            leaf,
            updates,
            directions,
            state.previous_direction,
            state.alignment_ema,
            state.reversal_ema,
            state.log_multiplier,
        )

        def pick(field):
            return tree_map(
                lambda value: getattr(value, field),
                results,
                is_leaf=lambda value: isinstance(value, _AuraSLeaf),
            )

        return pick("update"), AuraSState(
            direction_state=direction_state,
            count=count,
            bias_correction=jnp.asarray(bias_correction, dtype=jnp.float32),
            previous_direction=directions,
            alignment_ema=pick("alignment_ema"),
            reversal_ema=pick("reversal_ema"),
            log_multiplier=pick("log_multiplier"),
        )

    return optax.GradientTransformation(init_fn, update_fn)


class AuraSignState(NamedTuple):
    direction_state: optax.OptState
    previous_direction: optax.Updates | None
    alignment: optax.Updates
    log_multiplier: optax.Updates


class _AuraSignLeaf(NamedTuple):
    update: jax.Array
    alignment: jax.Array
    log_multiplier: jax.Array


def scale_by_aura_sign(
    direction: optax.GradientTransformation,
    *,
    beta: float,
    kappa_plus: float,
    kappa_minus: float,
    brake_threshold: float,
    leak: float,
    gamma_min: float,
    gamma_max: float,
    momentum_of=None,
) -> optax.GradientTransformation:
    """AURA-sign: the simplified AURA-S gate around a direction transform."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if kappa_plus < 0.0 or kappa_minus < 0.0:
        raise ValueError("kappa_plus and kappa_minus must be non-negative")
    if not 0.0 < brake_threshold <= 1.0:
        raise ValueError("brake_threshold must be in (0, 1]")
    if not 0.0 < leak <= 1.0:
        raise ValueError("leak must be in (0, 1]")
    if not 0.0 < gamma_min <= 1.0 <= gamma_max:
        raise ValueError("need 0 < gamma_min <= 1 <= gamma_max")

    log_gamma_min = math.log(gamma_min)
    log_gamma_max = math.log(gamma_max)

    def init_fn(params):
        real_zeros = tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params)
        return AuraSignState(
            direction_state=direction.init(params),
            previous_direction=None if momentum_of is not None else tree_map(jnp.zeros_like, params),
            alignment=real_zeros,
            log_multiplier=real_zeros,
        )

    def update_fn(updates, state, params=None):
        if momentum_of is not None:
            reference = momentum_of(state.direction_state)
        else:
            reference = state.previous_direction
        directions, direction_state = direction.update(updates, state.direction_state, params)

        def leaf(gradient, current, ref, alignment, log_multiplier):
            agreement = jnp.real(gradient) * jnp.real(ref) + jnp.imag(gradient) * jnp.imag(ref)
            alignment = beta * alignment + (1.0 - beta) * jnp.sign(agreement)
            drive = jnp.where(alignment >= 0.0, kappa_plus * alignment, kappa_minus * alignment)
            drive = drive - jnp.where(alignment <= -brake_threshold, kappa_plus, 0.0)
            log_multiplier = jnp.clip(leak * log_multiplier + drive, log_gamma_min, log_gamma_max)
            return _AuraSignLeaf(jnp.exp(log_multiplier) * current, alignment, log_multiplier)

        results = tree_map(
            leaf, updates, directions, reference, state.alignment, state.log_multiplier
        )

        def pick(field):
            return tree_map(
                lambda value: getattr(value, field),
                results,
                is_leaf=lambda value: isinstance(value, _AuraSignLeaf),
            )

        return pick("update"), AuraSignState(
            direction_state=direction_state,
            previous_direction=None if momentum_of is not None else directions,
            alignment=pick("alignment"),
            log_multiplier=pick("log_multiplier"),
        )

    return optax.GradientTransformation(init_fn, update_fn)


class AuraSnrState(NamedTuple):
    direction_state: optax.OptState
    previous_direction: optax.Updates | None
    consistency: optax.Updates
    log_multiplier: optax.Updates


class _AuraSnrLeaf(NamedTuple):
    update: jax.Array
    consistency: jax.Array
    log_multiplier: jax.Array


def scale_by_aura_snr(
    direction: optax.GradientTransformation,
    *,
    beta_zeta: float,
    epsilon_e: float,
    opposition_threshold: float,
    gamma_min: float,
    gamma_max: float,
    kappa_plus: float | None = None,
    kappa_minus: float | None = None,
    leak: float | None = None,
    eta_plus: float | None = None,
    eta_minus: float | None = None,
    momentum_of=None,
) -> optax.GradientTransformation:
    """AURA-SNR: AURA's consistency gate with a zero noise floor, wrapped around a
    direction transform."""

    if not 0.0 <= beta_zeta < 1.0:
        raise ValueError("beta_zeta must be in [0, 1)")
    if epsilon_e <= 0.0:
        raise ValueError("epsilon_e must be positive")
    leaky_rule = (kappa_plus, kappa_minus, leak)
    aura_rule = (eta_plus, eta_minus)
    if all(value is not None for value in leaky_rule) and all(value is None for value in aura_rule):
        if kappa_plus < 0.0 or kappa_minus < 0.0:
            raise ValueError("kappa_plus and kappa_minus must be non-negative")
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must be in (0, 1]")
    elif all(value is None for value in leaky_rule) and all(value is not None for value in aura_rule):
        if not 0.0 < eta_minus < 1.0:
            raise ValueError("eta_minus must be in (0, 1)")
        if eta_plus <= 1.0:
            raise ValueError("eta_plus must exceed 1")
    else:
        raise ValueError(
            "give kappa_plus, kappa_minus and leak (leaky log-multiplier) "
            "or eta_plus and eta_minus (AURA's rule), not a mix"
        )
    if not 0.0 < opposition_threshold <= 1.0:
        raise ValueError("opposition_threshold must be in (0, 1]")
    if not 0.0 < gamma_min <= 1.0 <= gamma_max:
        raise ValueError("need 0 < gamma_min <= 1 <= gamma_max")

    log_gamma_min = math.log(gamma_min)
    log_gamma_max = math.log(gamma_max)

    if eta_plus is None:

        def next_log_multiplier(log_multiplier, chi, opposed):
            drive = jnp.where(chi >= 0.0, kappa_plus * chi, kappa_minus * chi)
            drive = drive - jnp.where(opposed, kappa_plus, 0.0)
            return leak * log_multiplier + drive

    else:
        log_eta_plus = math.log(eta_plus)
        log_eta_minus = math.log(eta_minus)

        def next_log_multiplier(log_multiplier, chi, opposed):
            grow = jnp.logical_and(chi >= 0.0, jnp.logical_not(opposed))
            return log_multiplier + jnp.where(opposed, log_eta_minus, jnp.where(grow, log_eta_plus, 0.0))

    def init_fn(params):
        return AuraSnrState(
            direction_state=direction.init(params),
            previous_direction=None if momentum_of is not None else tree_map(jnp.zeros_like, params),
            consistency=tree_map(jnp.zeros_like, params),
            log_multiplier=tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params),
        )

    def update_fn(updates, state, params=None):
        if momentum_of is not None:
            reference = momentum_of(state.direction_state)
        else:
            reference = state.previous_direction
        directions, direction_state = direction.update(updates, state.direction_state, params)

        def leaf(gradient, current, ref, consistency, log_multiplier):
            # Moduli taken separately: |g|^2 |r|^2 underflows long before |g| |r| does.
            phasor = gradient * jnp.conj(ref) / (jnp.abs(gradient) * jnp.abs(ref) + epsilon_e)
            consistency = beta_zeta * consistency + (1.0 - beta_zeta) * phasor
            chi = jnp.real(consistency)
            psi = jnp.imag(consistency)
            opposed = jnp.logical_or(chi <= -opposition_threshold, jnp.abs(psi) >= opposition_threshold)
            log_multiplier = jnp.clip(
                next_log_multiplier(log_multiplier, chi, opposed), log_gamma_min, log_gamma_max
            )
            return _AuraSnrLeaf(jnp.exp(log_multiplier) * current, consistency, log_multiplier)

        results = tree_map(
            leaf, updates, directions, reference, state.consistency, state.log_multiplier
        )

        def pick(field):
            return tree_map(
                lambda value: getattr(value, field),
                results,
                is_leaf=lambda value: isinstance(value, _AuraSnrLeaf),
            )

        return pick("update"), AuraSnrState(
            direction_state=direction_state,
            previous_direction=None if momentum_of is not None else directions,
            consistency=pick("consistency"),
            log_multiplier=pick("log_multiplier"),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _muon_hybrid_momentum(direction_state):
    """First moments behind ``_muon_hybrid_direction``: Muon's momentum on the
    matrices, Adam's first moment on the biases."""

    is_masked = lambda value: isinstance(value, optax.MaskedNode)
    return tree_map(
        lambda matrix_moment, bias_moment: bias_moment if is_masked(matrix_moment) else matrix_moment,
        direction_state[0].inner_state[0].mu,
        direction_state[1].inner_state.mu,
        is_leaf=is_masked,
    )


def scale_by_astra(
    *,
    beta_1: float,
    beta_zeta: float,
    epsilon_e: float,
    chi_alignment: float,
    chi_opposition: float,
    psi_alignment: float,
    psi_opposition: float,
    eta_minus: float,
    eta_plus: float,
    gamma_min: float,
    gamma_max: float,
    leak_rate: float,
    leak_exponent: float = 1.0,
    gamma_init: float = 1.0,
) -> optax.GradientTransformation:
    """ASTRA: AURA with modified innovation and multiplier blocks; chain it
    immediately after ``optax.scale_by_adam``."""

    if not 0.0 <= beta_1 < 1.0:
        raise ValueError("beta_1 must be in [0, 1)")
    if not 0.0 <= beta_zeta < 1.0:
        raise ValueError("beta_zeta must be in [0, 1)")
    if epsilon_e <= 0.0:
        raise ValueError("epsilon_e must be positive")
    if not -1.0 <= chi_opposition < chi_alignment <= 1.0:
        raise ValueError("chi_opposition and chi_alignment must satisfy -1 <= chi_opposition < chi_alignment <= 1")
    if not 0.0 <= psi_alignment < psi_opposition <= 1.0:
        raise ValueError("psi_alignment and psi_opposition must satisfy 0 <= psi_alignment < psi_opposition <= 1")
    if not 0.0 < eta_minus < 1.0:
        raise ValueError("eta_minus must be in (0, 1)")
    if eta_plus <= 1.0:
        raise ValueError("eta_plus must exceed 1")
    if not 0.0 < gamma_min <= 1.0:
        raise ValueError("gamma_min must be in (0, 1]")
    if gamma_max < 1.0:
        raise ValueError("gamma_max must be at least 1")
    if not 0.0 < leak_rate <= 1.0:
        raise ValueError("leak_rate must be in (0, 1]")
    if not 0.0 < leak_exponent <= 1.0:
        raise ValueError("leak_exponent must be in (0, 1]")
    if leak_rate < 1.0 and leak_exponent < 1.0:
        raise ValueError("use either leak_rate or leak_exponent, not both")
    if not gamma_min <= gamma_init <= gamma_max:
        raise ValueError("gamma_init must be in [gamma_min, gamma_max]")

    def init_fn(params):
        zeros = tree_map(jnp.zeros_like, params)
        multiplier = tree_map(
            lambda param: jnp.full_like(jnp.real(param), gamma_init), params
        )
        zeros_real = tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params)
        return AuraState(
            count=jnp.zeros((), dtype=jnp.int32),
            previous_direction=zeros,
            zeta_ema=zeros,
            chi=zeros_real,
            psi=zeros_real,
            multiplier=multiplier,
        )

    def update_fn(updates, state, params=None):
        del params
        count = state.count + 1
        bias_correction = 1.0 - beta_zeta**count

        def innovation(direction, previous_direction, zeta_ema_previous):
            fresh = direction - beta_1 * previous_direction
            z = fresh * jnp.conj(previous_direction)
            # Cosine, not AURA's Dice denominator: |fresh| ~ (1-beta_1)|direction| would crush zeta.
            zeta = z / (jnp.abs(fresh) * jnp.abs(previous_direction) + epsilon_e)
            zeta_ema = beta_zeta * zeta_ema_previous + (1.0 - beta_zeta) * zeta
            zeta_hat = zeta_ema / bias_correction
            chi = jnp.real(zeta_hat)
            psi = jnp.imag(zeta_hat)
            return _AuraInnovation(zeta_ema, chi, psi)

        results = tree_map(
            innovation, updates, state.previous_direction, state.zeta_ema
        )

        def is_innovation_result(value):
            return isinstance(value, _AuraInnovation)

        zeta_ema = tree_map(
            lambda value: value.zeta_ema, results, is_leaf=is_innovation_result
        )
        chi = tree_map(
            lambda value: value.chi, results, is_leaf=is_innovation_result
        )
        psi = tree_map(
            lambda value: value.psi, results, is_leaf=is_innovation_result
        )

        def next_multiplier(gamma, chi_value, psi_value):
            shrink = jnp.logical_or(chi_value <= chi_opposition, jnp.abs(psi_value) >= psi_opposition)
            grow = jnp.logical_and(
                jnp.logical_not(shrink),
                jnp.logical_and(chi_value >= chi_alignment, jnp.abs(psi_value) <= psi_alignment),
            )
            # A constant leak_rate cannot balance the gate's drift; leak_exponent's decay can.
            if leak_exponent < 1.0:
                held = jnp.power(gamma, leak_exponent)
            else:
                held = jnp.where(
                    gamma > 1.0,
                    jnp.maximum(leak_rate * gamma, 1.0),
                    jnp.minimum(gamma / leak_rate, 1.0),
                )
            return jnp.where(
                shrink,
                jnp.maximum(eta_minus * gamma, gamma_min),
                jnp.where(grow, jnp.minimum(eta_plus * gamma, gamma_max), held),
            )

        multiplier = tree_map(next_multiplier, state.multiplier, chi, psi)
        modulated = tree_map(
            lambda direction, gamma: gamma * direction, updates, multiplier
        )
        return modulated, AuraState(
            count=count,
            previous_direction=updates,
            zeta_ema=zeta_ema,
            chi=chi,
            psi=psi,
            multiplier=multiplier,
        )

    return optax.GradientTransformation(init_fn, update_fn)


class EclipseState(NamedTuple):
    count: jax.Array
    m: optax.Updates
    v: optax.Updates
    previous_gradient: optax.Updates
    cross_ema: optax.Updates
    chi: optax.Updates
    psi: optax.Updates
    multiplier: optax.Updates


class _EclipseInnovation(NamedTuple):
    m: jax.Array
    v: jax.Array
    direction: jax.Array
    cross_ema: jax.Array
    chi: jax.Array
    psi: jax.Array


def scale_by_eclipse(
    *,
    beta_1: float,
    beta_2: float,
    epsilon_a: float,
    epsilon_e: float,
    chi_alignment: float,
    chi_opposition: float,
    psi_alignment: float,
    psi_opposition: float,
    eta_minus: float,
    eta_plus: float,
    gamma_min: float,
    gamma_max: float,
    leak_rate: float = 1.0,
    gamma_init: float = 1.0,
) -> optax.GradientTransformation:
    """ECLIPSE: innovation gate on consecutive raw gradients with its own Adam
    moments; use it as the first transform in the chain."""

    if not 0.0 <= beta_1 < 1.0:
        raise ValueError("beta_1 must be in [0, 1)")
    if not 0.0 <= beta_2 < 1.0:
        raise ValueError("beta_2 must be in [0, 1)")
    if epsilon_a <= 0.0:
        raise ValueError("epsilon_a must be positive")
    if epsilon_e <= 0.0:
        raise ValueError("epsilon_e must be positive")
    if not -1.0 <= chi_opposition < chi_alignment <= 1.0:
        raise ValueError("chi_opposition and chi_alignment must satisfy -1 <= chi_opposition < chi_alignment <= 1")
    if not 0.0 <= psi_alignment < psi_opposition <= 1.0:
        raise ValueError("psi_alignment and psi_opposition must satisfy 0 <= psi_alignment < psi_opposition <= 1")
    if not 0.0 < eta_minus < 1.0:
        raise ValueError("eta_minus must be in (0, 1)")
    if eta_plus <= 1.0:
        raise ValueError("eta_plus must exceed 1")
    if not 0.0 < gamma_min <= 1.0:
        raise ValueError("gamma_min must be in (0, 1]")
    if gamma_max < 1.0:
        raise ValueError("gamma_max must be at least 1")
    if not 0.0 < leak_rate <= 1.0:
        raise ValueError("leak_rate must be in (0, 1]")
    if not gamma_min <= gamma_init <= gamma_max:
        raise ValueError("gamma_init must be in [gamma_min, gamma_max]")

    def init_fn(params):
        zeros_complex = tree_map(jnp.zeros_like, params)
        zeros_real = tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params)
        multiplier = tree_map(
            lambda param: jnp.full_like(jnp.real(param), gamma_init), params
        )
        return EclipseState(
            count=jnp.zeros((), dtype=jnp.int32),
            m=zeros_complex,
            v=zeros_real,
            previous_gradient=zeros_complex,
            cross_ema=zeros_complex,
            chi=zeros_real,
            psi=zeros_real,
            multiplier=multiplier,
        )

    def update_fn(gradients, state, params=None):
        del params
        count = state.count + 1
        bias_1 = 1.0 - beta_1**count
        bias_2 = 1.0 - beta_2**count
        safeguard = epsilon_e * bias_2

        def innovation(gradient, m_previous, v_previous, previous_gradient, cross_ema_previous):
            m = beta_1 * m_previous + (1.0 - beta_1) * gradient
            v = beta_2 * v_previous + (1.0 - beta_2) * jnp.abs(gradient) ** 2
            direction = (m / bias_1) / (jnp.sqrt(v / bias_2) + epsilon_a)
            cross = beta_2 * cross_ema_previous + (1.0 - beta_2) * gradient * jnp.conj(previous_gradient)
            energy = 0.5 * (v + v_previous)
            zeta_hat = cross / (energy + safeguard)
            chi = jnp.real(zeta_hat)
            psi = jnp.imag(zeta_hat)
            return _EclipseInnovation(m, v, direction, cross, chi, psi)

        results = tree_map(
            innovation, gradients, state.m, state.v, state.previous_gradient, state.cross_ema
        )

        def is_innovation_result(value):
            return isinstance(value, _EclipseInnovation)

        m = tree_map(lambda value: value.m, results, is_leaf=is_innovation_result)
        v = tree_map(lambda value: value.v, results, is_leaf=is_innovation_result)
        direction = tree_map(lambda value: value.direction, results, is_leaf=is_innovation_result)
        cross_ema = tree_map(lambda value: value.cross_ema, results, is_leaf=is_innovation_result)
        chi = tree_map(lambda value: value.chi, results, is_leaf=is_innovation_result)
        psi = tree_map(lambda value: value.psi, results, is_leaf=is_innovation_result)

        def next_multiplier(gamma, chi_value, psi_value):
            shrink = jnp.logical_or(chi_value <= chi_opposition, jnp.abs(psi_value) >= psi_opposition)
            grow = jnp.logical_and(
                jnp.logical_not(shrink),
                jnp.logical_and(chi_value >= chi_alignment, jnp.abs(psi_value) <= psi_alignment),
            )
            held = jnp.where(
                gamma > 1.0,
                jnp.maximum(leak_rate * gamma, 1.0),
                jnp.minimum(gamma / leak_rate, 1.0),
            )
            return jnp.where(
                shrink,
                jnp.maximum(eta_minus * gamma, gamma_min),
                jnp.where(grow, jnp.minimum(eta_plus * gamma, gamma_max), held),
            )

        multiplier = tree_map(next_multiplier, state.multiplier, chi, psi)
        modulated = tree_map(
            lambda one_direction, gamma: gamma * one_direction, direction, multiplier
        )
        return modulated, EclipseState(
            count=count,
            m=m,
            v=v,
            previous_gradient=gradients,
            cross_ema=cross_ema,
            chi=chi,
            psi=psi,
            multiplier=multiplier,
        )

    return optax.GradientTransformation(init_fn, update_fn)


class PulsarState(NamedTuple):
    count: jax.Array
    m: optax.Updates  # raw, not bias-corrected
    v: optax.Updates  # bias-corrected
    multiplier: optax.Updates


class _PulsarLeafUpdate(NamedTuple):
    m: jax.Array
    v: jax.Array
    direction: jax.Array
    multiplier: jax.Array


def scale_by_pulsar(
    *,
    beta_1: float,
    beta_2: float,
    epsilon_a: float,
    epsilon_e: float,
    kappa: float,
) -> optax.GradientTransformation:
    """PULSAR: ADAM whose step is gated from scratch at every step; keeps its own
    Adam state, so use it as the first transform in the chain."""

    if not 0.0 <= beta_1 < 1.0:
        raise ValueError("beta_1 must be in [0, 1)")
    if not 0.0 <= beta_2 < 1.0:
        raise ValueError("beta_2 must be in [0, 1)")
    if epsilon_a <= 0.0:
        raise ValueError("epsilon_a must be positive")
    if epsilon_e <= 0.0:
        raise ValueError("epsilon_e must be positive")
    if not 0.0 <= kappa < 1.0:
        raise ValueError("kappa must be in [0, 1)")

    def init_fn(params):
        return PulsarState(
            count=jnp.zeros((), dtype=jnp.int32),
            m=tree_map(jnp.zeros_like, params),
            v=tree_map(lambda param: jnp.zeros_like(jnp.real(param)), params),
            multiplier=tree_map(lambda param: jnp.ones_like(jnp.real(param)), params),
        )

    def update_fn(gradients, state, params=None):
        del params
        count = state.count + 1
        delta = (1.0 - beta_2) / (1.0 - beta_2**count)
        first_moment_correction = 1.0 / (1.0 - beta_1**count)

        def leaf_update(gradient, m_previous, v_previous):
            energy = jnp.abs(gradient) ** 2
            normalizer = (
                energy
                + jnp.maximum(v_previous, jnp.abs(m_previous) ** 2)
                + epsilon_e
            )
            agreement = 2.0 * gradient * jnp.conj(m_previous)
            control = jnp.maximum(
                -normalizer,
                jnp.real(agreement) - jnp.abs(jnp.imag(agreement)),
            )
            m = beta_1 * m_previous + (1.0 - beta_1) * gradient
            v = v_previous + delta * (energy - v_previous)
            multiplier = (normalizer + kappa * control) / (normalizer - kappa * control)
            direction = (
                (first_moment_correction * m) / (jnp.sqrt(v) + epsilon_a) * multiplier
            )
            return _PulsarLeafUpdate(m, v, direction, multiplier)

        results = tree_map(leaf_update, gradients, state.m, state.v)

        def is_leaf_update(value):
            return isinstance(value, _PulsarLeafUpdate)

        m = tree_map(lambda value: value.m, results, is_leaf=is_leaf_update)
        v = tree_map(lambda value: value.v, results, is_leaf=is_leaf_update)
        direction = tree_map(lambda value: value.direction, results, is_leaf=is_leaf_update)
        multiplier = tree_map(lambda value: value.multiplier, results, is_leaf=is_leaf_update)
        return direction, PulsarState(
            count=count, m=m, v=v, multiplier=multiplier
        )

    return optax.GradientTransformation(init_fn, update_fn)


class HCSCGMState(NamedTuple):
    count: jax.Array
    params: optax.Params
    grad: optax.Updates
    direction: optax.Updates
    theta: jax.Array


def scale_by_hscgm(
    *,
    theta_max: float,
    delta_1: float,
) -> optax.GradientTransformation:
    """Hybrid complex spectral conjugate gradient direction (Zhang et al., 2024)."""

    if theta_max <= 1.0:
        raise ValueError("theta_max must exceed 1")
    if delta_1 < 0.25:
        raise ValueError("delta_1 must be at least 1/4")

    def init_fn(params):
        zeros = tree_map(jnp.zeros_like, params)
        return HCSCGMState(
            count=jnp.zeros((), dtype=jnp.int32),
            params=zeros,
            grad=zeros,
            direction=zeros,
            theta=jnp.asarray(1.0),
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_by_hscgm requires the current parameters")
        grads = updates

        def first_direction(_):
            direction = tree_map(lambda g: -g, grads)
            return direction, state.theta

        def later_direction(_):
            s = optax.tree.sub(params, state.params)
            y = optax.tree.sub(grads, state.grad)
            s_norm_sq = optax.tree.norm(s, squared=True)
            y_dot_s = optax.tree.real(optax.tree.vdot(y, s))
            raw_theta = jnp.where(
                y_dot_s != 0.0, s_norm_sq / y_dot_s, state.theta
            )
            in_bounds = jnp.logical_and(
                raw_theta >= 1.0 / theta_max, raw_theta <= theta_max
            )
            theta = jnp.where(in_bounds, raw_theta, state.theta)

            residual = optax.tree.sub(optax.tree.scale(theta, y), s)
            grad_dot_residual = optax.tree.real(
                optax.tree.vdot(grads, residual)
            )
            y_dot_previous = optax.tree.real(
                optax.tree.vdot(y, state.direction)
            )
            beta_sp = jnp.where(
                y_dot_previous != 0.0, grad_dot_residual / y_dot_previous, 0.0
            )
            residual_norm_sq = optax.tree.norm(residual, squared=True)
            mu = jnp.where(
                y_dot_previous != 0.0,
                delta_1 * residual_norm_sq / (theta * jnp.square(y_dot_previous)),
                0.0,
            )
            grad_dot_previous = optax.tree.real(
                optax.tree.vdot(grads, state.direction)
            )
            beta_hsp = beta_sp - mu * grad_dot_previous

            direction = optax.tree.add_scale(
                optax.tree.scale(-theta, grads), beta_hsp, state.direction
            )
            return direction, theta

        direction, theta = jax.lax.cond(
            state.count > 0, later_direction, first_direction, operand=None
        )
        next_state = HCSCGMState(
            count=state.count + 1,
            params=params,
            grad=grads,
            direction=direction,
            theta=theta,
        )
        return direction, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_weak_wolfe_linesearch(
    *,
    sigma_1: float,
    sigma_2: float,
    max_linesearch_steps: int,
    initial_step_size: float = 1.0,
) -> optax.GradientTransformationExtraArgs:
    """Bisection line search enforcing the weak complex Wolfe conditions, eqs.
    (11)-(12) of Zhang et al. (2024)."""

    if not 0.0 < sigma_1 < sigma_2 < 1.0:
        raise ValueError("sigma_1 and sigma_2 must satisfy 0 < sigma_1 < sigma_2 < 1")

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, *, value, grad, value_fn, **extra_args):
        del extra_args
        if params is None:
            raise ValueError("scale_by_weak_wolfe_linesearch requires params")
        direction = updates
        slope_0 = optax.tree.real(optax.tree.vdot(grad, direction))
        value_and_grad_fn = jax.value_and_grad(value_fn)

        def body(_, carry):
            alpha, alpha_low, alpha_high, accepted = carry
            candidate = optax.tree.add_scale(params, alpha, direction)
            candidate_value, candidate_grad = value_and_grad_fn(candidate)
            candidate_grad = tree_map(jnp.conj, candidate_grad)
            slope = optax.tree.real(optax.tree.vdot(candidate_grad, direction))

            armijo_ok = candidate_value - value <= sigma_1 * alpha * slope_0
            curvature_ok = slope >= sigma_2 * slope_0
            newly_accepted = armijo_ok & curvature_ok

            next_high = jnp.where(armijo_ok, alpha_high, alpha)
            next_low = jnp.where(armijo_ok & ~curvature_ok, alpha, alpha_low)
            bisected = 0.5 * (alpha_low + next_high)
            grown = jnp.where(
                jnp.isinf(alpha_high), 2.0 * alpha, 0.5 * (next_low + alpha_high)
            )
            candidate_alpha = jnp.where(armijo_ok, grown, bisected)
            next_alpha = jnp.where(
                accepted | newly_accepted, alpha, candidate_alpha
            )
            return (
                next_alpha,
                jnp.where(accepted, alpha_low, next_low),
                jnp.where(accepted, alpha_high, next_high),
                accepted | newly_accepted,
            )

        init_carry = (
            jnp.asarray(initial_step_size),
            jnp.asarray(0.0),
            jnp.asarray(jnp.inf),
            jnp.asarray(False),
        )
        alpha, _, _, _ = jax.lax.fori_loop(0, max_linesearch_steps, body, init_carry)

        scaled_updates = optax.tree.scale(alpha, direction)
        return scaled_updates, optax.EmptyState()

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


class AdamWarmupLBFGSState(NamedTuple):
    count: jax.Array
    adam_state: optax.OptState
    lbfgs_state: optax.OptState


def chain_adam_warmup_then_lbfgs(
    *,
    adam: optax.GradientTransformation,
    lbfgs: optax.GradientTransformationExtraArgs,
    warmup_steps: int,
) -> optax.GradientTransformationExtraArgs:
    """Run Adam for ``warmup_steps`` steps, then switch permanently to L-BFGS."""

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    def init_fn(params):
        return AdamWarmupLBFGSState(
            count=jnp.zeros((), dtype=jnp.int32),
            adam_state=adam.init(params),
            lbfgs_state=lbfgs.init(params),
        )

    def update_fn(
        updates, state, params=None, *, value=None, grad=None, value_fn=None, **extra_args
    ):
        del extra_args

        def warmup_branch(_):
            adam_updates, next_adam_state = adam.update(
                updates, state.adam_state, params
            )
            return adam_updates, next_adam_state, state.lbfgs_state

        def lbfgs_branch(_):
            lbfgs_updates, next_lbfgs_state = lbfgs.update(
                updates,
                state.lbfgs_state,
                params,
                value=value,
                grad=grad,
                value_fn=value_fn,
            )
            return lbfgs_updates, state.adam_state, next_lbfgs_state

        final_updates, next_adam_state, next_lbfgs_state = jax.lax.cond(
            state.count < warmup_steps, warmup_branch, lbfgs_branch, operand=None
        )
        next_state = AdamWarmupLBFGSState(
            state.count + 1, next_adam_state, next_lbfgs_state
        )
        return final_updates, next_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _muon_hybrid_direction(config) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Muon-AURA's direction as two masked stages: Muon plus width scaling on the 2D
    matrices, bias-corrected ADAM on the biases."""

    is_matrix = lambda tree: tree_map(lambda leaf: leaf.ndim == 2, tree)
    is_not_matrix = lambda tree: tree_map(lambda leaf: leaf.ndim != 2, tree)
    # 2D-only mask (scale_by_muon rejects rank < 2); tree_map skips the MaskedNode leaves.
    matrix_dimension_numbers = lambda tree: tree_map(
        lambda leaf: optax.contrib.MuonDimensionNumbers(), tree
    )
    muon_matrix_direction = optax.masked(
        optax.chain(
            optax.contrib.scale_by_muon(
                ns_steps=config.ns_steps,
                beta=config.beta,
                eps=config.epsilon,
                nesterov=config.nesterov,
                weight_dimension_numbers=matrix_dimension_numbers,
            ),
            # Optax's default Muon width scaling, without its private _muon.scale_by_shape.
            optax.stateless(
                lambda updates, params: tree_map(
                    lambda leaf: jnp.sqrt(
                        jnp.maximum(1.0, leaf.shape[1] / leaf.shape[0])
                    )
                    * leaf,
                    updates,
                )
            ),
            optax.scale(config.learning_rate_scale),
        ),
        is_matrix,
    )
    adam_bias_direction = optax.masked(
        optax.scale_by_adam(
            b1=config.beta_1,
            b2=config.beta_2,
            eps=config.epsilon,
            nesterov=False,
        ),
        is_not_matrix,
    )
    return muon_matrix_direction, adam_bias_direction


def build_optimizer(
    method: str,
    learning_rate: float,
    *,
    steps_per_epoch: int = 1,
    total_steps: int | None = None,
    adam_config: optimizer_config.AdamConfig = optimizer_config.ADAM_CONFIG,
    sgd_config: optimizer_config.SgdConfig = optimizer_config.SGD_CONFIG,
    rprop_config: optimizer_config.RpropConfig = optimizer_config.RPROP_CONFIG,
    aura_config: optimizer_config.AdamAuraConfig = optimizer_config.AURA_CONFIG,
    astra_config: optimizer_config.AstraConfig = optimizer_config.ASTRA_CONFIG,
    eclipse_config: optimizer_config.EclipseConfig = optimizer_config.ECLIPSE_CONFIG,
    pulsar_config: optimizer_config.PulsarConfig = optimizer_config.PULSAR_CONFIG,
    aura_light_config: optimizer_config.AuraLightConfig = optimizer_config.AURA_LIGHT_CONFIG,
    nadamw_config: optimizer_config.NadamWConfig = optimizer_config.NADAMW_CONFIG,
    nadam_config: optimizer_config.NadamConfig = optimizer_config.NADAM_CONFIG,
    cvamsgrad_config: optimizer_config.CvAMSGradConfig = optimizer_config.CVAMSGRAD_CONFIG,
    muon_config: optimizer_config.MuonConfig = optimizer_config.MUON_CONFIG,
    muon_aura_config: optimizer_config.MuonAuraConfig = optimizer_config.MUON_AURA_CONFIG,
    aura_s_config: optimizer_config.AdamAuraSConfig = optimizer_config.AURA_S_CONFIG,
    muon_aura_s_config: optimizer_config.MuonAuraSConfig = optimizer_config.MUON_AURA_S_CONFIG,
    aura_sn_config: optimizer_config.AdamAuraSNConfig = optimizer_config.AURA_SN_CONFIG,
    muon_aura_sn_config: optimizer_config.MuonAuraSNConfig = optimizer_config.MUON_AURA_SN_CONFIG,
    aura_sign_config: optimizer_config.AdamAuraSignConfig = optimizer_config.AURA_SIGN_CONFIG,
    muon_aura_sign_config: optimizer_config.MuonAuraSignConfig = optimizer_config.MUON_AURA_SIGN_CONFIG,
    aura_snr_config: optimizer_config.AdamAuraSnrConfig = optimizer_config.AURA_SNR_CONFIG,
    muon_aura_snr_config: optimizer_config.MuonAuraSnrConfig = optimizer_config.MUON_AURA_SNR_CONFIG,
    aura_snr_ablation_config: optimizer_config.AdamAuraSnrAblationConfig = optimizer_config.AURA_SNR_ABLATION_CONFIG,
    aura_spring_config: optimizer_config.AdamAuraSpringAblationConfig = optimizer_config.AURA_SPRING_CONFIG,
    muon_aura_spring_config: optimizer_config.MuonAuraSpringAblationConfig = optimizer_config.MUON_AURA_SPRING_CONFIG,
    aura_spring_eb_config: optimizer_config.AdamAuraSpringEbConfig = optimizer_config.AURA_SPRING_EB_CONFIG,
    hcscgm_config: optimizer_config.HCSCGMConfig = optimizer_config.HCSCGM_CONFIG,
    lbfgs_config: optimizer_config.LBFGSConfig = optimizer_config.LBFGS_CONFIG,
) -> optax.GradientTransformation:
    """Build one method with a shared base learning rate for a fair ablation."""

    if (
        method not in optimizer_config.METHODS
        and method not in optimizer_config.TIMING_BASELINE_METHODS
        and method not in optimizer_config.EXPERIMENTAL_METHODS
    ):
        raise ValueError(
            f"Unknown method {method!r}; expected one of "
            f"{optimizer_config.METHODS + optimizer_config.TIMING_BASELINE_METHODS}"
        )
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if method == optimizer_config.METHOD_ADAM_VARIABLE_LR and total_steps is None:
        raise ValueError("total_steps is required for METHOD_ADAM_VARIABLE_LR")

    if method == optimizer_config.METHOD_SGD:
        return optax.sgd(
            learning_rate,
            momentum=sgd_config.momentum,
            nesterov=sgd_config.nesterov,
        )

    if method == optimizer_config.METHOD_NADAMW:
        return optax.nadamw(
            learning_rate,
            b1=nadamw_config.beta_1,
            b2=nadamw_config.beta_2,
            eps=nadamw_config.epsilon,
            weight_decay=nadamw_config.weight_decay,
        )

    if method == optimizer_config.METHOD_NADAM:
        return optax.nadam(
            learning_rate,
            b1=nadam_config.beta_1,
            b2=nadam_config.beta_2,
            eps=nadam_config.epsilon,
        )

    if method == optimizer_config.METHOD_CvAMSGrad:
        return optax.chain(
            scale_by_cvamsgrad(
                beta_1=cvamsgrad_config.beta_1,
                beta_2=cvamsgrad_config.beta_2,
                epsilon=cvamsgrad_config.epsilon,
                bias_correction=cvamsgrad_config.bias_correction,
            ),
            optax.scale_by_learning_rate(learning_rate),
        )

    if method == optimizer_config.METHOD_MUON:
        return optax.contrib.muon(
            learning_rate * muon_config.learning_rate_scale,
            ns_steps=muon_config.ns_steps,
            beta=muon_config.beta,
            eps=muon_config.epsilon,
            weight_decay=muon_config.weight_decay,
            nesterov=muon_config.nesterov,
            adam_b1=muon_config.beta_1,
            adam_b2=muon_config.beta_2,
            adam_weight_decay=muon_config.adam_weight_decay,
            adam_learning_rate=learning_rate,
        )

    if method == optimizer_config.METHOD_MUON_AURA:
        muon_direction = _muon_hybrid_direction(muon_aura_config)
        muon_aura_modulation = scale_by_aura(
            beta_zeta=muon_aura_config.beta_zeta,
            epsilon_e=muon_aura_config.epsilon_e,
            chi_alignment=muon_aura_config.chi_alignment,
            chi_opposition=muon_aura_config.chi_opposition,
            psi_alignment=muon_aura_config.psi_alignment,
            psi_opposition=muon_aura_config.psi_opposition,
            eta_minus=muon_aura_config.eta_minus,
            eta_plus=muon_aura_config.eta_plus,
            gamma_min=muon_aura_config.gamma_min,
            gamma_max=muon_aura_config.gamma_max,
            gamma_init=muon_aura_config.gamma_init,
        )
        # Kept flat so AuraState stays at chain index 2, where the diagnostics read it.
        return optax.chain(
            *muon_direction,
            muon_aura_modulation,
            optax.add_decayed_weights(muon_aura_config.weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )

    if method == optimizer_config.METHOD_HCSCGM:
        # Needs a full-batch value_fn: the direction assumes one fixed objective.
        return optax.chain(
            scale_by_hscgm(
                theta_max=hcscgm_config.theta_max,
                delta_1=hcscgm_config.delta_1,
            ),
            scale_by_weak_wolfe_linesearch(
                sigma_1=hcscgm_config.sigma_1,
                sigma_2=hcscgm_config.sigma_2,
                max_linesearch_steps=hcscgm_config.max_linesearch_steps,
            ),
        )

    if method == optimizer_config.METHOD_LBFGS:
        # Needs a fixed-objective loss closure, like HCSCGM.
        warmup_steps = lbfgs_config.adam_warmup_epochs * steps_per_epoch
        return chain_adam_warmup_then_lbfgs(
            adam=optax.chain(
                optax.scale_by_adam(
                    b1=adam_config.beta_1,
                    b2=adam_config.beta_2,
                    eps=adam_config.epsilon,
                    nesterov=False,
                ),
                optax.scale_by_learning_rate(learning_rate),
            ),
            lbfgs=optax.lbfgs(
                memory_size=lbfgs_config.memory_size,
                scale_init_precond=lbfgs_config.scale_init_precond,
            ),
            warmup_steps=warmup_steps,
        )

    if method == optimizer_config.METHOD_RPROP:
        return optax.chain(
            scale_by_componentwise_rprop(
                learning_rate,
                eta_minus=rprop_config.eta_minus,
                eta_plus=rprop_config.eta_plus,
                min_step_size=rprop_config.min_step_size,
                max_step_size=rprop_config.max_step_size,
            ),
            optax.scale(-1.0),
        )

    adam_scaling = optax.scale_by_adam(
        b1=adam_config.beta_1,
        b2=adam_config.beta_2,
        eps=adam_config.epsilon,
        nesterov=False,
    )
    if method == optimizer_config.METHOD_ADAM:
        return optax.chain(
            adam_scaling, optax.scale_by_learning_rate(learning_rate)
        )
    if method == optimizer_config.METHOD_ADAM_VARIABLE_LR:
        drop_at_step = round(total_steps * optimizer_config.ADAM_VARIABLE_LR_DROP_AT_FRACTION)
        schedule = optax.piecewise_constant_schedule(
            init_value=learning_rate,
            boundaries_and_scales={drop_at_step: 1.0 / optimizer_config.ADAM_VARIABLE_LR_DROP_FACTOR},
        )
        return optax.chain(adam_scaling, optax.scale_by_learning_rate(schedule))
    if method == optimizer_config.METHOD_AURA_LIGHT:
        # Modulation at chain index 1, as for AURA, so the diagnostics read the same slot.
        aura_light_modulation = scale_by_aura(
            beta_zeta=aura_light_config.beta_zeta,
            epsilon_e=aura_light_config.epsilon_e,
            chi_alignment=aura_light_config.chi_alignment,
            chi_opposition=aura_light_config.chi_opposition,
            psi_alignment=aura_light_config.psi_alignment,
            psi_opposition=aura_light_config.psi_opposition,
            eta_minus=aura_light_config.eta_minus,
            eta_plus=aura_light_config.eta_plus,
            gamma_min=aura_light_config.gamma_min,
            gamma_max=aura_light_config.gamma_max,
            gamma_init=aura_light_config.gamma_init,
        )
        return optax.chain(
            optax.ema(decay=aura_light_config.beta_1, debias=True),
            aura_light_modulation,
            optax.add_decayed_weights(aura_light_config.weight_decay),
            optax.scale_by_learning_rate(
                learning_rate * aura_light_config.learning_rate_scale
            ),
        )
    if method == optimizer_config.METHOD_ADAM_AURA_SIGN:
        gate = aura_sign_config
        return optax.chain(
            scale_by_aura_sign(
                adam_scaling,
                beta=gate.beta,
                kappa_plus=gate.kappa_plus,
                kappa_minus=gate.kappa_minus,
                brake_threshold=gate.brake_threshold,
                leak=gate.leak,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                momentum_of=lambda direction_state: direction_state.mu,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate * gate.learning_rate_scale),
        )
    if method == optimizer_config.METHOD_MUON_AURA_SIGN:
        gate = muon_aura_sign_config
        return optax.chain(
            scale_by_aura_sign(
                optax.chain(*_muon_hybrid_direction(gate)),
                beta=gate.sign_beta,
                kappa_plus=gate.kappa_plus,
                kappa_minus=gate.kappa_minus,
                brake_threshold=gate.brake_threshold,
                leak=gate.leak,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                momentum_of=_muon_hybrid_momentum if gate.momentum_reference else None,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
    if method == optimizer_config.METHOD_ADAM_AURA_SNR:
        gate = aura_snr_config
        return optax.chain(
            scale_by_aura_snr(
                adam_scaling,
                beta_zeta=gate.beta_zeta,
                epsilon_e=gate.epsilon_e,
                kappa_plus=gate.kappa_plus,
                kappa_minus=gate.kappa_minus,
                opposition_threshold=gate.opposition_threshold,
                leak=gate.leak,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                momentum_of=lambda direction_state: direction_state.mu,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate * gate.learning_rate_scale),
        )
    if method == optimizer_config.METHOD_MUON_AURA_SNR:
        gate = muon_aura_snr_config
        return optax.chain(
            scale_by_aura_snr(
                optax.chain(*_muon_hybrid_direction(gate)),
                beta_zeta=gate.beta_zeta,
                epsilon_e=gate.epsilon_e,
                kappa_plus=gate.kappa_plus,
                kappa_minus=gate.kappa_minus,
                opposition_threshold=gate.opposition_threshold,
                leak=gate.leak,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                momentum_of=_muon_hybrid_momentum if gate.momentum_reference else None,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
    if method == optimizer_config.METHOD_ADAM_AURA_SNR_ABLATION:
        gate = aura_snr_ablation_config
        return optax.chain(
            scale_by_aura_snr(
                adam_scaling,
                beta_zeta=gate.beta_zeta,
                epsilon_e=gate.epsilon_e,
                opposition_threshold=gate.opposition_threshold,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                eta_plus=gate.eta_plus,
                eta_minus=gate.eta_minus,
                momentum_of=lambda direction_state: direction_state.mu,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate * gate.learning_rate_scale),
        )
    if method in optimizer_config.AURA_S_FAMILY:
        gate = {
            optimizer_config.METHOD_ADAM_AURA_S: aura_s_config,
            optimizer_config.METHOD_MUON_AURA_S: muon_aura_s_config,
            optimizer_config.METHOD_ADAM_AURA_SN: aura_sn_config,
            optimizer_config.METHOD_MUON_AURA_SN: muon_aura_sn_config,
        }[method]
        on_adam = method in (optimizer_config.METHOD_ADAM_AURA_S, optimizer_config.METHOD_ADAM_AURA_SN)
        direction = adam_scaling if on_adam else optax.chain(*_muon_hybrid_direction(gate))
        # Muon applies learning_rate_scale inside its direction (matrix path only).
        step_scale = gate.learning_rate_scale if on_adam else 1.0
        return optax.chain(
            scale_by_aura_s(
                direction,
                beta_zeta=gate.beta_zeta,
                epsilon_e=gate.epsilon_e,
                kappa_plus=gate.kappa_plus,
                kappa_minus=gate.kappa_minus,
                rotation_penalty=gate.rotation_penalty,
                leak=gate.leak,
                chi_opposition=gate.chi_opposition,
                psi_opposition=gate.psi_opposition,
                kappa_brake=gate.kappa_brake,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                gamma_init=gate.gamma_init,
                noise_band=getattr(gate, "noise_band", None),
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate * step_scale),
        )
    if method in (optimizer_config.METHOD_ADAM_AURA, optimizer_config.METHOD_ADAM_AURA_COSINE_ABLATION):
        aura_modulation = scale_by_aura(
            beta_zeta=aura_config.beta_zeta,
            epsilon_e=aura_config.epsilon_e,
            chi_alignment=aura_config.chi_alignment,
            chi_opposition=aura_config.chi_opposition,
            psi_alignment=aura_config.psi_alignment,
            psi_opposition=aura_config.psi_opposition,
            eta_minus=aura_config.eta_minus,
            eta_plus=aura_config.eta_plus,
            gamma_min=aura_config.gamma_min,
            gamma_max=aura_config.gamma_max,
            gamma_init=aura_config.gamma_init,
            measure="cosine" if method == optimizer_config.METHOD_ADAM_AURA_COSINE_ABLATION else "dice",
        )
        return optax.chain(
            optax.scale_by_adam(
                b1=aura_config.beta_1,
                b2=aura_config.beta_2,
                eps=aura_config.epsilon,
                nesterov=False,
            ),
            aura_modulation,
            # Decay after scale_by_adam, so it is not rescaled by 1/(sqrt(v)+eps).
            optax.add_decayed_weights(aura_config.weight_decay),
            optax.scale_by_learning_rate(
                learning_rate * aura_config.learning_rate_scale
            ),
        )
    if method == optimizer_config.METHOD_ADAM_AURA_SPRING_ABLATION:
        gate = aura_spring_config
        return optax.chain(
            adam_scaling,
            scale_by_aura(
                beta_zeta=gate.beta_zeta,
                epsilon_e=gate.epsilon_e,
                chi_opposition=gate.chi_opposition,
                psi_opposition=gate.psi_opposition,
                kappa_plus=gate.kappa_plus,
                kappa_minus=gate.kappa_minus,
                leak=gate.leak,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                gamma_init=gate.gamma_init,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate * gate.learning_rate_scale),
        )
    if method == optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION:
        gate = muon_aura_spring_config
        return optax.chain(
            *_muon_hybrid_direction(gate),
            scale_by_aura(
                beta_zeta=gate.beta_zeta,
                epsilon_e=gate.epsilon_e,
                chi_opposition=gate.chi_opposition,
                psi_opposition=gate.psi_opposition,
                kappa_plus=gate.kappa_plus,
                kappa_minus=gate.kappa_minus,
                leak=gate.leak,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                gamma_init=gate.gamma_init,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
    if method == optimizer_config.METHOD_ADAM_AURA_SPRING_EB:
        gate = aura_spring_eb_config
        return optax.chain(
            adam_scaling,
            scale_by_aura(
                beta_zeta=gate.beta_zeta,
                epsilon_e=gate.epsilon_e,
                chi_alignment=gate.chi_alignment,
                chi_opposition=gate.chi_opposition,
                psi_alignment=gate.psi_alignment,
                psi_opposition=gate.psi_opposition,
                eta_minus=gate.eta_minus,
                eta_plus=gate.eta_plus,
                leak=gate.leak,
                gamma_min=gate.gamma_min,
                gamma_max=gate.gamma_max,
                gamma_init=gate.gamma_init,
            ),
            optax.add_decayed_weights(gate.weight_decay),
            optax.scale_by_learning_rate(learning_rate * gate.learning_rate_scale),
        )
    if method == optimizer_config.METHOD_ASTRA:
        astra_modulation = scale_by_astra(
            beta_1=adam_config.beta_1,
            beta_zeta=astra_config.beta_zeta,
            epsilon_e=astra_config.epsilon_e,
            chi_alignment=astra_config.chi_alignment,
            chi_opposition=astra_config.chi_opposition,
            psi_alignment=astra_config.psi_alignment,
            psi_opposition=astra_config.psi_opposition,
            eta_minus=astra_config.eta_minus,
            eta_plus=astra_config.eta_plus,
            gamma_min=astra_config.gamma_min,
            gamma_max=astra_config.gamma_max,
            leak_rate=astra_config.leak_rate,
            leak_exponent=astra_config.leak_exponent,
            gamma_init=astra_config.gamma_init,
        )
        return optax.chain(
            adam_scaling,
            astra_modulation,
            optax.add_decayed_weights(astra_config.weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
    if method == optimizer_config.METHOD_ECLIPSE:
        eclipse_modulation = scale_by_eclipse(
            beta_1=adam_config.beta_1,
            beta_2=adam_config.beta_2,
            epsilon_a=adam_config.epsilon,
            epsilon_e=eclipse_config.epsilon_e,
            chi_alignment=eclipse_config.chi_alignment,
            chi_opposition=eclipse_config.chi_opposition,
            psi_alignment=eclipse_config.psi_alignment,
            psi_opposition=eclipse_config.psi_opposition,
            eta_minus=eclipse_config.eta_minus,
            eta_plus=eclipse_config.eta_plus,
            gamma_min=eclipse_config.gamma_min,
            gamma_max=eclipse_config.gamma_max,
            leak_rate=eclipse_config.leak_rate,
            gamma_init=eclipse_config.gamma_init,
        )
        return optax.chain(
            eclipse_modulation,
            optax.add_decayed_weights(eclipse_config.weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
    if method == optimizer_config.METHOD_PULSAR:
        pulsar_modulation = scale_by_pulsar(
            beta_1=adam_config.beta_1,
            beta_2=adam_config.beta_2,
            epsilon_a=adam_config.epsilon,
            epsilon_e=pulsar_config.epsilon_e,
            kappa=pulsar_config.kappa,
        )
        return optax.chain(
            pulsar_modulation,
            optax.add_decayed_weights(pulsar_config.weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
    raise ValueError(f"Unknown method {method!r}")


def optimizer_diagnostics(method: str, state) -> jax.Array:
    """Return ``[gamma_min, gamma_mean, gamma_max, nan, nan, nan, nan, mean_chi,
    mean_abs_psi]``; entries 3-6 keep the legacy 9-column layout."""

    if method in (
        optimizer_config.METHOD_ADAM,
        optimizer_config.METHOD_ADAM_VARIABLE_LR,
        optimizer_config.METHOD_NADAMW,
        optimizer_config.METHOD_NADAM,
        optimizer_config.METHOD_CvAMSGrad,
        optimizer_config.METHOD_MUON,
        optimizer_config.METHOD_HCSCGM,
        optimizer_config.METHOD_LBFGS,
        optimizer_config.METHOD_SGD,
    ):
        return jnp.asarray(
            [1.0, 1.0, 1.0, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan]
        )

    if method == optimizer_config.METHOD_RPROP:
        rprop_state = state[0]
        step_sizes = jnp.concatenate(
            [
                jnp.ravel(value)
                for tree in (
                    rprop_state.step_size_real,
                    rprop_state.step_size_imag,
                )
                for value in tree_leaves(tree)
            ]
        )
        return jnp.asarray(
            [
                jnp.min(step_sizes),
                jnp.mean(step_sizes),
                jnp.max(step_sizes),
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
            ]
        )

    if method in (
        optimizer_config.METHOD_ADAM_AURA,
        optimizer_config.METHOD_ADAM_AURA_COSINE_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_ABLATION,
        optimizer_config.METHOD_ADAM_AURA_SPRING_EB,
        optimizer_config.METHOD_ASTRA,
        optimizer_config.METHOD_AURA_LIGHT,
        optimizer_config.METHOD_MUON_AURA,
        optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION,
    ):
        aura_state = (
            state[2]
            if method in (
                optimizer_config.METHOD_MUON_AURA,
                optimizer_config.METHOD_MUON_AURA_SPRING_ABLATION,
            )
            else state[1]
        )
        multipliers = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(aura_state.multiplier)]
        )
        chi_values = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(aura_state.chi)]
        )
        psi_values = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(aura_state.psi)]
        )
        return jnp.asarray(
            [
                jnp.min(multipliers),
                jnp.mean(multipliers),
                jnp.max(multipliers),
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.mean(chi_values),
                jnp.mean(jnp.abs(psi_values)),
            ]
        )

    if method in optimizer_config.AURA_SNR_FAMILY:
        aura_snr_state = state[0]
        multipliers = jnp.exp(
            jnp.concatenate([jnp.ravel(value) for value in tree_leaves(aura_snr_state.log_multiplier)])
        )
        consistency = jnp.concatenate([jnp.ravel(value) for value in tree_leaves(aura_snr_state.consistency)])
        return jnp.asarray(
            [
                jnp.min(multipliers),
                jnp.mean(multipliers),
                jnp.max(multipliers),
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.mean(jnp.real(consistency)),
                jnp.mean(jnp.abs(jnp.imag(consistency))),
            ]
        )

    if method in optimizer_config.AURA_SIGN_FAMILY:
        aura_sign_state = state[0]
        multipliers = jnp.exp(
            jnp.concatenate([jnp.ravel(value) for value in tree_leaves(aura_sign_state.log_multiplier)])
        )
        alignment = jnp.concatenate([jnp.ravel(value) for value in tree_leaves(aura_sign_state.alignment)])
        return jnp.asarray(
            [
                jnp.min(multipliers),
                jnp.mean(multipliers),
                jnp.max(multipliers),
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.mean(alignment),
                jnp.nan,
            ]
        )

    if method in optimizer_config.AURA_S_FAMILY:
        aura_s_state = state[0]
        multipliers = jnp.exp(
            jnp.concatenate([jnp.ravel(value) for value in tree_leaves(aura_s_state.log_multiplier)])
        )
        alignment = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(aura_s_state.alignment_ema)]
        ) / jnp.maximum(aura_s_state.bias_correction, 1e-12)
        return jnp.asarray(
            [
                jnp.min(multipliers),
                jnp.mean(multipliers),
                jnp.max(multipliers),
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.mean(jnp.real(alignment)),
                jnp.mean(jnp.abs(jnp.imag(alignment))),
            ]
        )

    if method == optimizer_config.METHOD_PULSAR:
        multipliers = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(state[0].multiplier)]
        )
        return jnp.asarray(
            [
                jnp.min(multipliers),
                jnp.mean(multipliers),
                jnp.max(multipliers),
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
            ]
        )

    if method == optimizer_config.METHOD_ECLIPSE:
        eclipse_state = state[0]
        multipliers = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(eclipse_state.multiplier)]
        )
        chi_values = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(eclipse_state.chi)]
        )
        psi_values = jnp.concatenate(
            [jnp.ravel(value) for value in tree_leaves(eclipse_state.psi)]
        )
        return jnp.asarray(
            [
                jnp.min(multipliers),
                jnp.mean(multipliers),
                jnp.max(multipliers),
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.mean(chi_values),
                jnp.mean(jnp.abs(psi_values)),
            ]
        )

    raise ValueError(f"Unknown method {method!r}")

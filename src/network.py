"""Generic complex-valued MLP utilities: layer init, forward pass, activations."""


from collections.abc import Sequence

import jax
import jax.numpy as jnp


def _real_dtype(complex_dtype):
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex64):
        return jnp.float32
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex128):
        return jnp.float64
    raise ValueError(f"Expected complex64 or complex128, got {complex_dtype}")


def complex_exp(z: jax.Array) -> jax.Array:
    """The entire activation exp(z), holomorphic on all of C."""

    return jnp.exp(z)


def componentwise_silu(z: jax.Array) -> jax.Array:
    """Apply SiLU independently to the real and imaginary components."""

    return jax.nn.silu(jnp.real(z)) + 1j * jax.nn.silu(jnp.imag(z))


def componentwise_tanh(z: jax.Array) -> jax.Array:
    """Apply tanh independently to the real and imaginary components."""

    return jnp.tanh(jnp.real(z)) + 1j * jnp.tanh(jnp.imag(z))


ACTIVATIONS = {
    "complex_exp": complex_exp,
    "componentwise_silu": componentwise_silu,
    "componentwise_tanh": componentwise_tanh,
}


def _init_layers(architecture, layer_keys, component_std_fn, dtype):
    real_dtype = _real_dtype(dtype)
    layers = []
    for layer_key, fan_in, fan_out in zip(
        layer_keys, architecture[:-1], architecture[1:]
    ):
        real_key, imag_key = jax.random.split(layer_key)
        component_std = component_std_fn(fan_in, fan_out, real_dtype)
        real = component_std * jax.random.normal(
            real_key, (fan_in, fan_out), dtype=real_dtype
        )
        imag = component_std * jax.random.normal(
            imag_key, (fan_in, fan_out), dtype=real_dtype
        )
        layers.append(
            {
                "weight": (real + 1j * imag).astype(dtype),
                "bias": jnp.zeros((fan_out,), dtype=dtype),
            }
        )
    return tuple(layers)


def initialize_network(
    key: jax.Array,
    architecture: Sequence[int],
    *,
    activation_name: str,
    dtype=jnp.complex64,
    beta: float = 0.5,
):
    """Initialize complex affine layers with zero biases."""

    if activation_name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation_name {activation_name!r}; expected one of {tuple(ACTIVATIONS)}")
    architecture = tuple(int(width) for width in architecture)
    if len(architecture) < 2 or any(width < 1 for width in architecture):
        raise ValueError("architecture must contain at least two positive widths")
    if architecture[-1] != 1:
        raise ValueError("this benchmark requires output width 1")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    layer_keys = jax.random.split(key, len(architecture) - 1)
    if activation_name == "complex_exp":
        beta_array = jnp.asarray(beta, _real_dtype(dtype))
        component_std_fn = lambda fan_in, fan_out, real_dtype: jnp.sqrt(
            beta_array / (2.0 * fan_in * jnp.exp(beta_array))
        )
    else:
        component_std_fn = lambda fan_in, fan_out, real_dtype: jnp.sqrt(
            jnp.asarray(2.0 / (fan_in + fan_out), real_dtype)
        )
    return _init_layers(architecture, layer_keys, component_std_fn, dtype)


def forward(params, z: jax.Array, *, activation_name: str) -> jax.Array:
    """Evaluate a complex MLP built by ``initialize_network``."""

    activate = ACTIVATIONS[activation_name]
    input_width = params[0]["weight"].shape[0]
    values = jnp.asarray(z).reshape(-1, input_width)
    for layer in params[:-1]:
        values = activate(values @ layer["weight"] + layer["bias"])
    output = values @ params[-1]["weight"] + params[-1]["bias"]
    return output[:, 0]


def mean_squared_error(params, z: jax.Array, targets: jax.Array, *, activation_name: str) -> jax.Array:
    residual = forward(params, z, activation_name=activation_name) - targets
    return jnp.mean(jnp.square(jnp.abs(residual)))


def relative_l2_error(params, z: jax.Array, targets: jax.Array, *, activation_name: str) -> jax.Array:
    residual_norm = jnp.linalg.norm(forward(params, z, activation_name=activation_name) - targets)
    target_norm = jnp.maximum(jnp.linalg.norm(targets), jnp.finfo(targets.real.dtype).tiny)
    return residual_norm / target_norm


def holomorphicity_residual(params, *, activation_name: str, z: complex = 0.2 + 0.1j) -> jax.Array:
    """Numerically check Df(i v) = i Df(v), a Cauchy--Riemann identity."""

    dtype = params[0]["weight"].dtype
    point = jnp.asarray([z], dtype=dtype)
    one = jnp.ones_like(point)
    function = lambda value: forward(params, value, activation_name=activation_name)
    _, derivative_real = jax.jvp(function, (point,), (one,))
    _, derivative_imag = jax.jvp(function, (point,), (1j * one,))
    scale = jnp.maximum(jnp.linalg.norm(derivative_real), 1.0)
    return jnp.linalg.norm(derivative_imag - 1j * derivative_real) / scale

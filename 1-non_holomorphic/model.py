"""Complex-valued MLP with componentwise SiLU for the 1-non_holomorphic benchmark."""


from collections.abc import Sequence

import jax
import jax.numpy as jnp

ACTIVATION_NAME = "componentwise_silu"


def _real_dtype(complex_dtype):
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex64):
        return jnp.float32
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex128):
        return jnp.float64
    raise ValueError(f"Expected complex64 or complex128, got {complex_dtype}")


def _activation(z: jax.Array) -> jax.Array:
    """Apply SiLU independently to the real and imaginary components."""

    return jax.nn.silu(jnp.real(z)) + 1j * jax.nn.silu(jnp.imag(z))


def _init_layers(architecture, layer_keys, component_std, dtype):
    real_dtype = _real_dtype(dtype)
    layers = []
    for layer_key, fan_in, fan_out in zip(layer_keys, architecture[:-1], architecture[1:]):
        real_key, imag_key = jax.random.split(layer_key)
        std = component_std(fan_in, fan_out)
        real = std * jax.random.normal(real_key, (fan_in, fan_out), dtype=real_dtype)
        imag = std * jax.random.normal(imag_key, (fan_in, fan_out), dtype=real_dtype)
        layers.append(
            {
                "weight": (real + 1j * imag).astype(dtype),
                "bias": jnp.zeros((fan_out,), dtype=dtype),
            }
        )
    return tuple(layers)


def init_model(
    key: jax.Array,
    architecture: Sequence[int],
    *,
    dtype=jnp.complex64,
    beta: float = 0.5,
):
    """Initialize complex affine layers with zero biases."""

    del beta
    architecture = tuple(int(width) for width in architecture)
    if len(architecture) < 2 or any(width < 1 for width in architecture):
        raise ValueError("architecture must contain at least two positive widths")
    if architecture[-1] != 1:
        raise ValueError("this benchmark requires output width 1")

    layer_keys = jax.random.split(key, len(architecture) - 1)
    real_dtype = _real_dtype(dtype)
    component_std = lambda fan_in, fan_out: jnp.sqrt(jnp.asarray(2.0 / (fan_in + fan_out), real_dtype))
    return _init_layers(architecture, layer_keys, component_std, dtype)


def forward(params, z: jax.Array) -> jax.Array:
    """Evaluate the complex MLP built by ``init_model``."""

    input_width = params[0]["weight"].shape[0]
    values = jnp.asarray(z).reshape(-1, input_width)
    for layer in params[:-1]:
        values = _activation(values @ layer["weight"] + layer["bias"])
    output = values @ params[-1]["weight"] + params[-1]["bias"]
    return output[:, 0]


def mean_squared_error(params, z: jax.Array, targets: jax.Array) -> jax.Array:
    residual = forward(params, z) - targets
    return jnp.mean(jnp.square(jnp.abs(residual)))


def relative_l2_error(params, z: jax.Array, targets: jax.Array) -> jax.Array:
    residual_norm = jnp.linalg.norm(forward(params, z) - targets)
    target_norm = jnp.maximum(jnp.linalg.norm(targets), jnp.finfo(targets.real.dtype).tiny)
    return residual_norm / target_norm


def holomorphicity_residual(params, *, z: complex = 0.2 + 0.1j) -> jax.Array:
    """Numerically check Df(i v) = i Df(v), a Cauchy-Riemann identity."""

    dtype = params[0]["weight"].dtype
    point = jnp.asarray([z], dtype=dtype)
    one = jnp.ones_like(point)
    function = lambda value: forward(params, value)
    _, derivative_real = jax.jvp(function, (point,), (one,))
    _, derivative_imag = jax.jvp(function, (point,), (1j * one,))
    scale = jnp.maximum(jnp.linalg.norm(derivative_real), 1.0)
    return jnp.linalg.norm(derivative_imag - 1j * derivative_real) / scale

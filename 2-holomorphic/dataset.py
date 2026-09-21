"""Target function and dataset generation for the 2-holomorphic benchmark."""


import jax
import jax.numpy as jnp

TARGET_INPUT_WIDTH = 1

# Unused here; kept so every 1-* case's dataset.py has the same shape.
C4_DOMAIN_MIN = 0.5
C4_DOMAIN_MAX = 1.0


def _real_dtype(complex_dtype):
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex64):
        return jnp.float32
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex128):
        return jnp.float64
    raise ValueError(f"Expected complex64 or complex128, got {complex_dtype}")


def target_function(z: jax.Array) -> jax.Array:
    """Evaluate the entire multi-scale holomorphic target."""

    return (
        jnp.exp((0.39 - 0.22j) * z)
        + 0.12 * jnp.exp(3.5 * z)
        + 0.03 * jnp.exp((-2.8 + 3.3j) * z)
        + 0.002 * jnp.exp((-5.3 - 4.6j) * z)
        + 0.0015 * jnp.exp((5.5 + 5.0j) * z)
        + 0.015 * jnp.sin(5.5 * z)
        + 0.059 * z**4
        - 0.040j * z**5
    )


def target_derivative(z: jax.Array) -> jax.Array:
    """Evaluate the target's analytic complex derivative."""

    return (
        (0.39 - 0.22j) * jnp.exp((0.39 - 0.22j) * z)
        + 0.12 * 3.5 * jnp.exp(3.5 * z)
        + 0.03 * (-2.8 + 3.3j) * jnp.exp((-2.8 + 3.3j) * z)
        + 0.002 * (-5.3 - 4.6j) * jnp.exp((-5.3 - 4.6j) * z)
        + 0.0015 * (5.5 + 5.0j) * jnp.exp((5.5 + 5.0j) * z)
        + 0.015 * 5.5 * jnp.cos(5.5 * z)
        + 4.0 * 0.059 * z**3
        - 5.0 * 0.040j * z**4
    )


def _sample_complex(key, shape, minval, maxval, dtype):
    """Sample uniform complex points with real and imaginary parts independent."""

    real_dtype = _real_dtype(dtype)
    key_real, key_imag = jax.random.split(key)
    real = jax.random.uniform(key_real, shape, minval=minval, maxval=maxval, dtype=real_dtype)
    imag = jax.random.uniform(key_imag, shape, minval=minval, maxval=maxval, dtype=real_dtype)
    return (real + 1j * imag).astype(dtype)


def make_datasets(
    key: jax.Array,
    *,
    train_size: int,
    test_size: int,
    half_width: float,
    dtype=jnp.complex64,
):
    """Create random training points and an independent random test set."""

    if train_size < 1:
        raise ValueError("train_size must be positive")
    if test_size < 1:
        raise ValueError("test_size must be positive")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")

    key_train, key_test = jax.random.split(key)
    z_train = _sample_complex(key_train, (train_size,), -half_width, half_width, dtype)
    z_test = _sample_complex(key_test, (test_size,), -half_width, half_width, dtype)
    return z_train, target_function(z_train), z_test, target_function(z_test)

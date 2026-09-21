"""Target function and dataset generation for the 1-non_holomorphic benchmark."""


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
    """Evaluate a purely non-holomorphic target with z/conj(z) coupling."""

    z_conjugate = jnp.conj(z)
    return (
        jnp.exp((0.30 - 0.20j) * z * z_conjugate)
        + 0.25 * jnp.sin(z) * jnp.cos(z_conjugate)
        + 0.10 * z**2 * z_conjugate
        + 0.08 * z * z_conjugate**2
        - 0.05j * (z * z_conjugate) ** 2
    )


def target_derivative(z: jax.Array) -> jax.Array:
    """Evaluate the target's derivative along the real input direction."""

    z_conjugate = jnp.conj(z)
    return (
        (0.30 - 0.20j) * (z + z_conjugate) * jnp.exp((0.30 - 0.20j) * z * z_conjugate)
        + 0.25 * (jnp.cos(z) * jnp.cos(z_conjugate) - jnp.sin(z) * jnp.sin(z_conjugate))
        + 0.10 * (2 * z * z_conjugate + z**2)
        + 0.08 * (z_conjugate**2 + 2 * z * z_conjugate)
        - 0.10j * z * z_conjugate * (z + z_conjugate)
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

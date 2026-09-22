"""Target function and dataset generation for the 3-multivariate_c4 benchmark."""


import jax
import jax.numpy as jnp

TARGET_INPUT_WIDTH = 4

# Away from zero: the target divides by x1.
C4_DOMAIN_MIN = 0.5
C4_DOMAIN_MAX = 1.0


def _real_dtype(complex_dtype):
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex64):
        return jnp.float32
    if jnp.dtype(complex_dtype) == jnp.dtype(jnp.complex128):
        return jnp.float64
    raise ValueError(f"Expected complex64 or complex128, got {complex_dtype}")


def target_function(x: jax.Array) -> jax.Array:
    """Evaluate the C^4 -> C function approximation target."""

    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return (x2**2 / x1 + x3 + 10.0 * x1 * x4) / 1.5


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

    del half_width
    if train_size < 1:
        raise ValueError("train_size must be positive")
    if test_size < 1:
        raise ValueError("test_size must be positive")

    key_train, key_test = jax.random.split(key)
    x_train = _sample_complex(key_train, (train_size, TARGET_INPUT_WIDTH), C4_DOMAIN_MIN, C4_DOMAIN_MAX, dtype)
    x_test = _sample_complex(key_test, (test_size, TARGET_INPUT_WIDTH), C4_DOMAIN_MIN, C4_DOMAIN_MAX, dtype)
    return x_train, target_function(x_train), x_test, target_function(x_test)

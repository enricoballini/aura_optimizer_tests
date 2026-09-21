"""Standalone reproduction of Section 5.1 of Zhang et al. (2024)."""


import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


N_IN = 4
N_HIDDEN = 62
N_TRAIN = 62
DOMAIN_HALF_WIDTH = 0.5
WEIGHT_INIT_RANGE = 0.2
DTYPE = jnp.complex128


def target_function(x: jax.Array) -> jax.Array:
    """f(x) = (x2^2/x1 + x3 + 10*x1*x4) / 1.5, for x = (x1,x2,x3,x4) in C^4."""

    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return (x2**2 / x1 + x3 + 10.0 * x1 * x4) / 1.5


def sample_complex(key, shape, half_width):
    """Uniform samples in [-half_width, half_width] for both real and imag parts."""

    key_real, key_imag = jax.random.split(key)
    real = jax.random.uniform(
        key_real, shape, minval=-half_width, maxval=half_width, dtype=jnp.float64
    )
    imag = jax.random.uniform(
        key_imag, shape, minval=-half_width, maxval=half_width, dtype=jnp.float64
    )
    return (real + 1j * imag).astype(DTYPE)


def make_training_set(key):
    """62 random points in C^4, as in Section 5.1."""

    x_train = sample_complex(key, (N_TRAIN, N_IN), DOMAIN_HALF_WIDTH)
    return x_train, target_function(x_train)


def init_weights(key):
    """A single hidden-layer 4-62-1 CVNN, weights uniform in [-0.2, 0.2] (no biases)."""

    key_v, key_u = jax.random.split(key)
    v = sample_complex(key_v, (N_HIDDEN, N_IN), WEIGHT_INIT_RANGE)
    u = sample_complex(key_u, (N_HIDDEN,), WEIGHT_INIT_RANGE)
    return pack(v, u)


def pack(v: jax.Array, u: jax.Array) -> jax.Array:
    """Flatten the two weight arrays into the single vector w used by HCSCGM."""

    return jnp.concatenate([v.ravel(), u.ravel()])


def unpack(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    v = w[: N_HIDDEN * N_IN].reshape(N_HIDDEN, N_IN)
    u = w[N_HIDDEN * N_IN :]
    return v, u


def forward(w: jax.Array, x: jax.Array) -> jax.Array:
    """y_n = u . F(V x_n), with F = tanh, for a batch of inputs x (N, 4)."""

    v, u = unpack(w)
    hidden = jnp.tanh(x @ v.T)
    return hidden @ u


def training_error(w: jax.Array, x_train: jax.Array, d_train: jax.Array) -> jax.Array:
    """E(w) = mean_n |e_n|^2, e_n = y_n - d_n (Algorithm 1, line 7)."""

    residual = forward(w, x_train) - d_train
    return jnp.mean(jnp.real(residual * jnp.conj(residual)))


def complex_gradient(w: jax.Array, x_train: jax.Array, d_train: jax.Array) -> jax.Array:
    """The complex gradient grad_w E(w), matching the paper's convention."""

    return jnp.conj(jax.grad(training_error)(w, x_train, d_train))


def hcscgm_direction(
    grad: jax.Array,
    previous_grad: jax.Array,
    previous_direction: jax.Array,
    step: jax.Array,
    previous_theta: float,
    *,
    theta_max: float,
    delta_1: float,
) -> tuple[jax.Array, float]:
    """One step of the HCSCGM direction update (paper eqs. (6), (10), (13), (15))."""

    s = step
    y = grad - previous_grad

    y_dot_s = float(jnp.real(jnp.vdot(y, s)))
    raw_theta = float(jnp.real(jnp.vdot(s, s))) / y_dot_s if y_dot_s != 0.0 else previous_theta
    theta = raw_theta if (1.0 / theta_max <= raw_theta <= theta_max) else previous_theta

    residual = theta * y - s
    y_dot_previous_direction = float(jnp.real(jnp.vdot(y, previous_direction)))
    if y_dot_previous_direction != 0.0:
        beta_sp = float(jnp.real(jnp.vdot(grad, residual))) / y_dot_previous_direction
    else:
        beta_sp = 0.0

    if y_dot_previous_direction != 0.0:
        mu = (
            delta_1
            * float(jnp.real(jnp.vdot(residual, residual)))
            / (theta * y_dot_previous_direction**2)
        )
    else:
        mu = 0.0
    grad_dot_previous_direction = float(jnp.real(jnp.vdot(grad, previous_direction)))
    beta_hsp = beta_sp - mu * grad_dot_previous_direction

    direction = -theta * grad + beta_hsp * previous_direction
    return direction, theta


def wolfe_line_search(
    w: jax.Array,
    direction: jax.Array,
    grad: jax.Array,
    x_train: jax.Array,
    d_train: jax.Array,
    *,
    sigma_1: float,
    sigma_2: float,
    max_iterations: int,
    initial_step: float = 1.0,
) -> float:
    """Bisection bracketing search satisfying eqs. (11)-(12)."""

    loss_0 = float(training_error(w, x_train, d_train))
    slope_0 = float(jnp.real(jnp.vdot(grad, direction)))

    alpha_low, alpha_high = 0.0, None
    alpha = initial_step
    for _ in range(max_iterations):
        candidate = w + alpha * direction
        loss_candidate = float(training_error(candidate, x_train, d_train))

        if loss_candidate - loss_0 > sigma_1 * alpha * slope_0:
            alpha_high = alpha
            alpha = 0.5 * (alpha_low + alpha_high)
            continue

        slope_candidate = float(
            jnp.real(jnp.vdot(complex_gradient(candidate, x_train, d_train), direction))
        )
        if slope_candidate < sigma_2 * slope_0:
            alpha_low = alpha
            alpha = 2.0 * alpha if alpha_high is None else 0.5 * (alpha_low + alpha_high)
            continue

        return alpha

    return alpha


def train(
    *,
    seed: int = 0,
    max_iterations: int = 25_000,
    theta_0: float = 1.0,
    delta_1: float = 0.5,
    theta_max: float = 1e4,
    sigma_1: float = 0.001,
    sigma_2: float = 0.9,
    target_error: float = 0.0,
    print_every: int = 100,
):
    """Run HCSCGM on the Section 5.1 function-approximation problem."""

    key = jax.random.PRNGKey(seed)
    data_key, weight_key = jax.random.split(key)
    x_train, d_train = make_training_set(data_key)
    w = init_weights(weight_key)

    grad = complex_gradient(w, x_train, d_train)
    direction = -grad
    theta = theta_0

    history = []
    start_time = time.perf_counter()
    for iteration in range(max_iterations):
        error = float(training_error(w, x_train, d_train))
        history.append(error)
        if print_every and iteration % print_every == 0:
            print(f"iteration {iteration:6d} | training error {error:.6e}")
        if error <= target_error:
            break

        alpha = wolfe_line_search(
            w,
            direction,
            grad,
            x_train,
            d_train,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            max_iterations=30,
        )
        w_next = w + alpha * direction
        grad_next = complex_gradient(w_next, x_train, d_train)
        direction, theta = hcscgm_direction(
            grad_next,
            grad,
            direction,
            w_next - w,
            theta,
            theta_max=theta_max,
            delta_1=delta_1,
        )
        w, grad = w_next, grad_next

    elapsed = time.perf_counter() - start_time
    final_error = float(training_error(w, x_train, d_train))
    print(
        f"\nFinished after {len(history)} iterations in {elapsed:.3f} s "
        f"| final training error {final_error:.6e}"
    )
    return w, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=25_000)
    parser.add_argument(
        "--target-error",
        type=float,
        default=0.0,
        help="Stop early once training error drops to this value (0 = run "
        "the full --max-iterations, matching the paper's Fig. 1; use "
        "0.0018 for the paper's Table 2 stopping point).",
    )
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()

    train(
        seed=args.seed,
        max_iterations=args.max_iterations,
        target_error=args.target_error,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()

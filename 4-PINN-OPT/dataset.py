"""Boundary-point sampling for the PIHNN plate-with-hole test case: square plate with a
circular hole under uniaxial tension."""


import jax.numpy as jnp
import numpy as np

PLATE_L = 2.5
HOLE_R = 1.0
TENSION = 1.0

# Matches Figs. 5-6, where the modeled quadrant spans the full L (the text suggests L/2).
HALF_L = PLATE_L

TOTAL_LENGTH = 2 * HALF_L + HOLE_R * np.pi / 2.0 + 2 * (HALF_L - HOLE_R)
ALPHA_N = (2 * HALF_L + HOLE_R * np.pi / 2.0) / TOTAL_LENGTH
ALPHA_S = (2 * (HALF_L - HOLE_R)) / TOTAL_LENGTH


def _allocate_counts(lengths, n_total):
    """Split n_total points across segments proportionally to their length
    (largest-remainder method)."""

    lengths = np.asarray(lengths, dtype=np.float64)
    exact = n_total * lengths / lengths.sum()
    counts = np.floor(exact).astype(int)
    remainder = n_total - counts.sum()
    order = np.argsort(-(exact - counts))
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _sample_boundary(n_total, rng):
    seg_lengths = [HALF_L, HALF_L, HOLE_R * np.pi / 2.0, HALF_L - HOLE_R, HALF_L - HOLE_R]
    n_left, n_top, n_arc, n_right, n_bottom = _allocate_counts(seg_lengths, n_total)

    y = rng.uniform(0.0, HALF_L, n_left)
    z_left = -HALF_L + 1j * y
    n_left_vec = np.tile([-1.0, 0.0], (n_left, 1))
    t0_left = np.tile([-TENSION, 0.0], (n_left, 1))

    x = rng.uniform(-HALF_L, 0.0, n_top)
    z_top = x + 1j * HALF_L
    n_top_vec = np.tile([0.0, 1.0], (n_top, 1))
    t0_top = np.zeros((n_top, 2))

    theta = rng.uniform(np.pi / 2.0, np.pi, n_arc)
    z_arc = HOLE_R * np.cos(theta) + 1j * HOLE_R * np.sin(theta)
    n_arc_vec = np.stack([-np.cos(theta), -np.sin(theta)], axis=1)
    t0_arc = np.zeros((n_arc, 2))

    z_neumann = np.concatenate([z_left, z_top, z_arc])
    n_neumann = np.concatenate([n_left_vec, n_top_vec, n_arc_vec])
    t0_neumann = np.concatenate([t0_left, t0_top, t0_arc])

    y = rng.uniform(HOLE_R, HALF_L, n_right)
    z_right = 0.0 + 1j * y
    n_right_vec = np.tile([1.0, 0.0], (n_right, 1))

    x = rng.uniform(-HALF_L, -HOLE_R, n_bottom)
    z_bottom = x + 1j * 0.0
    n_bottom_vec = np.tile([0.0, -1.0], (n_bottom, 1))

    z_sym = np.concatenate([z_right, z_bottom])
    n_sym = np.concatenate([n_right_vec, n_bottom_vec])

    return (
        jnp.asarray(z_neumann, dtype=jnp.complex64),
        jnp.asarray(n_neumann, dtype=jnp.float32),
        jnp.asarray(t0_neumann, dtype=jnp.float32),
        jnp.asarray(z_sym, dtype=jnp.complex64),
        jnp.asarray(n_sym, dtype=jnp.float32),
    )

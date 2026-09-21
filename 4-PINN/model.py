"""Holomorphic PIHNN (Calafa et al., 2024) and the Kolosov-Muskhelishvili
stress/displacement representation."""


import jax
import jax.numpy as jnp

MU = 1.0
LAMBDA_T = 1.0
GAMMA = (LAMBDA_T + 3.0 * MU) / (LAMBDA_T + MU)

# LAYER_SIZES = [1, 10, 10, 10, 1]
LAYER_SIZES = [1, 64, 64, 64, 1]
BETA = 0.5
GAUSS = 3


def _forward_one(params, z):
    x = jnp.reshape(z, (1,))
    for W, b in params[:-1]:
        x = jnp.exp(x @ W + b)
    W, b = params[-1]
    x = x @ W + b
    return x[0]


def net_values(params, z, order_phi, order_psi):
    """Return ([phi(z), phi'(z), ...], [psi(z), psi'(z), ...]) for a batch z via
    holomorphic autodiff."""

    phi_f = lambda zz: _forward_one(params["phi"], zz)
    psi_f = lambda zz: _forward_one(params["psi"], zz)

    phi_values = [jax.vmap(phi_f)(z)]
    d = phi_f
    for _ in range(order_phi):
        d = jax.grad(d, holomorphic=True)
        phi_values.append(jax.vmap(d)(z))

    psi_values = [jax.vmap(psi_f)(z)]
    d = psi_f
    for _ in range(order_psi):
        d = jax.grad(d, holomorphic=True)
        psi_values.append(jax.vmap(d)(z))

    return phi_values, psi_values


def _init_one_layer(key, n_in, n_out, std):
    key, k_re, k_im = jax.random.split(key, 3)
    W = std * jax.random.normal(k_re, (n_in, n_out)) + 1j * std * jax.random.normal(k_im, (n_in, n_out))
    b = jnp.zeros((n_out,), dtype=jnp.complex64)
    return key, (W.astype(jnp.complex64), b)


def init_network(key, beta, x0):
    """Algorithm 1 of the paper: PIHNN weight initialization with phi(z) = e^z, for
    both networks."""

    n_layers = len(LAYER_SIZES) - 1
    x_phi = x0.reshape(-1, 1)
    x_psi = x0.reshape(-1, 1)
    phi_params, psi_params = [], []
    for l in range(n_layers):
        n_in, n_out = LAYER_SIZES[l], LAYER_SIZES[l + 1]
        if l < GAUSS:
            pooled_sq_abs = jnp.concatenate([jnp.abs(x_phi).reshape(-1), jnp.abs(x_psi).reshape(-1)]) ** 2
            m_l = jnp.mean(pooled_sq_abs)
            p_l = beta / (2 * n_in * m_l)
        else:
            p_l = beta / (2 * n_in * jnp.exp(beta))
        std = jnp.sqrt(p_l)

        key, phi_layer = _init_one_layer(key, n_in, n_out, std)
        key, psi_layer = _init_one_layer(key, n_in, n_out, std)
        phi_params.append(phi_layer)
        psi_params.append(psi_layer)

        if l < n_layers - 1:
            W_phi, b_phi = phi_layer
            W_psi, b_psi = psi_layer
            x_phi = jnp.exp(x_phi @ W_phi + b_phi)
            x_psi = jnp.exp(x_psi @ W_psi + b_psi)
    return {"phi": phi_params, "psi": psi_params}


def stresses_displacements(z, phi, dphi, ddphi, psi, dpsi):
    """Kolosov-Muskhelishvili representation (Eq. 3)."""

    sxx = jnp.real(2 * dphi - jnp.conj(z) * ddphi - dpsi)
    syy = jnp.real(2 * dphi + jnp.conj(z) * ddphi + dpsi)
    sxy = jnp.imag(jnp.conj(z) * ddphi + dpsi)
    ux = jnp.real(GAMMA * phi - z * jnp.conj(dphi) - jnp.conj(psi)) / (2 * MU)
    uy = jnp.imag(GAMMA * phi - z * jnp.conj(dphi) - jnp.conj(psi)) / (2 * MU)
    return sxx, syy, sxy, ux, uy

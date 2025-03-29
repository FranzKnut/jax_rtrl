import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn


"""
ExpExp_nu parameterization: Learn nu_log and theta_log. r = exp(-exp(nu_log)). By design r is always less than 1
"""


@jax.jit
def g_phi_params(nu_log, theta_log, eps=1e-8):
    nu = jnp.exp(nu_log)
    r = jnp.exp(-nu)
    theta = jnp.exp(theta_log)
    g = r * jnp.cos(theta)
    phi = r * jnp.sin(theta)
    norm = jnp.sqrt(1 - r**2) + eps
    return g, phi, norm


def initialize_exp_exp_r(key, shape, r_max=1, r_min=0):
    u1 = jax.random.uniform(key, shape=shape)
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    return nu_log


def initialize_theta_log(key, shape, max_phase=6.28):
    u2 = jax.random.uniform(key, shape=shape)
    theta_log = jnp.log(max_phase * u2)
    return theta_log


# Derivatives of g and phi w.r.t w_r and w_theta
@jax.jit
def d_g_phi_exp_exp_nu_params(w_r, w_theta, g, phi, norm):
    d_g_w_r = -jnp.exp(w_r) * g
    d_g_w_theta = -phi
    d_phi_w_r = -jnp.exp(w_r) * phi
    d_phi_w_theta = g
    d_norm_w_r = jnp.exp(w_r) * jnp.exp(-2 * jnp.exp(w_r)) / norm
    return d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r


@jax.jit
def linear_act(x):
    return x


@jax.jit
def drelu(x):
    return lax.select(x > 0, lax.full_like(x, 1), lax.full_like(x, 0))


@jax.jit
def dtanh(x):
    return 1 - jnp.tanh(x) ** 2


@jax.jit
def dlinear(x):
    return lax.full_like(x, 1)


act_options = {"relu": nn.relu, "tanh": nn.tanh, "linear": linear_act}

d_act = {"relu": drelu, "tanh": dtanh, "linear": dlinear}

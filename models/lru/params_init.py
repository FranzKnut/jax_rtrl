import jax
import jax.numpy as jnp

"""
Initializations functions
"""


def nu_log_init(key, shape, r_max=1, r_min=0):
    u1 = jax.random.uniform(key, shape=shape)
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    return nu_log


def theta_log_init(key, shape, max_phase=6.28):
    u2 = jax.random.uniform(key, shape=shape)
    theta_log = jnp.log(max_phase * u2)
    return theta_log


def gamma_log_init(key, shape, nu_log, theta_log):
    nu = jnp.exp(nu_log)
    theta = jnp.exp(theta_log)
    diag_lambda = jnp.exp(-nu + 1j * theta)
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


# Glorot initialization
def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization

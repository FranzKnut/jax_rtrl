"""Linear Recurrent Units built with Jax.

Stolen from the paper Real-Time Recurrent Learning using Trace Units in Reinforcement Learning
by Elelimy et. al.
NeurIPS 2024

"""

from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Any, Tuple
from functools import partial

from jax_rtrl.models.lru.params_init import gamma_log_init, matrix_init, nu_log_init, theta_log_init

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def get_lambda(nu_log, theta_log):
    nu = jnp.exp(nu_log)
    theta = jnp.exp(theta_log)
    Lambda = jnp.exp(-nu + 1j * theta)
    return Lambda


class LRU(nn.Module):
    r_max: float = 1.0
    r_min: float = 0.0
    max_phase: float = 6.28

    @nn.compact
    def __call__(self, carry, inputs):
        h_tminus1 = carry
        input_dim = inputs.shape[-1]
        hidden_dim = h_tminus1.shape[-1]

        nu_log = self.param("nu_log", nu_log_init, (1, hidden_dim), self.r_max, self.r_min)
        theta_log = self.param("theta_log", theta_log_init, (1, hidden_dim), self.max_phase)

        B_real = self.param(
            "B_real",
            partial(matrix_init, normalization=jnp.sqrt(2 * input_dim)),
            (hidden_dim, input_dim),
        )

        B_img = self.param(
            "B_img",
            partial(matrix_init, normalization=jnp.sqrt(2 * input_dim)),
            (hidden_dim, input_dim),
        )

        gamma_log = self.param("gamma_log", gamma_log_init, (1, hidden_dim), nu_log, theta_log)

        Lambda = get_lambda(nu_log, theta_log)
        B = B_real + 1j * B_img

        B_norm = B * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)

        h_t = (Lambda * h_tminus1) + (inputs @ B_norm.squeeze().transpose())

        return h_t, h_t  # carry, output


class LRULayer(nn.Module):
    d_output: int

    @nn.compact
    def __call__(self, carry, x_t):
        h_tminus1 = carry
        hidden_dim = h_tminus1.shape[-1]

        C_real = self.param(
            "C_real",
            partial(matrix_init, normalization=jnp.sqrt(hidden_dim)),
            (self.d_output, hidden_dim),
        )

        C_img = self.param(
            "C_img",
            partial(matrix_init, normalization=jnp.sqrt(hidden_dim)),
            (self.d_output, hidden_dim),
        )

        D = self.param("D", matrix_init, (self.d_output, x_t.shape[-1]))
        online_lru = LRU()
        carry, h_t = online_lru(carry, x_t)
        C = C_real + 1j * C_img
        y_t = (h_t @ C.transpose()).real + D @ x_t
        return carry, y_t  # carry, output


class BPTTLRUs(nn.Module):
    num_units: int
    d_output: int

    @nn.compact
    def __call__(self, c, xs):
        model = nn.scan(
            LRULayer,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        return model(d_output=self.d_output)(c, xs)

    def initialize_state(self, batch_size):
        hidden_init = jnp.zeros((batch_size, self.num_units), dtype=jnp.complex64)
        return hidden_init


if __name__ == "__main__":
    d_input = 3
    d_hidden = 2
    seq_len = 20
    batch_size = 10
    key = jax.random.PRNGKey(42)

    init_x = jnp.zeros((seq_len, batch_size, d_input))  # batch_size,seq_len,d_input

    # BPTT Linear RTUs
    bptt_lrtu = BPTTLRUs(num_units=d_hidden, d_output=1)
    hidden_init = bptt_lrtu.initialize_state(batch_size)
    bptt_lrtu_params = bptt_lrtu.init(key, hidden_init, init_x)

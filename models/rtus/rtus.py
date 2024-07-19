from flax import linen as nn
import jax.numpy as jnp
from typing import Any, Tuple

from online_lru.rtus.rtus_utils import act_options
from online_lru.rtus.linear_rtus import RealTimeLinearRTUs
from online_lru.rtus.non_linear_rtus import RealTimeNonLinearRTUs


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


# real-time rtus expect inputs of shape (batch_size, n_features)
"""
A Consice interface to Real-Time Non-Linear RTUs
Non-linear recurrence
"""


class RealTimeNonLinearRTUsLayer(nn.Module):
    n_hidden: int  # number of hidden features
    activation: str = "relu"

    @nn.compact
    def __call__(self, carry, x_t):
        update_gate = RealTimeNonLinearRTUs(self.n_hidden, self.activation)
        carry, h_t = update_gate(carry, x_t)
        return carry, h_t  # carry, output

    @staticmethod
    def initialize_state(batch_size, d_rec, d_input):
        hidden_init = (jnp.zeros((batch_size, d_rec)), jnp.zeros((batch_size, d_rec)))
        memory_grad_init = (
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
        )
        return (hidden_init, memory_grad_init)


"""
A Consice interface to Real-Time Linear RTUs
Linear recurrence + non-linear output
"""


class RealTimeLinearRTUsLayer(nn.Module):
    n_hidden: int  # number of hidden features
    activation: str = "relu"

    @nn.compact
    def __call__(self, carry, x_t):
        update_gate = RealTimeLinearRTUs(self.n_hidden)
        carry, (h_t_c1, h_t_c2) = update_gate(carry, x_t)
        h_t = act_options[self.activation](jnp.concatenate((h_t_c1, h_t_c2), axis=-1))
        return carry, h_t  # carry, output

    @staticmethod
    def initialize_state(batch_size, d_rec, d_input):
        hidden_init = (jnp.zeros((batch_size, d_rec)), jnp.zeros((batch_size, d_rec)))
        memory_grad_init = (
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
            jnp.zeros((batch_size, d_input, d_rec)),
        )
        return (hidden_init, memory_grad_init)

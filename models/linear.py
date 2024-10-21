from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from models.seq_util import binary_operator


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return random.normal(key=key, shape=shape, dtype=dtype) / normalization


def truncated_normal_matrix_init(key, shape, dtype=jnp.float_, normalization=1):
    return random.truncated_normal(key, -2.0, 2.0, shape, dtype) / normalization


class LinearRNN(nn.Module):
    """
    RNN layer that updates internal elegibility traces to allow online
    learning.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    seq_length: int  # time sequence length
    activation: str = "linear"  # activation function
    training_mode: str = "bptt"  # which learning algorithm that will be used
    scaling_hidden: float = 1.0  # additional scaling for the A matrix in the RNN

    def setup(self):
        # Check that desired approximation is handled
        assert self.training_mode in [
            "bptt",
            "online_spatial",
            "online_1truncated",
            "online_snap1",
        ]
        self.online = "online" in self.training_mode  # whether we compute the gradient online
        if self.online:
            self.approximation_type = self.training_mode[7:]
        else:
            self.approximation_type = "bptt"

        # Truncated normal to match haiku's initialization
        self.A = self.param(
            "A",
            partial(truncated_normal_matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_hidden, self.d_hidden),
        )
        self.B = self.param(
            "B",
            partial(matrix_init, normalization=jnp.sqrt(self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C = self.param(
            "C",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.D = self.param("D", matrix_init, (self.d_model,))
        if self.activation == "linear":
            self.act_fun = lambda x: x
        elif self.activation == "tanh":
            self.act_fun = jax.nn.tanh
        elif self.activation == "relu":
            self.act_fun = jax.nn.relu
        else:
            raise ValueError("Activation function not supported")

        # Internal variables of the model needed for updating the gradient
        if self.approximation_type in ["snap1"]:
            self.traces_A = self.variable("traces", "A", jnp.zeros, (self.seq_length, self.d_hidden, self.d_hidden))
            self.traces_B = self.variable("traces", "B", jnp.zeros, (self.seq_length, self.d_hidden, self.d_model))
            self.pert_hidden_states = self.variable(
                "perturbations",
                "hidden_states",
                partial(jnp.zeros, dtype=jnp.float32),
                (self.seq_length, self.d_hidden),
            )

    def get_hidden_states(self, inputs):
        """
        Compute the hidden states corresponding to inputs.

        Return:
            hidden_states array[T, N]
        """

        def _step(state, Bu):
            if self.training_mode in ["bptt"]:
                new_state = self.A @ self.act_fun(state) + Bu
            elif self.approximation_type in ["1truncated"]:
                new_state = self.A @ jax.lax.stop_gradient(self.act_fun(state)) + Bu
            else:
                new_state = jax.lax.stop_gradient(self.A @ self.act_fun(state)) + Bu
            return new_state, new_state

        Bu_elements = jax.vmap(lambda u: self.B @ u)(inputs)
        _, hidden_states = jax.lax.scan(_step, jnp.zeros(self.d_hidden), Bu_elements)

        return hidden_states

    def to_output(self, inputs, hidden_states):
        """
        Compute output given inputs and hidden states.

        Args:
            inputs array[T, H].
            hidden_states array[T, N].
        """
        return jax.vmap(lambda x, u: self.C @ x + self.D * u)(hidden_states, inputs)

    def __call__(self, inputs):
        """
        Forward pass. If in training mode, additionally computes the
        eligibility traces that will be needed to compute the gradient estimate
        in backward.
        """
        # Compute hidden states and output
        hidden_states = self.get_hidden_states(inputs)
        if self.approximation_type in ["snap1"]:
            # To obtain the spatially backpropagated errors sent to hidden_states
            hidden_states += self.pert_hidden_states.value
        output = self.to_output(inputs, hidden_states)

        # Update traces
        if self.approximation_type in ["snap1"]:
            # Repeat diagonal of A T times
            diags_A = jnp.repeat(jnp.diagonal(self.A)[None, ...], inputs.shape[0], axis=0)
            # Add the rho'(x) to it
            der_act_fun = jax.grad(self.act_fun)
            rho_primes = jax.vmap(lambda x: jax.vmap(der_act_fun)(x))(hidden_states)

            A_rho_prime_elements_N = jax.vmap(lambda x: jnp.outer(x, jnp.ones((self.d_hidden,))))(diags_A * rho_primes)
            A_rho_prime_elements_H = jax.vmap(lambda x: jnp.outer(x, jnp.ones((self.d_model,))))(diags_A * rho_primes)

            # Compute the trace of A
            # with tA_{t+1} = (diag A * rho'(x_t)) 1^T * tA_t + 1 rho(x_t)^T
            rho_x_elements = jax.vmap(lambda x: jnp.outer(jnp.ones((self.d_hidden,)), x))(self.act_fun(hidden_states))
            _, self.traces_A.value = jax.lax.associative_scan(
                partial(binary_operator),
                (A_rho_prime_elements_N, rho_x_elements),
            )

            # Compute the trace of B
            # with tB_{t+1} = (diag A * rho'(x_t)) 1^T * tB_t + 1 rho(x_t)^T
            u_elements = jax.vmap(lambda u: jnp.outer(jnp.ones((self.d_hidden,)), u))(inputs)
            _, self.traces_B.value = jax.lax.associative_scan(
                partial(binary_operator), (A_rho_prime_elements_H, u_elements)
            )
        return output

    def update_gradients(self, grad):
        """
        Eventually combine traces and perturbations to compute the (online) gradient.
        """
        if self.approximation_type not in ["snap1"]:
            return grad

        # We need to change the gradients for A, and B
        grad["A"] = jnp.sum(
            jax.vmap(lambda dx, trace: dx.reshape(-1, 1) * trace)(
                self.pert_hidden_states.value[1:], self.traces_A.value[:-1]
            ),
            axis=0,
        )
        grad["B"] = jnp.sum(
            jax.vmap(lambda dx, trace: dx.reshape(-1, 1) * trace)(
                self.pert_hidden_states.value[1:], self.traces_B.value[:-1]
            ),
            axis=0,
        )
        return grad

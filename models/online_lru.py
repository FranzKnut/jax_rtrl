"""https://github.com/NicolasZucchet/Online-learning-LR-dependencies"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core.frozen_dict import unfreeze
from models.linear import binary_operator_diag, matrix_init


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32, log=True):
    u = random.uniform(key=key, shape=shape, dtype=dtype)
    nu = -0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2)
    if log:
        nu = jnp.log(nu)
    return nu


def theta_init(key, shape, max_phase, dtype=jnp.float32, log=True):
    u = random.uniform(key, shape=shape, dtype=dtype)
    theta = max_phase * u
    if log:
        theta = jnp.log(theta)
    return theta


def gamma_log_init(key, lamb, log=True):
    nu, theta = lamb
    if log:
        nu = jnp.exp(nu)
        theta = jnp.exp(theta)
    diag_lambda = jnp.exp(-nu + 1j * theta)
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


# Parallel scan operations
@jax.vmap
def binary_operator_diag_spatial(q_i, q_j):
    """Same as above but stop the gradient for the recurrent connection"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, jax.lax.stop_gradient(A_j * b_i) + b_j


class LRU(nn.Module):
    """
    LRU layer that updates internal elegibility traces to allow online learning.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    seq_length: int  # time sequence length
    gamma_norm: bool = True  # use gamma normalization
    exp_param: bool = True  # exponential parametrization for lambda
    r_min: float = 0.0  # smallest eigenvalue norm
    r_max: float = 1.0  # largest eigenvalue norm
    max_phase: float = 6.28  # max phase eigenvalue
    training_mode: str = "bptt"  # which learning algorithm that will be used
    training: bool = False  # TODO remove, for debugging purposes

    def get_diag_lambda(self, nu=None, theta=None):
        """
        Transform parameters nu and theta into the diagonal of the recurrent
        Lambda matrix.

        Args:
            nu, theta array[N]: when set to their default values, None, the
                parameters will take the values of the Module.
                NOTE: these arguments are added in order to backpropagate through this
                transformation.
        """
        if nu is None:
            nu = self.nu
        if theta is None:
            theta = self.theta
        if self.exp_param:
            theta = jnp.exp(theta)
            nu = jnp.exp(nu)
        return jnp.exp(-nu + 1j * theta)

    def get_diag_gamma(self):
        """
        Transform parameters gamma_log into the diagonal terms of the modulation matrix gamma.
        """
        if self.gamma_norm:
            return jnp.exp(self.gamma_log)
        else:
            return jnp.ones((self.d_hidden,))

    def get_B(self):
        """
        Get input to hidden matrix B.
        """
        return self.B_re + 1j * self.B_im

    def get_B_norm(self):
        """
        Get modulated input to hidden matrix gamma B.
        """
        return self.get_B() * jnp.expand_dims(self.get_diag_gamma(), axis=-1)

    def to_output(self, inputs, hidden_states):
        """
        Compute output given inputs and hidden states.

        Args:
            inputs array[T, H].
            hidden_states array[T, N].
        """
        C = self.C_re + 1j * self.C_im
        D = self.D
        y = jax.vmap(lambda x, u: (C @ x).real + D * u)(hidden_states, inputs)
        return y

    def get_hidden_states(self, inputs):
        """
        Compute the hidden states corresponding to inputs

        Return:
            hidden_states array[T, N]
        """
        # Materializing the diagonal of Lambda and projections
        diag_lambda = self.get_diag_lambda()
        B_norm = self.get_B_norm()

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        elements = (Lambda_elements, Bu_elements)
        if self.training_mode == "bptt":
            _, hidden_states = jax.lax.associative_scan(binary_operator_diag, elements)
        else:
            _, hidden_states = jax.lax.associative_scan(binary_operator_diag_spatial, elements)

        return hidden_states

    def setup(self):
        # Check that desired approximation is handled
        if self.training_mode == "online_snap1":
            raise NotImplementedError("SnAp-1 not implemented for LRU")
        assert self.training_mode in [
            "bptt",
            "online_full",
            "online_full_rec",
            "online_full_rec_simpleB",
            "online_snap1",  # same as online_full
            "online_spatial",
            "online_1truncated",
            "online_reservoir",
        ]
        self.online = "online" in self.training_mode  # whether we compute the gradient online
        if self.online:
            self.approximation_type = self.training_mode[7:]

        # NOTE if exp_param is true, self.theta and self.nu actually represent the log of nu and
        # theta lambda is initialized uniformly in complex plane
        self.theta = self.param(
            "theta",
            partial(theta_init, max_phase=self.max_phase, log=self.exp_param),
            (self.d_hidden,),
        )  # phase of lambda in [0, max_phase]
        self.nu = self.param(
            "nu",
            partial(nu_init, r_min=self.r_min, r_max=self.r_max, log=self.exp_param),
            (self.d_hidden,),
        )  # norm of lambda in [r_min, r_max]
        if self.gamma_norm:
            self.gamma_log = self.param("gamma_log", partial(gamma_log_init, log=self.exp_param), (self.nu, self.theta))

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.param(
            "B_re",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C_re = self.param(
            "C_re",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.C_im = self.param(
            "C_im",
            partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.D = self.param("D", matrix_init, (self.d_model,))

        # Internal variables of the model needed for updating the gradient
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            self.pert_hidden_states = self.variable(
                "perturbations",
                "hidden_states",
                partial(jnp.zeros, dtype=jnp.complex64),
                (self.seq_length, self.d_hidden),
            )

            self.traces_gamma = self.variable("traces", "gamma", jnp.zeros, (self.seq_length, self.d_hidden))
            self.traces_lambda = self.variable("traces", "lambda", jnp.zeros, (self.seq_length, self.d_hidden))

            if self.approximation_type in ["full", "snap1", "full_rec_simpleB"]:
                self.traces_B = self.variable("traces", "B", jnp.zeros, (self.seq_length, self.d_hidden, self.d_model))

    def __call__(self, inputs):
        """
        Forward pass. If in training mode, additionally computes the eligibility traces that
        will be needed to compute the gradient estimate in backward.
        """
        # Compute hidden states and outputs
        hidden_states = self.get_hidden_states(inputs)
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            # To obtain the spatially backpropagated errors sent to hidden_states
            # NOTE: only works if pert_hidden_states is equal to 0
            hidden_states += self.pert_hidden_states.value
        output = self.to_output(inputs, hidden_states)

        # Compute and update traces if needed (i.e. if we are in online training mode)
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            Bu_elements = jax.vmap(lambda u: self.get_B() @ u)(inputs)
            # Update traces for B, lambda and gamma
            if self.approximation_type in ["1truncated"]:
                self.traces_lambda.value = hidden_states[:-1]
                self.traces_gamma.value = Bu_elements
            elif self.approximation_type in ["full", "full_rec", "full_rec_simpleB", "snap1"]:
                Lambda_elements = jnp.repeat(self.get_diag_lambda()[None, ...], inputs.shape[0], axis=0)
                # Update for trace lambda
                _, self.traces_lambda.value = jax.lax.associative_scan(
                    binary_operator_diag,
                    (Lambda_elements[:-1], hidden_states[:-1]),
                )
                # Update for trace gamma
                Bu_elements_gamma = Bu_elements
                _, self.traces_gamma.value = jax.lax.associative_scan(
                    binary_operator_diag, (Lambda_elements, Bu_elements_gamma)
                )

            # Update trace for B
            if self.approximation_type in ["full", "snap1"]:
                full_Lambda_elements = jnp.repeat(
                    jnp.expand_dims(self.get_diag_lambda(), axis=-1)[None, ...],
                    inputs.shape[0],
                    axis=0,
                )  # same as left multiplying by diag(lambda), but same shape as B (to allow for
                #    element-wise multiplication in the associative scan)
                gammau_elements = jax.vmap(lambda u: jnp.outer(self.get_diag_gamma(), u))(inputs).astype(jnp.complex64)
                _, self.traces_B.value = jax.lax.associative_scan(
                    binary_operator_diag,
                    (full_Lambda_elements, gammau_elements + 0j),
                )
            elif self.approximation_type in ["full_rec_simpleB"]:
                self.traces_B.value = jax.vmap(lambda u: jnp.outer(self.get_diag_gamma(), u))(inputs).astype(
                    jnp.complex64
                )
        return output

    def update_gradients(self, grad):
        """
        Eventually combine traces and perturbations to compute the (online) gradient.
        """
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        # We need to change the gradients for lambda, gamma and B
        # The others are automatically computed with spatial backpropagation
        # NOTE: self.pert_hidden_states contains dL/dhidden_states

        # Grads for lambda
        delta_lambda = jnp.sum(self.pert_hidden_states.value[1:] * self.traces_lambda.value, axis=0)
        _, dl = jax.vjp(
            lambda nu, theta: self.get_diag_lambda(nu=nu, theta=theta),
            self.nu,
            self.theta,
        )
        grad_nu, grad_theta = dl(delta_lambda)
        grad["nu"] = grad_nu
        grad["theta"] = grad_theta

        # Grads for gamma if needed
        if self.gamma_norm:
            delta_gamma = jnp.sum((self.pert_hidden_states.value * self.traces_gamma.value).real, axis=0)
            # as dgamma/dgamma_log = exp(gamma_log) = gamma
            grad["gamma_log"] = delta_gamma * self.get_diag_gamma()

        # Grads for B
        if self.approximation_type in ["snap1", "full", "full_rec_simpleB"]:
            grad_B = jnp.sum(
                jax.vmap(lambda dx, trace: dx.reshape(-1, 1) * trace)(
                    self.pert_hidden_states.value, self.traces_B.value
                ),
                axis=0,
            )
            grad["B_re"] = grad_B.real
            grad["B_im"] = -grad_B.imag  # Comes from the use of Writtinger derivatives

        return grad

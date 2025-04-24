"""Linear Recurrent Units built with Jax.

author: jlemmel
Stolen from the paper Real-Time Recurrent Learning using Trace Units in Reinforcement Learning
by Elelimy et. al.
NeurIPS 2024
"""

from functools import partial
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
from flax import linen as nn

from ..linear import binary_operator
from ..seq_util import binary_operator_reset

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def get_lambda(nu_log, theta_log):
    """Construct lambda from nu and theta."""
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    return Lambda


def get_B_norm(B_real, B_img, gamma_log):
    """Get modulated input to hidden matrix gamma B."""
    return (B_real + 1j * B_img) * jnp.exp(jnp.expand_dims(gamma_log, axis=-1))


def nu_log_init(key, shape, r_max=1, r_min=0):
    """Initialize nu log as log(-0.5 log(x (r_max^2 - r_min^2) + r_min^2), x ~ U[0, 1]."""
    u1 = jax.random.uniform(key, shape=shape)
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    return nu_log


def theta_log_init(key, shape, max_phase=6.28):
    """Initialize theta log as log(max_phase * x), x ~ U[0, 1]."""
    u2 = jax.random.uniform(key, shape=shape)
    theta_log = jnp.log(max_phase * u2)
    return theta_log


def gamma_log_init(key, shape, nu_log, theta_log):
    """Initialize gamma log from nu and theta."""
    nu = jnp.exp(nu_log)
    theta = jnp.exp(theta_log)
    diag_lambda = jnp.exp(-nu + 1j * theta)
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    """Glorot initialization."""
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


class LRUCell(nn.Module):
    """Linear Recurrent Unit Cell."""

    d_hidden: int
    r_max: float = 1.0
    r_min: float = 0.0
    max_phase: float = 6.28
    """
    grad memory: dh_{t-1}/d lambda, dh_{t-1}/d gamma #1,
                 dh_c_{t-1}/d B #2
    """

    def setup(self):
        """Create parameters."""
        self.nu_log = self.param(
            "nu_log", nu_log_init, (self.d_hidden,), self.r_max, self.r_min
        )
        self.theta_log = self.param(
            "theta_log", theta_log_init, (self.d_hidden,), self.max_phase
        )
        self.gamma_log = self.param(
            "gamma_log", gamma_log_init, (self.d_hidden,), self.nu_log, self.theta_log
        )

    @nn.compact
    def __call__(self, carry, inputs, resets=None):
        """Compute output for given input sequence and optionally reset hidden state.

        Assumes that the input is a sequence of shape (steps, input_dim) or a single vector of shape (input_dim,).
        """
        h_tminus1 = carry
        _is_sequence = len(inputs.shape) > 1
        input_dim = inputs.shape[-1]
        hidden_dim = h_tminus1.shape[-1]

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

        Lambda = get_lambda(self.nu_log, self.theta_log)
        B_norm = get_B_norm(B_real, B_img, self.gamma_log)

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        inputs = jnp.reshape(inputs, (-1, input_dim))
        Lambda_elements = jnp.repeat(Lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        if resets is None:
            _, hidden_states = jax.lax.associative_scan(
                binary_operator, (Lambda_elements, Bu_elements)
            )
        else:
            _, hidden_states, _ = jax.lax.associative_scan(
                binary_operator_reset,
                (Lambda_elements, Bu_elements, resets.astype(jnp.int32)),
            )
        return hidden_states[-1], hidden_states if _is_sequence else hidden_states[-1]

    def _to_lambda(self, x):
        return get_lambda(self.nu_log, self.theta_log)


class OnlineLRUCell(nn.RNNCellBase):
    """LRU cell with online gradient computation."""

    d_hidden: int
    plasticity: str = "bptt"

    @nn.compact
    def __call__(self, carry, x_t, resets=None, force_trace_compute=False):
        """Call the LRU cell with online gradient computation.

        Parameters
        ----------
        carry : any
            LRU cell state.
        x_t : array
            Input sequence.
        resets : array, optional
            Boolean array with length of input sequence, by default None
        force_trace_compute : bool, optional
            Force updating the gradient traces even when autograd is not required, by default False

        Returns
        -------
        Tuple : carry, output
        """
        model_fn = LRUCell(self.d_hidden)
        if self.plasticity == "bptt":
            return model_fn(carry, x_t, resets=resets)

        def _trace_update(carry, _p, x_t):
            h, grad_memory = carry
            Lambda = get_lambda(_p["nu_log"], _p["theta_log"])
            B = _p["B_real"] + 1j * _p["B_img"]

            new_grad_lambda = Lambda * grad_memory[0] + h
            new_grad_gamma = (
                Lambda * grad_memory[1] + (x_t @ jnp.swapaxes(B, -1, -2)).squeeze()
            )
            new_grad_B = (jnp.expand_dims(Lambda, axis=-1)) * grad_memory[
                2
            ] + jnp.outer(_p["gamma_log"], x_t)

            return (
                new_grad_lambda,
                new_grad_gamma,
                new_grad_B,
            )

        def f(mdl, carry, x_t):
            h, *traces = carry
            # Vmap in case of batched carry
            if len(carry[0].shape) > 1:
                h_next, out = jax.vmap(mdl)(h, x_t, resets=resets)
            else:
                h_next, out = mdl(h, x_t, resets=resets)
            if force_trace_compute:
                traces = _trace_update(carry, mdl.variables["params"], x_t)
            return (h_next, *traces), out

        def fwd(mdl: LRUCell, carry, x_t):
            f_out, vjp_func = nn.vjp(f, mdl, carry, x_t)
            _, vjp_to_lambda = nn.vjp(lambda m, x: m._to_lambda(x), mdl, x_t)
            traces = _trace_update(carry, mdl.variables["params"], x_t)
            return f_out, (
                vjp_func,
                traces,
                vjp_to_lambda,
                mdl.gamma_log,
            )  # output, residual

        def bwd(residuals, y_t):
            # y_t =(partial{output}/partial{h_{t}},ignore the rest
            # grad_memory = \partial{h_{t-1}} \partial{lambda},
            # \partial{h_{t-1},c1} \partial{gamma},\partial{h_{t-1}} \partial{B}
            return self.rtrl_gradient(residuals, y_t)

        online_lru_cell_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        return online_lru_cell_grad(model_fn, carry, x_t)  # carry, output

    @staticmethod
    def rtrl_gradient(residuals, y_t, plasticity="rtrl"):
        """Compute RTRL gradient."""
        vjp_func, new_grad_memory, vjp_to_lambda, gamma_log = residuals

        if plasticity != "rtrl":
            raise ValueError("Unknown plasticity for LRU: " + plasticity)
        params_t, *inputs_t = vjp_func(y_t)
        d_output_d_h = y_t[1][0]

        d_output_d_lambda = d_output_d_h * new_grad_memory[0]
        d_params_rec = vjp_to_lambda(d_output_d_lambda)[0]
        correct_nu_log, correct_theta_log = (
            d_params_rec["params"]["nu_log"],
            d_params_rec["params"]["theta_log"],
        )

        correct_gamma_log = (d_output_d_h * new_grad_memory[1]).real * jnp.exp(
            gamma_log
        )
        grad_B = jnp.expand_dims(d_output_d_h, -1) * new_grad_memory[2]
        # correct_B_re = (jnp.expand_dims(d_output_d_h,-1) * new_grad_memory[2]).real
        # correct_B_img = (jnp.expand_dims(d_output_d_h,-1) * new_grad_memory[2]).imag
        params_t1 = flax.core.unfreeze(params_t)
        params_t1["params"]["nu_log"] = correct_nu_log
        params_t1["params"]["theta_log"] = correct_theta_log
        params_t1["params"]["gamma_log"] = correct_gamma_log.real
        params_t1["params"]["B_real"] = grad_B.real  # jnp.sum(correct_B_re,0)
        params_t1["params"]["B_img"] = -grad_B.imag  # jnp.sum(correct_B_img,0)
        return params_t1, *inputs_t


class OnlineLRULayer(nn.RNNCellBase):
    """OnlineLRU layer with linear projection afterwards."""

    d_output: int
    d_hidden: int = None
    plasticity: str = "bptt"

    def __init__(self, d_output, d_hidden=None, plasticity="bptt", **kwargs):
        """Initialize the model with the specified output dimension, hidden dimension, and plasticity type.

        Args:
            d_output (int): The dimension of the output layer.
            d_hidden (int, optional): The dimension of the hidden layer. Defaults to the value of d_output if not provided.
            plasticity (str): The type of plasticity to use. Defaults to "bptt".
            kwargs: Additional arguments for the LRU cell.
        """
        self.d_output = d_output
        self.d_hidden = d_hidden or d_output
        self.plasticity = plasticity
        super().__init__(**kwargs)

    @nn.compact
    def __call__(self, carry, x_t, *args, **kwargs):
        """Apply the LRU cell to the input and computes the output projections.

        Args:
            carry (Union[Array, Tuple[Array, ...]]): The carry state from the previous time step.
            x_t (Array): The input at the current time step.
            resets (Optional[Array]): Optional reset signals.
            **args: Additional arguments for the LRU cell.

        Returns:
            Tuple[Union[Array, Tuple[Array, ...]], Array]: The updated carry state and the output at the current time step.
        """
        h_tminus1 = carry if self.plasticity == "bptt" else carry[0]
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

        online_lru = OnlineLRUCell(self.d_hidden or self.d_output, self.plasticity)
        carry, h_t = online_lru(carry, x_t, *args, **kwargs)
        C = C_real + 1j * C_img
        y_t = (h_t @ C.transpose()).real + x_t @ D.transpose()
        return carry, y_t

    @staticmethod
    def rtrl_gradient(*args, **kwargs):
        """Compute the Real-Time Recurrent Learning (RTRL) gradient for the LRU cell."""
        return OnlineLRUCell.rtrl_gradient(*args, **kwargs)

    def initialize_carry(self, rng, input_shape):
        """Initialize the carry state for the LRU cell including gradient traces."""
        d_hidden = self.d_hidden or self.d_output
        batch_size = input_shape[0:1] if len(input_shape) > 1 else ()
        hidden_init = jnp.zeros((*batch_size, d_hidden), dtype=jnp.complex64)
        if self.plasticity == "bptt":
            return hidden_init
        memory_grad_init = (
            jnp.zeros((*batch_size, d_hidden), dtype=jnp.complex64),
            jnp.zeros((*batch_size, d_hidden), dtype=jnp.complex64),
            jnp.zeros((*batch_size, d_hidden, input_shape[-1]), dtype=jnp.complex64),
        )
        return (hidden_init, memory_grad_init)


if __name__ == "__main__":
    input_dim = 2
    d_hidden = 5
    seq_len = 2
    batch_size = 2

    inputs = jnp.ones((batch_size, input_dim), dtype=jnp.float32)
    model = OnlineLRULayer(d_hidden=d_hidden, d_output=1)
    h_init, grad_momory_init = model.initialize_carry(None, (batch_size, input_dim))
    params = model.init(jax.random.PRNGKey(0), (h_init, grad_momory_init), inputs[0])
    print(params)

    test_x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, batch_size, input_dim))

    # print(test_x)
    def _apply_rtrl(rt_rtu_params, test_x, rt_hidden_init, mem_grad_init):
        rt_carry = (rt_hidden_init, mem_grad_init)
        hs_c1 = []
        for i in range(seq_len):
            rt_carry, out = jax.vmap(partial(model.apply, rt_rtu_params))(
                rt_carry, test_x[i, :, :]
            )
            hs_c1.append(out)
        error = (1 - jnp.mean(jnp.stack(hs_c1))) ** 2
        return error

    rtrl_out = _apply_rtrl(params, test_x, h_init, grad_momory_init)
    grad_rtrl = jax.grad(_apply_rtrl)(params, test_x, h_init, grad_momory_init)
    print(rtrl_out)
    print(grad_rtrl)

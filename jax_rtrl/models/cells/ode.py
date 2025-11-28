from dataclasses import field
from typing import Literal
from chex import PRNGKey
import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_rtrl.models.wirings import make_mask_initializer
from jax_rtrl.util.jax_util import tree_stack


class ODECell(nn.RNNCellBase):
    """Simple CTRNN cell."""

    num_units: int
    dt: float = 1.0
    T: float = 1.0
    solver: Literal["euler"] = "euler"
    wiring: str | None = None
    wiring_kwargs: dict = field(default_factory=dict)

    def _f(self, h, x):
        """Compute the derivative of the state."""
        raise NotImplementedError("_f must be implemented in subclasses.")

    def _make_params(self, x):
        """Create parameters for ODECell."""
        # Define params
        w_shape = (self.num_units, x.shape[-1] + self.num_units + 1)
        if self.wiring is not None:
            self.variable(
                "wiring",
                "mask",
                make_mask_initializer(self.wiring, **self.wiring_kwargs),
                self.make_rng() if self.has_rng("params") else None,
                w_shape,
                int,
            ).value

    def solve(self, h, x, return_sequences=False):
        """Solve ODE over time T with step dt."""
        outs = []
        if self.solver == "euler":
            # Euler integration steps with dt
            for _step in jnp.arange(0, self.T, self.dt):
                h_dot = self._f(h, x)
                h = jax.tree.map(lambda a, b: a + b * self.dt, h, h_dot)
                if return_sequences:
                    outs.append(h)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        if return_sequences:
            return tree_stack(outs)
        return h

    @nn.compact
    def __call__(self, h, x, return_sequences=False):  # noqa
        """Call ODE solver."""
        # Initialize hidden state
        if h is None:
            h = self.initialize_carry(self.make_rng(), x.shape)
        # Initialize parameters
        self._make_params(x)
        # Solve
        out = self.solve(h, x, return_sequences)
        return out, out

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


class OnlineODECell(ODECell):
    """Online CTRNN module."""

    plasticity: str = "rtrl"

    @nn.compact
    def __call__(self, carry, x, return_sequences=False):  # noqa
        """Call ODE solver."""
        # Initialize hidden state
        if carry is None:
            carry = self.initialize_carry(self.make_rng(), x.shape)
        # Initialize parameters
        self._make_params(x)
        # Solve ODE
        return self.solve(carry, x, return_sequences)

    def solve(self, carry, x, return_sequences=False, force_trace_compute=False):
        """Solve ODE over time T with step dt."""
        if self.plasticity == "bptt":
            out = super().solve(carry, x, return_sequences)
            return out, out
        outs = []
        if self.solver == "euler":

            def f(mdl, _carry, _x):
                h = _carry[0]
                h_dot = mdl._f(h, _x)
                h_next = jax.tree.map(lambda a, b: a + b * mdl.dt, h, h_dot)
                if force_trace_compute:
                    traces = mdl._trace_update(_carry, mdl.variables["params"], _x)
                else:
                    traces = _carry[1:]
                return (h_next, *traces), h_next

            def fwd(mdl, _carry, _x):
                """Forward pass with tmp for backward pass."""
                h = _carry[0]
                out = jax.tree.map(lambda a, b: a + b * mdl.dt, h, mdl._f(h, _x))
                # during init, traces are None -> avoid computing them
                if all([(_c is None) for _c in _carry[1:]]):
                    print("Skipping trace computation during init")
                    traces = _carry[1:]
                else:
                    traces = mdl._trace_update(_carry, mdl.variables["params"], _x)
                return (
                    (out, *traces),
                    (out, *traces) if force_trace_compute else out,
                ), (
                    out,
                    *traces,
                )

            def bwd(tmp, y_bar):
                """Backward pass using RTRL."""
                # carry, jp, jx, hebb = tmp
                df_dy = y_bar[-1]
                df_dy += y_bar[-2][0]  # Also include carry grad
                grads_p, grads_x = self.online_gradient(
                    tmp, df_dy, plasticity=self.plasticity
                )
                # grads_p['W'] += hebb
                carry = jax.tree.map(jnp.zeros_like, tmp)
                return ({"params": grads_p}, carry, grads_x)

            # Euler integration steps with dt
            f_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
            for _step in jnp.arange(0, self.T, self.dt):
                carry, h = f_grad(self, carry, x)
                if return_sequences:
                    outs.append((carry, h))
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        if return_sequences:
            return tree_stack(outs)
        return carry, h

    def _trace_update(self, carry, _p, x):
        raise NotImplementedError("_trace_update must be implemented in subclasses.")

    @staticmethod
    def online_gradient(carry, df_dy, plasticity="rflo"):
        """Compute RTRL gradient."""
        h, jp, jx = carry
        if plasticity == "rtrl":
            grads_p = jax.tree.map(lambda t: df_dy @ t, jp)
        else:
            grads_p = jax.tree.map(lambda t: (df_dy.T * t.T).T, jp)
        if len(df_dy.shape) > 1:
            # has batch dim
            grads_p = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads_p)
        grads_x = jnp.einsum("...h,...hi->...i", df_dy, jx)
        return grads_p, grads_x

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the carry with jacobian traces."""
        h = super().initialize_carry(rng, input_shape)
        if self.plasticity == "bptt":
            return h

        # jh = jnp.zeros(h.shape[:-1] + (h.shape[-1], h.shape[-1]))
        jx = jnp.zeros(input_shape[:-1] + (h.shape[-1], input_shape[-1]))

        # initialize to get the parameter shapes
        _h = h
        leading_shape = input_shape[:-1]
        for _ in leading_shape:
            # Reduce to single example for init
            _h = _h[0]

        params = self.lazy_init(rng, (_h, None, None), jnp.zeros(input_shape[-1:]))

        if self.plasticity == "rtrl":
            leading_shape = h.shape

        # Initialize the jacobian traces
        jp = jax.tree.map(
            lambda x: jnp.zeros(leading_shape + x.shape), params["params"]
        )
        return h, jp, jx


def snap0(cell, carry, params, x, ode):
    """Compute jacobian trace update for RTRL."""
    h, jp, jx = carry
    # immediate jacobian (only this step)
    df_dw, df_dx = jax.jacrev(ode, argnums=[0, 2])(params, h, x)
    return jax.tree.map(lambda p: p * cell.dt, (df_dw, df_dx))


def rtrl(cell, carry, params, x, ode):
    """Compute jacobian trace update for RTRL."""
    h, jp, jx = carry

    # immediate jacobian (this step)
    df_dw, df_dh, df_dx = jax.jacrev(ode, argnums=[0, 1, 2])(params, h, x)

    # dh/dh = d(h + f(h) * dt)/dh = I + df/dh * dt
    dh_dh = df_dh * cell.dt + jnp.identity(cell.num_units)

    # jacobian trace (previous step * dh_h)
    comm = jax.tree.map(lambda p: jnp.matmul(dh_dh, p), jp)
    comm_x = jnp.matmul(dh_dh, jx)

    def rtrl_step(rec, dh):
        return rec + dh * cell.dt

    # Update dh_dw approximation
    dh_dw = jax.tree.map(rtrl_step, comm, df_dw)

    # Update dh_dx approximation
    dh_dx = jax.tree.map(rtrl_step, comm_x, df_dx)
    return dh_dw, dh_dx

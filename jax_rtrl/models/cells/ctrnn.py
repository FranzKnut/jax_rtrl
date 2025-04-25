"""CTRNN implementation."""

from dataclasses import field
from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.interpreters
import jax.numpy as jnp
import jax.random as jrand
from chex import PRNGKey
from flax.linen import nowrap

from ..wirings import make_mask_initializer


def ctrnn_ode(params, h, x):
    """Compute euler integration step or CTRNN ODE."""
    W, tau = params
    # Concatenate input and hidden state and ones for bias
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    u = y @ W.T
    act = jnp.tanh(u)
    # Subtract decay and divide by tau
    return (act - h) / tau


def ctrnn_tg(params, h, x):
    """Compute euler integration step or CTRNN ODE."""
    W, W_tau = params
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    act = jnp.tanh(y @ W.T)
    # tau = jax.nn.softplus(y @ W_tau.T) + 1
    # tau = jax.nn.softmax(y @ W_tau.T)
    tau = jax.nn.sigmoid(y @ W_tau.T)
    # Subtract decay and divide by tau
    return (act - h) * tau


class CTRNNCell(nn.RNNCellBase):
    """Simple CTRNN cell."""

    num_units: int
    dt: float = 1.0
    ode_type: str = "murray"
    wiring: str | None = "fully_connected"
    wiring_kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, h, x):  # noqa
        """Compute euler integration step or CTRNN ODE."""
        if h is None:
            h = self.initialize_carry(self.make_rng(), x.shape)
        # Define params
        w_shape = (self.num_units, x.shape[-1] + self.num_units + 1)

        def _initializer(key, *_):
            _w_in = nn.initializers.orthogonal(0.1)(key, (self.num_units, x.shape[-1]))
            _w_rec = nn.initializers.orthogonal(0.1)(
                key, (self.num_units, self.num_units)
            )
            _bias = jnp.zeros((self.num_units, 1))
            return jnp.concatenate([_w_in, _w_rec, _bias], axis=-1)

        W = self.param("W", _initializer, w_shape)

        if self.wiring is not None:
            mask = self.variable(
                "wiring",
                "mask",
                make_mask_initializer(self.wiring, **self.wiring_kwargs),
                self.make_rng() if self.has_rng("params") else None,
                w_shape,
                int,
            ).value
            W = jax.lax.stop_gradient(mask) * W
        # Compute updates
        if self.ode_type == "murray":
            tau = self.param(
                "tau", partial(jrand.uniform, minval=3, maxval=8), (self.num_units,)
            )
            df_dt = ctrnn_ode((W, tau), h, x)
        elif self.ode_type == "tg":
            W_tau = self.param(
                "W_tau",
                nn.initializers.he_normal(in_axis=-1, out_axis=-2),
                (self.num_units, x.shape[-1] + self.num_units + 1),
            )
            df_dt = ctrnn_tg((W, W_tau), h, x)
        # Euler integration step with dt
        out = jax.tree.map(lambda a, b: a + b * self.dt, h, df_dt)
        return out, out

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        return jnp.zeros(tuple(input_shape)[:-1] + (self.num_units,))

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @staticmethod
    def clip_tau(params):
        """HACK: clip tau to > 1.0"""
        params["tau"] = jnp.clip(params["tau"], min=1.0)
        return params


def rtrl_ctrnn(cell, carry, params, x, ode=ctrnn_ode):
    """Compute jacobian trace update for RTRL."""
    h, jp, jx = carry

    # immediate jacobian (this step)
    W, tau = params.values()
    df_dw, df_dh, df_dx = jax.jacrev(ode, argnums=[0, 1, 2])((W, tau), h, x)
    df_dw = {"W": df_dw[0], "tau": df_dw[1]}  # , "b": df_dw[1]

    # dh/dh = d(h + f(h) * dt)/dh = I + df/dh * dt
    dh_dh = df_dh * cell.dt  # + jnp.identity(cell.num_units)

    # jacobian trace (previous step * dh_h)
    comm = jax.tree.map(lambda p: jnp.tensordot(dh_dh, p, axes=1), jp)

    def rtrl_step(p, rec, dh):
        return p + rec + dh * cell.dt

    # Update dh_dw approximation
    dh_dw = jax.tree.map(rtrl_step, jp, comm, df_dw)

    # Update dh_dx approximation
    dh_dx = df_dx + jx

    return dh_dw, dh_dx


def hebbian(pre, post):
    return jnp.outer(post, pre)


def rflo_murray(cell: CTRNNCell, carry, params, x):
    """Compute jacobian trace for RFLO."""
    h, jp, jx = carry
    W, tau = params.values()

    jw = jp["W"]
    jtau = jp["tau"]

    # immediate jacobian (this step)
    v = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    u = v @ W.T
    # df_dh = jax.jacfwd(jax.nn.tanh)(u)
    # df_dh = jax.jacrev(jax.nn.tanh)(u)
    df_dh = 1 - jnp.tanh(u) ** 2
    # post = jnp.tanh(u)

    # hebb = hebbian(v, post)

    # Outer product the get Immediate Jacobian
    # M_immediate = jnp.einsum('ij,k', df_dh, v)
    M_immediate = df_dh[..., None] * v[None]

    # Update eligibility traces
    jw += (1 / tau)[:, None] * (M_immediate - jw)
    dh_dtau = ((h - jnp.tanh(u)) / tau) - jtau
    jtau += dh_dtau / tau

    df_dw = {"W": jw, "tau": jtau}
    dh_dx = jnp.outer(
        df_dh,
        (
            jnp.concatenate(
                [jnp.ones_like(x), jnp.zeros_like(h), jnp.zeros(x.shape[:-1] + (1,))],
                axis=-1,
            )
            @ W.T
        )[..., : x.shape[-1]],
    )
    # dh_dh = df_dh @ W.T[x.shape[-1]:x.shape[-1]+h.shape[-1]]
    return df_dw, dh_dx  # , hebb


# def rflo_tg(cell: CTRNNCell, carry, params, x):
#     """Compute jacobian trace for RFLO."""
#     h, jp, jx = carry
#     W, tau = params

#     jw = jp['params']['W']
#     jtau = jp['params']['W_tau']

#     # immediate jacobian (this step)
#     v = jnp.concatenate([x, h, jnp.ones(x.shape[:-1]+(1,))])
#     u = W @ v
#     # df_dh = jax.jacfwd(jax.nn.tanh)(u)
#     # df_dh = jax.jacrev(jax.nn.tanh)(u)
#     df_dh = jnp.eye(u.shape[-1]) * (1-jnp.tanh(u)**2)

#     # Outer product the get Immediate Jacobian
#     # M_immediate = jnp.einsum('ij,k', df_dh, v)
#     M_immediate = df_dh[..., None] * v[None, None]

#     # Update eligibility traces
#     jw += (1 / tau)[:, None, None] * (M_immediate - jw)
#     dh_dtau = ((h - jnp.tanh(u)) * 1 / tau) * jnp.eye(tau.shape[-1]) - jtau
#     jtau += (1 / tau)[:, None] * dh_dtau

#     df_dw = {"params": {"W": jw, "tau": jtau}}
#     dh_dx = jx
#     return df_dw, dh_dx


class OnlineCTRNNCell(CTRNNCell):
    """Online CTRNN module."""

    plasticity: str = "rflo"

    @nn.compact
    def __call__(self, carry, x, force_trace_compute=False):  # noqa
        if carry is None:
            carry = self.initialize_carry(self.make_rng(), x.shape)

        if self.plasticity == "bptt":
            return CTRNNCell.__call__(self, carry, x)

        def _trace_update(carry, _p, x):
            if self.plasticity == "rtrl":
                traces = rtrl_ctrnn(self, carry, _p, x)
            elif self.plasticity == "rflo":
                traces = rflo_murray(self, carry, _p, x)
            else:
                raise ValueError(f"Plasticity mode {self.plasticity} not recognized.")
            return traces

        def f(mdl, carry, x):
            h_next, out = CTRNNCell.__call__(mdl, carry[0], x)
            if force_trace_compute:
                traces = _trace_update(carry, mdl.variables["params"], x)
            else:
                traces = carry[1:]
            return (h_next, *traces), out

        def fwd(mdl, carry, x):
            """Forward pass with tmp for backward pass."""
            out, _ = CTRNNCell.__call__(mdl, carry[0], x)
            traces = _trace_update(carry, mdl.variables["params"], x)
            return (
                (out, *traces),
                (out, *traces) if force_trace_compute else out,
            ), (out, *traces)

        def bwd(tmp, y_bar):
            """Backward pass using RTRL."""
            # carry, jp, jx, hebb = tmp
            df_dy = y_bar[-1]
            grads_p, grads_x = self.rtrl_gradient(
                tmp, df_dy, plasticity=self.plasticity
            )
            # grads_p['W'] += hebb
            carry = jax.tree.map(jnp.zeros_like, tmp)  # [:-1]
            return ({"params": grads_p}, carry, grads_x)

        f_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        return f_grad(self, carry, x)

    @staticmethod
    def rtrl_gradient(carry, df_dy, plasticity="rflo"):
        """Compute RTRL gradient."""
        h, jp, jx = carry
        if plasticity == "rflo":
            grads_p = jax.tree.map(lambda t: (df_dy.T * t.T).T, jp)
        else:
            grads_p = jax.tree.map(lambda t: df_dy @ t, jp)
        if len(df_dy.shape) > 1:
            # has batch dim
            grads_p = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads_p)
        grads_x = jnp.einsum("...h,...hi->...i", df_dy, jx)
        return grads_p, grads_x

    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry with jacobian traces."""
        h = super().initialize_carry(rng, input_shape)
        if self.plasticity == "bptt":
            return h

        # jh = jnp.zeros(h.shape[:-1] + (h.shape[-1], h.shape[-1]))
        jx = jnp.zeros(h.shape[:-1] + (h.shape[-1], input_shape[-1]))

        # HACK: if we are inside a batched setting, we need to replicate for self.init
        _h = h
        if hasattr(rng, "_trace") and hasattr(rng._trace, "axis_data"):
            _outer_batch_size = rng._trace.axis_data.size
            _h = jnp.tile(h, (_outer_batch_size,)+ (1,) * len(h.shape))
            _h = _h.reshape(h.shape[:-1]+ ( _outer_batch_size, -1))
            input_shape = input_shape[:-1] + (_outer_batch_size,) + input_shape[-1:]
        # Lazy initialize to get the parameter shapes
        params = self.lazy_init(
            rng,
            (_h, None, None),
            jnp.zeros(input_shape),
        )
        # Now we also have to "unbatch" the params
        if hasattr(rng, "_trace"):
            params = jax.tree.map(lambda x: x[0], params)
            
        # Initialize the jacobian traces
        leading_shape = h.shape[:-1] if self.plasticity == "rflo" else h.shape
        jp = jax.tree.map(
            lambda x: jnp.zeros(leading_shape + x.shape), params["params"]
        )
        return h, jp, jx


if __name__ == "__main__":
    import numpy as np
    import optax

    import matplotlib.pyplot as plt

    key = jrand.PRNGKey(0)
    key, key_model, key_data, key_train = jrand.split(key, 4)
    x = jnp.linspace(0, 5 * np.pi, 100)[:, None]
    y = jnp.sin(x) + 2

    cell = CTRNNCell(32)
    carry = cell.initialize_carry(jrand.PRNGKey(0), (1,))
    params = cell.init(jrand.PRNGKey(0), carry, x[0])

    def loss_mse(y_hat, _y):
        """MSE loss function."""
        return jax.numpy.mean((y_hat - _y) ** 2)

    def loss_rnn(p, c, __x, __y):
        """RNN loss."""
        c, y_hat = cell.apply(p, c, __x)
        return loss_mse(y_hat, __y)

    jax.grad(loss_rnn)(params, carry, x[0], y[0])

    model = nn.Sequential(
        [
            nn.RNN(cell),
            nn.Dense(1),
        ]
    )

    params = model.init(key_model, x, mutable=True)

    def loss_mlp(p, __x, __y):
        """MLP loss function."""
        y_hat = model.apply(p, __x)
        return loss_mse(y_hat, __y)

    loss_mlp(params, x, y)

    def print_progress(i, loss):
        """Print inside jit."""
        if i % 1000 == 0:
            print(f"Iteration {i} | Loss: {loss:.3f}")

    def train(_loss_fn, _params, data, _key, num_steps=10_000, lr=1e-4, batch_size=64):
        """Train network. We use Stochastic Gradient Descent with a constant learning rate."""
        _x, _y = data
        optimizer = optax.lion(lr)
        opt_state = optimizer.init(_params)

        def step(carry, n):
            __params, _opt_state, _key = carry
            _key, key_batch = jrand.split(_key)
            # batch = jrand.choice(key_batch, jnp.hstack([_x, _y]), (batch_size,))
            current_loss, grads = jax.value_and_grad(_loss_fn)(__params, _x, _y)
            updates, _opt_state = optimizer.update(grads, _opt_state, __params)
            __params = optax.apply_updates(__params, updates)
            jax.debug.callback(print_progress, n, current_loss)
            return (__params, _opt_state, _key), current_loss

        (_params, *_), _losses = jax.lax.scan(
            step, (_params, opt_state, _key), jnp.arange(num_steps, dtype=np.int32)
        )
        print(f"Final loss: {_losses[-1]:.3f}")
        return _params, _losses

    key, key_train = jrand.split(key_data)
    params, losses = train(loss_mlp, params, (x, y), key_train)

    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)

    # Plot the trained model output
    plt.subplot(1, 2, 2)
    y_hat = model.apply(params, x)
    plt.plot(x, y, label="target")
    plt.plot(x, y_hat, label="trained")
    plt.legend()
    plt.show()

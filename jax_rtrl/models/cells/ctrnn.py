"""CTRNN implementation."""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
from chex import PRNGKey

from jax_rtrl.models.cells.ode import ODECell, OnlineODECell, rtrl, snap0
from jax_rtrl.util.jax_util import (
    set_matching_leaves,
    get_matching_leaves,
)

## CTRNN ODE functions


def ctrnn_ode(params, h, x):
    """Compute euler integration step or CTRNN ODE."""
    # Concatenate input and hidden state and ones for bias
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    u = y @ params["W"].T
    act = jnp.tanh(u)
    # Subtract decay and divide by tau
    return (act - h) / params["tau"]


def ctrnn_ode_tau_softplus(params, h, x, min_tau=1.0):
    """Compute euler integration step or CTRNN ODE."""
    # Concatenate input and hidden state and ones for bias
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    u = y @ params["W"].T
    act = jnp.tanh(u)
    # Subtract decay and divide by tau
    return (act + params["h0"] - h) / (jax.nn.softplus(params["tau"]) + min_tau)


def ctrnn_tg(params, h, x):
    """Compute euler integration step or CTRNN ODE."""
    W, W_tau = params
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    act = jnp.tanh(y @ W.T)
    tau = jax.nn.softplus(y @ W_tau.T) + 1
    # tau = jax.nn.softmax(y @ W_tau.T)
    # tau = jax.nn.sigmoid(y @ W_tau.T)
    # Subtract decay and divide by tau
    return (act - h) * tau


def clip_tau(params, tau_min=1.0):
    """HACK: clip tau to >= tau_min."""
    return set_matching_leaves(
        params,
        ".*tau.*",
        jax.tree.map(
            partial(jnp.clip, min=tau_min), get_matching_leaves(params, ".*tau.*")
        ),
    )


class CTRNNCell(ODECell):
    """Simple CTRNN cell."""

    # num_units: int
    ode_type: str = "tau_softplus"
    # wiring: str | None = None
    # wiring_kwargs: dict = field(default_factory=dict)
    tau_min: float = 1.0  # minimum value for tau used in ctrnn_ode_tau_softplus

    def _make_params(self, x):
        # Define params
        ODECell._make_params(self, x)
        w_shape = (self.num_units, x.shape[-1] + self.num_units + 1)

        def _initializer(key, *_):
            _w_in = nn.initializers.lecun_uniform(in_axis=-1, out_axis=-2)(
                key, (self.num_units, x.shape[-1])
            )
            _w_rec = nn.initializers.lecun_uniform(in_axis=-1, out_axis=-2)(
                key, (self.num_units, self.num_units)
            )
            _bias = jnp.zeros((self.num_units, 1))
            return jnp.concatenate([_w_in, _w_rec, _bias], axis=-1)

        W = self.param("W", _initializer, w_shape)
        if self.wiring is not None:
            W = jax.lax.stop_gradient(self.variables["wiring"]["mask"]) * W

        if self.ode_type in ["murray", "tau_softplus"]:
            tau = self.param(
                "tau", partial(jrand.uniform, minval=1, maxval=8), (self.num_units,)
            )
        elif self.ode_type == "tg":
            tau = self.param(
                "W_tau",
                nn.initializers.lecun_uniform(in_axis=-1, out_axis=-2),
                (self.num_units, x.shape[-1] + self.num_units + 1),
            )

        if self.ode_type in ["tau_softplus"]:
            h0 = self.param(
                "h0", nn.initializers.zeros, x.shape[:-1] + (self.num_units,)
            )
            params = (h0, W, tau)
        else:
            params = (W, tau)
        return params

    def _f(self, h, x):
        """Compute the derivative of the state."""
        params = self.variables["params"]
        if self.wiring is not None:
            params["W"] = (
                jax.lax.stop_gradient(self.variables["wiring"]["mask"]) * params["W"]
            )
        if self.ode_type == "murray":
            df_dt = ctrnn_ode(params, h, x)
        elif self.ode_type == "tau_softplus":
            df_dt = ctrnn_ode_tau_softplus(params, h, x, min_tau=self.tau_min)
        elif self.ode_type == "tg":
            df_dt = ctrnn_tg(params, h, x)
        return df_dt

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize neuron states."""
        if self.variables and "h0" in self.variables["params"]:
            h0 = self.variables["params"]["h0"]
            return jnp.broadcast_to(h0, input_shape[:-1] + (self.num_units,))
        else:
            return jnp.zeros(tuple(input_shape)[:-1] + (self.num_units,))

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


## CTRNN Jacobian functions


def _rflo_murray(cell: CTRNNCell, carry, params, x, ode=ctrnn_ode):
    """Inefficient version using jacrev."""
    h, jp, jx = carry

    # immediate jacobian (this step)
    df_dw, df_dh, df_dx = jax.jacrev(ode, argnums=[0, 1, 2])(params, h, x)
    jp = {
        "W": jax.tree.map(
            lambda p, d: p * (1 - cell.dt / params["tau"])[:, None]
            + d.sum(axis=0) * cell.dt,
            jp["W"],
            df_dw["W"],
        ),
        "tau": jax.tree.map(
            lambda p, d: p * (1 - cell.dt / params["tau"]) + d.sum(axis=0) * cell.dt,
            jp["tau"],
            df_dw["tau"],
        ),
    }
    jx = jax.tree.map(
        lambda p, d: p * (1 - cell.dt / params["tau"])[:, None]
        + d.sum(axis=0) * cell.dt,
        jx,
        df_dx,
    )
    return jp, jx  # , hebb


def rflo_murray(cell: CTRNNCell, carry, params, x):
    """Compute jacobian trace for RFLO."""
    h, jp, jx = carry
    W, tau = params.values()

    jw = jp["W"]
    jtau = jp["tau"]

    # immediate jacobian (this step)
    xi = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    a = xi @ W.T
    phi_prime = 1 - jnp.tanh(a) ** 2

    # Update W eligibility trace
    # Outer product the get Immediate Jacobian
    w_immediate = phi_prime[..., None] * xi[None]
    jw += (cell.dt / tau)[:, None] * (w_immediate - jw)

    # Update W eligibility trace
    dh_dtau = ((h - jnp.tanh(a)) / tau) - jtau
    jtau += cell.dt * dh_dtau / tau

    df_dw = {"W": jw, "tau": jtau}
    jx += (1 / tau)[:, None] * (jnp.diag(phi_prime) @ W[..., : x.shape[-1]] - jx)
    return df_dw, jx  # , hebb


def eprop_ctrnn(cell: CTRNNCell, carry, params, x):
    """Compute jacobian trace for RFLO."""
    h, jp, jx = carry
    W, tau = params.values()

    jw = jp["W"]
    jtau = jp["tau"]

    # immediate jacobian (this step)
    xi = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    a = xi @ W.T
    phi_prime = 1 - jnp.tanh(a) ** 2

    # hebb = hebbian(v, post)

    # Outer product the get Immediate Jacobian
    # M_immediate = jnp.einsum('ij,k', df_dh, v)
    comm_local = phi_prime * jnp.diag(W[:, x.shape[-1] : -1])

    # Update W eligibility trace
    w_immediate = phi_prime[..., None] * xi[None]
    jw += (cell.dt / tau)[:, None] * (w_immediate + (comm_local - 1)[:, None] * jw)

    # Update W eligibility trace
    tau_immediate = (h - jnp.tanh(a)) / tau
    jtau += (cell.dt / tau) * (tau_immediate + (comm_local - 1) * jtau)

    jx += (1 / tau)[:, None] * (jnp.diag(phi_prime) @ W[..., : x.shape[-1]] - jx)
    return {"W": jw, "tau": jtau}, jx


def rflo_tau_softplus(cell: CTRNNCell, carry, params, x):
    """Compute jacobian trace for RFLO."""
    h, jp, jx = carry
    W, h0, b_tau = params.values()
    tau = jax.nn.softplus(b_tau) + cell.tau_min

    jw = jp["W"]
    jtau = jp["tau"]
    jh0 = jp["h0"]

    # immediate jacobian (this step)
    v = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    u = v @ W.T
    # df_dh = jax.jacfwd(jax.nn.tanh)(u)
    # df_dh = jax.jacrev(jax.nn.tanh)(u)
    phi_prime = 1 - jnp.tanh(u) ** 2
    df_dh = jnp.diag(phi_prime) @ W
    # post = jnp.tanh(u)

    # hebb = hebbian(v, post)

    # Outer product the get Immediate Jacobian
    # M_immediate = jnp.einsum('ij,k', df_dh, v)
    M_immediate = phi_prime[..., None] * v[None]

    # Update eligibility traces
    jw += (1 / tau)[:, None] * (M_immediate - jw) * cell.dt

    dh_dtau = (
        ((h - jnp.tanh(u)) * jax.nn.sigmoid(b_tau) / tau)
        - jtau
        + jtau @ df_dh[:, x.shape[-1] : -1]
    )

    jtau += dh_dtau / tau * cell.dt

    dh_dh0 = 1 - jh0  # + jh0 @ df_dh[:, x.shape[-1] : -1]
    jh0 += cell.dt * dh_dh0 / tau

    df_dw = {"W": jw, "tau": jtau, "h0": jh0}
    jx += (1 / tau)[:, None] * (df_dh[..., : x.shape[-1]] - jx) * cell.dt
    return df_dw, jx  # , hebb


def eprop_ctrnn_tau_softplus(cell: CTRNNCell, carry, params, x):
    """Compute jacobian trace for RFLO."""
    h, jp, jx = carry
    W, h0, b_tau = params.values()
    tau = jax.nn.softplus(b_tau) + cell.tau_min

    jw = jp["W"]
    jtau = jp["tau"]
    jh0 = jp["h0"]

    # immediate jacobian (this step)
    v = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    u = v @ W.T
    # df_dh = jax.jacfwd(jax.nn.tanh)(u)
    # df_dh = jax.jacrev(jax.nn.tanh)(u)
    phi_prime = 1 - jnp.tanh(u) ** 2
    df_dh = jnp.diag(phi_prime) @ W
    # post = jnp.tanh(u)

    # hebb = hebbian(v, post)

    # Outer product the get Immediate Jacobian
    # M_immediate = jnp.einsum('ij,k', df_dh, v)
    M_immediate = phi_prime[..., None] * v[None]

    # Update eligibility traces
    jw += (1 / tau)[:, None] * (M_immediate - jw) * cell.dt

    dh_dtau = (
        ((h - jnp.tanh(u)) * jax.nn.sigmoid(b_tau) / tau)
        - jtau
        + jtau @ df_dh[:, x.shape[-1] : -1]
    )

    jtau += dh_dtau / tau * cell.dt

    dh_dh0 = 1 - jh0  # + jh0 @ df_dh[:, x.shape[-1] : -1]
    jh0 += cell.dt * dh_dh0 / tau

    df_dw = {"W": jw, "tau": jtau, "h0": jh0}
    jx += (1 / tau)[:, None] * (df_dh[..., : x.shape[-1]] - jx) * cell.dt
    return df_dw, jx  # , hebb


# def hebbian(pre, post):
#     # TODO, also: infomax
#     return jnp.outer(post, pre)

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


class OnlineCTRNNCell(OnlineODECell, CTRNNCell):
    """Online CTRNN module."""

    plasticity: str = "rflo"

    def _trace_update(self, carry, _p, x):
        if self.wiring is not None:
            _p["W"] = jax.lax.stop_gradient(self.variables["wiring"]["mask"]) * _p["W"]
        # Select ODE function
        if self.ode_type == "murray":
            _ode = ctrnn_ode
        elif self.ode_type == "tau_softplus":
            _ode = ctrnn_ode_tau_softplus
        elif self.ode_type == "tg":
            _ode = ctrnn_tg
        else:
            raise ValueError(f"ODE type {self.ode_type} not recognized.")
        # Compute trace update
        if self.plasticity == "rtrl":
            traces = rtrl(self, carry, _p, x, ode=_ode)
        elif self.plasticity == "snap0":
            traces = snap0(self, carry, _p, x, ode=_ode)
        elif self.plasticity == "rflo":
            if self.ode_type == "murray":
                traces = rflo_murray(self, carry, _p, x)
            elif self.ode_type == "tau_softplus":
                traces = rflo_tau_softplus(self, carry, _p, x)
            else:
                raise ValueError(f"ODE type {self.ode_type} not supported for rflo.")
        elif self.plasticity == "eprop":
            if self.ode_type == "murray":
                traces = eprop_ctrnn(self, carry, _p, x)
            elif self.ode_type == "tau_softplus":
                traces = eprop_ctrnn_tau_softplus(self, carry, _p, x)
            else:
                raise ValueError(f"ODE type {self.ode_type} not supported.")
        else:
            raise ValueError(f"Plasticity mode {self.plasticity} not supported.")
        if self.wiring is not None:
            traces[0]["W"] = (
                jax.lax.stop_gradient(self.variables["wiring"]["mask"]) * traces[0]["W"]
            )
        return traces


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
        optimizer = optax.adam(lr)
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

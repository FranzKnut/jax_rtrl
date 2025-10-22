"""LTC implementation. TODO!"""

from typing import Literal
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
from chex import PRNGKey
from flax.linen import nowrap

from jax_rtrl.models.cells.ctrnn import rtrl_ctrnn
from jax_rtrl.models.cells.ode import ODECell, OnlineODECell


def ltc_hasani(params, h, x):
    """."""
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    gating = jax.nn.sigmoid(params["a"] * y[..., None,:] + params["b"])
    potential = params["e"] - y[..., None, :]  # shape (..., 1, input+hidden+1)
    w = params["W"]
    # Optional: enforce positivity on weights
    # w = jnp.exp(w)
    # w = jax.nn.softplus(w)
    y_dot = jnp.mean(w * gating * potential, axis=-1)
    return y_dot


def ltc_farsang(params, h, x):
    """h' = -fi hi + ui eli.

    fi = Pm+n  \sum gji sigmoid(aji yj + bji) + gli
    ui = Pm+n  \sum kji sigmoid(aji yj + bji) + gli
    kji = gji eji/eli
    """
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    k = params["W"] * params["e"] / params["e_l"][:, None]
    g = jnp.stack([params["W"], k], axis=0)
    g_l = jnp.stack([params["g_l"], params["g_l"]], axis=0)
    gating = jax.nn.sigmoid(params["a"] * y[None] + params["b"])
    gating = jnp.stack([gating, gating], axis=0)
    f_u = jnp.mean(g * gating + g_l[..., None], axis=-1)
    f, u = f_u
    y_dot = -f * h + u * params["e_l"]
    return y_dot


def lrc_ode(params, h, x):
    """Compute euler integration step or CTRNN ODE."""
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1] + (1,))], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    k = params["W"] * params["e"] / params["e_l"][:, None]
    g = jnp.stack([params["W"], k], axis=0)
    g_l = jnp.stack([params["g_l"], params["g_l"]], axis=0)
    gating = jax.nn.sigmoid(params["a"] * y[..., None, :] + params["b"])
    batched = y.ndim > 1
    gating = jnp.stack([gating, gating], axis=0)
    if batched:
        gating = gating.swapaxes(0, 1)
    f_u = jnp.sum(g * gating + g_l[..., None], axis=-1)
    if batched:
        f_u = f_u.swapaxes(0, 1)
    f, u = f_u
    y_dot = -jax.nn.sigmoid(f) * h + jax.nn.tanh(u) * params["e_l"]
    capacitance = jax.nn.sigmoid(params["o"] @ y.T).T
    y_dot = y_dot * capacitance
    return y_dot


class LTCCell(ODECell):
    """Simple CTRNN cell."""

    ode_type: Literal["hasani", "farsang"] = "hasani"

    def _make_params(self, x):
        # Define params
        w_shape = (self.num_units, x.shape[-1] + self.num_units + 1)
        self.param("a", nn.initializers.ones, w_shape)
        self.param("b", nn.initializers.zeros, w_shape)
        self.param("e", nn.initializers.lecun_normal(in_axis=-1, out_axis=-2), w_shape)
        self.param("W", nn.initializers.uniform(1), w_shape)
        self.param("e_l", nn.initializers.uniform(-1), self.num_units)
        self.param("g_l", nn.initializers.uniform(1), self.num_units)
        if self.ode_type == "lrc":
            self.param("o", nn.initializers.uniform(-1), w_shape)
            # self.param("p", nn.initializers.uniform(-1), self.num_units)
        super()._make_params(x)

    def _f(self, h, x):  # noqa
        """Compute euler integration step or CTRNN ODE."""
        params = self.variables["params"]

        if "mask" in self.variables:
            mask = jax.lax.stop_gradient(self.variables["mask"])
            params = jax.tree.map(lambda W: W * mask, params)
        if self.ode_type in ["hasani", "ltc"]:
            df_dt = ltc_hasani(params, h, x)
        elif self.ode_type in ["farsang", "fltc"]:
            df_dt = ltc_farsang(params, h, x)
        elif self.ode_type == "lrc":
            df_dt = lrc_ode(params, h, x)
        else:
            raise ValueError(f"Unknown ode_type: {self.ode_type}")
        return df_dt

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize neuron states."""
        return jnp.zeros(input_shape[:-1] + (self.num_units,))

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


def rflo_ltc(cell: LTCCell, carry, params, x):
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


class OnlineLTCCell(OnlineODECell, LTCCell):
    """Online LTC module."""

    def _trace_update(self, carry, _p, x):
        if self.plasticity == "rtrl":
            if self.ode_type in ["farsang", "fltc"]:
                _ode_fn = ltc_farsang
            elif self.ode_type in ["hasani", "ltc"]:
                _ode_fn = ltc_hasani
            elif self.ode_type == "lrc":
                _ode_fn = lrc_ode
            else:
                raise ValueError(f"ODE type {self.ode_type} not recognized.")
            traces = rtrl_ctrnn(self, carry, _p, x, ode=_ode_fn)
        elif self.plasticity == "rflo":
            traces = rflo_ltc(self, carry, _p, x)
        else:
            raise ValueError(f"Plasticity mode {self.plasticity} not recognized.")
        return traces


if __name__ == "__main__":
    import numpy as np
    import optax

    import matplotlib.pyplot as plt

    key = jrand.PRNGKey(0)
    key, key_model, key_data, key_train = jrand.split(key, 4)
    x = jnp.linspace(0, 5 * np.pi, 100)[:, None]
    y = jnp.sin(x) + 2

    cell = LTCCell(32)
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

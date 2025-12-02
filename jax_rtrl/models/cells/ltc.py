"""Liquid Time Constant (LTC) model flax implementation."""

from typing import Literal
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.nn import sigmoid
from chex import PRNGKey
from flax.linen import nowrap

from jax_rtrl.models.cells.ode import ODECell, OnlineODECell, rtrl, snap0


def ltc_hasani(params, h, x):
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    gating = sigmoid(params["a"] * y[..., None, :] + params["b"])
    potential = params["e"] - h[..., :, None]  # shape (..., hidden, 1)
    # enforce positivity on weights
    # w = jax.nn.softplus(params["W"])
    w = params["W"]
    # gl =  jax.nn.sigmoid(params["g_l"])
    # gl = jax.nn.softplus(params["g_l"])
    gl = params["g_l"]
    syn = jnp.sum(w * gating * potential, axis=-1)
    leak = gl * (params["e_l"] - h)
    # Ensure G_L is negative (last column of W)
    y_dot = syn + leak
    return y_dot


def ltc_farsang(params, h, x):
    """LTC as in Farsang et al. 2024.

    h' = -fi hi + ui eli.

    fi = Pm+n  \sum gji sigmoid(aji yj + bji) + gli
    ui = Pm+n  \sum kji sigmoid(aji yj + bji) + gli
    kji = gji eji/eli
    """
    # Concatenate input and hidden state
    y = jnp.concatenate([x, h, jnp.ones(x.shape[:-1])], axis=-1)
    # This way we only need one FC layer for recurrent and input connections
    k = params["W"] * params["e"] / params["e_l"][:, None]
    g = jnp.stack([params["W"], k], axis=0)
    g_l = jnp.stack([params["g_l"], params["g_l"]], axis=0)
    gating = sigmoid(params["a"] * y[None] + params["b"])
    gating = jnp.stack([gating, gating], axis=0)
    f_u = jnp.mean(g * gating + g_l[..., None], axis=-1)
    f, u = f_u
    y_dot = -f * h + u * params["e_l"]
    return y_dot


class LTCCell(ODECell):
    """Generic LTC cell."""

    ode_type: Literal["hasani", "farsang"] = "hasani"

    def _make_params(self, x):
        # Define params
        w_shape = (self.num_units, x.shape[-1] + self.num_units)
        self.param("a", nn.initializers.normal(), w_shape)
        self.param("b", nn.initializers.normal(), w_shape)
        self.param("e", nn.initializers.lecun_normal(in_axis=-1, out_axis=-2), w_shape)
        self.param("W", nn.initializers.uniform(0.01), w_shape)
        self.param("e_l", nn.initializers.uniform(-1), self.num_units)
        self.param("g_l", nn.initializers.uniform(1.0), self.num_units)

        if self.ode_type == "lrc":
            self.param("o", nn.initializers.uniform(-1), w_shape)
            # self.param("p", nn.initializers.uniform(-1), self.num_units)
        super()._make_params(x)

    def _f(self, h, x):  # noqa
        """Compute euler integration step or CTRNN ODE."""
        params = self.variables["params"]

        if self.wiring is not None:
            mask = jax.lax.stop_gradient(self.variables["wiring"]["mask"])
            params = jax.tree.map(lambda W: W * mask, params)
        if self.ode_type in ["hasani", "ltc"]:
            df_dt = ltc_hasani(params, h, x)
        elif self.ode_type in ["farsang", "fltc"]:
            df_dt = ltc_farsang(params, h, x)
        # elif self.ode_type == "lrc":
        #     df_dt = lrc_ode(params, h, x)
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

    y = jnp.concatenate([x, h], axis=-1)
    syn_in = params["a"] * y[..., None, :] + params["b"]
    syn = jax.nn.sigmoid(syn_in)
    # pointwise derivatives
    syn_diff = jax.vmap(jax.vmap(jax.grad(jax.nn.sigmoid)))(syn_in)
    ltc_diff = params["e"] - h[..., None]  # shape (..., 1, input+hidden)

    dh_dh = (
        1.0 - params["g_l"]
        # + (params["W"] * (ltc_diff * syn_diff * params["a"] - syn))[:, -h.shape[-1] :]
    )
    # dh_dh = jnp.sum(dh_dh, axis=-1)

    # comm = jax.tree.map(lambda p: p * dh_dh, jp)
    dh_dG_L = cell.dt * (params["e_l"] - h) + dh_dh * jp["g_l"]
    dh_dE_L = cell.dt * params["g_l"] + dh_dh * jp["e_l"]

    # dh_dG_s = jnp.sum(ltc_diff * syn, axis=-1)[:, None] + dh_dh[:, None] * jp["W"]
    dh_dG_s = cell.dt * ltc_diff * syn + dh_dh[:, None] * jp["W"]
    # dh_dE_s = jnp.sum(params["W"] * syn, axis=-1)[:, None] + dh_dh[:, None] * jp["e"]
    dh_dE_s = cell.dt * params["W"] * syn + dh_dh[:, None] * jp["e"]

    _tmp = params["W"] * ltc_diff * syn_diff

    # dh_da = jnp.sum(_tmp * y, axis=-1)[:, None] + dh_dh[:, None] * jp["a"]
    dh_da = _tmp * y + dh_dh[:, None] * jp["a"]
    # dh_da = dh_dh[:, None] * jp["a"]
    # dh_db = jnp.sum(_tmp, axis=-1)[:, None] + dh_dh[:, None] * jp["b"]
    dh_db = _tmp + dh_dh[:, None] * jp["b"]
    # dh_db = dh_dh[:, None] * jp["b"]

    # Infomax
    # dh_db = 1 - 2 * syn
    # dh_da = 1 / params["a"] + y * dh_db

    dh_dx = (_tmp * params["a"] - params["W"] * syn)[..., : x.shape[-1]] + dh_dh[
        :, None
    ] * jx
    jp = {
        "W": dh_dG_s,
        "e": dh_dE_s,
        "g_l": dh_dG_L,
        "e_l": dh_dE_L,
        "a": dh_da,
        "b": dh_db,
    }
    return jp, dh_dx  # , hebb


class OnlineLTCCell(OnlineODECell, LTCCell):
    """Online LTC module."""

    def _trace_update(self, carry, _p, x):
        if self.plasticity in ["rtrl", "snap0"]:
            if self.ode_type in ["farsang", "fltc"]:
                _ode_fn = ltc_farsang
            elif self.ode_type in ["hasani", "ltc"]:
                _ode_fn = ltc_hasani
            # elif self.ode_type == "lrc":
            #     _ode_fn = lrc_ode
            else:
                raise ValueError(f"ODE type {self.ode_type} not recognized.")
            if self.plasticity == "rtrl":
                traces = rtrl(self, carry, _p, x, ode=_ode_fn)
            else:
                traces = snap0(self, carry, _p, x, ode=_ode_fn)

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

    def train(_loss_fn, _params, data, _key, num_steps=10_000, lr=1e-3):
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

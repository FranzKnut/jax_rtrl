"""Liquid Resistance Liquid Capacitance (LRC) model flax implementation."""

from typing import Literal
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.nn import sigmoid
from chex import PRNGKey
from flax.linen import nowrap

from jax_rtrl.models.cells.ode import ODECell, OnlineODECell, rtrl, snap0
from jax_rtrl.util.jax_util import symmetric_uniform_init


def lrc_ode(params, h, x, use_symmetric=False):
    # Calculate A
    sensory_syn = sigmoid(params["sigma_in"] * (x - params["mu_in"]))
    sensory_syn_w = (params["w_in"] @ sensory_syn.T).T  # neural activation

    syn = sigmoid(params["sigma"] * (h - params["mu"]))
    syn_w = params["w"] * syn

    f = params["g_l"] + sensory_syn_w + syn_w
    A = -jax.nn.sigmoid(f)

    # Calculate B
    # sensory_syn = sigmoid(params["sigma_in"] * (x - params["mu_in"]))
    sensory_syn_h = (params["h_in"] @ sensory_syn.T).T  # neural activation

    # syn = sigmoid(params["sigma"] * (h - params["mu"]))
    syn_h = params["h"] * syn

    g = params["g_l"] + sensory_syn_h + syn_h
    B = params["v_l"] * jnp.tanh(g)

    # This could be also input dependent, not only state dep.
    elastance_term = params["w_e"] * h + params["b_e"]

    if use_symmetric:  # type of elastance
        elastance = sigmoid(elastance_term + params["s_e"]) - sigmoid(
            elastance_term - params["s_e"]
        )
    else:
        elastance = sigmoid(elastance_term)

    v_prime = h * A + B
    return elastance * v_prime


class LRCCell(ODECell):
    """LRC cell."""

    ode_type: Literal["lrc"] = "lrc"
    use_symmetric: bool = False

    def _make_params(self, x):
        # Define params
        input_size = x.shape[-1]
        sensory_shape = (self.num_units, input_size)
        lim = jnp.sqrt(1 / self.num_units)

        # Sensory parameters
        self.param("mu_in", symmetric_uniform_init(lim), input_size)
        self.param("sigma_in", symmetric_uniform_init(lim), input_size)
        self.param("w_in", symmetric_uniform_init(lim), sensory_shape)
        self.param("h_in", symmetric_uniform_init(lim), sensory_shape)

        # Recurrent parameters
        self.param("mu", symmetric_uniform_init(lim), self.num_units)
        self.param("sigma", symmetric_uniform_init(lim), self.num_units)
        self.param("w", symmetric_uniform_init(lim), self.num_units)
        self.param("h", symmetric_uniform_init(lim), self.num_units)

        # Leak parameters
        self.param("v_l", symmetric_uniform_init(lim), self.num_units)
        self.param("g_l", symmetric_uniform_init(lim), self.num_units)

        # Eslastance parameters
        self.param("w_e", symmetric_uniform_init(lim), self.num_units)
        self.param("b_e", symmetric_uniform_init(lim), self.num_units)
        if self.use_symmetric:
            self.param("s_e", symmetric_uniform_init(lim), self.num_units)

        super()._make_params(x)

    def _f(self, h, x):  # noqa
        """Compute euler integration step or CTRNN ODE."""
        params = self.variables["params"]

        if self.wiring is not None:
            mask = jax.lax.stop_gradient(self.variables["wiring"]["mask"])
            params = jax.tree.map(lambda W: W * mask, params)
        if self.ode_type == "lrc":
            df_dt = lrc_ode(params, h, x, self.use_symmetric)
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


def rflo_lrc(cell: LRCCell, carry, params, x):
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


class OnlineLRCCell(OnlineODECell, LRCCell):
    """Online LTC module."""

    def _trace_update(self, carry, _p, x):
        if self.plasticity in ["rtrl", "snap0"]:
            if self.ode_type == "lrc":
                _ode_fn = lrc_ode
            else:
                raise ValueError(f"ODE type {self.ode_type} not recognized.")
            if self.plasticity == "rtrl":
                traces = rtrl(self, carry, _p, x, ode=_ode_fn)
            else:
                traces = snap0(self, carry, _p, x, ode=_ode_fn)

        elif self.plasticity == "rflo":
            traces = rflo_lrc(self, carry, _p, x)
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

    cell = LRCCell(32)
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

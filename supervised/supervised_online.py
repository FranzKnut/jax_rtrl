import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
import flax.linen as nn

import matplotlib.pyplot as plt
from tqdm import trange

from jax_rtrl.models.neural_networks import RNNEnsemble

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)


class Model(nn.Module):
    outsize: int
    out_dist: str = None  # 'Normal'
    hidden_size: int = 32
    num_modules: int = 1
    dt: float = 1
    plasticity: str = "rflo"
    dropout_rate: float = 0

    @nn.nowrap
    def _make_rnn(self):
        return RNNEnsemble(
            out_size=self.outsize,
            num_modules=self.num_modules,
            out_dist=self.out_dist,
            kwargs={"num_units": self.hidden_size, "dt": self.dt, "plasticity": self.plasticity},
            name="rnn",
        )

    @nn.compact
    def __call__(self, x, carry=None, key=None, training=False):
        cell = self._make_rnn()
        if carry is None:
            carry = cell.initialize_carry(key, x.shape)
            key = jrand.fold_in(key, key[0])
        carry, out = cell(carry, x, training=training)
        return carry, out

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self._make_rnn().initialize_carry(key, input_shape)


def make_model(initial_input, key, kwargs={}):
    key, key_model = jrand.split(key)
    model = Model(1, **kwargs)
    params = model.init(key_model, initial_input, key=key_model)
    h0 = model.initialize_carry(key_model, initial_input.shape[-1:])
    return model, params, h0


def train(_loss_fn, _params, data, _key, h0, num_steps=1_000, lr=1e-2):
    # We use Stochastic Gradient Descent with a constant learning rate
    _x, _y = data
    # mask = jax.tree.map(lambda x: True, _params)
    # mask['params']['cell']['tau'] = False
    # lr = optax.warmup_cosine_decay_schedule(init_value=lr/10, peak_value=lr, warmup_steps=500, decay_steps=2500)
    optimizer = optax.chain(
        # optax.clip_by_global_norm(1),
        optax.sgd(lr),
        # optax.sgd(lr, momentum=0.8, nesterov=True),
        # optax.adam(lr),
    )
    opt_state = optimizer.init(_params)
    pbar = trange(num_steps, maxinterval=2)

    def run_episode(ep_carry, n):
        def step(carry, _data):
            __params, _opt_state, _key, h = carry
            __x, __y = _data
            # _key, key_batch = jrand.split(_key)
            (current_loss, (_key, h)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(__params, __x, __y, (_key, h))
            updates, _opt_state = optimizer.update(grads, _opt_state, __params)
            __params = optax.apply_updates(__params, updates)

            # HACK: Clip params to ensure damping tau
            for s in __params["params"]["rnn"]:
                if "tau" in s:
                    __params["params"]["rnn"][s]["tau"] = jnp.clip(__params["params"]["rnn"][s]["tau"], min=1)

            # Grad clipping for the Jacobian traces
            # h = (h[0], *jax.tree.map(lambda x: optax.clip_by_global_norm(.1).update(x, None)[0], h[1:]))

            return (__params, _opt_state, _key, h), current_loss

        # Reset only hidden state, keep traces
        # step_carry = (*ep_carry[:-1], (h0[0], *ep_carry[-1][1:]))
        step_carry = (*ep_carry[:-1], h0)

        step_carry, __losses = jax.lax.scan(step, step_carry, (_x, _y))
        current_loss = __losses.sum()

        def print_progress(i, loss):
            pbar.update()
            pbar.set_description(f"Iteration {i} | Loss: {loss.mean():.3f}", refresh=False)

        jax.debug.callback(print_progress, n, current_loss)
        return step_carry, current_loss

    (_params, *_), _losses = jax.lax.scan(
        run_episode, (_params, opt_state, _key, h0), jnp.arange(num_steps, dtype=np.int32)
    )
    return _params, _losses


if __name__ == "__main__":
    key = jrand.PRNGKey(1)
    key, key_train = jrand.split(key)

    x = jnp.linspace(0, 5 * np.pi, 100)[:, None]
    y = jnp.sin(x) + 2

    model, params, h0 = make_model(x[0], key)

    def loss(p, __x, __y, carry=None):
        # MSE loss
        key, rnn_state = carry
        rnn_state, y_hat = model.apply(p, __x, rnn_state, rngs={"dropout": key}, training=True)
        key = jrand.fold_in(key, key[0])
        if model.out_dist is None:
            loss = jnp.mean((__y - y_hat) ** 2)
        else:
            loss = jnp.mean(-y_hat.log_prob(__y))
        return loss, (key, rnn_state)

    params, losses = train(loss, params, (x, y), key_train, h0)

    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)

    # Plot the trained model output
    plt.subplot(1, 2, 2)

    def predict(params, x):
        def _step(carry, _x):
            return model.apply(params, _x, carry, training=False)

        outs = jax.lax.scan(_step, h0, x)[1]
        if model.out_dist:
            outs = outs.sample(seed=key)
        return outs

    y_hat = predict(params, x)
    print(f"Final loss: {jnp.mean((y-y_hat)**2):.3f}")
    plt.plot(x, y, label="target")
    plt.plot(x, y_hat, label="trained")
    plt.legend()
    plt.show()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/sinewave.png")

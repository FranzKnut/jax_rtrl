"""Supervised training loops."""

import jax
import optax
from jax import numpy as jnp
from flax import linen as nn
from tqdm import trange


def train_rnn_online(
    loss_fn,
    optimizer,
    params,
    data,
    key,
    h0,
    num_steps=10_000,
    param_post_update_fn=None,
):
    """Train RNN using Stochastic Gradient Descent with a constant learning rate."""
    _x, _y = data
    # mask = jax.tree.map(lambda x: True, _params)
    # mask['params']['cell']['tau'] = False
    opt_state = optimizer.init(params)
    pbar = trange(int(num_steps), maxinterval=2)

    def run_episode(ep_carry, n):
        print("Tracing run_episode.")
        def step(carry, _data):
            __params, _opt_state, _key, h = carry
            __x, __y = _data
            # _key, key_batch = jrand.split(_key)
            (current_loss, h), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                __params, __x, __y, h
            )
            updates, _opt_state = optimizer.update(grads, _opt_state, __params)
            __params = optax.apply_updates(__params, updates)

            if param_post_update_fn is not None:
                __params["params"] = param_post_update_fn(__params["params"])

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
            pbar.set_description(
                f"Iteration {i} | Loss: {loss.mean():.3f}", refresh=False
            )

        jax.debug.callback(print_progress, n, current_loss)
        return step_carry, current_loss

    (params, *_), _losses = jax.lax.scan(
        run_episode,
        (params, opt_state, key, h0),
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    pbar.close()
    return params, _losses


def print_progress(i, loss):
    if i % 1000 == 0:
        print(f"Iteration {i} | Loss: {loss:.3f}")


def train_rnn_offline(
    _loss_fn,
    optimizer,
    _params,
    data,
    _key,
    num_steps=10_000,
):
    # We use Stochastic Gradient Descent with a constant learning rate
    _x, _y = data

    # mask = jax.tree.map(lambda x: True, _params)
    # mask["params"]["layers_0"]["cell"]["tau"] = False
    # optimizer = optax.adamw(lr, mask=mask) # Mask tau from weight decay

    opt_state = optimizer.init(_params)

    def step(carry, n):
        __params, _opt_state, _key = carry
        # _key, key_batch = jrand.split(_key)
        current_loss, grads = jax.value_and_grad(_loss_fn)(__params, _x, _y)
        updates, _opt_state = optimizer.update(grads, _opt_state, __params)
        __params = optax.apply_updates(__params, updates)
        jax.debug.callback(print_progress, n, current_loss)
        return (__params, _opt_state, _key), current_loss

    (_params, *_), _losses = jax.lax.scan(
        step, (_params, opt_state, _key), jnp.arange(num_steps, dtype=jnp.int32)
    )
    print(f"Final loss: {_losses[-1]:.3f}")
    return _params, _losses


def predict(model: nn.RNNCellBase, params, *inputs):
    """Predict a sequence of outputs given an input sequence."""

    def _step(carry, _x):
        return model.apply(params, *_x, carry)

    h0 = model.apply(
        params,
        jax.random.PRNGKey(0),
        inputs[0].shape[1:],
        method=model.initialize_carry,
    )
    outs = jax.lax.scan(_step, h0, inputs)[1]
    return outs

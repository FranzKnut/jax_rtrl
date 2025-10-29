"""Supervised training loops."""

import jax
import jax.random as jrand
import optax
from jax import numpy as jnp
from flax import linen as nn
from tqdm import trange

from jax_rtrl.models.seq_models import (
    RNNEnsemble,
    RNNEnsembleConfig,
    make_batched_model,
    scan_rnn,
)
from jax_rtrl.supervised import example_datasets


def get_data(dataset):
    x, y = getattr(example_datasets, dataset)()

    # add missing time and feature dims
    t = x.shape[-2]
    if x.ndim <= 2:
        # Assume single batch
        y = y * jnp.ones((1, 1, 1))
    elif y.ndim == 2:
        assert y.shape[0] == x.shape[0]
        # Assume one label per sequence
        # Repeat along batch and time dimension
        y = y[:, None] * jnp.ones((x.shape[0], t, 1))

    if x.ndim == 2:
        x = x[None]

    batch_size = x.shape[0]
    if batch_size > 1:
        (
            (x_train, y_train),
            (x_test, y_test),
        ) = example_datasets.split_train_test(
            (x, y),
            percent_eval=0.05,
            shuffle=True,
        )
    else:
        x_train, y_train = x, y
        x_test, y_test = x[0], y[0]

    # Transpose to time dim first
    x_train = x_train.transpose(1, 0, 2)
    y_train = y_train.transpose(1, 0, 2)
    return x_train, y_train, x_test, y_test


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
        current_loss = __losses.mean()

        def print_progress(i, loss):
            if i % 10 == 0:
                pbar.set_description(
                    f"Iteration {i} | Loss: {loss.mean():.3f}", refresh=False
                )
                pbar.update(10)

        jax.debug.callback(print_progress, n, current_loss)
        return step_carry, current_loss

    (params, *_), _losses = jax.lax.scan(
        run_episode,
        (params, opt_state, key, h0),
        jnp.arange(num_steps, dtype=jnp.int32),
    )
    pbar.close()
    return params, _losses


def train_rnn_offline(
    _loss_fn,
    optimizer,
    _params,
    data,
    _key,
    num_steps=100,
):
    # We use Stochastic Gradient Descent with a constant learning rate
    _x, _y = data

    # mask = jax.tree.map(lambda x: True, _params)
    # mask["params"]["layers_0"]["cell"]["tau"] = False
    # optimizer = optax.adamw(lr, mask=mask) # Mask tau from weight decay

    opt_state = optimizer.init(_params)
    pbar = trange(int(num_steps), maxinterval=2)

    def print_progress(i, loss):
        if i % 10 == 0:
            pbar.set_description(
                f"Iteration {i} | Loss: {loss.mean():.3f}", refresh=False
            )
            pbar.update(10)

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
    pbar.close()
    print(f"Final loss: {_losses[-1]:.3f}")
    return _params, _losses


def predict(model: nn.RNNCellBase, params, *inputs):
    """Predict a sequence of outputs given an input sequence."""
    _, y_hat = scan_rnn(model, params, None, *inputs)
    return y_hat


def make_model(initial_input, key, kwargs: RNNEnsembleConfig):
    key_model = jrand.split(key, initial_input.shape[0])
    model = make_batched_model(RNNEnsemble)(kwargs)
    params = model.init(key_model[0], None, initial_input)
    h0 = model.apply(
        params, key_model[0], initial_input.shape, method=model.initialize_carry
    )
    return model, params, h0

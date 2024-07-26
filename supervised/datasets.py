"""Datasets for supervised learning."""

import os

import jax
from jax import numpy as jnp
from jax import random as jrandom


def split_train_test(inputs, target, percent_eval=0.2):
    """Split the dataset into train and test sets along the first axis.

    Train episodes are taken from the beginning of the dataset, test episodes from the end.
    :param percent_eval:
    :param inputs:
    :param target:
    :return: train and eval tuples of (inputs, target)
    """
    train_size = int(inputs.shape[0] * (1 - percent_eval))
    inputs_train = inputs[:train_size]
    inputs_eval = inputs[train_size:]

    target_train = target[:train_size]
    target_eval = target[train_size:]
    return (inputs_train, target_train), (inputs_eval, target_eval)


def dataloader(arrays, batch_size: int, *, key=None, permute=False):
    """Dataloader that returns a tuple of batches from the given arrays.

    Args:
        arrays (_type_): List of data arrays
        batch_size (int): Batch size
        key (_type_): jax random key
        permute (bool, optional): use random permutations. Defaults to False.

    Yields:
        _type_: _description_
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        if permute:
            indices = jrandom.permutation(key, indices)
            (key,) = jrandom.split(key, 1)
        start = 0
        end = min(batch_size, dataset_size)
        while end <= dataset_size:
            batch_perm = indices[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def cut_sequences(set_x: jax.Array, set_y: jax.Array, seq_len: int, overlap=0, set_t=None):
    """Cut the given sequences into subsequences of length seq_len.

    Sequences may overlap. If set_t is given, it is also cut into subsequences.
    Args:
        set_x (Array): Sequence of x values
        set_y (Array): Sequence of y values
        seq_len (int): Sequence length
        overlap (int, optional): Overlap for the sequences. Defaults to 0.
        set_t (Array, optional): An optional Array containing timestamps. Defaults to None.

    Returns:
        Tuple[Array, Array] or Tuple[Array, Array, Array]: if set_t is None, returns (x, y), else (x, t, y)
    """
    times = []
    starts = jnp.arange(len(set_x) - seq_len + 1, step=seq_len - overlap)
    x = jax.vmap(lambda start: jax.lax.dynamic_slice(set_x, (start,), (seq_len,)))(starts)
    y = jax.vmap(lambda start: jax.lax.dynamic_slice(set_y, (start,), (seq_len,)))(starts)
    if set_t is None:
        return (jnp.stack(x, axis=0), jnp.stack(y, axis=0))
    else:
        t = jax.vmap(lambda start: jax.lax.dynamic_slice(set_y, (start,), (seq_len,)))(starts)
        return (
            jnp.stack(x, axis=0),
            jnp.expand_dims(jnp.stack(t, axis=0), axis=-1),
            jnp.stack(y, axis=0),
        )


# Toy datasets -----------------------------------------------------------------


def spirals(dataset_size, key):
    """Create a dataset of two spirals."""
    t = jnp.linspace(0, 2 * jnp.pi, 16)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * jnp.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return x, y


# Gym Simulations ---------------------------------------------------------------


def rollouts(name, with_time=False):
    """Load a dataset from the BulletEnv simulator.

    :param name: Environment name
    :param y_mode: 'obs' for learning observations (auto-regression), 'act' for learning actions
    :param act_in_obs: If True, the actions are included in the observations
    :return:
    """
    path = "artifacts/rollouts/" + name
    files = sorted([os.path.join(path, d) for d in os.listdir(path) if d.endswith(".npz")])

    all_x = []
    if with_time:
        all_t = []
    all_y = []
    all_dones = []
    for f in files:
        loaded = jnp.load(f, allow_pickle=True)
        obs_arr = loaded["obs"]
        act_arr = loaded["act"]
        x = obs_arr.astype(jnp.float32)
        y = act_arr
        x_times = jnp.ones(x.shape[0])
        # Only add sequences that have maximum length
        all_x.append(x)
        if with_time:
            all_t.append(x_times)
        all_y.append(y)
        all_dones.append(jnp.array([False for _ in range(x.shape[0] - 1)] + [True]))

    all_x = jnp.concatenate(all_x, axis=0)
    all_y = jnp.concatenate(all_y, axis=0)
    all_dones = jnp.concatenate(all_dones, axis=0)
    if with_time:
        all_t = jnp.concatenate(all_t, axis=0)

    print("Read {} files containing {:d} steps total".format(len(files), len(all_x)))
    if with_time:
        return all_x, all_y, all_t, all_dones
    return all_x, all_y, all_dones

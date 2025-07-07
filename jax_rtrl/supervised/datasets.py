"""Datasets for supervised learning."""

import os
import re
from collections.abc import Iterable
import jax

from jax import numpy as jnp
from jax import random as jrandom
from tqdm import tqdm


def split_train_test(dataset, percent_eval: float = 0.2):
    """Split the dataset into train and test sets along the first axis.

    Train episodes are taken from the beginning of the dataset, test episodes from the end.
    :param dataset: Pytree
    :param percent_eval: float
    :return: train and eval tuples of (inputs, target)
    """
    dataset_size = jax.tree.flatten(dataset)[0][0].shape[0]
    train_size = int(dataset_size * (1 - percent_eval))
    dataset_train = jax.tree.map(lambda x: x[:train_size], dataset)
    dataset_eval = jax.tree.map(lambda x: x[train_size:], dataset)

    return dataset_train, dataset_eval


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


def cut_sequences(*data: Iterable[jax.Array] | jax.Array, seq_len: int, overlap=0):
    """Cut the given sequences into subsequences of length seq_len.

    Sequences may overlap.
    Args:
        data (Array | Iterable): the data to be split
        seq_len (int): Sequence length
        overlap (int, optional): Overlap for the sequences. Defaults to 0.

    Returns:
        Tuple[Array, ...]
    """
    first_set = data if isinstance(data, jax.Array) else data[0]
    starts = jnp.arange(len(first_set) - seq_len + 1, step=seq_len - overlap)
    sliced = [
        jax.vmap(lambda start: jax.lax.dynamic_slice(d, (start,), (seq_len,)))(starts)
        for d in data
    ]
    return [jnp.stack(s, axis=0) for s in sliced]


def load_np_files_from_folder(path, is_npz=True, num_files: int = None):
    """Load a set of npz or npz files from a folder.

    :param name: Environment name
    :param is_npz: If True, the files in the folder are assumed to be .npz files containing multiple fields
    :param num_files: If not None, only loads the specified number of files
    :return:
    """
    files = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if d.endswith(".npz" if is_npz else ".npy")
    ]
    # Sort numbered files
    files.sort(key=lambda f: int(re.sub(r"\D", "", f)))
    print(f"Loading {len(files[:num_files])} files from {path}")
    data = []
    with jax.default_device(jax.devices("cpu")[0]):
        for f in tqdm(files[:num_files]):
            d = jnp.load(f, allow_pickle=True)
            if is_npz:
                d = dict(d)
            data.append(d)

        if is_npz:
            output = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *data)
            num_steps = len(output[list(output.keys())[0]])
            num_steps_per_file = [len(d[list(d.keys())[0]]) for d in data]
        else:
            output = jnp.concatenate(data, axis=0)
            num_steps_per_file = [len(d) for d in data]
            file_starts = jnp.zeros(num_steps)
            num_steps = len(output)

        # An Array that is zero everywhere except at the start of each file
        start_indices = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(jnp.array(num_steps_per_file[:-1]))], dtype=int
        )
        file_starts = jnp.zeros(num_steps).at[start_indices].set(1)

    print(f"Files contained {num_steps:d} steps total")
    return output, file_starts


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


def sine(length=100, offset=2, num_periods=3):
    """A simple sine-wave."""
    x = jnp.linspace(0, num_periods * 2 * jnp.pi, length)[:, None]
    y = jnp.sin(x) + offset
    return x, y


# Gym Simulations ---------------------------------------------------------------


def rollouts(data_folder, with_time=False):
    """Load a dataset from the BulletEnv simulator.
    
    TODO: redundant with load_np_files_from_folder, refactor to use it.

    :param name: Environment name
    :param y_mode: 'obs' for learning observations (auto-regression), 'act' for learning actions
    :param act_in_obs: If True, the actions are included in the observations
    :return:
    """
    files = sorted(
        [
            os.path.join(data_folder, d)
            for d in os.listdir(data_folder)
            if d.endswith(".npz")
        ]
    )

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

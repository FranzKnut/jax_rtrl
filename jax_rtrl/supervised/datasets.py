"""Datasets for supervised learning."""

import os
import re
import zipfile
from collections.abc import Iterable
import jax

from typing import NamedTuple
import numpy as np
from numpy.lib._version import NumpyVersion
if NumpyVersion(np.__version__) >= "2.0.0":
    from numpy.lib.format import read_array_header_1_0 as _read_array_header
else:
    from numpy.lib.format import _read_array_header
from jax import numpy as jnp
from jax import random as jrandom
from tqdm import tqdm
import flashbax as fbx
from flashbax.vault import Vault


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
    for f in tqdm(files[:num_files]):
        d = np.load(f, allow_pickle=True, mmap_mode="r")
        if is_npz:
            d = dict(d)
        data.append(d)

    if is_npz:
        output = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *data)
        num_steps = len(output[list(output.keys())[0]])
        num_steps_per_file = [len(d[list(d.keys())[0]]) for d in data]
    else:
        output = np.concatenate(data, axis=0)
        num_steps_per_file = [len(d) for d in data]
        file_starts = np.zeros(num_steps)
        num_steps = len(output)

    # An Array that is zero everywhere except at the start of each file
    start_indices = np.concatenate(
        [np.zeros(1), np.cumsum(np.array(num_steps_per_file[:-1]))], dtype=int
    )
    file_starts = np.zeros(num_steps).at[start_indices].set(1)

    print(f"Files contained {num_steps:d} steps total")
    return output, file_starts


def npy_headers(f):
    """
    Takes a .npy file handle.
    Generates a tuple of (shape, np.dtype).
    """
    version = np.lib.format.read_magic(f)
    shape, fortran, dtype = _read_array_header(f, version)
    return shape, dtype


def npz_headers(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue

            npy = archive.open(name)
            shape, dtype = npy_headers(npy)
            yield name[:-4], shape, dtype


def load_into_vault(
    path, vault_name, is_npz=True, num_files: int = None, vault_uid=None
) -> tuple[Vault, jax.Array]:
    """Load a set of npz or npz files from a folder and store them in a flashbax Vault.

    :param name: Path of rollout files
    :param is_npz: If True, the files in the folder are assumed to be .npz files containing multiple fields
    :param num_files: If not None, only loads the specified number of files
    :param write_freq: How often to write to the vault
    :return: (Vault, file_starts)
    """
    files = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if d.endswith(".npz" if is_npz else ".npy")
    ]

    # Sort numbered files
    files.sort(key=lambda f: int(re.sub(r"\D", "", f)))
    print(f"Loading {len(files[:num_files])} files from {path}")
    num_steps_per_file = []
    print("Determining total length of files...")

    for f in tqdm(files[:num_files]):
        if is_npz:
            # Best effort assuming all contained have the same leading dimension
            num_steps_per_file.append(list(npz_headers(f))[0][1][0])
        else:
            # TODO: Test this
            with open(f, "rb") as _file:
                num_steps_per_file.append(npy_headers(_file)[0][0])

    total_num_steps = sum(num_steps_per_file)
    # add_batch_size = np.gcd.reduce(num_steps_per_file)

    with jax.default_device(jax.devices("cpu")[0]):
        buffer = fbx.make_trajectory_buffer(
            add_batch_size=1,
            max_length_time_axis=max(num_steps_per_file) + 1,
            sample_batch_size=1,
            sample_sequence_length=1,
            min_length_time_axis=1,
            period=1,
        )

        add_batch = jax.jit(buffer.add, donate_argnums=0)

        d = jnp.load(files[0], allow_pickle=True)

        class Data(NamedTuple):
            img: jnp.ndarray
            observation: jnp.ndarray
            action: jnp.ndarray

        def _prep_data(_d):
            if is_npz:
                _d = dict(_d)
            _d = {
                k: jnp.array(v)
                for k, v in _d.items()
                if k in ["observation", "action", "data"]
            }
            _d["img"] = _d["data"]
            del _d["data"]
            _d = jax.tree_util.tree_map(lambda x: x[None], _d)
            return Data(**_d)

        init_data = jax.tree_util.tree_map(lambda x: x[0][0], _prep_data(d))
        buffer_state = buffer.init(init_data)

        # Create vault
        vault = Vault(
            vault_name=vault_name,
            experience_structure=buffer_state.experience,
            vault_uid=vault_uid,
        )
        if vault.vault_index:
            return vault, None  # Already exists

        for f in tqdm(files[:num_files]):
            d = np.load(f, allow_pickle=True, mmap_mode="r")
            data = _prep_data(d)
            buffer_state = add_batch(buffer_state, data)
            vault.write(buffer_state)

        # An Array that is zero everywhere except at the start of each file
        start_indices = np.concatenate(
            [np.zeros(1), np.cumsum(np.array(num_steps_per_file[:-1]))], dtype=int
        )
        file_starts = np.zeros(total_num_steps).at[start_indices].set(1)

    print(f"Files contained {total_num_steps:d} steps total")
    return vault, file_starts


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


if __name__ == "__main__":
    load_into_vault("data/dvs_only_256p_100hz", vault_name="MMDVS")

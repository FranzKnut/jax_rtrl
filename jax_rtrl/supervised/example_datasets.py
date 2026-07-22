"""Datasets for supervised learning."""

import os
import re
import zipfile
from collections.abc import Iterable
import jax

import numpy as np
from jax import numpy as jnp
from jax import random as jrandom
from tqdm import tqdm

from jax_rtrl.supervised.sequential_vault import make_vault


def tree_stack(trees, axis=0, concatenate=False):
    """Take a list of trees and stack every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function. Taken from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves, treedef = jax.tree.flatten(trees[0])
    leaves_list = [leaves]
    for tree in trees[1:]:
        leaves, _ = jax.tree.flatten(tree)
        leaves_list.append(leaves)

    result_leaves = []
    leaves_list = list(zip(*leaves_list))
    for leaf_id, leaf in enumerate(leaves_list):
        _op = jnp.concatenate if concatenate else jnp.stack
        leaf = [jnp.atleast_1d(leaf_item) for leaf_item in leaf]
        result_leaves.append(_op(leaf, axis=axis))
    return treedef.unflatten(result_leaves)


def split_train_test(
    dataset, fraction_eval: float = 0.2, shuffle: bool = False, axis=0
):
    """Split the dataset into train and test sets along the first axis.

    Train episodes are taken from the beginning of the dataset, test episodes from the end.
    :param dataset: Pytree
    :param fraction_eval: float
    :param shuffle: Whether to shuffle the dataset before splitting
    :param axis: Axis along which to split the dataset
    :return: train and eval tuples of (inputs, target)
    """
    if shuffle:
        key = jrandom.PRNGKey(0)
        perm = jrandom.permutation(
            key, jnp.arange(jax.tree_util.tree_leaves(dataset)[0].shape[0])
        )
        dataset = jax.tree.map(lambda x: x[perm], dataset)
    dataset_size = jax.tree.flatten(dataset)[0][0].shape[axis]
    train_size = int(dataset_size * (1 - fraction_eval))

    # Split dataset along axis
    dataset_train = jax.tree.map(
        lambda x: jnp.take(x, jnp.arange(train_size), axis=axis), dataset
    )
    dataset_eval = jax.tree.map(
        lambda x: jnp.take(x, jnp.arange(train_size, dataset_size), axis=axis),
        dataset,
    )

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
    if isinstance(first_set, dict):
        first_set = list(first_set.values())[0]
    starts = jnp.arange(len(first_set) - seq_len + 1, step=seq_len - overlap)
    sliced = jax.tree.map(
        lambda d: jnp.stack([d[s : s + seq_len] for s in starts]),
        data,
    )
    return sliced


def load_np_files_from_folder(
    path,
    is_npz=True,
    num_files: int = None,
    stack=False,
    collapse_first_two_cols=False,
):
    """Load a set of npz or npz files from a folder.

    :param name: Environment name
    :param is_npz: If True, the files in the folder are assumed to be .npz files containing multiple fields
    :param num_files: If not None, only loads the specified number of files
    :param stack: If True, stacks the data along the first axis else concatenates it.
    :param collapse_first_two_cols: If True, collapses the first two columns after joining.
    :return: A tuple of (data, file_starts) where data is a pytree of arrays and file_starts is an array indicating the start of each file.
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

    _op = np.stack if stack else np.concatenate
    output = jax.tree.map(lambda *x: _op(x, axis=0), *data)
    if is_npz:
        num_steps = len(output[list(output.keys())[0]])
        num_steps_per_file = [len(d[list(d.keys())[0]]) for d in data]
    else:
        num_steps_per_file = [len(d) for d in data]
        num_steps = len(output)

    file_starts = np.zeros(num_steps)
    # An Array that is zero everywhere except at the start of each file
    start_indices = np.concatenate(
        [np.zeros(1, dtype=int), np.cumsum(np.array(num_steps_per_file[:-1]))]
    ).astype(int)
    file_starts[start_indices] = 1

    return output, file_starts


def read_array_header(fobj):
    version = np.lib.format.read_magic(fobj)
    func_name = "read_array_header_" + "_".join(str(v) for v in version)
    func = getattr(np.lib.format, func_name)
    return func(fobj)


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
            shape, fortan, dtype = read_array_header(npy)
            yield name[:-4], shape, dtype


def load_into_vault(
    path,
    vault_name,
    is_npz=True,
    num_files: int = None,
    vault_uid=None,
    data_transform_fn=None,
    resume_load=False,
):
    """Load NPZ files from a folder and store them in a flashbax Vault.

    Parameters
    ----------
    path : str
        Path to the directory containing rollout files.
    vault_name : str
        Name of the vault to create.
    is_npz : bool, default=True
        If True, the files in the folder are assumed to be ``.npz`` files
        containing multiple fields.
    num_files : int, optional
        If not None, only the specified number of files is loaded.
    vault_uid : str or None, optional
        Optional unique identifier for the vault. If None, a random UID is used.
    data_transform_fn : callable, optional
        A function to transform the loaded data. After transformation, the data
        should be a pytree of arrays with shape (batch_size, num_steps, *).
    resume_load : bool, default=False
        If True, resume filling a partially created vault.

    Returns
    -------
    tuple
        A tuple of ``(Vault, file_starts)``.
    """
    import flashbax as fbx
    from flashbax.vault import Vault

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
            num_steps_per_file.append(list(npz_headers(f))[0][1][1])
        else:
            # TODO: Test this
            with open(f, "rb") as _file:
                num_steps_per_file.append(read_array_header(_file)[0][0])

    total_num_steps = sum(num_steps_per_file)
    print(f"Files contain {total_num_steps:d} steps total")
    # add_batch_size = np.gcd.reduce(num_steps_per_file)

    first_batch = dict(np.load(files[0], allow_pickle=True))
    if data_transform_fn is not None:
        first_batch = data_transform_fn(first_batch)

    def batch_generator(start_index):
        for f in tqdm(files[start_index:num_files]):
            _data = dict(np.load(f, allow_pickle=True))
            if data_transform_fn is not None:
                _data = data_transform_fn(_data)
            yield _data

    return make_vault(
        vault_name=vault_name,
        total_num_steps=total_num_steps,
        batch_generator=batch_generator,
        vault_uid=vault_uid,
        resume_load=resume_load,
        batch_size=first_batch["obs"].shape[0],
        steps_per_batch=num_steps_per_file[0],
        max_length_time_axis=max(num_steps_per_file) + 1,
    )


# Toy datasets -----------------------------------------------------------------


def spirals(dataset_size=100, time=16, key=jrandom.PRNGKey(0)):
    """Create a dataset of two spirals.

    Creates x: [dataset_size, time, 2] and y: [dataset_size, 1]
    where the two classes in y are spirals that wind in opposite directions.
    """
    t = jnp.linspace(0, 2 * jnp.pi, time)
    offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * jnp.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)
    return x, jax.nn.one_hot(y, num_classes=2, axis=-1).squeeze()


def sine(length=100, offset=2, num_periods=3):
    """A simple sine-wave."""
    x = jnp.linspace(0, num_periods * 2 * jnp.pi, length)[:, None]
    y = jnp.sin(x) + offset
    return x, y


# Gym Simulations ---------------------------------------------------------------


def legacy_rollouts(data_folder="data/spring/halfcheetah", with_time=False):
    """Load a dataset from the BulletEnv simulator."""
    outputs, _ = load_np_files_from_folder(data_folder, is_npz=True, stack=False)
    obs = outputs["obs"]
    return obs, obs


def rollouts(data_folder="data/cheetah", with_time=False):
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
    load_into_vault("data", vault_name="test", vault_uid="test")

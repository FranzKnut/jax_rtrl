"""Utilities for writing and reading Flashbax vaults."""

import itertools
from collections.abc import Iterable
from typing import Any

from chex import PRNGKey
import jax
from jax import numpy as jnp, random as jrandom
import numpy as np
from flashbax.vault import Vault
from flashbax.buffers.trajectory_buffer import make_trajectory_buffer

from jax_rtrl.util.jax_util import tree_stack


def make_vault(
    vault_name: str,
    max_length_time_axis: int,
    total_num_steps: int,
    steps_per_batch: int | list[int],
    batch_generator: Iterable[Any],
    batch_size: int = 1,
    vault_uid: str | None = None,
    resume_load: bool = True,
) -> tuple[Vault, jax.Array | None]:
    """Create a vault and write data to it from a batch generator.

    Parameters
    ----------
    vault_name : str
        Name of the vault to be created. The vault will be created at the path `./vaults/{vault_name}/{vault_uid}`.
    max_length_time_axis : int
        The maximum length of the time axis for each batch of data to be written to the vault.
    total_num_steps : int
        The total number of steps to be written to the vault.
    steps_per_batch : int | list[int]
        The number of steps in each batch to be written to the vault. If an integer is provided, it will be used for all batches.
        The sum of the elements in this list must be equal to `total_num_steps`.
    batch_generator : Iterable[Any]
        A generator that yields batches of data to be written to the vault.
        The batches should be pytrees whose leaves have shape (batch_size, time, ...).
    batch_size : int, optional
        by default 1
    vault_uid : str | None, optional
        by default None
    resume_load : bool, optional
        by default True

    Returns
    -------
    tuple[Vault, jax.Array | None]
        Returns the created vault and an array indicating the start of each batch that was written to the vault.

    Raises
    ------
    ValueError
    """
    if isinstance(steps_per_batch, list):
        assert total_num_steps == np.sum(steps_per_batch), (
            "total_num_steps must be equal to the sum of steps_per_batch."
        )

    with jax.default_device(jax.devices("cpu")[0]):
        buffer = make_trajectory_buffer(
            add_batch_size=batch_size,
            max_length_time_axis=max_length_time_axis,
            sample_batch_size=1,
            sample_sequence_length=1,
            min_length_time_axis=1,
            period=1,
        )

        if isinstance(steps_per_batch, list):
            num_batches = len(steps_per_batch)
        else:
            num_batches = np.ceil(total_num_steps / steps_per_batch)

        steps_per_batch = np.reshape(steps_per_batch, -1)
        steps_per_batch_cumsum = np.cumsum(steps_per_batch)

        iterator = iter(batch_generator(0))
        try:
            first_batch = next(iterator)
        except StopIteration:
            raise ValueError("Batch generator is empty. Cannot write to vault.")

        init_data = jax.tree_util.tree_map(lambda x: x[0][0], first_batch)

        # Create vault
        vault = Vault(
            vault_name=vault_name,
            experience_structure=init_data,
            vault_uid=vault_uid,
        )
        print(f"Vault: {vault._base_path}")
        print("Index is at:", vault.vault_index)
        if (vault.vault_index > 0 and not resume_load) or (
            vault.vault_index > total_num_steps
        ):
            return vault, None  # Already (partially) exists
        else:
            _indices = np.where(steps_per_batch_cumsum > vault.vault_index)[0]
            start_index = _indices[0] if len(_indices) > 0 else 0
            if num_batches is None or start_index < num_batches:
                print(f"Resuming from file {start_index}")

        iterator = iter(batch_generator(start_index))
        try:
            first_batch = next(iterator)
        except StopIteration:
            raise ValueError(
                "total_num_steps is greater than the number of steps in the batch generator!"
            )

        # Write batches to vault
        add_batch = jax.jit(buffer.add, donate_argnums=0)
        buffer_state = buffer.init(init_data)

        written_batches = 0
        for batch in itertools.chain([first_batch], iterator):
            buffer_state = add_batch(buffer_state, batch)
            vault.write(buffer_state)
            written_batches += 1

        # An Array that is zero everywhere except at the start of each batch
        start_indices = np.concatenate(
            [np.zeros(1, dtype=int), np.cumsum(np.array(steps_per_batch[:-1]))]
        ).astype(int)
        batch_starts = np.zeros(total_num_steps, dtype=int)
        batch_starts[start_indices] = 1

        print(f"Vault contains {vault.vault_index} samples.")
        return vault, batch_starts


def read_vault_data(vault, start_id: int, num_steps: int = 1):
    """Read a chunk of data from the vault.

    :param start_id: Which start_id to read from (0-indexed)
    :param chunk_size: How much percent of the data should be read
    :return: The data chunk
    """
    start_percent = start_id * 100 / vault.vault_index
    end_percent = start_percent + num_steps * 100 / vault.vault_index
    assert start_percent >= 0, "Requested data chunk is negative."
    assert start_percent < end_percent, "start_percent must be < end_percent."

    # Making sure to do all adjustments of the data in CPU or else we risk out of memory Errors!
    with jax.default_device(jax.devices("cpu")[0]):
        _buff_state = vault.read(percentiles=(start_percent, end_percent))
        data = _buff_state.experience
        data = jax.tree.map(jnp.nan_to_num, data)
        data = jax.tree.map(lambda d: d[:, :num_steps], data)
        # data = jax.tree.map(partial(jnp.swapaxes, axis1=0, axis2=1), data)
        # Discard speed control from action

    return _buff_state.replace(experience=data)


def vault_generator(
    key: PRNGKey,
    vault: Any,
    batch_size: int,
    seq_len: int,
    num_samples: int = 1,
    eval_size: int = 0,
):
    for i in range(num_samples):
        key_sample, key = jrandom.split(key)
        batch_ids = jrandom.randint(
            key_sample,
            batch_size,
            0,
            vault.vault_index - seq_len - eval_size,
        )
        # HACK: Read a bit too much to avoid too small sequences
        batch_ids = [max(b - 0.1, 0) for b in batch_ids]
        jit_tree_stack = jax.jit(tree_stack, static_argnames=["concatenate"])
        sample = jit_tree_stack(
            [read_vault_data(vault, start_id=c, num_steps=seq_len) for c in batch_ids],
            concatenate=True,
        ).experience
        yield sample

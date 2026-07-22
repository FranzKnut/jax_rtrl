"""Utilities for writing Flashbax vaults from sequential generators."""

import itertools
from collections.abc import Callable, Iterable
from typing import Any

import jax
from flashbax.vault import Vault
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer


class SequentialVault(Vault):
    """Vault that can be filled sequentially from a generator of batches."""

    def write_numpy_batches(
        self,
        batch_generator: Iterable[Any],
        *,
        buffer: TrajectoryBuffer,
        data_transform_fn: Callable[[Any], Any] | None = None,
        init_data_fn: Callable[[Any], Any] | None = None,
    ) -> int:
        """Write batches from a generator into the vault in order."""

        iterator = iter(batch_generator)
        try:
            first_batch = next(iterator)
        except StopIteration:
            return 0

        if data_transform_fn is not None:
            first_batch = data_transform_fn(first_batch)

        if init_data_fn is None:

            def init_data_fn(batch):
                return jax.tree_util.tree_map(lambda x: x[0, 0], batch)

        buffer_state = buffer.init(init_data_fn(first_batch))
        add_batch = jax.jit(buffer.add, donate_argnums=0)

        written_batches = 0
        for batch in itertools.chain([first_batch], iterator):
            buffer_state = add_batch(buffer_state, batch)
            self.write(buffer_state)
            written_batches += 1

        return written_batches

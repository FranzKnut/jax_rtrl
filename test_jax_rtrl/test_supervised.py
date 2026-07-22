import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import jax.numpy as jnp

from jax_rtrl.supervised.sequential_vault import SequentialVault


class TestSupervised(unittest.TestCase):
    def test_sequential_vault_writes_batches_in_order(self):
        class FakeBuffer:
            def __init__(self):
                self.init_calls = []
                self.add_calls = []

            def init(self, init_data):
                self.init_calls.append(init_data)
                return SimpleNamespace(experience=init_data, current_index=0)

            def add(self, buffer_state, batch):
                self.add_calls.append(batch)
                return SimpleNamespace(
                    experience=batch, current_index=buffer_state.current_index + 1
                )

        batches = [
            {"obs": jnp.array([[1, 2], [3, 4]])},
            {"obs": jnp.array([[5, 6], [7, 8]])},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                buffer = FakeBuffer()

                with patch("jax_rtrl.supervised.sequential_vault.jax.jit", side_effect=lambda fn, **kwargs: fn), patch.object(SequentialVault, "write", autospec=True) as write_mock:
                    vault = SequentialVault(
                        vault_name="test_vault",
                        experience_structure=jnp.zeros((1, 1)),
                        vault_uid="test_uid",
                    )

                    written = vault.write_numpy_batches(
                        iter(batches),
                        buffer=buffer,
                        init_data_fn=lambda batch: batch["obs"][0],
                    )

                self.assertEqual(written, 2)
                self.assertEqual(len(buffer.init_calls), 1)
                self.assertTrue(np.array_equal(buffer.init_calls[0], batches[0]["obs"][0]))
                self.assertEqual(len(buffer.add_calls), 2)
                self.assertTrue(np.array_equal(buffer.add_calls[0]["obs"], batches[0]["obs"]))
                self.assertTrue(np.array_equal(buffer.add_calls[1]["obs"], batches[1]["obs"]))
                self.assertEqual(write_mock.call_count, 2)
                self.assertTrue(np.array_equal(write_mock.call_args_list[0].args[1].experience["obs"], batches[0]["obs"]))
                self.assertTrue(np.array_equal(write_mock.call_args_list[1].args[1].experience["obs"], batches[1]["obs"]))
            finally:
                os.chdir(cwd)
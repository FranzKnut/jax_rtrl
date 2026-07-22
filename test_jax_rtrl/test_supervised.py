import tempfile
import unittest

import jax.numpy as jnp

from jax_rtrl.supervised.sequential_vault import (
    make_vault,
    vault_generator,
    read_vault_data,
)

import shutil
from jax import random as jrandom


class TestSequentialVault(unittest.TestCase):
    """Test cases for sequential_vault module."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        cls.key = jrandom.PRNGKey(42)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def _create_batch_generator(self, num_batches=3, batch_size=2, steps=2):
        """Create a simple batch generator for testing."""

        def generator(start_index=0):
            for i in range(start_index, num_batches):
                yield {
                    "experience": jnp.ones((batch_size, steps, 4)),
                    "reward": jnp.zeros((batch_size, steps)),
                }

        return generator

    def test_make_vault_basic(self):
        """Test basic vault creation and writing."""
        generator = self._create_batch_generator()

        vault, batch_starts = make_vault(
            vault_name="test_vault_basic",
            max_length_time_axis=5,
            total_num_steps=15,
            steps_per_batch=5,
            batch_generator=generator,
            batch_size=2,
            base_path=self.test_dir,
            resume_load=False,
        )

        self.assertIsNotNone(vault)
        self.assertEqual(vault.vault_index, 6)  # 3 batches * 2 steps each
        self.assertIsNotNone(batch_starts)

    def test_make_vault_steps_per_batch_list(self):
        """Test vault creation with steps_per_batch as list."""
        generator = self._create_batch_generator()

        vault, batch_starts = make_vault(
            vault_name="test_vault_list",
            max_length_time_axis=5,
            total_num_steps=15,
            steps_per_batch=[5, 5, 5],
            batch_generator=generator,
            batch_size=2,
            base_path=self.test_dir,
            resume_load=False,
        )

        self.assertEqual(vault.vault_index, 6)

    def test_make_vault_empty_generator(self):
        """Test error handling for empty batch generator."""

        def empty_generator(start_index=0):
            return iter([])

        with self.assertRaises(ValueError):
            make_vault(
                vault_name="test_empty",
                max_length_time_axis=5,
                total_num_steps=5,
                steps_per_batch=5,
                batch_generator=empty_generator,
                base_path=self.test_dir,
                resume_load=False,
            )

    def test_make_vault_steps_mismatch(self):
        """Test error when total_num_steps doesn't match steps_per_batch sum."""
        generator = self._create_batch_generator()

        with self.assertRaises(AssertionError):
            make_vault(
                vault_name="test_mismatch",
                max_length_time_axis=5,
                total_num_steps=20,  # Doesn't match [5, 5, 5] = 15
                steps_per_batch=[5, 5, 5],
                batch_generator=generator,
                base_path=self.test_dir,
                resume_load=False,
            )

    def test_read_vault_data(self):
        """Test reading data from vault."""
        generator = self._create_batch_generator()

        vault, _ = make_vault(
            vault_name="test_vault_read",
            max_length_time_axis=5,
            total_num_steps=15,
            steps_per_batch=5,
            batch_generator=generator,
            batch_size=2,
            base_path=self.test_dir,
            resume_load=False,
        )

        data = read_vault_data(vault, start_id=0, num_steps=5)

        self.assertIsNotNone(data)
        self.assertIn("experience", data)

    def test_read_vault_data_invalid_percent(self):
        """Test reading with invalid percentage raises assertion."""
        generator = self._create_batch_generator()

        vault, _ = make_vault(
            vault_name="test_vault_invalid",
            max_length_time_axis=5,
            total_num_steps=15,
            steps_per_batch=5,
            batch_generator=generator,
            batch_size=2,
            base_path=self.test_dir,
            resume_load=False,
        )

        with self.assertRaises(AssertionError):
            read_vault_data(vault, start_id=-1, num_steps=5)

    def test_vault_generator(self):
        """Test vault sample generation."""
        generator = self._create_batch_generator()

        vault, _ = make_vault(
            vault_name="test_vault_gen",
            max_length_time_axis=5,
            total_num_steps=15,
            steps_per_batch=5,
            batch_generator=generator,
            batch_size=2,
            base_path=self.test_dir,
            resume_load=False,
        )

        samples = list(
            vault_generator(
                key=self.key, vault=vault, batch_size=1, seq_len=5, num_samples=2
            )
        )

        self.assertEqual(len(samples), 2)
        self.assertIn("experience", samples[0])

    def test_resume_load(self):
        """Test vault resume functionality."""
        vault_name = "test_vault_resume"
        generator = self._create_batch_generator()

        # Create vault first time
        vault1, _ = make_vault(
            vault_name=vault_name,
            max_length_time_axis=5,
            total_num_steps=15,
            steps_per_batch=5,
            batch_generator=generator,
            batch_size=2,
            base_path=self.test_dir,
            resume_load=True,
        )

        first_index = vault1.vault_index

        # Create vault second time (should resume)
        vault2, _ = make_vault(
            vault_name=vault_name,
            max_length_time_axis=5,
            total_num_steps=15,
            steps_per_batch=5,
            batch_generator=generator,
            batch_size=2,
            base_path=self.test_dir,
            resume_load=True,
        )

        self.assertEqual(vault2.vault_index, first_index + 6)

if __name__ == "__main__":
    unittest.main()

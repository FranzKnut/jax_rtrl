import unittest

import jax
import jax.numpy as jnp
from jax_rtrl.models.cells.attention import AttentionCell, CausalAttentionCell


class AttentionCellTestBase(unittest.TestCase):
    """Base class for attention cell tests with common setup logic."""

    cell_cls = AttentionCell

    def get_cell_kwargs(self):
        """Override in subclasses to provide specific cell parameters."""
        return {
            "d_hidden": 6,
            "num_heads": 2,
        }

    def setUp(self):
        self.cell = self.cell_cls(**self.get_cell_kwargs())
        self.steps = 5
        self.input_dim = 4
        self.input_data = jax.random.normal(
            jax.random.PRNGKey(0), (self.steps, self.input_dim)
        )
        self.params = self.cell.init(jax.random.PRNGKey(1), None, self.input_data)

    def jacobian(self, resets=None):
        """Jacobian of the output sequence w.r.t. the input sequence."""

        def forward(x):
            return self.cell.apply(self.params, None, x, resets)[1]

        return jax.jacobian(forward)(self.input_data)


class AttentionCellTest(AttentionCellTestBase):
    """Tests for the bidirectional (non-causal) attention cell."""

    cell_cls = AttentionCell

    def test_output_shape(self):
        """Test output shape for a sequence input."""
        carry, out = self.cell.apply(self.params, None, self.input_data)
        self.assertEqual(out.shape, (self.steps, self.cell.d_hidden))

    def test_single_step_output_shape(self):
        """Test output shape for a single-step (non-sequence) input."""
        carry, out = self.cell.apply(self.params, None, self.input_data[0])
        self.assertEqual(out.shape, (self.cell.d_hidden,))

    def test_carry_passthrough(self):
        """Test that the (unused) carry is passed through unchanged."""
        carry_in = jnp.ones(())
        carry_out, _ = self.cell.apply(self.params, carry_in, self.input_data)
        self.assertTrue(jnp.array_equal(carry_in, carry_out))

    def test_attends_to_future(self):
        """Test that outputs depend on both past and future positions."""
        jac = self.jacobian()
        has_future_dependency = any(
            not jnp.allclose(jac[i, :, j, :], 0.0)
            for i in range(self.steps)
            for j in range(i + 1, self.steps)
        )
        self.assertTrue(
            has_future_dependency,
            "Bidirectional attention should let outputs depend on future inputs",
        )

    def test_resets_block_cross_episode_attention(self):
        """Test that a reset prevents attention across the episode boundary in either direction."""
        resets = jnp.array([False, False, True, False, False])
        jac = self.jacobian(resets=resets)
        for i in range(2, self.steps):
            for j in range(0, 2):
                self.assertTrue(
                    jnp.allclose(jac[i, :, j, :], 0.0),
                    f"Output at step {i} should not depend on pre-reset input at step {j}",
                )
                self.assertTrue(
                    jnp.allclose(jac[j, :, i, :], 0.0),
                    f"Output at step {j} should not depend on post-reset input at step {i}",
                )
        # Steps within the same episode should still attend to each other.
        self.assertFalse(jnp.allclose(jac[1, :, 0, :], 0.0))
        self.assertFalse(jnp.allclose(jac[3, :, 2, :], 0.0))


class CausalAttentionCellTest(AttentionCellTestBase):
    """Tests for the causal attention cell."""

    cell_cls = CausalAttentionCell

    def test_output_shape(self):
        """Test output shape for a sequence input."""
        carry, out = self.cell.apply(self.params, None, self.input_data)
        self.assertEqual(out.shape, (self.steps, self.cell.d_hidden))

    def test_single_step_output_shape(self):
        """Test output shape for a single-step (non-sequence) input."""
        carry, out = self.cell.apply(self.params, None, self.input_data[0])
        self.assertEqual(out.shape, (self.cell.d_hidden,))

    def test_causal_mask_blocks_future(self):
        """Test that outputs never depend on future inputs."""
        jac = self.jacobian()
        for i in range(self.steps):
            for j in range(i + 1, self.steps):
                self.assertTrue(
                    jnp.allclose(jac[i, :, j, :], 0.0),
                    f"Output at step {i} should not depend on future input at step {j}",
                )

    def test_attends_to_past_and_present(self):
        """Test that outputs do depend on past and present inputs."""
        jac = self.jacobian()
        for i in range(self.steps):
            for j in range(0, i + 1):
                self.assertFalse(
                    jnp.allclose(jac[i, :, j, :], 0.0),
                    f"Output at step {i} should depend on input at step {j}",
                )

    def test_resets_block_cross_episode_attention(self):
        """Test that a reset prevents attention to inputs before the episode boundary."""
        resets = jnp.array([False, False, True, False, False])
        jac = self.jacobian(resets=resets)
        for i in range(2, self.steps):
            for j in range(0, 2):
                self.assertTrue(
                    jnp.allclose(jac[i, :, j, :], 0.0),
                    f"Output at step {i} should not depend on pre-reset input at step {j}",
                )
        # Steps within the same episode should still attend to the past.
        self.assertFalse(jnp.allclose(jac[1, :, 0, :], 0.0))
        self.assertFalse(jnp.allclose(jac[3, :, 2, :], 0.0))


if __name__ == "__main__":
    unittest.main()

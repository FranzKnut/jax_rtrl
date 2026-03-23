"""Tests for Hopfield Network layer implementations."""

import unittest

import jax
import jax.numpy as jnp

from jax_rtrl.models.cells.hopfield import (
    HopfieldLayer,
    classical_update,
    hebbian_weights,
    modern_update,
)

A_TOL = 1e-5


class TestHebbianWeights(unittest.TestCase):
    """Unit tests for the hebbian_weights helper function."""

    def test_shape(self):
        """Weight matrix has shape (num_units, num_units)."""
        patterns = jnp.ones((3, 8))
        W = hebbian_weights(patterns)
        self.assertEqual(W.shape, (8, 8))

    def test_symmetry(self):
        """Weight matrix is symmetric."""
        patterns = jax.random.normal(jax.random.PRNGKey(0), (5, 10))
        W = hebbian_weights(patterns)
        self.assertTrue(jnp.allclose(W, W.T, atol=A_TOL))

    def test_zero_diagonal(self):
        """Diagonal of weight matrix is zero (no self-connections)."""
        patterns = jax.random.normal(jax.random.PRNGKey(0), (5, 10))
        W = hebbian_weights(patterns)
        self.assertTrue(jnp.allclose(jnp.diag(W), jnp.zeros(10), atol=A_TOL))


class TestClassicalUpdate(unittest.TestCase):
    """Unit tests for the classical_update helper function."""

    def test_output_values(self):
        """Output is strictly bipolar {-1, +1}."""
        W = jnp.eye(4)
        x = jnp.array([0.5, -0.5, 0.0, -0.1])
        out = classical_update(x, W)
        self.assertTrue(jnp.all(jnp.abs(out) == 1.0))

    def test_non_negative_maps_to_plus_one(self):
        """Non-negative net-input maps to +1."""
        W = jnp.eye(4)
        x = jnp.array([1.0, 0.0, 0.0, 0.0])
        out = classical_update(x, W)
        self.assertEqual(float(out[0]), 1.0)

    def test_negative_maps_to_minus_one(self):
        """Negative net-input maps to -1."""
        W = jnp.eye(4)
        x = jnp.array([-1.0, 0.0, 0.0, 0.0])
        out = classical_update(x, W)
        self.assertEqual(float(out[0]), -1.0)


class TestModernUpdate(unittest.TestCase):
    """Unit tests for the modern_update helper function."""

    def test_output_shape(self):
        """Output has the same shape as the query."""
        patterns = jax.random.normal(jax.random.PRNGKey(0), (6, 8))
        x = jax.random.normal(jax.random.PRNGKey(1), (8,))
        out = modern_update(x, patterns)
        self.assertEqual(out.shape, (8,))

    def test_batched_output_shape(self):
        """Output has correct shape with a batch dimension."""
        patterns = jax.random.normal(jax.random.PRNGKey(0), (6, 8))
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
        out = modern_update(x, patterns)
        self.assertEqual(out.shape, (4, 8))

    def test_high_beta_retrieves_closest(self):
        """With very high beta the output is the pattern closest to the query."""
        # Use identity matrix as patterns so each column is a basis vector.
        d = 8
        patterns = jnp.eye(d)
        # Query is the 3rd basis vector with slight noise.
        x = patterns[2] + 0.01 * jax.random.normal(jax.random.PRNGKey(7), (d,))
        out = modern_update(x, patterns, beta=1000.0)
        self.assertTrue(jnp.allclose(out, patterns[2], atol=1e-3))


class TestClassicalHopfieldLayer(unittest.TestCase):
    """Tests for HopfieldLayer in classical mode."""

    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.num_units = 8

    def test_output_shape(self):
        """Output has shape (..., num_units)."""
        layer = HopfieldLayer(num_units=self.num_units, mode="classical")
        x = jax.random.normal(self.rng, (self.num_units,))
        patterns = jax.random.normal(self.rng, (4, self.num_units))
        params = layer.init(self.rng, x, patterns)
        out = layer.apply(params, x, patterns)
        self.assertEqual(out.shape, (self.num_units,))

    def test_batched_output_shape(self):
        """Output preserves a leading batch dimension."""
        layer = HopfieldLayer(num_units=self.num_units, mode="classical")
        x = jax.random.normal(self.rng, (5, self.num_units))
        patterns = jax.random.normal(self.rng, (4, self.num_units))
        params = layer.init(self.rng, x, patterns)
        out = layer.apply(params, x, patterns)
        self.assertEqual(out.shape, (5, self.num_units))

    def test_retrieves_stored_pattern(self):
        """Exact stored pattern is a fixed point of the classical network."""
        # Two orthogonal bipolar patterns of length 4.
        patterns = jnp.array([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]])
        layer = HopfieldLayer(num_units=4, mode="classical", num_steps=10)
        params = layer.init(self.rng, patterns[0], patterns)
        retrieved = layer.apply(params, patterns[0], patterns)
        self.assertTrue(
            jnp.allclose(retrieved, patterns[0], atol=A_TOL),
            "Exact stored pattern should be retrieved unchanged.",
        )

    def test_output_is_bipolar(self):
        """All output values are in {-1, +1}."""
        layer = HopfieldLayer(num_units=self.num_units, mode="classical", num_steps=3)
        x = jax.random.normal(self.rng, (self.num_units,))
        patterns = jax.random.normal(self.rng, (4, self.num_units))
        params = layer.init(self.rng, x, patterns)
        out = layer.apply(params, x, patterns)
        self.assertTrue(jnp.all(jnp.abs(out) == 1.0))

    def test_learned_weight_matrix(self):
        """Classical layer with learned weights (no patterns argument) runs and has the right shape."""
        layer = HopfieldLayer(num_units=self.num_units, mode="classical")
        x = jax.random.normal(self.rng, (self.num_units,))
        params = layer.init(self.rng, x)
        out = layer.apply(params, x)
        self.assertEqual(out.shape, (self.num_units,))
        # The weight param should exist and be square.
        self.assertIn("W", params["params"])
        self.assertEqual(params["params"]["W"].shape, (self.num_units, self.num_units))


class TestModernHopfieldLayer(unittest.TestCase):
    """Tests for HopfieldLayer in modern mode."""

    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.num_units = 16

    def test_output_shape(self):
        """Output has shape (..., num_units)."""
        layer = HopfieldLayer(num_units=self.num_units, mode="modern")
        x = jax.random.normal(self.rng, (self.num_units,))
        patterns = jax.random.normal(self.rng, (8, self.num_units))
        params = layer.init(self.rng, x, patterns)
        out = layer.apply(params, x, patterns)
        self.assertEqual(out.shape, (self.num_units,))

    def test_batched_output_shape(self):
        """Output preserves a leading batch dimension."""
        layer = HopfieldLayer(num_units=self.num_units, mode="modern")
        x = jax.random.normal(self.rng, (4, self.num_units))
        patterns = jax.random.normal(self.rng, (8, self.num_units))
        params = layer.init(self.rng, x, patterns)
        out = layer.apply(params, x, patterns)
        self.assertEqual(out.shape, (4, self.num_units))

    def test_learned_patterns_shape(self):
        """Learned patterns parameter has shape (num_stored_patterns, num_units)."""
        num_stored = 32
        layer = HopfieldLayer(
            num_units=self.num_units, mode="modern", num_stored_patterns=num_stored
        )
        x = jax.random.normal(self.rng, (self.num_units,))
        params = layer.init(self.rng, x)
        self.assertIn("patterns", params["params"])
        self.assertEqual(
            params["params"]["patterns"].shape, (num_stored, self.num_units)
        )

    def test_default_learned_patterns_shape(self):
        """Without num_stored_patterns, the default is num_units patterns."""
        layer = HopfieldLayer(num_units=self.num_units, mode="modern")
        x = jax.random.normal(self.rng, (self.num_units,))
        params = layer.init(self.rng, x)
        self.assertEqual(
            params["params"]["patterns"].shape, (self.num_units, self.num_units)
        )

    def test_high_beta_retrieves_closest_pattern(self):
        """With high beta, the layer retrieves the closest stored pattern."""
        d = self.num_units
        patterns = jnp.eye(d)  # orthonormal basis
        layer = HopfieldLayer(num_units=d, mode="modern", beta=500.0)
        # Query is the 5th basis vector with slight noise.
        x = patterns[5] + 0.01 * jax.random.normal(self.rng, (d,))
        params = layer.init(self.rng, x, patterns)
        out = layer.apply(params, x, patterns)
        self.assertTrue(
            jnp.allclose(out, patterns[5], atol=1e-3),
            "With high beta the closest pattern should be retrieved.",
        )

    def test_gradient_flow_external_patterns(self):
        """Gradients flow through the layer when patterns are provided externally."""
        layer = HopfieldLayer(num_units=self.num_units, mode="modern")
        x = jax.random.normal(self.rng, (self.num_units,))
        patterns = jax.random.normal(self.rng, (8, self.num_units))
        params = layer.init(self.rng, x, patterns)

        def loss_fn(params):
            out = layer.apply(params, x, patterns)
            return jnp.sum(out**2)

        grads = jax.grad(loss_fn)(params)
        # No learned params in this case – gradient computation should not error.
        self.assertIsNotNone(grads)

    def test_gradient_flow_learned_patterns(self):
        """Gradients flow through learned stored patterns."""
        layer = HopfieldLayer(num_units=self.num_units, mode="modern")
        x = jax.random.normal(self.rng, (self.num_units,))
        params = layer.init(self.rng, x)

        def loss_fn(params):
            out = layer.apply(params, x)
            return jnp.sum(out**2)

        grads = jax.grad(loss_fn)(params)
        self.assertTrue(
            jnp.any(grads["params"]["patterns"] != 0),
            "Gradients w.r.t. learned patterns should be non-zero.",
        )

    def test_multiple_steps_converges(self):
        """More retrieval steps should not increase the energy."""
        d = self.num_units
        patterns = jax.random.normal(jax.random.PRNGKey(1), (d, d))
        beta = 2.0
        x_init = jax.random.normal(self.rng, (d,))

        def energy(x):
            scores = beta * patterns @ x
            return -jax.nn.logsumexp(scores) + 0.5 * jnp.dot(x, x)

        layer_1 = HopfieldLayer(num_units=d, mode="modern", beta=beta, num_steps=1)
        layer_5 = HopfieldLayer(num_units=d, mode="modern", beta=beta, num_steps=5)
        # Both layers have no learnable params when patterns are provided.
        params = layer_1.init(self.rng, x_init, patterns)

        x_1 = layer_1.apply(params, x_init, patterns)
        x_5 = layer_5.apply(params, x_init, patterns)

        e_init = float(energy(x_init))
        e_1 = float(energy(x_1))
        e_5 = float(energy(x_5))

        self.assertLessEqual(e_1, e_init + 1e-5, "One step should not increase energy.")
        self.assertLessEqual(e_5, e_init + 1e-5, "Five steps should not increase energy.")


class TestHopfieldLayerModeSwitch(unittest.TestCase):
    """Tests ensuring mode selection works correctly."""

    def setUp(self):
        self.rng = jax.random.PRNGKey(42)
        self.num_units = 8

    def test_modern_mode_default(self):
        """Default mode is 'modern'."""
        layer = HopfieldLayer(num_units=self.num_units)
        self.assertEqual(layer.mode, "modern")

    def test_classical_mode_attribute(self):
        """Classical mode attribute is set correctly."""
        layer = HopfieldLayer(num_units=self.num_units, mode="classical")
        self.assertEqual(layer.mode, "classical")

    def test_classical_output_differs_from_modern(self):
        """Classical and modern modes produce different outputs."""
        x = jax.random.normal(self.rng, (self.num_units,))
        patterns = jax.random.normal(self.rng, (4, self.num_units))

        layer_c = HopfieldLayer(num_units=self.num_units, mode="classical")
        layer_m = HopfieldLayer(num_units=self.num_units, mode="modern")

        params_c = layer_c.init(self.rng, x, patterns)
        params_m = layer_m.init(self.rng, x, patterns)

        out_c = layer_c.apply(params_c, x, patterns)
        out_m = layer_m.apply(params_m, x, patterns)

        # Outputs should differ: classical is bipolar, modern is continuous.
        self.assertFalse(
            jnp.allclose(out_c, out_m, atol=A_TOL),
            "Classical and modern modes should produce different outputs.",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for Hopfield Network cell implementations."""

import unittest

import jax
import jax.numpy as jnp

from jax_rtrl.models.cells.hopfield import (
    HopfieldCell,
    classical_hopfield,
    modern_hopfield,
)
from jax_rtrl.models.seq_models import scan_rnn
from jax_rtrl.util.jax_util import mse_loss

A_TOL = 1e-5


## ODE function tests


class TestClassicalHopfieldODE(unittest.TestCase):
    """Tests for the classical_hopfield standalone ODE function."""

    def setUp(self):
        self.num_units = 5
        self.input_dim = 3
        key = jax.random.PRNGKey(0)
        w_shape = (self.num_units, self.input_dim + self.num_units + 1)
        self.params = {
            "W": jax.random.normal(key, w_shape),
            "tau": jnp.ones(self.num_units),
        }
        self.h = jax.random.normal(jax.random.PRNGKey(1), (self.num_units,))
        self.x = jax.random.normal(jax.random.PRNGKey(2), (self.input_dim,))

    def test_output_shape(self):
        """ODE derivative has shape (num_units,)."""
        dh = classical_hopfield(self.params, self.h, self.x)
        self.assertEqual(dh.shape, (self.num_units,))

    def test_fixed_point_is_bipolar(self):
        """At a fixed point (dh/dt = 0), the hidden state must be bipolar."""
        # At a fixed point: h = sign(W @ [x, h, 1]), so h is bipolar.
        # Simulate convergence by many Euler steps.
        h = jnp.sign(self.h)  # start close to a bipolar state
        tau = jnp.ones(self.num_units) * 0.1
        params = {**self.params, "tau": tau}
        for _ in range(200):
            dh = classical_hopfield(params, h, self.x)
            h = h + 0.1 * dh
        # After convergence the state should be approximately bipolar.
        self.assertTrue(jnp.allclose(jnp.abs(h), jnp.ones_like(h), atol=0.01))


class TestModernHopfieldODE(unittest.TestCase):
    """Tests for the modern_hopfield standalone ODE function."""

    def setUp(self):
        self.num_units = 8
        self.input_dim = 3
        self.num_stored = 8
        key = jax.random.PRNGKey(0)
        w_shape = (self.num_units, self.input_dim + self.num_units + 1)
        self.params = {
            "W": jax.random.normal(key, w_shape),
            "tau": jnp.ones(self.num_units),
            "patterns": jnp.eye(self.num_stored),  # orthonormal patterns
        }
        self.h = jax.random.normal(jax.random.PRNGKey(1), (self.num_units,))
        self.x = jnp.zeros(self.input_dim)

    def test_output_shape(self):
        """ODE derivative has shape (num_units,)."""
        dh = modern_hopfield(self.params, self.h, self.x)
        self.assertEqual(dh.shape, (self.num_units,))

    def test_high_beta_attracts_to_nearest_pattern(self):
        """With very high beta and zero input, ODE converges to nearest pattern."""
        # Zero W so query = W @ [x, h, 1] = h (only recurrent part).
        # Use identity patterns so similarity = h[i].
        w_shape = (self.num_units, self.input_dim + self.num_units + 1)
        W_zero_input = jnp.zeros(w_shape)
        # Make the recurrent block an identity so query = h.
        W_zero_input = W_zero_input.at[:, self.input_dim : self.input_dim + self.num_units].set(
            jnp.eye(self.num_units)
        )
        params = {**self.params, "W": W_zero_input, "tau": jnp.ones(self.num_units) * 0.1}
        # Start hidden state near pattern 3.
        h = jnp.eye(self.num_units)[3] * 0.9
        for _ in range(300):
            dh = modern_hopfield(params, h, self.x, beta=100.0)
            h = h + 0.1 * dh
        self.assertTrue(
            jnp.allclose(h, jnp.eye(self.num_units)[3], atol=0.05),
            "ODE should converge to the nearest stored pattern.",
        )


## Cell tests


class HopfieldCellTestBase(unittest.TestCase):
    """Base class for HopfieldCell tests with common setup."""

    def get_cell_kwargs(self):
        return {"num_units": 5, "ode_type": "modern"}

    def setUp(self):
        self.cell = HopfieldCell(**self.get_cell_kwargs())
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        self.target = jax.random.normal(jax.random.PRNGKey(1), (5,))
        self.params = self.cell.init(jax.random.PRNGKey(3), None, self.input_data[0])
        self.initialize_carry = lambda: self.cell.apply(
            self.params,
            jax.random.PRNGKey(4),
            self.input_data[0].shape,
            method=self.cell.initialize_carry,
        )


class TestModernHopfieldCell(HopfieldCellTestBase):
    def get_cell_kwargs(self):
        return {"num_units": 5, "ode_type": "modern", "beta": 2.0}

    def test_output_shape(self):
        """Cell output has shape (num_units,)."""
        h = self.initialize_carry()
        carry, out = self.cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(out.shape, (5,))

    def test_carry_shape(self):
        """Carry has the same shape as the output."""
        h = self.initialize_carry()
        carry, out = self.cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(carry.shape, out.shape)

    def test_params_structure(self):
        """Modern mode creates W, tau, and patterns parameters."""
        self.assertIn("W", self.params["params"])
        self.assertIn("tau", self.params["params"])
        self.assertIn("patterns", self.params["params"])

    def test_patterns_shape(self):
        """Stored patterns have shape (num_stored_patterns, num_units)."""
        self.assertEqual(
            self.params["params"]["patterns"].shape, (5, 5)
        )

    def test_custom_num_stored_patterns(self):
        """num_stored_patterns attribute controls the patterns parameter shape."""
        cell = HopfieldCell(num_units=5, ode_type="modern", num_stored_patterns=12)
        params = cell.init(jax.random.PRNGKey(0), None, self.input_data[0])
        self.assertEqual(params["params"]["patterns"].shape, (12, 5))

    def test_gradient_flow(self):
        """Gradients flow through all parameters."""
        h = self.initialize_carry()
        loss_fn = jax.grad(
            lambda params, h: mse_loss(
                self.cell.apply(params, h, self.input_data[0])[1], self.target
            )
        )
        grads = loss_fn(self.params, h)["params"]
        for key in grads:
            self.assertFalse(
                jnp.all(grads[key] == 0),
                f"Gradient for '{key}' should be non-zero.",
            )

    def test_multi_step_gradient_flow(self):
        """Gradients flow over multiple sequence steps."""
        h = self.initialize_carry()
        loss_fn = jax.grad(
            lambda params, _h: mse_loss(
                scan_rnn(self.cell, params, self.input_data, init_carry=_h)[1],
                self.target,
            )
        )
        grads = loss_fn(self.params, h)["params"]
        self.assertIsNotNone(grads)


class TestClassicalHopfieldCell(HopfieldCellTestBase):
    def get_cell_kwargs(self):
        return {"num_units": 5, "ode_type": "classical"}

    def test_output_shape(self):
        """Cell output has shape (num_units,)."""
        h = self.initialize_carry()
        carry, out = self.cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(out.shape, (5,))

    def test_carry_shape(self):
        """Carry has the same shape as the output."""
        h = self.initialize_carry()
        carry, out = self.cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(carry.shape, out.shape)

    def test_params_structure(self):
        """Classical mode creates W and tau parameters (no patterns)."""
        self.assertIn("W", self.params["params"])
        self.assertIn("tau", self.params["params"])
        self.assertNotIn("patterns", self.params["params"])

    def test_w_shape(self):
        """W has shape (num_units, input_dim + num_units + 1)."""
        input_dim = self.input_data.shape[-1]
        expected = (5, input_dim + 5 + 1)
        self.assertEqual(self.params["params"]["W"].shape, expected)

    def test_gradient_flow(self):
        """tau gradient flows; W gradient is zero because sign() is non-differentiable."""
        h = self.initialize_carry()
        loss_fn = jax.grad(
            lambda params, h: mse_loss(
                self.cell.apply(params, h, self.input_data[0])[1], self.target
            )
        )
        grads = loss_fn(self.params, h)["params"]
        # sign() has zero gradient, so W gradient is expected to be zero.
        self.assertTrue(
            jnp.all(grads["W"] == 0),
            "W gradient should be zero (sign() is non-differentiable).",
        )
        # tau gradient should be non-zero: d/d_tau = -(sign(u)-h)/tau^2
        self.assertFalse(
            jnp.all(grads["tau"] == 0),
            "tau gradient should be non-zero.",
        )

    def test_multi_step_gradient_flow(self):
        """Gradients flow over multiple sequence steps."""
        h = self.initialize_carry()
        loss_fn = jax.grad(
            lambda params, _h: mse_loss(
                scan_rnn(self.cell, params, self.input_data, init_carry=_h)[1],
                self.target,
            )
        )
        grads = loss_fn(self.params, h)["params"]
        self.assertIsNotNone(grads)


class TestHopfieldCellUpdateType(unittest.TestCase):
    """Tests for ode_type selection and cell-level differences."""

    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.num_units = 8
        self.input_data = jax.random.normal(self.rng, (10, 4))

    def test_default_ode_type_is_modern(self):
        """Default ode_type is 'modern'."""
        cell = HopfieldCell(num_units=self.num_units)
        self.assertEqual(cell.ode_type, "modern")

    def test_classical_ode_type(self):
        """ode_type='classical' attribute is stored correctly."""
        cell = HopfieldCell(num_units=self.num_units, ode_type="classical")
        self.assertEqual(cell.ode_type, "classical")

    def test_invalid_ode_type_raises(self):
        """Invalid ode_type raises ValueError at call time."""
        cell = HopfieldCell(num_units=self.num_units, ode_type="unknown")
        with self.assertRaises(ValueError):
            cell.init(self.rng, None, self.input_data[0])

    def test_classical_and_modern_produce_different_outputs(self):
        """Classical and modern modes produce different outputs for the same input."""
        x = jax.random.normal(self.rng, (self.input_data.shape[-1],))

        cell_c = HopfieldCell(num_units=self.num_units, ode_type="classical")
        cell_m = HopfieldCell(num_units=self.num_units, ode_type="modern")

        params_c = cell_c.init(self.rng, None, x)
        params_m = cell_m.init(self.rng, None, x)

        h_c = cell_c.apply(
            params_c, self.rng, x.shape, method=cell_c.initialize_carry
        )
        h_m = cell_m.apply(
            params_m, self.rng, x.shape, method=cell_m.initialize_carry
        )

        _, out_c = cell_c.apply(params_c, h_c, x)
        _, out_m = cell_m.apply(params_m, h_m, x)

        self.assertFalse(
            jnp.allclose(out_c, out_m, atol=A_TOL),
            "Classical and modern modes should produce different outputs.",
        )


if __name__ == "__main__":
    unittest.main()

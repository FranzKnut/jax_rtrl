import unittest

import jax
import jax.numpy as jnp
from jax_rtrl.models.cells.lrc import OnlineLRCCell
from jax_rtrl.models.seq_models import scan_rnn
from jax_rtrl.util.jax_util import mse_loss

A_TOL = 1e-5
# DEER converges iteratively; allow a slightly larger tolerance for gradient
# comparisons (the forward pass converges well within 10 iterations for small
# models, but the backward pass propagated through those iterations can
# accumulate minor floating-point differences).
DEER_TOL = 1e-3

# jax.config.update("jax_disable_jit", True)


class TestLRCGradients(unittest.TestCase):
    def setUp(self):
        self.cell = OnlineLRCCell(num_units=5, plasticity="bptt")
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        self.target = jax.random.normal(jax.random.PRNGKey(1), (5,))
        self.params = self.cell.init(jax.random.PRNGKey(3), None, self.input_data[0])
        self.initialize_carry = lambda: self.cell.apply(
            self.params,
            jax.random.PRNGKey(4),
            self.input_data[0].shape,
            method=self.cell.initialize_carry,
        )
        h = self.initialize_carry()

        self.loss_fn = jax.grad(
            lambda params, h: mse_loss(
                self.cell.apply(params, h, self.input_data[0])[1], self.target[0]
            )
        )

        self.bptt_grads_one_step = self.loss_fn(self.params, h)["params"]

        self.multi_step_loss_fn = jax.grad(
            lambda params, h: mse_loss(
                scan_rnn(self.cell, params, self.input_data, init_carry=h)[1], self.target
            )
        )

        self.bptt_grads_multi_step = self.multi_step_loss_fn(self.params, h)["params"]
        # Correct for masking
        # if "wiring" in self.params and "mask" in self.params["wiring"]:
        #     mask = self.params["wiring"]["mask"]
        #     self.bptt_grads_one_step["W"] *= mask
        #     self.bptt_grads_multi_step["W"] *= mask

    def test_rtrl_one_step(self):
        """Test gradients for one step RTRL."""

        # Test RTRL gradients
        self.cell.plasticity = "rtrl"
        h = self.initialize_carry()
        grad_test = self.loss_fn(self.params, h)["params"]
        for key in grad_test:
            self.assertTrue(
                jax.numpy.allclose(
                    grad_test[key], self.bptt_grads_one_step[key], atol=A_TOL
                ),
                f"Gradients do not match for key {key}",
            )

    def test_rtrl_multi_step(self):
        """Test that gradients for multiple steps RTRL."""

        # Test RTRL gradients
        self.cell.plasticity = "rtrl"
        h = self.initialize_carry()
        grad_test = self.multi_step_loss_fn(self.params, h)["params"]
        # if "wiring" in self.params and "mask" in self.params["wiring"]:
        #     mask = self.params["wiring"]["mask"]
        #     grad_test["W"] *= mask
        for key in grad_test:
            self.assertTrue(
                jax.numpy.allclose(
                    grad_test[key], self.bptt_grads_multi_step[key], atol=A_TOL
                ),
                f"Gradients do not match for key {key}",
            )


class TestDEERLRCCell(unittest.TestCase):
    """Tests for the DEER plasticity mode of OnlineLRCCell.

    The DEER mode computes the full hidden-state sequence in parallel using
    quasi-Newton (diagonal DEER) iterations backed by
    ``jax.lax.associative_scan``.  Because the LRC ODE has an exactly diagonal
    Jacobian w.r.t. the hidden state, the quasi-Newton approximation is exact
    and the forward pass converges to the sequential result.
    """

    def setUp(self):
        self.num_units = 5
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        self.target = jax.random.normal(jax.random.PRNGKey(1), (self.num_units,))

        # Build a BPTT cell and initialise parameters once – these are shared
        # with the DEER cell (same architecture, different forward pass).
        bptt_cell = OnlineLRCCell(num_units=self.num_units, plasticity="bptt")
        self.params = bptt_cell.init(
            jax.random.PRNGKey(3), None, self.input_data[0]
        )

        # Sequential BPTT hidden states (ground truth)
        bptt_h0 = bptt_cell.apply(
            self.params,
            jax.random.PRNGKey(4),
            self.input_data[0].shape,
            method=bptt_cell.initialize_carry,
        )
        _, self.bptt_hs = scan_rnn(
            bptt_cell, self.params, self.input_data, init_carry=bptt_h0
        )
        # BPTT gradients (ground truth)
        self.bptt_grads = jax.grad(
            lambda p: mse_loss(
                scan_rnn(bptt_cell, p, self.input_data, init_carry=bptt_h0)[1],
                self.target,
            )
        )(self.params)["params"]

        # DEER cell – shares parameter structure with the BPTT cell
        self.deer_cell = OnlineLRCCell(num_units=self.num_units, plasticity="deer")
        self.deer_h0 = self.deer_cell.apply(
            self.params,
            jax.random.PRNGKey(4),
            self.input_data[0].shape,
            method=self.deer_cell.initialize_carry,
        )

    def test_deer_initialize_carry_shape(self):
        """DEER carry should be a plain hidden-state array, not a tuple."""
        self.assertEqual(self.deer_h0.shape, (self.num_units,))

    def test_deer_forward_matches_sequential(self):
        """DEER hidden states should match sequential (BPTT) computation."""
        _, deer_hs = self.deer_cell.apply(
            self.params, self.deer_h0, self.input_data
        )
        self.assertEqual(deer_hs.shape, (len(self.input_data), self.num_units))
        self.assertTrue(
            jnp.allclose(deer_hs, self.bptt_hs, atol=DEER_TOL),
            f"DEER forward pass deviates from sequential. Max diff: "
            f"{jnp.max(jnp.abs(deer_hs - self.bptt_hs)):.2e}",
        )

    def test_deer_forward_via_scan_rnn(self):
        """scan_rnn should bypass jax.lax.scan and call DEER cell directly."""
        _, deer_hs = scan_rnn(
            self.deer_cell, self.params, self.input_data, init_carry=self.deer_h0
        )
        self.assertEqual(deer_hs.shape, self.bptt_hs.shape)
        self.assertTrue(
            jnp.allclose(deer_hs, self.bptt_hs, atol=DEER_TOL),
            f"scan_rnn DEER deviates from sequential. Max diff: "
            f"{jnp.max(jnp.abs(deer_hs - self.bptt_hs)):.2e}",
        )

    def test_deer_gradients_match_bptt(self):
        """DEER gradients should be close to BPTT gradients."""
        deer_grads = jax.grad(
            lambda p: mse_loss(
                self.deer_cell.apply(p, self.deer_h0, self.input_data)[1],
                self.target,
            )
        )(self.params)["params"]

        for key in deer_grads:
            self.assertTrue(
                jnp.allclose(deer_grads[key], self.bptt_grads[key], atol=DEER_TOL),
                f"DEER gradient mismatch for '{key}'. Max diff: "
                f"{jnp.max(jnp.abs(deer_grads[key] - self.bptt_grads[key])):.2e}",
            )

    def test_deer_single_input_fallback(self):
        """DEER mode should also accept a single (non-sequence) input."""
        x_single = self.input_data[0]  # shape (3,)
        final_h, all_h = self.deer_cell.apply(
            self.params, self.deer_h0, x_single
        )
        # With a single input the output should be (1, num_units)
        self.assertEqual(all_h.shape, (1, self.num_units))
        self.assertEqual(final_h.shape, (self.num_units,))


if __name__ == "__main__":
    unittest.main()

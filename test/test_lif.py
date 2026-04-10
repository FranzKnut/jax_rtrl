"""Gradient tests for LIFCell and OnlineLIFCell.

Tests verify that:
  1. Forward passes run without error for both LIF and adLIF.
  2. BPTT (with surrogate gradients) computes non-trivial gradients.
  3. E-prop one-step gradients match BPTT one-step gradients for all learnable
     parameters (the RFLO trace captures the exact immediate Jacobian).
  4. E-prop multi-step gradients differ from BPTT multi-step gradients
     (e-prop is a causal approximation that ignores cross-time gradient paths).
"""

import unittest

import jax
import jax.numpy as jnp

from jax_rtrl.models.cells.lif import LIFCell, OnlineLIFCell
from jax_rtrl.models.seq_models import scan_rnn
from jax_rtrl.util.jax_util import mse_loss

A_TOL = 1e-5


class LIFGradientsTestBase(unittest.TestCase):
    """Base class: sets up cells, parameters and reference BPTT gradients."""

    # Subclasses can override these
    lif_type = "lif"
    num_units = 5
    n_in = 3
    time_steps = 8

    def setUp(self):
        key = jax.random.PRNGKey(0)
        key_data, key_target, key_params = jax.random.split(key, 3)

        self.input_data = jax.random.normal(
            key_data, (self.time_steps, self.n_in)
        )
        self.target = jax.random.normal(key_target, (self.num_units,))

        # ---- BPTT cell (LIFCell, uses surrogate gradients) ----
        self.bptt_cell = LIFCell(
            num_units=self.num_units, lif_type=self.lif_type
        )
        self.params = self.bptt_cell.init(
            key_params, None, self.input_data[0]
        )

        def _bptt_carry():
            return self.bptt_cell.apply(
                self.params,
                jax.random.PRNGKey(42),
                self.input_data[0].shape,
                method=self.bptt_cell.initialize_carry,
            )

        self._bptt_carry = _bptt_carry

        # One-step BPTT reference
        h0 = _bptt_carry()
        self.bptt_grads_one_step = jax.grad(
            lambda p, h: mse_loss(
                self.bptt_cell.apply(p, h, self.input_data[0])[1],
                self.target,
            )
        )(self.params, h0)["params"]

        # Multi-step BPTT reference
        self.bptt_grads_multi_step = jax.grad(
            lambda p, h: mse_loss(
                scan_rnn(self.bptt_cell, p, self.input_data, init_carry=h)[1],
                self.target,
            )
        )(self.params, h0)["params"]

        # ---- Online cell (OnlineLIFCell, default plasticity="eprop") ----
        self.eprop_cell = OnlineLIFCell(
            num_units=self.num_units, lif_type=self.lif_type, plasticity="eprop"
        )

        def _eprop_carry():
            return self.eprop_cell.apply(
                self.params,
                jax.random.PRNGKey(42),
                self.input_data[0].shape,
                method=self.eprop_cell.initialize_carry,
            )

        self._eprop_carry = _eprop_carry


class TestLIFGradients(LIFGradientsTestBase):
    """Gradient tests for classical LIF."""

    lif_type = "lif"

    # ------------------------------------------------------------------
    # Smoke tests
    # ------------------------------------------------------------------

    def test_forward_pass_bptt(self):
        """LIFCell forward pass runs without error."""
        h = self._bptt_carry()
        new_h, spikes = self.bptt_cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(spikes.shape, (self.num_units,))

    def test_forward_pass_eprop(self):
        """OnlineLIFCell (eprop) forward pass runs without error."""
        h = self._eprop_carry()
        new_h, spikes = self.eprop_cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(spikes.shape, (self.num_units,))

    # ------------------------------------------------------------------
    # BPTT gradients are non-trivial (surrogate actually fires)
    # ------------------------------------------------------------------

    def test_bptt_grads_nontrivial(self):
        """BPTT gradients are not all zero (surrogate gradient is active)."""
        any_nonzero = any(
            jnp.any(g != 0)
            for g in jax.tree.leaves(self.bptt_grads_one_step)
        )
        self.assertTrue(any_nonzero, "All BPTT gradients are zero")

    # ------------------------------------------------------------------
    # E-prop one-step should match BPTT one-step (exact for step 1)
    # ------------------------------------------------------------------

    def test_eprop_one_step(self):
        """E-prop one-step gradients match BPTT one-step gradients."""
        h = self._eprop_carry()
        eprop_grads = jax.grad(
            lambda p, carry: mse_loss(
                self.eprop_cell.apply(p, carry, self.input_data[0])[1],
                self.target,
            )
        )(self.params, h)["params"]

        for key in eprop_grads:
            self.assertTrue(
                jnp.allclose(
                    eprop_grads[key], self.bptt_grads_one_step[key], atol=A_TOL
                ),
                f"One-step gradients do not match for param '{key}':\n"
                f"  eprop = {eprop_grads[key]}\n"
                f"  bptt  = {self.bptt_grads_one_step[key]}",
            )

    # ------------------------------------------------------------------
    # E-prop multi-step should DIFFER from BPTT multi-step
    # ------------------------------------------------------------------

    def test_eprop_multi_step_differs_from_bptt(self):
        """E-prop multi-step gradients differ from BPTT (causal approx)."""
        h = self._eprop_carry()
        eprop_grads = jax.grad(
            lambda p, carry: mse_loss(
                scan_rnn(self.eprop_cell, p, self.input_data, init_carry=carry)[1],
                self.target,
            )
        )(self.params, h)["params"]

        # At least W_in or W_rec should differ (they have cross-time paths)
        any_differs = any(
            not jnp.allclose(
                eprop_grads[k], self.bptt_grads_multi_step[k], atol=A_TOL
            )
            for k in ("W_in", "W_rec")
        )
        self.assertTrue(
            any_differs,
            "E-prop multi-step gradients are identical to BPTT – "
            "the online approximation seems to be exact (unexpected).",
        )

    # ------------------------------------------------------------------
    # Scan runs without error
    # ------------------------------------------------------------------

    def test_scan_rnn_bptt(self):
        """scan_rnn works for LIFCell."""
        h = self._bptt_carry()
        _, spikes = scan_rnn(self.bptt_cell, self.params, self.input_data, init_carry=h)
        self.assertEqual(spikes.shape, (self.time_steps, self.num_units))

    def test_scan_rnn_eprop(self):
        """scan_rnn works for OnlineLIFCell (eprop)."""
        h = self._eprop_carry()
        _, spikes = scan_rnn(
            self.eprop_cell, self.params, self.input_data, init_carry=h
        )
        self.assertEqual(spikes.shape, (self.time_steps, self.num_units))


class TestAdLIFGradients(LIFGradientsTestBase):
    """Gradient tests for adaptive LIF (adLIF)."""

    lif_type = "adlif"

    def test_forward_pass_bptt(self):
        """adLIFCell forward pass runs without error."""
        h = self._bptt_carry()
        new_h, spikes = self.bptt_cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(spikes.shape, (self.num_units,))

    def test_forward_pass_eprop(self):
        """OnlineLIFCell (adLIF, eprop) forward pass runs without error."""
        h = self._eprop_carry()
        new_h, spikes = self.eprop_cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(spikes.shape, (self.num_units,))

    def test_bptt_grads_nontrivial(self):
        """BPTT gradients are not all zero for adLIF."""
        any_nonzero = any(
            jnp.any(g != 0)
            for g in jax.tree.leaves(self.bptt_grads_one_step)
        )
        self.assertTrue(any_nonzero, "All adLIF BPTT gradients are zero")

    def test_eprop_one_step(self):
        """E-prop one-step gradients match BPTT one-step gradients for adLIF."""
        h = self._eprop_carry()
        eprop_grads = jax.grad(
            lambda p, carry: mse_loss(
                self.eprop_cell.apply(p, carry, self.input_data[0])[1],
                self.target,
            )
        )(self.params, h)["params"]

        for key in eprop_grads:
            self.assertTrue(
                jnp.allclose(
                    eprop_grads[key], self.bptt_grads_one_step[key], atol=A_TOL
                ),
                f"adLIF one-step gradients do not match for param '{key}':\n"
                f"  eprop = {eprop_grads[key]}\n"
                f"  bptt  = {self.bptt_grads_one_step[key]}",
            )

    def test_eprop_multi_step_differs_from_bptt(self):
        """E-prop multi-step gradients differ from BPTT for adLIF."""
        h = self._eprop_carry()
        eprop_grads = jax.grad(
            lambda p, carry: mse_loss(
                scan_rnn(self.eprop_cell, p, self.input_data, init_carry=carry)[1],
                self.target,
            )
        )(self.params, h)["params"]

        any_differs = any(
            not jnp.allclose(
                eprop_grads[k], self.bptt_grads_multi_step[k], atol=A_TOL
            )
            for k in ("W_in", "W_rec")
        )
        self.assertTrue(
            any_differs,
            "adLIF e-prop multi-step gradients are identical to BPTT (unexpected).",
        )

    def test_scan_rnn_bptt(self):
        """scan_rnn works for adLIFCell."""
        h = self._bptt_carry()
        _, spikes = scan_rnn(self.bptt_cell, self.params, self.input_data, init_carry=h)
        self.assertEqual(spikes.shape, (self.time_steps, self.num_units))

    def test_scan_rnn_eprop(self):
        """scan_rnn works for OnlineLIFCell with adLIF and eprop."""
        h = self._eprop_carry()
        _, spikes = scan_rnn(
            self.eprop_cell, self.params, self.input_data, init_carry=h
        )
        self.assertEqual(spikes.shape, (self.time_steps, self.num_units))


class TestLIFSubsteps(unittest.TestCase):
    """Tests for LIFCell / OnlineLIFCell with sub-step integration (dt < T)."""

    num_units = 5
    n_in = 3
    time_steps = 6
    dt = 0.5
    T = 1.0  # 2 sub-steps per external timestep

    def setUp(self):
        key = jax.random.PRNGKey(7)
        key_data, key_target, key_params = jax.random.split(key, 3)

        self.input_data = jax.random.normal(
            key_data, (self.time_steps, self.n_in)
        )
        self.target = jax.random.normal(key_target, (self.num_units,))

        # Cell with dt=0.5 (2 sub-steps per external step)
        self.cell = LIFCell(num_units=self.num_units, dt=self.dt, T=self.T)
        self.params = self.cell.init(key_params, None, self.input_data[0])

        # Reference cell with dt=1.0 (single step per external step)
        self.cell_dt1 = LIFCell(num_units=self.num_units, dt=1.0, T=1.0)

        # Online cell with sub-steps
        self.eprop_cell = OnlineLIFCell(
            num_units=self.num_units, dt=self.dt, T=self.T, plasticity="eprop"
        )

    def _carry(self, cell, params):
        return cell.apply(
            params,
            jax.random.PRNGKey(0),
            self.input_data[0].shape,
            method=cell.initialize_carry,
        )

    # ------------------------------------------------------------------
    # Forward pass smoke tests
    # ------------------------------------------------------------------

    def test_bptt_substep_forward(self):
        """LIFCell with dt=0.5 runs forward pass without error."""
        h = self._carry(self.cell, self.params)
        new_h, spikes = self.cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(spikes.shape, (self.num_units,))

    def test_eprop_substep_forward(self):
        """OnlineLIFCell with dt=0.5 runs forward pass without error."""
        h = self._carry(self.eprop_cell, self.params)
        new_h, spikes = self.eprop_cell.apply(self.params, h, self.input_data[0])
        self.assertEqual(spikes.shape, (self.num_units,))

    # ------------------------------------------------------------------
    # Sub-step dynamics differ from single-step dynamics
    # ------------------------------------------------------------------

    def test_substep_differs_from_single_step(self):
        """BPTT gradients for dt=0.5 (2 sub-steps) differ from dt=1.0 (1 sub-step).

        Note: in the linear (no-spike) regime, T/dt Euler sub-steps with alpha^dt
        exactly reproduce the same final membrane potential as 1 full step with
        alpha (property of the alpha^dt scaling with the Euler method).
        The difference manifests in the gradients because BPTT has 2x the
        intermediate gradient paths for dt=0.5.
        """
        h_sub = self._carry(self.cell, self.params)
        grads_sub = jax.grad(
            lambda p, carry: mse_loss(
                self.cell.apply(p, carry, self.input_data[0])[1], self.target
            )
        )(self.params, h_sub)["params"]

        h_one = self._carry(self.cell_dt1, self.params)
        grads_one = jax.grad(
            lambda p, carry: mse_loss(
                self.cell_dt1.apply(p, carry, self.input_data[0])[1], self.target
            )
        )(self.params, h_one)["params"]

        # With 2 sub-steps, BPTT traverses 2 gradient paths, so grads must differ
        any_differs = any(
            not jnp.allclose(grads_sub[k], grads_one[k], atol=A_TOL)
            for k in grads_sub
        )
        self.assertTrue(
            any_differs,
            "dt=0.5 and dt=1.0 BPTT gradients are identical – "
            "dt scaling may not be applied.",
        )

    # ------------------------------------------------------------------
    # BPTT gradients are non-trivial
    # ------------------------------------------------------------------

    def test_bptt_substep_grads_nontrivial(self):
        """BPTT with dt=0.5 computes non-trivial gradients."""
        h = self._carry(self.cell, self.params)
        grads = jax.grad(
            lambda p, carry: mse_loss(
                self.cell.apply(p, carry, self.input_data[0])[1], self.target
            )
        )(self.params, h)["params"]
        any_nonzero = any(jnp.any(g != 0) for g in jax.tree.leaves(grads))
        self.assertTrue(any_nonzero, "All sub-step BPTT gradients are zero")

    # ------------------------------------------------------------------
    # E-prop gradients differ from BPTT for multiple sub-steps
    # (cross-sub-step gradient paths mean eprop is only an approximation)
    # ------------------------------------------------------------------

    def test_eprop_multi_substep_differs_from_bptt(self):
        """E-prop grads differ from BPTT grads when T/dt > 1 (expected)."""
        h_bptt = self._carry(self.cell, self.params)
        bptt_grads = jax.grad(
            lambda p, carry: mse_loss(
                self.cell.apply(p, carry, self.input_data[0])[1], self.target
            )
        )(self.params, h_bptt)["params"]

        h_eprop = self._carry(self.eprop_cell, self.params)
        eprop_grads = jax.grad(
            lambda p, carry: mse_loss(
                self.eprop_cell.apply(p, carry, self.input_data[0])[1], self.target
            )
        )(self.params, h_eprop)["params"]

        # With 2 sub-steps, cross-sub-step gradient paths mean eprop != BPTT
        any_differs = any(
            not jnp.allclose(eprop_grads[k], bptt_grads[k], atol=A_TOL)
            for k in ("W_in", "W_rec")
        )
        self.assertTrue(
            any_differs,
            "E-prop and BPTT grads are identical for 2 sub-steps – "
            "cross-step approximation seems exact (unexpected).",
        )

    # ------------------------------------------------------------------
    # scan_rnn compatibility
    # ------------------------------------------------------------------

    def test_scan_rnn_substep_bptt(self):
        """scan_rnn works for LIFCell with dt=0.5."""
        h = self._carry(self.cell, self.params)
        _, spikes = scan_rnn(self.cell, self.params, self.input_data, init_carry=h)
        self.assertEqual(spikes.shape, (self.time_steps, self.num_units))

    def test_scan_rnn_substep_eprop(self):
        """scan_rnn works for OnlineLIFCell with dt=0.5."""
        h = self._carry(self.eprop_cell, self.params)
        _, spikes = scan_rnn(
            self.eprop_cell, self.params, self.input_data, init_carry=h
        )
        self.assertEqual(spikes.shape, (self.time_steps, self.num_units))


if __name__ == "__main__":
    unittest.main()

import unittest

import jax
from jax_rtrl.models.cells.ctrnn import CTRNNCell, OnlineCTRNNCell
from jax_rtrl.models.seq_models import scan_rnn
from jax_rtrl.util.jax_util import mse_loss

A_TOL = 1e-5

# jax.config.update("jax_disable_jit", True)


class TestCTRNNGradients(unittest.TestCase):
    def setUp(self):
        self.cell = OnlineCTRNNCell(
            num_units=5,
            plasticity="bptt",
            ode_type="murray",
            wiring="diagonal",
        )
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
                scan_rnn(self.cell, params, h, False, self.input_data)[1], self.target
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

    def test_rflo_one_step(self):
        """Test that gradients for one step RFLO."""

        # Test RTRL gradients
        self.cell.plasticity = "rflo"
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

    def test_rflo_multi_step(self):
        """Test that gradients for multiple steps RFLO."""

        # Test RTRL gradients
        self.cell.plasticity = "rflo"
        h = self.initialize_carry()
        grad_test = self.multi_step_loss_fn(self.params, h)["params"]
        for key in grad_test:
            self.assertTrue(
                jax.numpy.allclose(
                    grad_test[key], self.bptt_grads_multi_step[key], atol=A_TOL
                ),
                f"Gradients do not match for key {key}",
            )

    def test_eprop_one_step(self):
        """Test that gradients for one step e-prop."""

        # Test e-prop gradients
        self.cell.plasticity = "eprop"
        h = self.initialize_carry()
        grad_test = self.loss_fn(self.params, h)["params"]
        for key in grad_test:
            self.assertTrue(
                jax.numpy.allclose(
                    grad_test[key], self.bptt_grads_one_step[key], atol=A_TOL
                ),
                f"Gradients do not match for key {key}",
            )

    def test_eprop_multi_step(self):
        """Test that gradients for multiple steps e-prop."""

        # Test e-prop gradients
        self.cell.plasticity = "eprop"
        h = self.initialize_carry()
        grad_test = self.multi_step_loss_fn(self.params, h)["params"]
        for key in grad_test:
            self.assertTrue(
                jax.numpy.allclose(
                    grad_test[key], self.bptt_grads_multi_step[key], atol=A_TOL
                ),
                f"Gradients do not match for key {key}",
            )


if __name__ == "__main__":
    unittest.main()

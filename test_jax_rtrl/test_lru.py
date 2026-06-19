import unittest

import jax
import jax.numpy as jnp
from jax_rtrl.models.cells.lru import OnlineLRULayer
from jax_rtrl.models.seq_models import scan_rnn
from jax_rtrl.util.jax_util import get_keystr, mse_loss

A_TOL = 1e-5

# jax.config.update("jax_disable_jit", True)


def flatten_params(params):
    """Flatten the given params dictionary."""
    flattened, _ = jax.tree_util.tree_flatten_with_path(params)
    return {get_keystr(k): v for k, v in flattened}


class LRUGradientsTest(unittest.TestCase):
    """Base class for LRU gradient tests with common setup logic."""

    def get_cell_kwargs(self):
        """Override in subclasses to provide specific cell parameters."""
        return {
            "d_hidden": 5,
            "d_output": 4,
            "plasticity": "bptt",
        }

    def setUp(self):
        self.cell = OnlineLRULayer(**self.get_cell_kwargs())
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        self.target = jax.random.normal(jax.random.PRNGKey(1), (4,))

        self.params = self.cell.init(jax.random.PRNGKey(3), None, self.input_data[0])

        # Now get the actual initial carry
        self.initialize_carry = lambda: self.cell.apply(
            self.params,
            jax.random.PRNGKey(4),
            self.input_data[0].shape,
            method=self.cell.initialize_carry,
        )
        h = self.initialize_carry()

        # LRU returns complex outputs, take real part for loss
        # Compute reference grads with BPTT (parallel scan)
        grads = jax.grad(
            lambda params: mse_loss(
                jnp.real(self.cell.apply(params, h, self.input_data)[1]),
                self.target,
            )
        )(self.params)

        # Flatten nested params structure
        self.bptt_grads = flatten_params(grads["params"])

    def test_rtrl_multi_step(self):
        """Test gradients for multiple steps RTRL."""

        # Test RTRL gradients
        self.cell.plasticity = "rtrl"
        h = self.initialize_carry()
        _g = jax.grad(
            lambda params: mse_loss(
                jnp.real(scan_rnn(self.cell, params, self.input_data, init_carry=h)[1]),
                self.target,
            )
        )(self.params)
        grad_test = flatten_params(_g["params"])
        for key in grad_test:
            self.assertTrue(
                jax.numpy.allclose(grad_test[key], self.bptt_grads[key], atol=A_TOL),
                f"Gradients do not match for key {key}",
            )


if __name__ == "__main__":
    unittest.main()

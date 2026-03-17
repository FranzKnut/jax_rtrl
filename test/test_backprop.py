"""Tests that gradients backpropagate to the first layer in a two-layer RNN
when RTRL or RFLO plasticity is used on each layer cell.
"""

from functools import partial
import unittest

import jax
import jax.numpy as jnp

from jax_rtrl.models.cells.ctrnn import OnlineCTRNNCell
from jax_rtrl.models.cells.lrc import OnlineLRCCell
from jax_rtrl.models.cells.ltc import OnlineLTCCell
from jax_rtrl.models.seq_models import MultiLayerRNN, scan_rnn
from jax_rtrl.util.jax_util import mse_loss

A_TOL = 1e-5
_T, _INPUT_SIZE, _HIDDEN, _NUM_LAYERS = 5, 3, 6, 3


class _MultiLayerGradFlowBase(unittest.TestCase):
    """Base class for multi-layer gradient flow tests."""

    rnn_cls = None
    extra_kwargs = {}
    plasticities = []

    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.input_data = jax.random.normal(jax.random.PRNGKey(1), (_T, _INPUT_SIZE))
        self.target = jax.random.normal(jax.random.PRNGKey(2), (_HIDDEN,))
        x0 = self.input_data[0]
        self.bptt_model = MultiLayerRNN(
            sizes=[_HIDDEN] * _NUM_LAYERS,
            rnn_cls=self.rnn_cls,
            rnn_kwargs={"plasticity": "bptt", **self.extra_kwargs},
        )
        self.params = self.bptt_model.init(self.rng, None, x0)

        self.bptt_grads = jax.grad(
            partial(self._loss, self.bptt_model),
        )(self.params)["params"]["layer_0"]

        for i, g in enumerate(jax.tree.leaves(self.bptt_grads)):
            assert jnp.all(jnp.not_equal(g, 0)), f"BPTT zero grad at leaf {i}."

        self.online_models = {
            p: MultiLayerRNN(
                sizes=[_HIDDEN] * _NUM_LAYERS,
                rnn_cls=self.rnn_cls,
                rnn_kwargs={"plasticity": p, **self.extra_kwargs},
            )
            for p in self.plasticities
        }
        self.online_params = {
            p: self.online_models[p].init(self.rng, None, x0) for p in self.plasticities
        }

    def _loss(self, m: MultiLayerRNN, p):
        carry = m.apply(
            self.params,
            self.rng,
            self.input_data[0].shape,
            method=m.initialize_carry,
        )
        _, y_hats = scan_rnn(m, p, self.input_data, init_carry=carry)
        return mse_loss(y_hats, self.target)

    def _check_grad_flow(self, plasticity):
        """Assert that layer_0 parameters receive non-zero gradients."""

        online_grads = jax.grad(
            partial(
                self._loss,
                self.online_models[plasticity],
            )
        )(self.online_params[plasticity])["params"]["layer_0"]

        for i, g in enumerate(jax.tree.leaves(online_grads)):
            self.assertFalse(jnp.all(g == 0), f"{plasticity} zero grad at leaf {i}.")
        return online_grads

    def _check_matches_bptt(self, plasticity):
        """Assert that layer_0 gradients match BPTT."""
        online_grads = self._check_grad_flow(plasticity)
        for i, (g_online, g_bptt) in enumerate(
            zip(jax.tree.leaves(online_grads), jax.tree.leaves(self.bptt_grads))
        ):
            self.assertTrue(
                jnp.allclose(g_online, g_bptt, atol=A_TOL),
                f"{plasticity} grad mismatch at leaf {i}.",
            )


class TestCTRNNMultiLayer(_MultiLayerGradFlowBase):
    rnn_cls = OnlineCTRNNCell
    extra_kwargs = {"ode_type": "murray"}
    plasticities = ["rtrl", "rflo"]

    def test_rtrl_grad_flow(self):
        self._check_grad_flow("rtrl")

    def test_rflo_grad_flow(self):
        self._check_grad_flow("rflo")

    # def test_rtrl_matches_bptt(self):
    # # They won't
    #     self._check_matches_bptt("rtrl")


class TestLRCMultiLayer(_MultiLayerGradFlowBase):
    rnn_cls = OnlineLRCCell
    plasticities = ["rtrl"]

    def test_rtrl_grad_flow(self):
        self._check_grad_flow("rtrl")

    # def test_rtrl_matches_bptt(self):
    # # They won't
    #     self._check_matches_bptt("rtrl")


class TestLTCMultiLayer(_MultiLayerGradFlowBase):
    rnn_cls = OnlineLTCCell
    plasticities = ["rtrl", "rflo"]

    def test_rtrl_grad_flow(self):
        self._check_grad_flow("rtrl")

    def test_rflo_grad_flow(self):
        self._check_grad_flow("rflo")

    def test_rtrl_matches_bptt(self):
        self._check_matches_bptt("rtrl")


if __name__ == "__main__":
    unittest.main()
